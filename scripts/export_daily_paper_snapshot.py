#!/usr/bin/env python3
"""Export a Git-friendly daily snapshot of the forward paper experiment.

The exporter intentionally captures paper-trading state only. It never reads
.env, CoinEx credentials, or private account balances. The local SQLite file
remains the canonical runtime store; JSON/CSV/Markdown snapshots provide a
human-readable, auditable history that can safely be committed to Git.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen


PAPER_TABLES = (
    "paper_model_accounts",
    "paper_model_positions",
    "paper_model_trades",
    "paper_model_decisions",
    "paper_model_daily_risk_state",
    # Legacy single-account tables are included when present so no old paper
    # history is silently lost during the transition to isolated model ledgers.
    "paper_account",
    "paper_positions",
    "paper_trades",
    "paper_decisions",
    "paper_daily_risk_state",
)


def _http_json(url: str) -> dict[str, Any] | None:
    try:
        req = Request(url, headers={"User-Agent": "ai-trading-paper-snapshot/1.0"})
        with urlopen(req, timeout=10) as response:  # noqa: S310 - localhost URL is operator supplied
            return json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, json.JSONDecodeError):
        return None


def _git_commit(repo_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _table_rows(conn: sqlite3.Connection, table: str) -> list[dict[str, Any]]:
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?", (table,)
    ).fetchone()
    if not exists:
        return []
    rows = conn.execute(f'SELECT * FROM "{table}"').fetchall()
    return [dict(row) for row in rows]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _money(value: Any) -> str:
    try:
        return f"{float(value):,.2f}"
    except (TypeError, ValueError):
        return "—"


def _percent(value: Any) -> str:
    try:
        return f"{float(value) * 100:+.2f}%"
    except (TypeError, ValueError):
        return "—"


def _summary_markdown(snapshot: dict[str, Any]) -> str:
    generated = snapshot["generated_at_utc"]
    models_payload = snapshot.get("api", {}).get("models") or {}
    models = models_payload.get("models", []) if isinstance(models_payload, dict) else []
    lines = [
        "# Daily forward paper-trading snapshot",
        "",
        f"Generated: `{generated}`  ",
        f"Runtime git commit: `{snapshot.get('runtime_git_commit') or 'unknown'}`  ",
        "Real-order execution: **DISABLED**",
        "",
    ]
    if not models:
        lines.extend(
            [
                "> The FastAPI model endpoint was unavailable when this snapshot was made. ",
                "> Raw SQLite paper tables are still included in `snapshot.json`.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "| Model | Capital | P/L | Return | Trades | Closed | Win rate | Fees | Latest signal |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for model in models:
            portfolio = model.get("portfolio") or {}
            performance = model.get("performance") or {}
            latest = model.get("latest_decision") or {}
            win_rate = performance.get("win_rate")
            win_text = "—" if win_rate is None else f"{float(win_rate) * 100:.1f}%"
            lines.append(
                "| {name} | €{capital} | €{pnl} | {ret} | {trades} | {closed} | {win} | €{fees} | {signal} |".format(
                    name=model.get("short_name") or model.get("display_name") or model.get("model_id"),
                    capital=_money(portfolio.get("portfolio_value_usdt")),
                    pnl=_money(portfolio.get("total_pnl_usdt")),
                    ret=_percent(portfolio.get("total_return")),
                    trades=performance.get("trade_count", 0),
                    closed=performance.get("closed_trades", 0),
                    win=win_text,
                    fees=_money(performance.get("total_fees_usdt")),
                    signal=latest.get("signal", "WAITING"),
                )
            )
        lines.append("")
    btc = snapshot.get("api", {}).get("btc_quote")
    if isinstance(btc, dict):
        lines.extend(
            [
                f"BTCUSDT at snapshot: **{_money(btc.get('last'))} USDT**",
                "",
            ]
        )
    lines.extend(
        [
            "## Audit files",
            "",
            "- `snapshot.json`: complete captured paper state for this day.",
            "- `models/<model_id>/trades.csv`: that model's simulated fills.",
            "- `models/<model_id>/decisions.csv`: every recorded BUY/HOLD/SELL decision.",
            "- `models/<model_id>/positions.csv`: open simulated positions at export time.",
            "",
            "The SQLite database on the server remains the runtime source of truth. No API keys or `.env` values are exported.",
            "",
        ]
    )
    return "\n".join(lines)


def export_snapshot(repo_root: Path, db_path: Path, out_dir: Path, api_base: str) -> Path:
    now = datetime.now(timezone.utc)
    day = now.date().isoformat()
    day_dir = out_dir / day
    day_dir.mkdir(parents=True, exist_ok=True)

    tables: dict[str, list[dict[str, Any]]] = {}
    if db_path.exists():
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=5000")
        try:
            # A read transaction gives all table exports one consistent SQLite view.
            conn.execute("BEGIN")
            for table in PAPER_TABLES:
                tables[table] = _table_rows(conn, table)
        finally:
            conn.rollback()
            conn.close()
    else:
        for table in PAPER_TABLES:
            tables[table] = []

    api = {
        "health": _http_json(f"{api_base.rstrip('/')}/health"),
        "models": _http_json(f"{api_base.rstrip('/')}/api/paper/models"),
        "btc_quote": _http_json(f"{api_base.rstrip('/')}/api/market/BTCUSDT"),
    }

    snapshot: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": now.isoformat(),
        "runtime_git_commit": _git_commit(repo_root),
        "database_path": str(db_path.relative_to(repo_root)) if db_path.is_relative_to(repo_root) else str(db_path),
        "api": api,
        "database": {"tables": tables},
    }

    model_accounts = tables.get("paper_model_accounts", [])
    for account in model_accounts:
        model_id = str(account.get("model_id", "unknown"))
        model_dir = day_dir / "models" / model_id
        trades = [row for row in tables.get("paper_model_trades", []) if row.get("model_id") == model_id]
        decisions = [row for row in tables.get("paper_model_decisions", []) if row.get("model_id") == model_id]
        positions = [row for row in tables.get("paper_model_positions", []) if row.get("model_id") == model_id]
        _write_csv(model_dir / "trades.csv", trades)
        _write_csv(model_dir / "decisions.csv", decisions)
        _write_csv(model_dir / "positions.csv", positions)

    payload = json.dumps(snapshot, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    (day_dir / "snapshot.json").write_text(payload, encoding="utf-8")
    summary = _summary_markdown(snapshot)
    (day_dir / "summary.md").write_text(summary, encoding="utf-8")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "latest.json").write_text(payload, encoding="utf-8")
    (out_dir / "LATEST.md").write_text(summary, encoding="utf-8")
    return day_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--db", default="data/paper_trading.sqlite3")
    parser.add_argument("--out-dir", default="paper_snapshots")
    parser.add_argument("--api-base", default="http://127.0.0.1:8000")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    db_path = Path(args.db)
    if not db_path.is_absolute():
        db_path = repo_root / db_path
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir

    exported = export_snapshot(repo_root, db_path, out_dir, args.api_base)
    print(f"Paper snapshot exported: {exported}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
