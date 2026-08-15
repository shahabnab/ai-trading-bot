#!/usr/bin/env python3
"""Export a Git-friendly daily audit of every paper-trading algorithm.

Only local PAPER state and public API responses are exported. Credentials/.env are never read.
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

BRAIN_IDS = ("trader-brain-v1", "trader-brain-bandit-v1")
TABLES = (
    "paper_model_accounts", "paper_model_positions", "paper_model_trades", "paper_model_decisions",
    "paper_model_daily_risk_state", "trader_brain_experiences",
)


def http_json(url: str) -> dict[str, Any] | None:
    try:
        with urlopen(Request(url, headers={"User-Agent": "ai-trading-paper-snapshot/2.0"}), timeout=10) as response:  # noqa: S310 localhost operator input
            value = json.loads(response.read().decode("utf-8"))
            return value if isinstance(value, dict) else None
    except (OSError, URLError, json.JSONDecodeError):
        return None


def rows(conn: sqlite3.Connection, table: str) -> list[dict[str, Any]]:
    exists = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
    if not exists: return []
    return [dict(row) for row in conn.execute(f'SELECT * FROM "{table}"').fetchall()]


def write_csv(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not values:
        path.write_text("", encoding="utf-8"); return
    columns: list[str] = []
    for row in values:
        for key in row:
            if key not in columns: columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns); writer.writeheader(); writer.writerows(values)


def git_commit(repo_root: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError): return None


def money(value: Any) -> str:
    try: return f"{float(value):,.2f}"
    except (TypeError, ValueError): return "—"


def pct(value: Any) -> str:
    try: return f"{float(value)*100:+.2f}%"
    except (TypeError, ValueError): return "—"


def merge_models(api_base: str) -> dict[str, Any] | None:
    base = http_json(f"{api_base.rstrip('/')}/api/paper/models") or {"models": []}
    models = base.get("models") if isinstance(base.get("models"), list) else []
    known = {str(item.get("model_id")) for item in models if isinstance(item, dict)}
    for model_id in BRAIN_IDS:
        if model_id in known: continue
        item = http_json(f"{api_base.rstrip('/')}/api/paper/models/{model_id}")
        if isinstance(item, dict): models.append(item)
    base["models"] = models
    return base


def markdown(snapshot: dict[str, Any]) -> str:
    models_payload = snapshot.get("api", {}).get("models") or {}
    models = models_payload.get("models", []) if isinstance(models_payload, dict) else []
    lines = ["# Daily AI trading algorithm report", "", f"Generated: `{snapshot['generated_at_utc']}`  ", f"Runtime commit: `{snapshot.get('runtime_git_commit') or 'unknown'}`  ", "Real-order execution: **DISABLED**", ""]
    if models:
        lines += ["| Algorithm | Family | Equity | P/L | Return | Fills | Closed | Win rate | Fees | Latest |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
        for model in sorted(models, key=lambda m: float((m.get("portfolio") or {}).get("total_return", 0)), reverse=True):
            portfolio = model.get("portfolio") or {}; perf = model.get("performance") or {}; latest = model.get("latest_decision") or {}
            win = "—" if perf.get("win_rate") is None else f"{float(perf['win_rate'])*100:.1f}%"
            lines.append(f"| {model.get('display_name', model.get('model_id'))} | {model.get('algorithm_family', model.get('driver','—'))} | €{money(portfolio.get('portfolio_value_usdt'))} | €{money(portfolio.get('total_pnl_usdt'))} | {pct(portfolio.get('total_return'))} | {perf.get('trade_count',0)} | {perf.get('closed_trades',0)} | {win} | €{money(perf.get('total_fees_usdt'))} | {latest.get('signal','WAIT')} |")
        lines.append("")
    runtime = snapshot.get("trader_brain_runtime")
    if isinstance(runtime, dict) and isinstance(runtime.get("models"), list):
        lines += ["## Trader-Brain diagnostics", ""]
        for model in runtime["models"]:
            if not isinstance(model, dict): continue
            gate = model.get("gate") or {}; regime = model.get("regime") or {}; learning = model.get("learning") or {}
            lines += [f"### {model.get('display_name', model.get('model_id'))}", "", f"- Policy: `{model.get('policy_source','unknown')}`", f"- Gate: `{gate.get('gate_type','unknown')}`", f"- P(up/flat/down): `{float(gate.get('p_up',0)):.3f}` / `{float(gate.get('p_flat',0)):.3f}` / `{float(gate.get('p_down',0)):.3f}`", f"- Regime entropy: `{float(regime.get('entropy',0)):.3f}`", f"- Experiences/resolved: `{learning.get('experience_count',0)}` / `{learning.get('resolved_count',0)}`", f"- Requested/executed: `{model.get('requested_action','—')}` / `{model.get('executed_action','—')}`", ""]
    lines += ["## Audit files", "", "- `snapshot.json`: full captured PAPER state and latest Trader-Brain diagnostics.", "- `models/<model_id>/trades.csv`: simulated fills.", "- `models/<model_id>/decisions.csv`: every model decision/reason.", "- `models/<model_id>/positions.csv`: open paper positions.", "- `models/<model_id>/experiences.csv`: Trader-Brain state/action/reward rows when applicable.", "", "No API keys or `.env` values are exported.", ""]
    return "\n".join(lines)


def export_snapshot(repo_root: Path, db_path: Path, out_dir: Path, api_base: str) -> Path:
    now = datetime.now(timezone.utc); day_dir = out_dir / now.date().isoformat(); day_dir.mkdir(parents=True, exist_ok=True)
    tables: dict[str, list[dict[str, Any]]] = {name: [] for name in TABLES}
    if db_path.exists():
        conn = sqlite3.connect(db_path); conn.row_factory = sqlite3.Row; conn.execute("PRAGMA busy_timeout=5000")
        try:
            conn.execute("BEGIN")
            tables = {name: rows(conn, name) for name in TABLES}
        finally:
            conn.rollback(); conn.close()
    api = {"health": http_json(f"{api_base.rstrip('/')}/health"), "models": merge_models(api_base), "btc_quote": http_json(f"{api_base.rstrip('/')}/api/market/BTCUSDT")}
    runtime_path = repo_root / "state" / "trader_brain" / "latest.json"
    try: runtime = json.loads(runtime_path.read_text(encoding="utf-8")) if runtime_path.is_file() else None
    except (OSError, json.JSONDecodeError): runtime = None
    snapshot = {"schema_version": 2, "generated_at_utc": now.isoformat(), "runtime_git_commit": git_commit(repo_root), "api": api, "trader_brain_runtime": runtime, "database": {"tables": tables}}
    payload = json.dumps(snapshot, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    (day_dir / "snapshot.json").write_text(payload, encoding="utf-8"); (day_dir / "summary.md").write_text(markdown(snapshot), encoding="utf-8")
    out_dir.mkdir(parents=True, exist_ok=True); (out_dir / "latest.json").write_text(payload, encoding="utf-8"); (out_dir / "LATEST.md").write_text(markdown(snapshot), encoding="utf-8")
    accounts = tables.get("paper_model_accounts", [])
    for account in accounts:
        model_id = str(account.get("model_id", "unknown")); model_dir = day_dir / "models" / model_id
        write_csv(model_dir / "trades.csv", [r for r in tables.get("paper_model_trades", []) if str(r.get("model_id")) == model_id])
        write_csv(model_dir / "decisions.csv", [r for r in tables.get("paper_model_decisions", []) if str(r.get("model_id")) == model_id])
        write_csv(model_dir / "positions.csv", [r for r in tables.get("paper_model_positions", []) if str(r.get("model_id")) == model_id])
        write_csv(model_dir / "experiences.csv", [r for r in tables.get("trader_brain_experiences", []) if str(r.get("model_id")) == model_id])
    return day_dir


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--repo-root", default="."); parser.add_argument("--db", default="data/paper_trading.sqlite3"); parser.add_argument("--out-dir", default="paper_snapshots"); parser.add_argument("--api-base", default="http://127.0.0.1:8000"); args = parser.parse_args()
    root = Path(args.repo_root).resolve(); db = Path(args.db); out = Path(args.out_dir)
    if not db.is_absolute(): db = root / db
    if not out.is_absolute(): out = root / out
    print(f"Paper snapshot exported: {export_snapshot(root, db, out, args.api_base)}")
    return 0


if __name__ == "__main__": raise SystemExit(main())
