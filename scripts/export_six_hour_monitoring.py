#!/usr/bin/env python3
"""Export a compact six-hour observability snapshot for PAPER trading.

The snapshot is designed for machine review (including ChatGPT audits). It joins:
- current paper-account performance for every algorithm,
- real CoinEx BTC candles used as outcome truth,
- short-term decision/outcome telemetry,
- frozen V3 forward predictions scored after their intended horizon,
- Trader-Brain forward experiences/rewards,
- the latest server-ops status when available.

No API keys or .env values are read or exported.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable
from urllib.error import URLError
from urllib.request import Request, urlopen

HOUR_MS = 60 * 60 * 1000
MIN15_MS = 15 * 60 * 1000
BRAIN_IDS = ("trader-brain-v1", "trader-brain-bandit-v1")


def http_json(url: str) -> dict[str, Any] | None:
    try:
        request = Request(url, headers={"User-Agent": "ai-trading-six-hour-monitor/1.0"})
        with urlopen(request, timeout=15) as response:  # noqa: S310 localhost operator input
            value = json.loads(response.read().decode("utf-8"))
            return value if isinstance(value, dict) else None
    except (OSError, URLError, json.JSONDecodeError):
        return None


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def read_jsonl(path: Path, limit: int = 2000) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    rows: list[dict[str, Any]] = []
    for line in lines[-max(1, limit):]:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def git_commit(repo_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if number == number and abs(number) != float("inf") else default
    except (TypeError, ValueError):
        return default


def to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_iso(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if result.tzinfo is None:
            result = result.replace(tzinfo=UTC)
        return result.astimezone(UTC)
    except ValueError:
        return None


def compact_candle(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "created_at": to_int(row.get("created_at", row.get("timestamp"))),
        "open": to_float(row.get("open")),
        "high": to_float(row.get("high")),
        "low": to_float(row.get("low")),
        "close": to_float(row.get("close")),
        "volume": to_float(row.get("volume")),
        "value": to_float(row.get("value", row.get("quote_volume"))),
    }


def candle_rows(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    rows = (payload or {}).get("candles")
    if not isinstance(rows, list):
        return []
    result = [compact_candle(row) for row in rows if isinstance(row, dict)]
    return sorted((row for row in result if row["created_at"] > 0 and row["close"] > 0), key=lambda row: row["created_at"])


def close_by_end(candles: Iterable[dict[str, Any]], interval_ms: int) -> dict[int, float]:
    return {to_int(row["created_at"]) + interval_ms: to_float(row["close"]) for row in candles if to_float(row["close"]) > 0}


def close_at_or_before(end_map: dict[int, float], timestamp: int) -> tuple[int, float] | None:
    candidates = [ts for ts in end_map if ts <= timestamp]
    if not candidates:
        return None
    ts = max(candidates)
    return ts, end_map[ts]


def market_truth(c15: list[dict[str, Any]], c1h: list[dict[str, Any]], now: datetime) -> dict[str, Any]:
    end15 = close_by_end(c15, MIN15_MS)
    if not end15:
        return {"available": False, "reason": "No 15-minute CoinEx candles returned."}
    latest_end = max(end15)
    latest_close = end15[latest_end]

    def return_hours(hours: float) -> float | None:
        previous = close_at_or_before(end15, latest_end - int(hours * HOUR_MS))
        if previous is None or previous[1] <= 0:
            return None
        return latest_close / previous[1] - 1.0

    last_24 = [row for row in c15 if row["created_at"] + MIN15_MS > latest_end - 24 * HOUR_MS]
    last_72h = [row for row in c1h if row["created_at"] + HOUR_MS > latest_end - 72 * HOUR_MS]
    highs = [to_float(row["high"]) for row in last_24 if to_float(row["high"]) > 0]
    lows = [to_float(row["low"]) for row in last_24 if to_float(row["low"]) > 0]
    freshness = max(0.0, now.timestamp() - latest_end / 1000.0)
    return {
        "available": True,
        "latest_completed_candle_end": latest_end,
        "latest_close": latest_close,
        "freshness_seconds": freshness,
        "return_1h": return_hours(1),
        "return_2h": return_hours(2),
        "return_3h": return_hours(3),
        "return_4h": return_hours(4),
        "return_6h": return_hours(6),
        "return_12h": return_hours(12),
        "return_24h": return_hours(24),
        "high_24h": max(highs) if highs else None,
        "low_24h": min(lows) if lows else None,
        # Compact real-market truth retained with every six-hour audit. This is
        # intentionally OHLCV only; raw tick/order-book history stays on the VPS.
        "candles_15m_last_24h": last_24,
        "candles_1h_last_72h": last_72h,
    }


def merge_models(api_base: str) -> dict[str, Any]:
    base = http_json(f"{api_base.rstrip('/')}/api/paper/models") or {"models": []}
    models = base.get("models") if isinstance(base.get("models"), list) else []
    known = {str(item.get("model_id")) for item in models if isinstance(item, dict)}
    for model_id in BRAIN_IDS:
        if model_id in known:
            continue
        item = http_json(f"{api_base.rstrip('/')}/api/paper/models/{model_id}")
        if isinstance(item, dict):
            models.append(item)
    base["models"] = models
    return base


def summarize_short_term(repo_root: Path) -> dict[str, Any]:
    root = repo_root / "state" / "short_term"
    decisions = read_jsonl(root / "decision_diagnostics.jsonl", 4000)
    outcomes = read_jsonl(root / "decision_outcomes.jsonl", 4000)
    result: dict[str, Any] = {
        "decision_rows": len(decisions),
        "outcome_rows": len(outcomes),
        "models": {},
        "latest_decisions": decisions[-40:],
        "latest_outcomes": outcomes[-40:],
    }
    ids = sorted({str(row.get("model_id")) for row in decisions + outcomes if row.get("model_id")})
    for model_id in ids:
        ds = [row for row in decisions if str(row.get("model_id")) == model_id]
        os_ = [row for row in outcomes if str(row.get("model_id")) == model_id]
        classes = Counter(str(row.get("classification", "UNKNOWN")) for row in os_)
        result["models"][model_id] = {
            "decisions": len(ds),
            "resolved": len(os_),
            "pending": max(0, len(ds) - len(os_)),
            "setup_candidates": sum(bool(row.get("setup_ready")) for row in ds),
            "entry_signals": sum(str(row.get("decision_action")) == "ENTER_LONG" for row in ds),
            "classifications": dict(classes),
            "latest_decision": ds[-1] if ds else None,
            "latest_outcome": os_[-1] if os_ else None,
        }
    return result


def score_v3(repo_root: Path, models_payload: dict[str, Any], hour_candles: list[dict[str, Any]]) -> dict[str, Any]:
    predictions = read_jsonl(repo_root / "state" / "forward_v3" / "predictions.jsonl", 4000)
    models = models_payload.get("models") if isinstance(models_payload.get("models"), list) else []
    horizons = {str(m.get("model_id")): max(1, to_int(m.get("horizon_hours"), 1)) for m in models if isinstance(m, dict)}
    closes = close_by_end(hour_candles, HOUR_MS)
    latest_end = max(closes) if closes else 0
    scored: list[dict[str, Any]] = []
    for row in predictions:
        if not bool(row.get("policy_due")):
            continue
        model_id = str(row.get("model_id", ""))
        horizon = horizons.get(model_id, max(1, to_int(row.get("horizon_commitment_hours"), 1)))
        feature_ts = to_int(row.get("feature_timestamp"))
        target_ts = feature_ts + horizon * HOUR_MS
        if target_ts <= 0 or target_ts > latest_end:
            continue
        target = close_at_or_before(closes, target_ts)
        reference = to_float(row.get("paper_market_price"))
        if target is None or reference <= 0:
            continue
        actual_return = target[1] / reference - 1.0
        cost = max(0.0, to_float(row.get("one_way_cost_rate")))
        signal = str(row.get("signal", ""))
        position_before = str(row.get("position_before", "CASH"))
        if signal == "BUY":
            classification = "GOOD_ENTRY" if actual_return - 2.0 * cost > 0 else "BAD_ENTRY"
        elif signal == "SELL":
            classification = "GOOD_EXIT" if actual_return <= 0 else "EARLY_EXIT"
        elif position_before == "CASH":
            classification = "MISSED_LONG" if actual_return - 2.0 * cost > 0 else "GOOD_HOLD"
        else:
            classification = "POSITION_HOLD"
        scored.append({
            "model_id": model_id,
            "feature_timestamp": feature_ts,
            "target_timestamp": target_ts,
            "horizon_hours": horizon,
            "reference_price": reference,
            "resolved_price": target[1],
            "signal": signal,
            "position_before": position_before,
            "calibrated_probability": row.get("calibrated_probability"),
            "decision_ev": row.get("decision_ev"),
            "actual_return": actual_return,
            "actual_return_bps": actual_return * 10_000.0,
            "classification": classification,
        })
    by_model: dict[str, Any] = {}
    for model_id in sorted({str(row.get("model_id")) for row in predictions if row.get("model_id")}):
        rows = [row for row in scored if row["model_id"] == model_id]
        by_model[model_id] = {
            "resolved_policy_decisions": len(rows),
            "classifications": dict(Counter(row["classification"] for row in rows)),
            "latest_outcome": rows[-1] if rows else None,
        }
    return {
        "prediction_rows": len(predictions),
        "resolved_policy_rows": len(scored),
        "models": by_model,
        "latest_predictions": predictions[-30:],
        "latest_outcomes": scored[-30:],
    }


def trader_brain_evidence(db_path: Path) -> dict[str, Any]:
    if not db_path.is_file():
        return {"available": False, "reason": f"Database not found: {db_path}"}
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=5000")
    try:
        exists = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='trader_brain_experiences'").fetchone()
        if not exists:
            return {"available": False, "reason": "trader_brain_experiences table not found"}
        rows = [dict(row) for row in conn.execute(
            "SELECT * FROM trader_brain_experiences ORDER BY id DESC LIMIT 2000"
        ).fetchall()]
    finally:
        conn.close()
    rows.reverse()
    by_model: dict[str, Any] = {}
    for model_id in sorted({str(row.get("model_id")) for row in rows if row.get("model_id")}):
        mine = [row for row in rows if str(row.get("model_id")) == model_id]
        resolved = [row for row in mine if row.get("resolved_at")]
        action_counts = Counter(str(row.get("action", "UNKNOWN")) for row in mine)
        missed = 0
        for row in resolved:
            if str(row.get("position_before")) != "FLAT" or str(row.get("action")) not in {"NO_TRADE", "HOLD"}:
                continue
            try:
                shadow = json.loads(str(row.get("shadow_rewards_json") or "{}"))
                if to_float(shadow.get("LONG")) > 0:
                    missed += 1
            except json.JSONDecodeError:
                pass
        by_model[model_id] = {
            "experiences": len(mine),
            "resolved": len(resolved),
            "pending": len(mine) - len(resolved),
            "action_counts": dict(action_counts),
            "average_reward": (sum(to_float(row.get("reward")) for row in resolved) / len(resolved)) if resolved else None,
            "average_market_return": (sum(to_float(row.get("realized_return")) for row in resolved) / len(resolved)) if resolved else None,
            "profitable_flat_shadow_longs": missed,
            "latest_experience": mine[-1] if mine else None,
            "latest_resolved": resolved[-1] if resolved else None,
        }
    # Keep a bounded recent sample. JSON-encoded expert/gate fields are retained
    # because they are exactly what a later diagnostic audit needs.
    return {"available": True, "models": by_model, "latest_experiences": rows[-80:]}


def algorithm_table(models_payload: dict[str, Any]) -> list[dict[str, Any]]:
    models = models_payload.get("models") if isinstance(models_payload.get("models"), list) else []
    result: list[dict[str, Any]] = []
    for model in models:
        if not isinstance(model, dict):
            continue
        portfolio = model.get("portfolio") if isinstance(model.get("portfolio"), dict) else {}
        perf = model.get("performance") if isinstance(model.get("performance"), dict) else {}
        latest = model.get("latest_decision") if isinstance(model.get("latest_decision"), dict) else {}
        positions = portfolio.get("positions") if isinstance(portfolio.get("positions"), list) else []
        result.append({
            "model_id": model.get("model_id"),
            "display_name": model.get("display_name"),
            "driver": model.get("driver"),
            "policy_mode": model.get("policy_mode", "official"),
            "experimental": bool(model.get("experimental", False)),
            "horizon_hours": model.get("horizon_hours"),
            "equity": to_float(portfolio.get("portfolio_value_usdt")),
            "net_pnl": to_float(portfolio.get("total_pnl_usdt")),
            "return": to_float(portfolio.get("total_return")),
            "closed_trades": to_int(perf.get("closed_trades")),
            "executions": to_int(perf.get("trade_count")),
            "wins": to_int(perf.get("winning_trades")),
            "win_rate": perf.get("win_rate"),
            "fees": to_float(perf.get("total_fees_usdt")),
            "open_positions": positions,
            "latest_signal": latest.get("signal") if latest else None,
            "latest_decision_at": latest.get("created_at") if latest else None,
            "latest_reason": latest.get("reason") if latest else None,
        })
    return sorted(result, key=lambda row: row["return"], reverse=True)


def warnings_for(snapshot: dict[str, Any], now: datetime) -> list[dict[str, str]]:
    warnings: list[dict[str, str]] = []
    health = snapshot.get("system", {}).get("api_health")
    if not isinstance(health, dict) or health.get("status") != "ok":
        warnings.append({"severity": "critical", "code": "API_UNHEALTHY", "message": "Local FastAPI health endpoint is unavailable or not OK."})
    truth = snapshot.get("market_truth")
    if not isinstance(truth, dict) or not truth.get("available"):
        warnings.append({"severity": "critical", "code": "MARKET_TRUTH_MISSING", "message": "Real CoinEx candle truth could not be exported."})
    else:
        freshness = to_float(truth.get("freshness_seconds"), 1e9)
        if freshness > 30 * 60:
            warnings.append({"severity": "critical", "code": "MARKET_DATA_STALE", "message": f"Latest completed 15m candle is {freshness/60:.1f} minutes old."})
    ops = snapshot.get("system", {}).get("ops_status")
    if isinstance(ops, dict):
        severity = str(ops.get("overall_status", ops.get("status", ""))).lower()
        if severity in {"critical", "error", "failed", "unhealthy"}:
            warnings.append({"severity": "critical", "code": "OPS_CRITICAL", "message": "The server ops supervisor reports a critical/unhealthy state."})
    short = snapshot.get("evidence", {}).get("short_term", {})
    if isinstance(short, dict):
        for model_id, model in (short.get("models") or {}).items():
            if not isinstance(model, dict):
                continue
            pending = to_int(model.get("pending"))
            resolved = to_int(model.get("resolved"))
            if pending > 12 and resolved == 0:
                warnings.append({"severity": "warning", "code": "SHORT_OUTCOMES_PENDING", "message": f"{model_id} has {pending} diagnostic decisions but no resolved outcomes yet."})
    # Surface unusually stale latest decisions relative to each model's intended horizon.
    for model in snapshot.get("algorithms", []):
        if not isinstance(model, dict):
            continue
        created = parse_iso(model.get("latest_decision_at"))
        horizon = max(1.0, to_float(model.get("horizon_hours"), 1.0))
        if created and (now - created).total_seconds() > max(2 * HOUR_MS / 1000.0, 1.75 * horizon * 3600.0):
            warnings.append({
                "severity": "warning",
                "code": "DECISION_STALE",
                "message": f"{model.get('display_name')} has not recorded a decision for {(now-created).total_seconds()/3600:.1f}h.",
            })
    return warnings


def markdown(snapshot: dict[str, Any]) -> str:
    market = snapshot.get("market_truth") or {}
    lines = [
        "# Six-hour AI trading audit snapshot",
        "",
        f"Generated: `{snapshot['generated_at_utc']}`  ",
        f"Runtime commit: `{snapshot.get('runtime_git_commit') or 'unknown'}`  ",
        "Real-order execution: **DISABLED**",
        "",
        "## Real market",
        "",
    ]
    if market.get("available"):
        lines += [
            f"- BTC close: `${to_float(market.get('latest_close')):,.2f}`",
            f"- 6h return: `{(to_float(market.get('return_6h'))*100):+.2f}%`",
            f"- 24h return: `{(to_float(market.get('return_24h'))*100):+.2f}%`",
            f"- Candle freshness: `{to_float(market.get('freshness_seconds'))/60:.1f} min`",
            "",
        ]
    lines += [
        "## Algorithms",
        "",
        "| Algorithm | Policy | Equity | Net P/L | Return | Closed | Exec | Win rate | Fees | Position | Latest |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in snapshot.get("algorithms", []):
        win = "—" if row.get("win_rate") is None else f"{to_float(row.get('win_rate'))*100:.1f}%"
        position = "LONG" if row.get("open_positions") else "FLAT"
        lines.append(
            f"| {row.get('display_name')} | {row.get('policy_mode')} | €{to_float(row.get('equity')):,.2f} | "
            f"€{to_float(row.get('net_pnl')):,.2f} | {to_float(row.get('return'))*100:+.2f}% | "
            f"{row.get('closed_trades',0)} | {row.get('executions',0)} | {win} | €{to_float(row.get('fees')):,.2f} | "
            f"{position} | {row.get('latest_signal') or 'WAIT'} |"
        )
    lines += ["", "## Warnings", ""]
    warnings = snapshot.get("warnings") or []
    if warnings:
        lines += [f"- **{item.get('severity','info').upper()}** `{item.get('code')}` — {item.get('message')}" for item in warnings]
    else:
        lines.append("- No monitoring warning generated in this snapshot.")
    lines += [
        "",
        "## Evidence included",
        "",
        "- Exact recent CoinEx 15m/1h OHLCV used as real-market truth.",
        "- Short-term decision diagnostics and matured 2h outcomes.",
        "- Frozen V3 policy decisions scored against their real future horizon.",
        "- Trader-Brain experiences, resolved market returns, rewards and shadow rewards.",
        "- Current paper account P/L, fees, positions and latest audit reasons.",
        "",
    ]
    return "\n".join(lines)


def export_snapshot(repo_root: Path, db_path: Path, out_dir: Path, api_base: str) -> Path:
    now = datetime.now(UTC)
    health = http_json(f"{api_base.rstrip('/')}/health")
    models_payload = merge_models(api_base)
    ticker = http_json(f"{api_base.rstrip('/')}/api/market/BTCUSDT")
    candles15 = candle_rows(http_json(f"{api_base.rstrip('/')}/api/market/BTCUSDT/klines?period=15min&limit=400"))
    candles1h = candle_rows(http_json(f"{api_base.rstrip('/')}/api/market/BTCUSDT/klines?period=1hour&limit=400"))
    ops_status = read_json(repo_root / "state" / "ops" / "system_health.json")

    snapshot: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": now.isoformat(),
        "runtime_git_commit": git_commit(repo_root),
        "system": {
            "api_health": health,
            "ops_status": ops_status,
            "ticker": ticker,
        },
        "market_truth": market_truth(candles15, candles1h, now),
        "algorithms": algorithm_table(models_payload),
        "evidence": {
            "short_term": summarize_short_term(repo_root),
            "frozen_v3": score_v3(repo_root, models_payload, candles1h),
            "trader_brain": trader_brain_evidence(db_path),
        },
        "warnings": [],
        "audit_contract": {
            "paper_only": True,
            "raw_tick_and_orderbook_exported": False,
            "strategy_changes_allowed_from_single_snapshot": False,
            "operational_faults_should_be_fixed_immediately": True,
            "strategy_changes_require_repeated_evidence": True,
        },
    }
    snapshot["warnings"] = warnings_for(snapshot, now)

    day_dir = out_dir / now.date().isoformat()
    day_dir.mkdir(parents=True, exist_ok=True)
    stamp = now.strftime("%H%M%S")
    target = day_dir / f"{stamp}.json"
    payload = json.dumps(snapshot, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    target.write_text(payload, encoding="utf-8")
    (day_dir / f"{stamp}.md").write_text(markdown(snapshot), encoding="utf-8")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "latest.json").write_text(payload, encoding="utf-8")
    (out_dir / "LATEST.md").write_text(markdown(snapshot), encoding="utf-8")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description="Export six-hour PAPER observability snapshot.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--db", default="data/paper_trading.sqlite3")
    parser.add_argument("--out-dir", default="paper_monitoring")
    parser.add_argument("--api-base", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    root = Path(args.repo_root).resolve()
    db = Path(args.db)
    out = Path(args.out_dir)
    if not db.is_absolute():
        db = root / db
    if not out.is_absolute():
        out = root / out
    target = export_snapshot(root, db, out, args.api_base)
    print(f"Six-hour monitoring snapshot exported: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
