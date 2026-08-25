#!/usr/bin/env python3
"""Analyze PAPER monitoring snapshots without changing trading behavior.

This module consumes the JSON produced by ``export_six_hour_monitoring.py`` and
adds cost-aware research metrics. It is intentionally read-only: no strategy
thresholds, model artifacts, risk settings, positions, or orders are modified.
"""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_MIN_CLOSED_TRADES = 50
HIGH_FEE_BURDEN_RATIO = 0.50


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


def analyze_algorithm(row: dict[str, Any], min_closed_trades: int = DEFAULT_MIN_CLOSED_TRADES) -> dict[str, Any]:
    """Return cost-aware metrics for one algorithm row from a monitoring snapshot."""
    net_pnl = to_float(row.get("net_pnl"))
    fees = max(0.0, to_float(row.get("fees")))
    closed = max(0, to_int(row.get("closed_trades")))
    executions = max(0, to_int(row.get("executions")))

    # Monitoring's net P/L is after recorded fees. Adding fees back provides the
    # pre-fee account contribution without pretending it is a full trade-ledger
    # gross-profit statistic.
    pre_fee_pnl = net_pnl + fees
    net_per_closed_trade = net_pnl / closed if closed else None
    pre_fee_per_closed_trade = pre_fee_pnl / closed if closed else None
    fees_per_closed_trade = fees / closed if closed else None
    executions_per_closed_trade = executions / closed if closed else None

    fee_to_positive_pre_fee_ratio = None
    if pre_fee_pnl > 0:
        fee_to_positive_pre_fee_ratio = fees / pre_fee_pnl

    costs_erased_positive_pre_fee_edge = pre_fee_pnl > 0 and net_pnl <= 0
    high_fee_burden = (
        fee_to_positive_pre_fee_ratio is not None
        and fee_to_positive_pre_fee_ratio >= HIGH_FEE_BURDEN_RATIO
    )

    if closed < min_closed_trades:
        evidence_status = "INSUFFICIENT_SAMPLE"
    elif net_pnl > 0:
        evidence_status = "PRELIMINARY_POSITIVE"
    elif net_pnl < 0:
        evidence_status = "PRELIMINARY_NEGATIVE"
    else:
        evidence_status = "PRELIMINARY_FLAT"

    flags: list[str] = []
    if costs_erased_positive_pre_fee_edge:
        flags.append("COSTS_ERASED_PRE_FEE_EDGE")
    if high_fee_burden:
        flags.append("HIGH_FEE_BURDEN")
    if closed < min_closed_trades:
        flags.append("WAIT_FOR_MORE_CLOSED_TRADES")

    return {
        "model_id": row.get("model_id"),
        "display_name": row.get("display_name"),
        "policy_mode": row.get("policy_mode", "official"),
        "experimental": bool(row.get("experimental", False)),
        "closed_trades": closed,
        "executions": executions,
        "wins": max(0, to_int(row.get("wins"))),
        "win_rate": row.get("win_rate"),
        "net_pnl": net_pnl,
        "recorded_fees": fees,
        "pre_fee_pnl": pre_fee_pnl,
        "net_pnl_per_closed_trade": net_per_closed_trade,
        "pre_fee_pnl_per_closed_trade": pre_fee_per_closed_trade,
        "fees_per_closed_trade": fees_per_closed_trade,
        "executions_per_closed_trade": executions_per_closed_trade,
        "fee_to_positive_pre_fee_ratio": fee_to_positive_pre_fee_ratio,
        "costs_erased_positive_pre_fee_edge": costs_erased_positive_pre_fee_edge,
        "high_fee_burden": high_fee_burden,
        "minimum_closed_trades_for_preliminary_review": min_closed_trades,
        "evidence_status": evidence_status,
        "flags": flags,
    }


def analyze_snapshot(snapshot: dict[str, Any], min_closed_trades: int = DEFAULT_MIN_CLOSED_TRADES) -> dict[str, Any]:
    algorithms = snapshot.get("algorithms") if isinstance(snapshot.get("algorithms"), list) else []
    analyzed = [
        analyze_algorithm(row, min_closed_trades)
        for row in algorithms
        if isinstance(row, dict)
    ]

    official = [row for row in analyzed if not row.get("experimental")]
    experimental = [row for row in analyzed if row.get("experimental")]

    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source_generated_at_utc": snapshot.get("generated_at_utc"),
        "source_runtime_git_commit": snapshot.get("runtime_git_commit"),
        "read_only_analysis": True,
        "strategy_or_risk_parameters_changed": False,
        "minimum_closed_trades_for_preliminary_review": min_closed_trades,
        "algorithms": analyzed,
        "summary": {
            "algorithm_count": len(analyzed),
            "official_count": len(official),
            "experimental_count": len(experimental),
            "algorithms_with_costs_erasing_pre_fee_edge": [
                row.get("model_id") for row in analyzed if row.get("costs_erased_positive_pre_fee_edge")
            ],
            "algorithms_with_high_fee_burden": [
                row.get("model_id") for row in analyzed if row.get("high_fee_burden")
            ],
            "algorithms_ready_for_preliminary_review": [
                row.get("model_id") for row in analyzed if row.get("closed_trades", 0) >= min_closed_trades
            ],
        },
        "limitations": [
            "Pre-fee P/L is reconstructed as net P/L plus recorded fees; it is not gross profit from a per-trade ledger.",
            "Profit factor, max drawdown, MFE and MAE are not inferred when the source snapshot does not expose the required ledger/equity path.",
            "The analysis is diagnostic only and must not be used to auto-retune strategy or risk parameters.",
        ],
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# PAPER performance analysis",
        "",
        f"Generated: `{report['generated_at_utc']}`  ",
        f"Source snapshot: `{report.get('source_generated_at_utc') or 'unknown'}`  ",
        f"Source runtime commit: `{report.get('source_runtime_git_commit') or 'unknown'}`  ",
        "Trading behavior changed: **NO**",
        "",
        "| Algorithm | Closed | Net P/L | Pre-fee P/L | Fees | Net/trade | Fee burden | Evidence | Flags |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in report.get("algorithms", []):
        ratio = row.get("fee_to_positive_pre_fee_ratio")
        burden = "—" if ratio is None else f"{to_float(ratio) * 100:.1f}%"
        per_trade = row.get("net_pnl_per_closed_trade")
        per_trade_text = "—" if per_trade is None else f"€{to_float(per_trade):+.4f}"
        flags = ", ".join(row.get("flags") or []) or "—"
        lines.append(
            f"| {row.get('display_name') or row.get('model_id')} | {row.get('closed_trades', 0)} | "
            f"€{to_float(row.get('net_pnl')):+.2f} | €{to_float(row.get('pre_fee_pnl')):+.2f} | "
            f"€{to_float(row.get('recorded_fees')):.2f} | {per_trade_text} | {burden} | "
            f"{row.get('evidence_status')} | {flags} |"
        )
    lines += [
        "",
        "## Interpretation guardrails",
        "",
        f"- Fewer than {report.get('minimum_closed_trades_for_preliminary_review', DEFAULT_MIN_CLOSED_TRADES)} closed trades is marked as insufficient for preliminary strategy review.",
        "- Cost flags are diagnostic; they do not automatically lower or raise entry thresholds.",
        "- Profit factor, drawdown, MFE and MAE are intentionally not fabricated from incomplete data.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze a six-hour PAPER monitoring snapshot.")
    parser.add_argument("snapshot", nargs="?", default="paper_monitoring/latest.json")
    parser.add_argument("--out-json", default="paper_monitoring/performance_analysis.json")
    parser.add_argument("--out-md", default="paper_monitoring/PERFORMANCE_ANALYSIS.md")
    parser.add_argument("--min-closed-trades", type=int, default=DEFAULT_MIN_CLOSED_TRADES)
    args = parser.parse_args()

    source = Path(args.snapshot)
    if not source.is_file():
        raise SystemExit(f"Snapshot not found: {source}")
    snapshot = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(snapshot, dict):
        raise SystemExit("Snapshot root must be a JSON object")

    report = analyze_snapshot(snapshot, max(1, args.min_closed_trades))
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    out_md.write_text(markdown(report), encoding="utf-8")
    print(out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
