from __future__ import annotations

import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .strategies import ShortTermDecision

BUCKET_MS = 15 * 60 * 1000
DECISION_LOG = "decision_diagnostics.jsonl"
OUTCOME_LOG = "decision_outcomes.jsonl"
DEFAULT_SHADOW_THRESHOLDS_BPS = (55.0, 45.0, 35.0)
HORIZON_BUCKETS = {"15m": 1, "30m": 2, "1h": 4, "2h": 8}


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), separators=(",", ":"), sort_keys=True, default=str) + "\n")


def _read_jsonl(path: Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    if limit is not None and limit > 0:
        lines = lines[-limit:]
    for line in lines:
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if result != result or result in (float("inf"), float("-inf")):
        return default
    return result


def _timestamp(value: Any) -> int | None:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


def decision_key(model_id: str, feature_timestamp: int) -> str:
    return f"{model_id}:{feature_timestamp}"


def build_shadow_policies(
    decision: ShortTermDecision,
    *,
    entry_context: bool,
    official_threshold_bps: float,
    round_trip_cost_bps: float,
    shadow_thresholds_bps: Sequence[float] = DEFAULT_SHADOW_THRESHOLDS_BPS,
) -> list[dict[str, Any]]:
    """Return non-executing policy probes for the same raw setup.

    These probes never place paper fills. They isolate the entry edge hurdle from
    the strategy's setup logic so the frozen/official policy remains untouched.
    """
    candidates: list[tuple[str, float]] = [("official", float(official_threshold_bps))]
    seen = {round(float(official_threshold_bps), 9)}
    for threshold in shadow_thresholds_bps:
        threshold_f = float(threshold)
        key = round(threshold_f, 9)
        if key in seen:
            continue
        seen.add(key)
        candidates.append((f"shadow_{threshold_f:g}", threshold_f))
    candidates.append(("raw_setup", 0.0))

    setup_ready = bool(entry_context and decision.setup_ready)
    policies: list[dict[str, Any]] = []
    for name, threshold in candidates:
        would_enter = setup_ready and decision.edge_proxy_bps >= threshold
        policies.append(
            {
                "name": name,
                "threshold_bps": threshold,
                "would_enter": would_enter,
                "edge_proxy_bps": float(decision.edge_proxy_bps),
                "edge_gap_bps": float(decision.edge_proxy_bps) - threshold,
                "estimated_edge_after_cost_bps": float(decision.edge_proxy_bps) - float(round_trip_cost_bps),
            }
        )
    return policies


def append_decision_diagnostic(state_root: Path, payload: Mapping[str, Any]) -> None:
    _append_jsonl(state_root / DECISION_LOG, payload)


def _normalize_candles(candles: Iterable[Mapping[str, Any]]) -> dict[int, dict[str, float]]:
    normalized: dict[int, dict[str, float]] = {}
    for row in candles:
        ts = _timestamp(row.get("created_at"))
        if ts is None:
            continue
        close = _float(row.get("close"), -1.0)
        high = _float(row.get("high"), -1.0)
        low = _float(row.get("low"), -1.0)
        if close <= 0.0 or high <= 0.0 or low <= 0.0:
            continue
        normalized[ts] = {"close": close, "high": high, "low": low}
    return normalized


def _classify_hold(
    *,
    decision_action: str,
    setup_ready: bool,
    return_2h_bps: float,
    official_threshold_bps: float,
) -> str:
    if decision_action == "ENTER_LONG":
        return "ENTRY_SIGNAL"
    if not setup_ready:
        return "NO_SETUP"
    if return_2h_bps >= official_threshold_bps:
        return "MISSED_LONG"
    if return_2h_bps <= -official_threshold_bps:
        return "AVOIDED_LOSS"
    return "GOOD_HOLD"


def resolve_mature_outcomes(
    state_root: Path,
    candles: Iterable[Mapping[str, Any]],
    *,
    max_decisions: int = 5000,
) -> list[dict[str, Any]]:
    """Resolve +15m/+30m/+1h/+2h outcomes for mature decision telemetry.

    A decision is resolved only when the complete 2-hour future candle window is
    available. The outcome file is append-only and de-duplicated by decision key.
    """
    decisions = _read_jsonl(state_root / DECISION_LOG, limit=max_decisions)
    if not decisions:
        return []
    existing = _read_jsonl(state_root / OUTCOME_LOG)
    resolved_keys = {str(row.get("decision_key", "")) for row in existing}
    by_ts = _normalize_candles(candles)
    if not by_ts:
        return []

    new_rows: list[dict[str, Any]] = []
    for decision in decisions:
        key = str(decision.get("decision_key", ""))
        if not key or key in resolved_keys:
            continue
        feature_ts = _timestamp(decision.get("feature_timestamp"))
        if feature_ts is None:
            continue
        entry_price = _float(decision.get("signal_close"), -1.0)
        if entry_price <= 0.0:
            continue
        mature_ts = feature_ts + HORIZON_BUCKETS["2h"] * BUCKET_MS
        if mature_ts not in by_ts:
            continue

        returns_bps: dict[str, float] = {}
        complete = True
        for label, buckets in HORIZON_BUCKETS.items():
            candle = by_ts.get(feature_ts + buckets * BUCKET_MS)
            if candle is None:
                complete = False
                break
            returns_bps[label] = (candle["close"] / entry_price - 1.0) * 10_000.0
        if not complete:
            continue

        path_rows = [by_ts.get(feature_ts + i * BUCKET_MS) for i in range(1, HORIZON_BUCKETS["2h"] + 1)]
        if any(row is None for row in path_rows):
            continue
        valid_path = [row for row in path_rows if row is not None]
        mfe_bps = max((row["high"] / entry_price - 1.0) * 10_000.0 for row in valid_path)
        mae_bps = min((row["low"] / entry_price - 1.0) * 10_000.0 for row in valid_path)

        official_threshold = _float(decision.get("official_threshold_bps"), 0.0)
        round_trip_cost = _float(decision.get("round_trip_cost_bps"), 0.0)
        setup_ready = bool(decision.get("setup_ready", False))
        decision_action = str(decision.get("decision_action", "HOLD"))
        classification = _classify_hold(
            decision_action=decision_action,
            setup_ready=setup_ready,
            return_2h_bps=returns_bps["2h"],
            official_threshold_bps=official_threshold,
        )

        policy_results: list[dict[str, Any]] = []
        raw_policies = decision.get("shadow_policies", [])
        if isinstance(raw_policies, list):
            for policy in raw_policies:
                if not isinstance(policy, dict):
                    continue
                would_enter = bool(policy.get("would_enter", False))
                gross_bps = returns_bps["2h"] if would_enter else None
                net_bps = gross_bps - round_trip_cost if gross_bps is not None else None
                policy_results.append(
                    {
                        "name": str(policy.get("name", "unknown")),
                        "threshold_bps": _float(policy.get("threshold_bps"), 0.0),
                        "would_enter": would_enter,
                        "gross_return_2h_bps": gross_bps,
                        "net_return_2h_bps": net_bps,
                        "win_after_cost": bool(net_bps is not None and net_bps > 0.0),
                    }
                )

        row = {
            "decision_key": key,
            "model_id": str(decision.get("model_id", "")),
            "feature_timestamp": feature_ts,
            "resolved_at": datetime.now(UTC).isoformat(),
            "signal_close": entry_price,
            "decision_action": decision_action,
            "signal": str(decision.get("signal", "HOLD")),
            "confidence": _float(decision.get("confidence"), 0.0),
            "confirmation_score": _float(decision.get("confirmation_score"), 0.0),
            "setup_ready": setup_ready,
            "edge_proxy_bps": _float(decision.get("edge_proxy_bps"), 0.0),
            "official_threshold_bps": official_threshold,
            "round_trip_cost_bps": round_trip_cost,
            "returns_bps": returns_bps,
            "mfe_2h_bps": mfe_bps,
            "mae_2h_bps": mae_bps,
            "classification": classification,
            "shadow_results": policy_results,
        }
        _append_jsonl(state_root / OUTCOME_LOG, row)
        resolved_keys.add(key)
        new_rows.append(row)
    return new_rows


def _confidence_bucket(value: float) -> tuple[str, float, float]:
    edges = ((0.0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01))
    labels = ("<50%", "50-60%", "60-70%", "70-80%", "80-90%", "90-100%")
    for label, (low, high) in zip(labels, edges):
        if low <= value < high:
            return label, low, high
    return "90-100%", 0.9, 1.01


def build_diagnostics_summary(state_root: Path, *, limit: int = 2000) -> dict[str, Any]:
    decisions = _read_jsonl(state_root / DECISION_LOG, limit=limit)
    outcomes = _read_jsonl(state_root / OUTCOME_LOG, limit=limit)
    decision_by_key = {str(row.get("decision_key", "")): row for row in decisions if row.get("decision_key")}
    outcomes = [row for row in outcomes if str(row.get("decision_key", "")) in decision_by_key]

    model_ids = sorted({str(row.get("model_id", "")) for row in decisions if row.get("model_id")})
    models: list[dict[str, Any]] = []
    for model_id in model_ids:
        model_decisions = [row for row in decisions if str(row.get("model_id", "")) == model_id]
        model_outcomes = [row for row in outcomes if str(row.get("model_id", "")) == model_id]
        classifications: dict[str, int] = defaultdict(int)
        for row in model_outcomes:
            classifications[str(row.get("classification", "UNKNOWN"))] += 1

        policy_acc: dict[str, dict[str, Any]] = {}
        for row in model_outcomes:
            raw_results = row.get("shadow_results", [])
            if not isinstance(raw_results, list):
                continue
            for result in raw_results:
                if not isinstance(result, dict):
                    continue
                name = str(result.get("name", "unknown"))
                acc = policy_acc.setdefault(
                    name,
                    {
                        "name": name,
                        "threshold_bps": _float(result.get("threshold_bps"), 0.0),
                        "signals": 0,
                        "wins": 0,
                        "gross_sum_bps": 0.0,
                        "net_sum_bps": 0.0,
                    },
                )
                if bool(result.get("would_enter", False)):
                    acc["signals"] += 1
                    gross = _float(result.get("gross_return_2h_bps"), 0.0)
                    net = _float(result.get("net_return_2h_bps"), 0.0)
                    acc["gross_sum_bps"] += gross
                    acc["net_sum_bps"] += net
                    if bool(result.get("win_after_cost", False)):
                        acc["wins"] += 1

        policies: list[dict[str, Any]] = []
        order = {"official": 0, "shadow_55": 1, "shadow_45": 2, "shadow_35": 3, "raw_setup": 99}
        for acc in sorted(policy_acc.values(), key=lambda item: (order.get(str(item["name"]), 50), str(item["name"]))):
            signals = int(acc["signals"])
            wins = int(acc["wins"])
            policies.append(
                {
                    "name": acc["name"],
                    "threshold_bps": acc["threshold_bps"],
                    "signals": signals,
                    "wins": wins,
                    "win_rate": (wins / signals) if signals else None,
                    "avg_gross_2h_bps": (acc["gross_sum_bps"] / signals) if signals else None,
                    "avg_net_2h_bps": (acc["net_sum_bps"] / signals) if signals else None,
                    "cumulative_net_2h_bps": acc["net_sum_bps"],
                }
            )

        buckets: dict[str, list[float]] = defaultdict(list)
        bucket_positive: dict[str, int] = defaultdict(int)
        for row in model_outcomes:
            label, _, _ = _confidence_bucket(_float(row.get("confidence"), 0.0))
            returns = row.get("returns_bps", {})
            if not isinstance(returns, dict):
                continue
            ret = _float(returns.get("2h"), 0.0)
            buckets[label].append(ret)
            if ret > 0.0:
                bucket_positive[label] += 1
        confidence_rows: list[dict[str, Any]] = []
        for label in ("<50%", "50-60%", "60-70%", "70-80%", "80-90%", "90-100%"):
            values = buckets.get(label, [])
            confidence_rows.append(
                {
                    "label": label,
                    "samples": len(values),
                    "positive_2h": bucket_positive.get(label, 0),
                    "direction_accuracy": (bucket_positive.get(label, 0) / len(values)) if values else None,
                    "avg_return_2h_bps": (sum(values) / len(values)) if values else None,
                }
            )

        models.append(
            {
                "model_id": model_id,
                "decisions": len(model_decisions),
                "resolved": len(model_outcomes),
                "pending": max(0, len(model_decisions) - len(model_outcomes)),
                "setup_candidates": sum(1 for row in model_decisions if bool(row.get("setup_ready", False))),
                "official_entry_signals": sum(1 for row in model_decisions if str(row.get("decision_action", "")) == "ENTER_LONG"),
                "avg_edge_gap_bps": (
                    sum(_float(row.get("edge_gap_bps"), 0.0) for row in model_decisions) / len(model_decisions)
                    if model_decisions
                    else None
                ),
                "classifications": dict(classifications),
                "policies": policies,
                "confidence_buckets": confidence_rows,
            }
        )

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "window": {
            "decision_limit": limit,
            "decisions": len(decisions),
            "resolved": len(outcomes),
            "pending": max(0, len(decisions) - len(outcomes)),
        },
        "models": models,
        "recent_outcomes": list(reversed(outcomes[-20:])),
        "notes": [
            "Shadow policies are diagnostics only and never place paper or real orders.",
            "Policy returns assume entry at the signal candle close and a fixed 2-hour exit, then subtract the configured round-trip cost.",
            "Short-term baselines are currently long-only; missed-short opportunities are not classified here.",
        ],
    }
