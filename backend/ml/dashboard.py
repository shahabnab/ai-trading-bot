from __future__ import annotations

import json
import math
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable


HOUR_MS = 60 * 60 * 1000
MODEL_FAMILIES = ("xgboost", "lstm")


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _as_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _latest_completed_runs(run_roots: Iterable[Path]) -> dict[str, tuple[Path, dict[str, Any]]]:
    latest: dict[str, tuple[Path, dict[str, Any]]] = {}

    for root in run_roots:
        if not root.is_dir():
            continue
        for run_file in root.glob("*/run.json"):
            run = _read_json(run_file)
            if not run or run.get("status") != "completed":
                continue
            family = str(run.get("model_family", "")).lower()
            if family not in MODEL_FAMILIES:
                continue
            run_dir = run_file.parent
            if not (run_dir / "predictions.jsonl").is_file():
                continue
            started_at = str(run.get("started_at", ""))
            current = latest.get(family)
            if current is None or started_at > str(current[1].get("started_at", "")):
                latest[family] = (run_dir, run)

    return latest


def _tail_prediction_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    rows: deque[dict[str, Any]] = deque(maxlen=limit)
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    rows.append(payload)
    except OSError:
        return []
    return list(rows)


def _normalize_prediction(row: dict[str, Any]) -> dict[str, Any] | None:
    try:
        source_timestamp = int(row["timestamp"])
    except (KeyError, TypeError, ValueError):
        return None

    reference_price = _as_float(row.get("close"))
    predicted_log_return = _as_float(row.get("predicted_log_return_1h"))
    actual_log_return = _as_float(row.get("actual_log_return_1h"))
    actual_simple_return = _as_float(row.get("actual_simple_return_1h"))
    if (
        reference_price is None
        or reference_price <= 0.0
        or predicted_log_return is None
        or actual_log_return is None
        or actual_simple_return is None
    ):
        return None

    predicted_simple_return = math.expm1(predicted_log_return)
    predicted_price = reference_price * math.exp(predicted_log_return)
    actual_price = reference_price * (1.0 + actual_simple_return)
    error_log_return = predicted_log_return - actual_log_return

    return {
        "source_timestamp": source_timestamp,
        "target_timestamp": source_timestamp + HOUR_MS,
        "fold": int(row.get("fold", 0) or 0),
        "reference_price": reference_price,
        "predicted_price": predicted_price,
        "actual_price": actual_price,
        "predicted_log_return": predicted_log_return,
        "actual_log_return": actual_log_return,
        "predicted_simple_return": predicted_simple_return,
        "actual_simple_return": actual_simple_return,
        "error_log_return": error_log_return,
        "direction_correct": (predicted_log_return > 0.0) == (actual_log_return > 0.0),
    }


def _summary_metrics(run_dir: Path, family: str) -> dict[str, Any]:
    summary = _read_json(run_dir / "summary.json") or {}
    forecast = summary.get("forecast_metrics")
    strategies = summary.get("strategies")
    strategy_key = f"{family}_cost_aware_long_only"
    strategy = strategies.get(strategy_key) if isinstance(strategies, dict) else None
    return {
        "forecast": forecast if isinstance(forecast, dict) else {},
        "strategy": strategy if isinstance(strategy, dict) else {},
        "fold_count": summary.get("fold_count"),
        "oos_prediction_count": summary.get("oos_prediction_count"),
        "feature_version": summary.get("feature_version"),
    }


def collect_prediction_history(
    run_roots: Iterable[str | Path],
    *,
    limit: int = 500,
) -> dict[str, Any]:
    """Return dashboard-ready OOS prediction history for the latest model runs.

    This adapter intentionally exposes completed walk-forward predictions only.
    It does not pretend that fold models are deployment/live models. A future
    forward-inference recorder can extend the same response shape without
    changing the dashboard contract.
    """

    if limit <= 0:
        raise ValueError("limit must be positive")

    roots = [Path(root) for root in run_roots]
    latest = _latest_completed_runs(roots)
    models: dict[str, Any] = {}

    for family in MODEL_FAMILIES:
        selected = latest.get(family)
        if selected is None:
            continue
        run_dir, run = selected
        normalized: list[dict[str, Any]] = []
        for row in _tail_prediction_rows(run_dir / "predictions.jsonl", limit):
            point = _normalize_prediction(row)
            if point is not None:
                normalized.append(point)
        normalized.sort(key=lambda item: int(item["target_timestamp"]))

        models[family] = {
            "run_id": str(run.get("run_id", run_dir.name)),
            "started_at": run.get("started_at"),
            "finished_at": run.get("finished_at"),
            "mode": "walk_forward_oos",
            "metrics": _summary_metrics(run_dir, family),
            "predictions": normalized,
        }

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "source": "completed_walk_forward_runs",
        "models": models,
    }
