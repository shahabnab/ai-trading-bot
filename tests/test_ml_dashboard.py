import json
import math
from pathlib import Path

from backend.ml.dashboard import HOUR_MS, collect_prediction_history


def _write_run(
    root: Path,
    *,
    run_id: str,
    family: str,
    started_at: str,
    predicted: float,
    actual: float,
) -> None:
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    run = {
        "run_id": run_id,
        "status": "completed",
        "started_at": started_at,
        "finished_at": started_at,
        "model_family": family,
    }
    (run_dir / "run.json").write_text(json.dumps(run), encoding="utf-8")
    summary = {
        "forecast_metrics": {"direction_accuracy": 0.55, "mae_log_return": 0.001},
        "strategies": {f"{family}_cost_aware_long_only": {"sharpe": 1.2}},
        "fold_count": 2,
        "oos_prediction_count": 1,
        "feature_version": "btc-hourly-tech-v1",
    }
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    row = {
        "timestamp": 10 * HOUR_MS,
        "fold": 2,
        "close": 100.0,
        "predicted_log_return_1h": predicted,
        "actual_log_return_1h": actual,
        "actual_simple_return_1h": math.expm1(actual),
    }
    (run_dir / "predictions.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")


def test_collect_prediction_history_uses_latest_completed_run(tmp_path: Path) -> None:
    _write_run(
        tmp_path,
        run_id="xgb-old",
        family="xgboost",
        started_at="2026-01-01T00:00:00+00:00",
        predicted=0.001,
        actual=0.002,
    )
    _write_run(
        tmp_path,
        run_id="xgb-new",
        family="xgboost",
        started_at="2026-02-01T00:00:00+00:00",
        predicted=0.01,
        actual=0.005,
    )
    _write_run(
        tmp_path,
        run_id="lstm-new",
        family="lstm",
        started_at="2026-02-02T00:00:00+00:00",
        predicted=-0.004,
        actual=-0.002,
    )

    result = collect_prediction_history([tmp_path], limit=50)

    assert result["models"]["xgboost"]["run_id"] == "xgb-new"
    xgb = result["models"]["xgboost"]["predictions"][0]
    assert xgb["target_timestamp"] == 11 * HOUR_MS
    assert math.isclose(xgb["predicted_price"], 100.0 * math.exp(0.01))
    assert xgb["direction_correct"] is True
    assert result["models"]["lstm"]["metrics"]["strategy"]["sharpe"] == 1.2


def test_collect_prediction_history_handles_missing_roots(tmp_path: Path) -> None:
    result = collect_prediction_history([tmp_path / "does-not-exist"], limit=10)
    assert result["models"] == {}
