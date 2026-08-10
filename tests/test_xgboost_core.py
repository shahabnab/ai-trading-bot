from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

pytest.importorskip("xgboost")

from backend.ml.evaluation import WalkForwardConfig
from backend.ml.features import HOUR_MS
from backend.ml.xgboost_core import train_xgboost_walk_forward


def _write_dataset(path: Path, hours: int = 24 * 36) -> None:
    close = 50_000.0
    rows: list[dict[str, object]] = []
    for index in range(hours):
        previous = close
        current_return = 0.0002 + 0.001 * math.sin(index / 8.0)
        close = previous * (1.0 + current_return)
        next_return = 0.0002 + 0.001 * math.sin((index + 1) / 8.0)
        rows.append(
            {
                "timestamp": (index + 1) * HOUR_MS,
                "open": previous,
                "high": max(previous, close) * 1.001,
                "low": min(previous, close) * 0.999,
                "close": close,
                "volume": 1_000.0 + index,
                "quote_volume": (1_000.0 + index) * close,
                "number_of_trades": 500 + index,
                "taker_buy_quote_ratio": 0.5,
                "target_return_1h": next_return if index < hours - 1 else None,
            }
        )

    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_xgboost_pipeline_writes_oos_artifacts(tmp_path: Path) -> None:
    dataset_path = tmp_path / "btc_hourly.jsonl"
    _write_dataset(dataset_path)

    summary = train_xgboost_walk_forward(
        dataset_path=dataset_path,
        run_root=tmp_path / "runs",
        walk_forward=WalkForwardConfig(
            train_days=14,
            validation_days=3,
            test_days=3,
            step_days=3,
        ),
        fee_bps=1.0,
        slippage_bps=1.0,
        spread_bps=1.0,
        execution_lambda=1.0,
        num_boost_round=10,
        early_stopping_rounds=3,
    )

    run_dir = tmp_path / "runs" / summary["run_id"]
    assert summary["fold_count"] >= 1
    assert summary["oos_prediction_count"] > 0
    assert (run_dir / "summary.json").is_file()
    assert (run_dir / "predictions.jsonl").is_file()
    assert any((run_dir / "models").glob("fold_*.json"))
