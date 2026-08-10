import json
import math
from pathlib import Path

import pytest

pytest.importorskip("tensorflow")

from backend.ml.evaluation import WalkForwardConfig
from backend.ml.lstm_core import train_lstm_walk_forward


HOUR_MS = 60 * 60 * 1000


def _write_synthetic_dataset(path: Path, hours: int = 24 * 36) -> None:
    rows = []
    price = 30_000.0
    for idx in range(hours + 1):
        movement = 0.0015 * math.sin(idx / 8.0) + 0.0004 * math.cos(idx / 17.0)
        next_price = price * (1.0 + movement)
        rows.append(
            {
                "timestamp": idx * HOUR_MS,
                "open": price,
                "high": max(price, next_price) * 1.001,
                "low": min(price, next_price) * 0.999,
                "close": next_price,
                "volume": 100.0 + 10.0 * math.sin(idx / 5.0),
                "quote_volume": (100.0 + 10.0 * math.sin(idx / 5.0)) * next_price,
                "number_of_trades": 500 + idx % 50,
                "taker_buy_quote_ratio": 0.5 + 0.1 * math.sin(idx / 13.0),
                "target_return_1h": 0.0,
            }
        )
        price = next_price

    for idx in range(len(rows) - 1):
        rows[idx]["target_return_1h"] = rows[idx + 1]["close"] / rows[idx]["close"] - 1.0
    rows[-1]["target_return_1h"] = None

    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_lstm_walk_forward_smoke(tmp_path: Path) -> None:
    dataset = tmp_path / "btc_hourly.jsonl"
    _write_synthetic_dataset(dataset)
    run_root = tmp_path / "runs"

    summary = train_lstm_walk_forward(
        dataset_path=dataset,
        run_root=run_root,
        walk_forward=WalkForwardConfig(
            train_days=14,
            validation_days=3,
            test_days=3,
            step_days=6,
        ),
        sequence_length=6,
        lstm_units=8,
        dense_units=4,
        dropout=0.0,
        epochs=1,
        batch_size=32,
        early_stopping_patience=0,
    )

    run_dir = run_root / summary["run_id"]
    assert summary["fold_count"] >= 1
    assert summary["oos_prediction_count"] > 0
    assert (run_dir / "summary.json").is_file()
    assert (run_dir / "predictions.jsonl").is_file()
    assert any((run_dir / "models").glob("fold_*.keras"))
    assert any((run_dir / "scalers").glob("fold_*.json"))
    assert "direction_classification" in summary
    assert 0.0 <= summary["direction_classification"]["auc"] <= 1.0
    assert "lstm_probability_gated" in summary["strategies"]

    with (run_dir / "predictions.jsonl").open("r", encoding="utf-8") as handle:
        first_row = json.loads(handle.readline())
    assert 0.0 <= first_row["predicted_direction_prob"] <= 1.0
    assert first_row["lstm_prob_position"] in (0.0, 1.0)
