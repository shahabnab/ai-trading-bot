from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from scripts.export_six_hour_monitoring import HOUR_MS, MIN15_MS, market_truth, score_v3


def _candle(start: int, close: float, *, interval_ms: int) -> dict[str, object]:
    return {
        "created_at": start,
        "open": close,
        "high": close * 1.001,
        "low": close * 0.999,
        "close": close,
        "volume": 100.0,
        "value": 100.0 * close,
    }


def test_market_truth_keeps_real_candles_and_returns():
    base = 1_700_000_000_000
    candles15 = [_candle(base + i * MIN15_MS, 100.0 + i * 0.1, interval_ms=MIN15_MS) for i in range(100)]
    candles1h = [_candle(base + i * HOUR_MS, 100.0 + i * 0.4, interval_ms=HOUR_MS) for i in range(40)]
    latest_end = int(candles15[-1]["created_at"]) + MIN15_MS
    now = datetime.fromtimestamp((latest_end + 60_000) / 1000.0, UTC)

    truth = market_truth(candles15, candles1h, now)

    assert truth["available"] is True
    assert truth["latest_close"] == candles15[-1]["close"]
    assert truth["return_6h"] is not None
    assert len(truth["candles_15m_last_24h"]) <= 96
    assert len(truth["candles_1h_last_72h"]) == len(candles1h)
    assert float(truth["freshness_seconds"]) < 120


def test_v3_policy_is_scored_against_real_future_horizon(tmp_path: Path):
    base = 1_700_000_000_000
    feature_ts = base + HOUR_MS
    state = tmp_path / "state" / "forward_v3"
    state.mkdir(parents=True)
    row = {
        "model_id": "v3-test",
        "feature_timestamp": feature_ts,
        "policy_due": True,
        "signal": "HOLD",
        "position_before": "CASH",
        "paper_market_price": "100",
        "one_way_cost_rate": 0.001,
        "calibrated_probability": 0.55,
        "decision_ev": -0.001,
        "horizon_commitment_hours": 2,
    }
    (state / "predictions.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    hour_candles = [
        _candle(base + 0 * HOUR_MS, 100.0, interval_ms=HOUR_MS),
        _candle(base + 1 * HOUR_MS, 101.0, interval_ms=HOUR_MS),
        _candle(base + 2 * HOUR_MS, 103.0, interval_ms=HOUR_MS),
        _candle(base + 3 * HOUR_MS, 104.0, interval_ms=HOUR_MS),
    ]
    models = {"models": [{"model_id": "v3-test", "horizon_hours": 2}]}

    result = score_v3(tmp_path, models, hour_candles)

    assert result["resolved_policy_rows"] == 1
    outcome = result["latest_outcomes"][0]
    assert outcome["classification"] == "MISSED_LONG"
    assert outcome["actual_return"] > 0.02
    assert outcome["target_timestamp"] == feature_ts + 2 * HOUR_MS
