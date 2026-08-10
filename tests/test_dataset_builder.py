import json
from pathlib import Path

import pytest

from backend.data_collection.dataset_builder import (
    HOUR_MS,
    MINUTE_MS,
    _add_market_features_and_targets,
    _aggregate_market,
)


def _candle(timestamp_ms: int, close: float) -> dict[str, object]:
    return {
        "open_time": timestamp_ms,
        "symbol": "BTCUSDT",
        "open": str(close),
        "high": str(close + 1),
        "low": str(close - 1),
        "close": str(close),
        "volume": "1",
        "quote_volume": str(close),
        "number_of_trades": 1,
        "taker_buy_base_volume": "0.5",
        "taker_buy_quote_volume": str(close / 2),
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_incomplete_hour_is_dropped_and_complete_hour_has_candle_count(tmp_path: Path) -> None:
    rows = [_candle(idx * MINUTE_MS, 100 + idx / 100) for idx in range(59)]
    rows += [
        _candle(HOUR_MS + idx * MINUTE_MS, 200 + idx / 100)
        for idx in range(60)
    ]
    path = tmp_path / "candles.jsonl"
    _write_jsonl(path, rows)

    hourly = _aggregate_market(path)

    assert len(hourly) == 1
    assert hourly[0]["timestamp"] == 2 * HOUR_MS
    assert hourly[0]["candle_count"] == 60
    assert hourly[0]["minute_count"] == 60


def test_targets_require_exact_future_timestamp_after_gap() -> None:
    rows = [
        {"timestamp": HOUR_MS, "open": 100, "high": 101, "low": 99, "close": 100},
        {"timestamp": 2 * HOUR_MS, "open": 100, "high": 102, "low": 99, "close": 101},
        {"timestamp": 4 * HOUR_MS, "open": 102, "high": 104, "low": 101, "close": 103},
    ]

    _add_market_features_and_targets(rows)

    assert rows[0]["target_return_1h"] == pytest.approx(0.01)
    assert rows[1]["target_return_1h"] is None
    assert rows[2]["return_1h"] is None
