from __future__ import annotations

import math

import pytest

from backend.ml.features import HOUR_MS, build_feature_dataset


def _rows(count: int = 500) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    close = 100.0
    for index in range(count):
        previous = close
        close = previous * (1.0 + 0.001 * math.sin(index / 8.0) + 0.0002)
        next_close = close * (1.0 + 0.001 * math.sin((index + 1) / 8.0) + 0.0002)
        rows.append(
            {
                "timestamp": (index + 1) * HOUR_MS,
                "open": previous,
                "high": max(previous, close) * 1.002,
                "low": min(previous, close) * 0.998,
                "close": close,
                "volume": 1000.0 + index,
                "quote_volume": (1000.0 + index) * close,
                "number_of_trades": 100 + index,
                "taker_buy_quote_ratio": 0.5 + 0.05 * math.sin(index / 6.0),
                "target_return_1h": next_close / close - 1.0 if index < count - 1 else None,
                "fear_greed_value": 50,
                "news_count": 0,
                "news_overall_sentiment_mean": 0.0,
                "news_btc_sentiment_mean": 0.0,
                "news_btc_relevance_mean": 0.0,
            }
        )
    return rows


def test_feature_builder_is_finite_after_causal_warmup() -> None:
    dataset = build_feature_dataset(_rows())
    assert dataset.size > 0
    assert dataset.X.shape[1] == len(dataset.feature_names)
    assert dataset.timestamps[0] >= 168 * HOUR_MS
    assert dataset.X.dtype.name == "float32"


def test_sentiment_features_are_opt_in() -> None:
    base = build_feature_dataset(_rows(), include_sentiment=False)
    with_sentiment = build_feature_dataset(_rows(), include_sentiment=True)
    assert with_sentiment.X.shape[1] == base.X.shape[1] + 5


def test_missing_hour_resets_time_dependent_feature_warmup() -> None:
    rows = _rows()
    missing_timestamp = 260 * HOUR_MS
    rows = [row for row in rows if row["timestamp"] != missing_timestamp]

    dataset = build_feature_dataset(rows)
    post_gap = dataset.timestamps[dataset.timestamps > missing_timestamp]

    assert len(post_gap) > 0
    assert int(post_gap[0]) >= missing_timestamp + 169 * HOUR_MS


def test_duplicate_timestamps_fail_closed() -> None:
    rows = _rows()
    rows.insert(10, dict(rows[10]))

    with pytest.raises(ValueError, match="strictly increasing and unique"):
        build_feature_dataset(rows)
