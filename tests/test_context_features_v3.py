import json
from pathlib import Path

import numpy as np

from backend.ml.context_features import build_context_feature_matrix

HOUR = 60 * 60 * 1000


def _write(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_full_context_matrix_is_aligned_and_finite(tmp_path):
    timestamps = np.arange(1, 31, dtype=np.int64) * HOUR
    btc = np.linspace(100.0, 130.0, len(timestamps))
    micro = []
    fut = []
    eth = []
    for i, ts in enumerate(timestamps):
        micro.append({
            "timestamp": int(ts), "agg_trade_count": 100 + i,
            "aggressive_quote_volume": 1_000_000 + i, "aggressive_imbalance": 0.1,
            "aggressive_buy_trade_ratio": 0.55, "agg_vwap": float(btc[i]),
            "avg_trade_quote": 1000.0, "p95_trade_quote": 5000.0, "max_trade_quote": 20000.0,
            "large_trade_quote_share": 0.2, "large_buy_quote_share": 0.6,
            "agg_price_range_pct": 0.01, "agg_intrahour_return": 0.002,
        })
        fut.append({"timestamp": int(ts), "close": float(btc[i] * 1.001), "quote_volume": 2_000_000 + i, "trade_count": 1000 + i, "taker_buy_quote_ratio": 0.52})
        eth.append({"timestamp": int(ts), "close": float(2000 + i * 5), "quote_volume": 3_000_000 + i, "trade_count": 2000 + i, "taker_buy_quote_ratio": 0.51})
    micro_path = tmp_path / "micro.jsonl"
    fut_path = tmp_path / "fut.jsonl"
    eth_path = tmp_path / "eth.jsonl"
    _write(micro_path, micro)
    _write(fut_path, fut)
    _write(eth_path, eth)

    matrix = build_context_feature_matrix(
        timestamps, btc, feature_set="full_context",
        micro_path=micro_path, futures_path=fut_path, eth_path=eth_path,
    )
    assert matrix.X.shape[0] == len(timestamps)
    assert matrix.X.shape[1] == len(matrix.feature_names)
    assert matrix.X.shape[1] > 20
    assert np.all(np.isfinite(matrix.X))


def test_technical_feature_set_adds_no_columns():
    ts = np.asarray([HOUR, 2 * HOUR], dtype=np.int64)
    btc = np.asarray([100.0, 101.0])
    matrix = build_context_feature_matrix(ts, btc, feature_set="technical")
    assert matrix.X.shape == (2, 0)
    assert matrix.feature_names == []
