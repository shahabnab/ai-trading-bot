from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

HOUR_MS = 60 * 60 * 1000

MICRO_FEATURE_NAMES = [
    "micro_log_agg_trade_count",
    "micro_log_aggressive_quote_volume",
    "micro_aggressive_imbalance",
    "micro_aggressive_buy_trade_ratio",
    "micro_vwap_gap",
    "micro_log_avg_trade_quote",
    "micro_log_p95_trade_quote",
    "micro_log_max_trade_quote",
    "micro_large_trade_quote_share",
    "micro_large_buy_quote_share",
    "micro_price_range_pct",
    "micro_intrahour_return",
    "micro_available",
]

FUTURES_FEATURE_NAMES = [
    "futures_basis",
    "futures_return_1h",
    "futures_return_6h",
    "futures_return_24h",
    "futures_log_quote_volume",
    "futures_quote_volume_z24",
    "futures_log_trade_count",
    "futures_taker_buy_quote_ratio",
    "futures_available",
]

CROSS_ASSET_FEATURE_NAMES = [
    "eth_return_1h",
    "eth_return_6h",
    "eth_return_24h",
    "eth_realized_vol_24h",
    "eth_log_quote_volume",
    "eth_quote_volume_z24",
    "eth_taker_buy_quote_ratio",
    "eth_log_trade_count",
    "btc_minus_eth_return_1h",
    "btc_minus_eth_return_6h",
    "eth_available",
]


@dataclass(frozen=True)
class ContextFeatureMatrix:
    X: np.ndarray
    feature_names: list[str]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and "timestamp" in row:
                rows.append(row)
    rows.sort(key=lambda row: int(row["timestamp"]))
    return rows


def _float(row: dict[str, Any] | None, key: str, default: float = 0.0) -> float:
    if row is None:
        return default
    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


def _maps(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(row["timestamp"]): row for row in rows}


def _return_at(mapping: dict[int, dict[str, Any]], timestamp: int, hours: int) -> float:
    current = mapping.get(timestamp)
    previous = mapping.get(timestamp - hours * HOUR_MS)
    c = _float(current, "close", 0.0)
    p = _float(previous, "close", 0.0)
    if c > 0.0 and p > 0.0:
        return c / p - 1.0
    return 0.0


def _rolling_z(mapping: dict[int, dict[str, Any]], timestamp: int, key: str, window: int = 24) -> float:
    values: list[float] = []
    for offset in range(window):
        row = mapping.get(timestamp - offset * HOUR_MS)
        if row is None:
            return 0.0
        values.append(np.log1p(max(_float(row, key, 0.0), 0.0)))
    array = np.asarray(values, dtype=np.float64)
    std = float(np.std(array))
    if std <= 1e-12:
        return 0.0
    return float((array[0] - np.mean(array)) / std)


def _realized_vol(mapping: dict[int, dict[str, Any]], timestamp: int, window: int = 24) -> float:
    values: list[float] = []
    for offset in range(window):
        t = timestamp - offset * HOUR_MS
        current = mapping.get(t)
        previous = mapping.get(t - HOUR_MS)
        c = _float(current, "close", 0.0)
        p = _float(previous, "close", 0.0)
        if c <= 0.0 or p <= 0.0:
            return 0.0
        values.append(float(np.log(c / p)))
    return float(np.std(np.asarray(values, dtype=np.float64)))


def build_context_feature_matrix(
    timestamps: np.ndarray,
    btc_closes: np.ndarray,
    *,
    feature_set: str,
    micro_path: Path = Path("data/processed/context/btc_spot_aggtrades_hourly.jsonl"),
    futures_path: Path = Path("data/processed/context/btc_um_futures_hourly.jsonl"),
    eth_path: Path = Path("data/processed/context/eth_spot_hourly.jsonl"),
) -> ContextFeatureMatrix:
    """Build causal auxiliary market features aligned to BTC timestamps.

    Supported feature sets:
      technical         -> no auxiliary columns
      technical_micro   -> BTC spot aggTrade microstructure
      full_context      -> microstructure + BTC perpetual futures + ETH spot
    """
    if feature_set not in {"technical", "technical_micro", "full_context"}:
        raise ValueError(f"unknown feature_set: {feature_set}")
    ts = np.asarray(timestamps, dtype=np.int64)
    closes = np.asarray(btc_closes, dtype=np.float64)
    if ts.shape != closes.shape:
        raise ValueError("timestamps and btc_closes must have identical shapes")
    if feature_set == "technical":
        return ContextFeatureMatrix(np.empty((len(ts), 0), dtype=np.float32), [])

    micro_rows = _read_jsonl(micro_path)
    micro = _maps(micro_rows)
    if not micro_rows:
        raise FileNotFoundError(f"microstructure context missing: {micro_path}")

    futures: dict[int, dict[str, Any]] = {}
    eth: dict[int, dict[str, Any]] = {}
    if feature_set == "full_context":
        futures_rows = _read_jsonl(futures_path)
        eth_rows = _read_jsonl(eth_path)
        if not futures_rows:
            raise FileNotFoundError(f"futures context missing: {futures_path}")
        if not eth_rows:
            raise FileNotFoundError(f"ETH context missing: {eth_path}")
        futures = _maps(futures_rows)
        eth = _maps(eth_rows)

    rows: list[list[float]] = []
    names = list(MICRO_FEATURE_NAMES)
    if feature_set == "full_context":
        names += FUTURES_FEATURE_NAMES + CROSS_ASSET_FEATURE_NAMES

    btc_close_map = {int(t): float(c) for t, c in zip(ts, closes, strict=True)}

    for timestamp, btc_close in zip(ts, closes, strict=True):
        t = int(timestamp)
        m = micro.get(t)
        m_available = 1.0 if m is not None else 0.0
        m_vwap = _float(m, "agg_vwap", btc_close)
        row = [
            float(np.log1p(max(_float(m, "agg_trade_count", 0.0), 0.0))),
            float(np.log1p(max(_float(m, "aggressive_quote_volume", 0.0), 0.0))),
            _float(m, "aggressive_imbalance", 0.0),
            _float(m, "aggressive_buy_trade_ratio", 0.5),
            m_vwap / btc_close - 1.0 if btc_close > 0.0 and m_vwap > 0.0 else 0.0,
            float(np.log1p(max(_float(m, "avg_trade_quote", 0.0), 0.0))),
            float(np.log1p(max(_float(m, "p95_trade_quote", 0.0), 0.0))),
            float(np.log1p(max(_float(m, "max_trade_quote", 0.0), 0.0))),
            _float(m, "large_trade_quote_share", 0.0),
            _float(m, "large_buy_quote_share", 0.5),
            _float(m, "agg_price_range_pct", 0.0),
            _float(m, "agg_intrahour_return", 0.0),
            m_available,
        ]

        if feature_set == "full_context":
            f = futures.get(t)
            f_available = 1.0 if f is not None else 0.0
            f_close = _float(f, "close", 0.0)
            row += [
                f_close / btc_close - 1.0 if f_close > 0.0 and btc_close > 0.0 else 0.0,
                _return_at(futures, t, 1),
                _return_at(futures, t, 6),
                _return_at(futures, t, 24),
                float(np.log1p(max(_float(f, "quote_volume", 0.0), 0.0))),
                _rolling_z(futures, t, "quote_volume", 24),
                float(np.log1p(max(_float(f, "trade_count", 0.0), 0.0))),
                _float(f, "taker_buy_quote_ratio", 0.5),
                f_available,
            ]

            e = eth.get(t)
            e_available = 1.0 if e is not None else 0.0
            eth_r1 = _return_at(eth, t, 1)
            eth_r6 = _return_at(eth, t, 6)
            eth_r24 = _return_at(eth, t, 24)
            btc_prev1 = btc_close_map.get(t - HOUR_MS, 0.0)
            btc_prev6 = btc_close_map.get(t - 6 * HOUR_MS, 0.0)
            btc_r1 = btc_close / btc_prev1 - 1.0 if btc_close > 0.0 and btc_prev1 > 0.0 else 0.0
            btc_r6 = btc_close / btc_prev6 - 1.0 if btc_close > 0.0 and btc_prev6 > 0.0 else 0.0
            row += [
                eth_r1,
                eth_r6,
                eth_r24,
                _realized_vol(eth, t, 24),
                float(np.log1p(max(_float(e, "quote_volume", 0.0), 0.0))),
                _rolling_z(eth, t, "quote_volume", 24),
                _float(e, "taker_buy_quote_ratio", 0.5),
                float(np.log1p(max(_float(e, "trade_count", 0.0), 0.0))),
                btc_r1 - eth_r1,
                btc_r6 - eth_r6,
                e_available,
            ]
        rows.append(row)

    matrix = np.asarray(rows, dtype=np.float32)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("context feature matrix contains non-finite values")
    return ContextFeatureMatrix(matrix, names)
