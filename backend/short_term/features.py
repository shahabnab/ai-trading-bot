from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class ShortTermFeatures:
    timestamp: int
    close: float
    return_15m: float
    return_30m: float
    return_1h: float
    return_2h: float
    ema_gap_bps: float
    ema_fast_slope_bps: float
    rsi_14: float
    atr_bps: float
    bollinger_z: float
    volume_z: float
    value_z: float
    vwap_distance_bps: float
    candle_range_bps: float
    candle_body_bps: float
    realized_vol_1h_bps: float
    trade_count: int
    buy_notional: float
    sell_notional: float
    trade_imbalance: float
    buy_ratio: float
    spread_bps: float | None
    book_imbalance: float | None
    microstructure_coverage: float
    quality: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _f(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _ema(values: np.ndarray, span: int) -> np.ndarray:
    if len(values) == 0:
        return values.copy()
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(values, dtype=np.float64)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


def _zscore_latest(values: np.ndarray, window: int) -> float:
    sample = values[-window:]
    if len(sample) < 2:
        return 0.0
    std = float(np.std(sample, ddof=0))
    if std <= 1e-12:
        return 0.0
    return float((sample[-1] - np.mean(sample)) / std)


def _rsi(values: np.ndarray, period: int = 14) -> float:
    if len(values) < period + 1:
        return 50.0
    diff = np.diff(values[-(period + 1):])
    gains = np.clip(diff, 0.0, None)
    losses = np.clip(-diff, 0.0, None)
    avg_gain = float(np.mean(gains))
    avg_loss = float(np.mean(losses))
    if avg_loss <= 1e-12:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return float(100.0 - 100.0 / (1.0 + rs))


def _return(values: np.ndarray, bars: int) -> float:
    if len(values) <= bars or values[-bars - 1] <= 0:
        return 0.0
    return float(values[-1] / values[-bars - 1] - 1.0)


def _micro_number(micro: Mapping[str, Any] | None, key: str, default: float = 0.0) -> float:
    return _f(micro.get(key), default) if micro else default


def build_short_term_features(
    candles: list[Mapping[str, Any]],
    microstructure: Mapping[str, Any] | None = None,
) -> ShortTermFeatures:
    """Build point-in-time 15-minute features from completed candles only.

    Candle volume/value capture activity at the bar level. The optional
    microstructure row adds taker buy/sell flow plus order-book spread and
    imbalance collected during the same 15-minute bucket.
    """
    rows: list[dict[str, float]] = []
    for row in candles:
        try:
            ts = int(row.get("created_at", row.get("timestamp", 0)))
            op = _f(row.get("open")); hi = _f(row.get("high")); lo = _f(row.get("low")); cl = _f(row.get("close"))
            vol = max(_f(row.get("volume")), 0.0); value = max(_f(row.get("value")), 0.0)
        except (TypeError, ValueError):
            continue
        if ts <= 0 or min(op, hi, lo, cl) <= 0:
            continue
        rows.append({"timestamp": float(ts), "open": op, "high": hi, "low": lo, "close": cl, "volume": vol, "value": value})
    rows.sort(key=lambda x: x["timestamp"])
    if len(rows) < 30:
        raise ValueError("At least 30 completed 15-minute candles are required")

    opens = np.asarray([r["open"] for r in rows], dtype=np.float64)
    highs = np.asarray([r["high"] for r in rows], dtype=np.float64)
    lows = np.asarray([r["low"] for r in rows], dtype=np.float64)
    closes = np.asarray([r["close"] for r in rows], dtype=np.float64)
    volumes = np.asarray([r["volume"] for r in rows], dtype=np.float64)
    values = np.asarray([r["value"] for r in rows], dtype=np.float64)

    fast = _ema(closes, 8)
    slow = _ema(closes, 21)
    ema_gap_bps = float((fast[-1] / slow[-1] - 1.0) * 10_000.0) if slow[-1] > 0 else 0.0
    ema_fast_slope_bps = float((fast[-1] / fast[-2] - 1.0) * 10_000.0) if len(fast) >= 2 and fast[-2] > 0 else 0.0

    prev_close = np.concatenate(([closes[0]], closes[:-1]))
    true_range = np.maximum(highs - lows, np.maximum(np.abs(highs - prev_close), np.abs(lows - prev_close)))
    atr = float(np.mean(true_range[-14:]))
    atr_bps = atr / closes[-1] * 10_000.0

    mean20 = float(np.mean(closes[-20:]))
    std20 = float(np.std(closes[-20:], ddof=0))
    boll_z = (closes[-1] - mean20) / std20 if std20 > 1e-12 else 0.0

    typical = (highs + lows + closes) / 3.0
    volume20 = volumes[-20:]
    volume_sum = float(np.sum(volume20))
    vwap = float(np.sum(typical[-20:] * volume20) / volume_sum) if volume_sum > 1e-12 else closes[-1]
    vwap_distance_bps = float((closes[-1] / vwap - 1.0) * 10_000.0) if vwap > 0 else 0.0

    log_returns = np.diff(np.log(closes[-5:])) if len(closes) >= 5 else np.asarray([], dtype=np.float64)
    realized_vol_1h_bps = float(np.std(log_returns, ddof=0) * np.sqrt(max(len(log_returns), 1)) * 10_000.0) if len(log_returns) else 0.0

    buy_notional = max(_micro_number(microstructure, "buy_notional"), 0.0)
    sell_notional = max(_micro_number(microstructure, "sell_notional"), 0.0)
    total_flow = buy_notional + sell_notional
    computed_imbalance = (buy_notional - sell_notional) / total_flow if total_flow > 1e-12 else 0.0
    trade_imbalance = _micro_number(microstructure, "trade_imbalance", computed_imbalance)
    buy_ratio = _micro_number(microstructure, "buy_ratio", buy_notional / total_flow if total_flow > 1e-12 else 0.5)
    spread_raw = microstructure.get("spread_bps") if microstructure else None
    book_raw = microstructure.get("book_imbalance") if microstructure else None
    spread_bps = _f(spread_raw) if spread_raw is not None else None
    book_imbalance = _f(book_raw) if book_raw is not None else None
    coverage = float(np.clip(_micro_number(microstructure, "coverage_fraction"), 0.0, 1.0))

    candle_quality = min(1.0, len(rows) / 96.0)
    micro_quality = coverage if microstructure else 0.0
    quality = float(np.clip(0.75 * candle_quality + 0.25 * micro_quality, 0.0, 1.0))

    return ShortTermFeatures(
        timestamp=int(rows[-1]["timestamp"]),
        close=float(closes[-1]),
        return_15m=_return(closes, 1),
        return_30m=_return(closes, 2),
        return_1h=_return(closes, 4),
        return_2h=_return(closes, 8),
        ema_gap_bps=ema_gap_bps,
        ema_fast_slope_bps=ema_fast_slope_bps,
        rsi_14=_rsi(closes, 14),
        atr_bps=float(atr_bps),
        bollinger_z=float(boll_z),
        volume_z=_zscore_latest(volumes, 20),
        value_z=_zscore_latest(values, 20),
        vwap_distance_bps=vwap_distance_bps,
        candle_range_bps=float((highs[-1] - lows[-1]) / closes[-1] * 10_000.0),
        candle_body_bps=float((closes[-1] / opens[-1] - 1.0) * 10_000.0),
        realized_vol_1h_bps=realized_vol_1h_bps,
        trade_count=int(_micro_number(microstructure, "trade_count")),
        buy_notional=buy_notional,
        sell_notional=sell_notional,
        trade_imbalance=float(np.clip(trade_imbalance, -1.0, 1.0)),
        buy_ratio=float(np.clip(buy_ratio, 0.0, 1.0)),
        spread_bps=spread_bps,
        book_imbalance=float(np.clip(book_imbalance, -1.0, 1.0)) if book_imbalance is not None else None,
        microstructure_coverage=coverage,
        quality=quality,
    )
