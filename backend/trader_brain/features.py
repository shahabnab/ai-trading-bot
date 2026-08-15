from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

HOUR_MS = 3_600_000
TECHNICAL_SIGNAL_NAMES = (
    "T1_trend_fast_slow",
    "T2_long_trend_location",
    "T3_breakout_strength",
    "T4_range_position",
    "T5_rsi14",
    "T6_macd_atr",
    "T7_bollinger_z",
    "T8_vol_expansion",
    "T9_volume_z",
    "T10_trend_efficiency",
)
REGIME_FEATURE_NAMES = (
    "btc_return_24h",
    "btc_realized_vol_24h",
    "btc_drawdown_7d",
    "volume_z_24h",
    "trend_efficiency_24h",
    "atr_pct",
)


@dataclass(frozen=True)
class TraderFeatureRow:
    timestamp: int
    close: float
    realized_vol_24h: float
    technical: dict[str, float]
    regime_vector: tuple[float, ...]


def normalize_candles(rows: Iterable[dict[str, Any]]) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    for item in rows:
        try:
            ts = int(item.get("created_at", item.get("timestamp", 0)))
            o = float(item["open"])
            h = float(item["high"])
            l = float(item["low"])
            c = float(item["close"])
            volume = float(item.get("volume", 0.0))
        except (KeyError, TypeError, ValueError):
            continue
        if ts > 0 and min(o, h, l, c) > 0 and np.all(np.isfinite([o, h, l, c, volume])):
            out.append({"timestamp": ts, "open": o, "high": h, "low": l, "close": c, "volume": max(volume, 0.0)})
    out.sort(key=lambda row: int(row["timestamp"]))
    dedup: dict[int, dict[str, float]] = {int(row["timestamp"]): row for row in out}
    return [dedup[key] for key in sorted(dedup)]


def _ema(values: np.ndarray, span: int) -> np.ndarray:
    out = np.full(len(values), np.nan, dtype=np.float64)
    if len(values) < span:
        return out
    alpha = 2.0 / (span + 1.0)
    out[span - 1] = float(np.mean(values[:span]))
    for idx in range(span, len(values)):
        out[idx] = alpha * values[idx] + (1.0 - alpha) * out[idx - 1]
    return out


def _rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(values), np.nan, dtype=np.float64)
    for idx in range(window - 1, len(values)):
        out[idx] = float(np.std(values[idx - window + 1 : idx + 1]))
    return out


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(values), np.nan, dtype=np.float64)
    for idx in range(window - 1, len(values)):
        out[idx] = float(np.mean(values[idx - window + 1 : idx + 1]))
    return out


def _rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    out = np.full(len(closes), np.nan, dtype=np.float64)
    if len(closes) <= period:
        return out
    delta = np.diff(closes)
    gains = np.maximum(delta, 0.0)
    losses = np.maximum(-delta, 0.0)
    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))
    for idx in range(period, len(closes)):
        if idx > period:
            avg_gain = ((period - 1) * avg_gain + gains[idx - 1]) / period
            avg_loss = ((period - 1) * avg_loss + losses[idx - 1]) / period
        out[idx] = 100.0 if avg_loss <= 1e-12 else 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    return out


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    tr = np.full(len(close), np.nan, dtype=np.float64)
    if not len(close):
        return tr
    tr[0] = high[0] - low[0]
    for idx in range(1, len(close)):
        tr[idx] = max(high[idx] - low[idx], abs(high[idx] - close[idx - 1]), abs(low[idx] - close[idx - 1]))
    out = np.full(len(close), np.nan, dtype=np.float64)
    if len(close) < period:
        return out
    out[period - 1] = float(np.mean(tr[:period]))
    for idx in range(period, len(close)):
        out[idx] = ((period - 1) * out[idx - 1] + tr[idx]) / period
    return out


def _safe(value: float, default: float = 0.0) -> float:
    return float(value) if np.isfinite(value) else default


def build_trader_feature_history(rows: Iterable[dict[str, Any]]) -> list[TraderFeatureRow]:
    candles = normalize_candles(rows)
    if len(candles) < 220:
        raise ValueError("Trader-Brain requires at least 220 complete hourly candles")
    timestamps = np.asarray([int(row["timestamp"]) for row in candles], dtype=np.int64)
    if np.any(np.diff(timestamps) != HOUR_MS):
        # Use only the newest continuous hourly segment so rolling indicators cannot jump over gaps.
        last_gap = np.flatnonzero(np.diff(timestamps) != HOUR_MS)
        start = int(last_gap[-1] + 1) if len(last_gap) else 0
        candles = candles[start:]
        if len(candles) < 220:
            raise ValueError("Newest continuous hourly segment is too short for Trader-Brain")
        timestamps = np.asarray([int(row["timestamp"]) for row in candles], dtype=np.int64)

    close = np.asarray([float(row["close"]) for row in candles], dtype=np.float64)
    high = np.asarray([float(row["high"]) for row in candles], dtype=np.float64)
    low = np.asarray([float(row["low"]) for row in candles], dtype=np.float64)
    volume = np.asarray([float(row["volume"]) for row in candles], dtype=np.float64)
    log_close = np.log(close)
    ret1 = np.diff(log_close, prepend=np.nan)
    ema12, ema20, ema26, ema100, ema200 = (_ema(close, span) for span in (12, 20, 26, 100, 200))
    macd = ema12 - ema26
    atr = _atr(high, low, close, 14)
    rsi = _rsi(close, 14)
    mean20 = _rolling_mean(close, 20)
    std20 = _rolling_std(close, 20)
    vol24 = _rolling_std(ret1, 24)
    vol168 = _rolling_std(ret1, 168)
    volume_log = np.log1p(volume)
    volume_mean = _rolling_mean(volume_log, 24)
    volume_std = _rolling_std(volume_log, 24)

    history: list[TraderFeatureRow] = []
    for idx in range(200, len(candles)):
        atr_i = max(_safe(atr[idx], close[idx] * 1e-4), close[idx] * 1e-6)
        previous_high = float(np.max(high[max(0, idx - 20) : idx]))
        previous_low = float(np.min(low[max(0, idx - 20) : idx]))
        range_width = max(previous_high - previous_low, 1e-12)
        trend_eff_num = close[idx] - close[max(0, idx - 24)]
        path = float(np.sum(np.abs(np.diff(close[max(0, idx - 24) : idx + 1]))))
        trend_eff = trend_eff_num / max(path, 1e-12)
        breakout = (close[idx] - previous_high) / atr_i if close[idx] >= previous_high else (close[idx] - previous_low) / atr_i if close[idx] <= previous_low else 0.0
        vol_ratio = _safe(vol24[idx] / max(_safe(vol168[idx], 1e-8), 1e-8) - 1.0)
        vol_z = _safe((volume_log[idx] - volume_mean[idx]) / max(_safe(volume_std[idx], 1e-8), 1e-8))
        boll_z = _safe((close[idx] - mean20[idx]) / max(_safe(std20[idx], 1e-8), 1e-8))
        technical = {
            "T1_trend_fast_slow": _safe((ema20[idx] - ema100[idx]) / atr_i),
            "T2_long_trend_location": _safe((close[idx] - ema200[idx]) / atr_i),
            "T3_breakout_strength": _safe(breakout),
            "T4_range_position": float(np.clip((close[idx] - previous_low) / range_width, 0.0, 1.0)),
            "T5_rsi14": _safe(rsi[idx], 50.0),
            "T6_macd_atr": _safe(macd[idx] / atr_i),
            "T7_bollinger_z": boll_z,
            "T8_vol_expansion": vol_ratio,
            "T9_volume_z": vol_z,
            "T10_trend_efficiency": float(np.clip(trend_eff, -1.0, 1.0)),
        }
        ret24 = _safe(log_close[idx] - log_close[idx - 24])
        peak168 = float(np.max(close[max(0, idx - 167) : idx + 1]))
        drawdown = close[idx] / max(peak168, 1e-12) - 1.0
        regime_vector = (
            ret24,
            max(_safe(vol24[idx]), 0.0),
            drawdown,
            vol_z,
            technical["T10_trend_efficiency"],
            atr_i / close[idx],
        )
        if np.all(np.isfinite(regime_vector)):
            history.append(
                TraderFeatureRow(
                    timestamp=int(timestamps[idx]),
                    close=float(close[idx]),
                    realized_vol_24h=max(_safe(vol24[idx]), 1e-6),
                    technical=technical,
                    regime_vector=tuple(float(value) for value in regime_vector),
                )
            )
    if not history:
        raise ValueError("No finite Trader-Brain feature rows were produced")
    return history
