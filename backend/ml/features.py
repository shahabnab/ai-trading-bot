from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


FEATURE_VERSION = "btc-hourly-tech-v2-gap-safe"
HOUR_MS = 60 * 60 * 1000

BASE_FEATURE_NAMES = [
    "log_return_1h",
    "return_2h",
    "return_3h",
    "return_6h",
    "return_12h",
    "return_24h",
    "return_168h",
    "realized_vol_6h",
    "realized_vol_24h",
    "realized_vol_168h",
    "candle_body_pct",
    "high_low_pct",
    "close_location",
    "log_volume",
    "volume_zscore_24h",
    "log_quote_volume",
    "quote_volume_zscore_24h",
    "log_trade_count",
    "taker_buy_quote_ratio",
    "ema_gap_10",
    "ema_gap_20",
    "ema_gap_50",
    "ema_gap_100",
    "ema_gap_200",
    "ema20_slope_3h",
    "ema50_slope_6h",
    "rsi_14",
    "macd_norm",
    "macd_signal_norm",
    "macd_hist_norm",
    "atr_14_norm",
    "bollinger_position_20",
    "bollinger_width_20",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
]

SENTIMENT_FEATURE_NAMES = [
    "fear_greed_centered",
    "log_news_count",
    "news_overall_sentiment_mean",
    "news_btc_sentiment_mean",
    "news_btc_relevance_mean",
]


@dataclass(frozen=True)
class FeatureDataset:
    timestamps: np.ndarray
    X: np.ndarray
    y_log_return: np.ndarray
    y_simple_return: np.ndarray
    closes: np.ndarray
    ema20: np.ndarray
    ema50: np.ndarray
    ema200: np.ndarray
    feature_names: list[str]

    @property
    def size(self) -> int:
        return int(self.X.shape[0])


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if isinstance(item, dict):
                rows.append(item)
    rows.sort(key=lambda row: int(row.get("timestamp", 0)))
    return rows


def _float(value: object, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _ema(values: np.ndarray, span: int) -> np.ndarray:
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(values, dtype=np.float64)
    out[:] = np.nan
    if len(values) == 0:
        return out
    out[0] = values[0]
    for idx in range(1, len(values)):
        out[idx] = alpha * values[idx] + (1.0 - alpha) * out[idx - 1]
    return out


def _rolling_mean_std(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    means = np.full(len(values), np.nan, dtype=np.float64)
    stds = np.full(len(values), np.nan, dtype=np.float64)
    for idx in range(window - 1, len(values)):
        segment = values[idx - window + 1 : idx + 1]
        if np.all(np.isfinite(segment)):
            means[idx] = float(np.mean(segment))
            stds[idx] = float(np.std(segment, ddof=0))
    return means, stds


def _rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    return _rolling_mean_std(values, window)[1]


def _rsi_wilder(closes: np.ndarray, period: int = 14) -> np.ndarray:
    out = np.full(len(closes), np.nan, dtype=np.float64)
    if len(closes) <= period:
        return out

    changes = np.diff(closes)
    gains = np.maximum(changes, 0.0)
    losses = np.maximum(-changes, 0.0)

    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))

    def value(gain: float, loss: float) -> float:
        if loss == 0.0:
            return 100.0 if gain > 0.0 else 50.0
        rs = gain / loss
        return 100.0 - 100.0 / (1.0 + rs)

    out[period] = value(avg_gain, avg_loss)
    for idx in range(period + 1, len(closes)):
        gain = gains[idx - 1]
        loss = losses[idx - 1]
        avg_gain = ((period - 1) * avg_gain + gain) / period
        avg_loss = ((period - 1) * avg_loss + loss) / period
        out[idx] = value(avg_gain, avg_loss)
    return out


def _atr_wilder(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    tr = np.full(len(close), np.nan, dtype=np.float64)
    if len(close) == 0:
        return tr
    tr[0] = high[0] - low[0]
    for idx in range(1, len(close)):
        tr[idx] = max(
            high[idx] - low[idx],
            abs(high[idx] - close[idx - 1]),
            abs(low[idx] - close[idx - 1]),
        )

    atr = np.full(len(close), np.nan, dtype=np.float64)
    if len(close) < period:
        return atr
    atr[period - 1] = float(np.mean(tr[:period]))
    for idx in range(period, len(close)):
        atr[idx] = ((period - 1) * atr[idx - 1] + tr[idx]) / period
    return atr


def _horizon_return(closes: np.ndarray, horizon: int) -> np.ndarray:
    out = np.full(len(closes), np.nan, dtype=np.float64)
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    for idx in range(horizon, len(closes)):
        previous = closes[idx - horizon]
        if previous > 0.0:
            out[idx] = closes[idx] / previous - 1.0
    return out


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    out = np.full(len(numerator), np.nan, dtype=np.float64)
    mask = np.isfinite(numerator) & np.isfinite(denominator) & (denominator != 0.0)
    out[mask] = numerator[mask] / denominator[mask]
    return out


def _continuous_slices(timestamps: np.ndarray) -> list[slice]:
    if len(timestamps) == 0:
        return []
    diffs = np.diff(timestamps)
    if np.any(diffs <= 0):
        raise ValueError("timestamps must be strictly increasing and unique")
    breaks = np.flatnonzero(diffs != HOUR_MS) + 1
    starts = np.concatenate(([0], breaks))
    ends = np.concatenate((breaks, [len(timestamps)]))
    return [slice(int(start), int(end)) for start, end in zip(starts, ends, strict=True)]


def build_feature_dataset(
    rows: Iterable[dict[str, Any]],
    *,
    include_sentiment: bool = False,
) -> FeatureDataset:
    ordered = sorted(rows, key=lambda row: int(row.get("timestamp", 0)))
    if not ordered:
        raise ValueError("training dataset is empty")

    timestamps = np.asarray([int(row["timestamp"]) for row in ordered], dtype=np.int64)
    segments = _continuous_slices(timestamps)
    opens = np.asarray([_float(row.get("open")) for row in ordered], dtype=np.float64)
    highs = np.asarray([_float(row.get("high")) for row in ordered], dtype=np.float64)
    lows = np.asarray([_float(row.get("low")) for row in ordered], dtype=np.float64)
    closes = np.asarray([_float(row.get("close")) for row in ordered], dtype=np.float64)
    volumes = np.asarray([max(_float(row.get("volume"), 0.0), 0.0) for row in ordered], dtype=np.float64)
    quote_volumes = np.asarray(
        [max(_float(row.get("quote_volume"), 0.0), 0.0) for row in ordered], dtype=np.float64
    )
    trade_counts = np.asarray(
        [max(_float(row.get("number_of_trades"), 0.0), 0.0) for row in ordered], dtype=np.float64
    )
    taker_ratio = np.asarray(
        [_float(row.get("taker_buy_quote_ratio"), 0.0) for row in ordered], dtype=np.float64
    )

    n = len(closes)
    log_return_1h = np.full(n, np.nan, dtype=np.float64)
    returns = {h: np.full(n, np.nan, dtype=np.float64) for h in (2, 3, 6, 12, 24, 168)}
    realized_vol = {h: np.full(n, np.nan, dtype=np.float64) for h in (6, 24, 168)}

    log_volume = np.log1p(volumes)
    log_quote_volume = np.log1p(quote_volumes)
    volume_z_24 = np.full(n, np.nan, dtype=np.float64)
    quote_z_24 = np.full(n, np.nan, dtype=np.float64)

    ema = {span: np.full(n, np.nan, dtype=np.float64) for span in (10, 20, 50, 100, 200)}
    ema20_slope_3h = np.full(n, np.nan, dtype=np.float64)
    ema50_slope_6h = np.full(n, np.nan, dtype=np.float64)
    rsi14 = np.full(n, np.nan, dtype=np.float64)
    macd_norm = np.full(n, np.nan, dtype=np.float64)
    macd_signal_norm = np.full(n, np.nan, dtype=np.float64)
    macd_hist_norm = np.full(n, np.nan, dtype=np.float64)
    atr14_norm = np.full(n, np.nan, dtype=np.float64)
    bb_position = np.full(n, np.nan, dtype=np.float64)
    bb_width = np.full(n, np.nan, dtype=np.float64)

    for segment in segments:
        seg_close = closes[segment]
        seg_high = highs[segment]
        seg_low = lows[segment]
        seg_log_volume = log_volume[segment]
        seg_log_quote_volume = log_quote_volume[segment]

        seg_log_return = np.full(len(seg_close), np.nan, dtype=np.float64)
        if len(seg_close) > 1:
            valid_prev = (seg_close[:-1] > 0.0) & (seg_close[1:] > 0.0)
            seg_log_return[1:][valid_prev] = np.log(seg_close[1:][valid_prev] / seg_close[:-1][valid_prev])
        log_return_1h[segment] = seg_log_return

        for horizon in returns:
            returns[horizon][segment] = _horizon_return(seg_close, horizon)
        for horizon in realized_vol:
            realized_vol[horizon][segment] = _rolling_std(seg_log_return, horizon)

        volume_mean_24, volume_std_24 = _rolling_mean_std(seg_log_volume, 24)
        quote_mean_24, quote_std_24 = _rolling_mean_std(seg_log_quote_volume, 24)
        volume_z_24[segment] = _safe_divide(seg_log_volume - volume_mean_24, volume_std_24)
        quote_z_24[segment] = _safe_divide(seg_log_quote_volume - quote_mean_24, quote_std_24)

        local_ema = {span: _ema(seg_close, span) for span in ema}
        for span in ema:
            ema[span][segment] = local_ema[span]

        local_ema20_slope = np.full(len(seg_close), np.nan, dtype=np.float64)
        local_ema50_slope = np.full(len(seg_close), np.nan, dtype=np.float64)
        if len(seg_close) > 3:
            local_ema20_slope[3:] = _safe_divide(local_ema[20][3:], local_ema[20][:-3]) - 1.0
        if len(seg_close) > 6:
            local_ema50_slope[6:] = _safe_divide(local_ema[50][6:], local_ema[50][:-6]) - 1.0
        ema20_slope_3h[segment] = local_ema20_slope
        ema50_slope_6h[segment] = local_ema50_slope

        rsi14[segment] = _rsi_wilder(seg_close, 14) / 100.0
        ema12 = _ema(seg_close, 12)
        ema26 = _ema(seg_close, 26)
        macd = ema12 - ema26
        macd_signal = _ema(macd, 9)
        macd_hist = macd - macd_signal
        macd_norm[segment] = _safe_divide(macd, seg_close)
        macd_signal_norm[segment] = _safe_divide(macd_signal, seg_close)
        macd_hist_norm[segment] = _safe_divide(macd_hist, seg_close)

        atr14 = _atr_wilder(seg_high, seg_low, seg_close, 14)
        atr14_norm[segment] = _safe_divide(atr14, seg_close)

        bb_mean_20, bb_std_20 = _rolling_mean_std(seg_close, 20)
        bb_position[segment] = _safe_divide(seg_close - bb_mean_20, 2.0 * bb_std_20)
        bb_width[segment] = _safe_divide(4.0 * bb_std_20, bb_mean_20)

    candle_body_pct = _safe_divide(closes - opens, opens)
    high_low_pct = _safe_divide(highs - lows, opens)
    close_location = _safe_divide(closes - lows, highs - lows)
    close_location = close_location * 2.0 - 1.0

    ema_gap = {span: _safe_divide(closes, values) - 1.0 for span, values in ema.items()}

    hours = ((timestamps // HOUR_MS) % 24).astype(np.float64)
    days = ((timestamps // (24 * HOUR_MS) + 4) % 7).astype(np.float64)
    hour_angle = 2.0 * np.pi * hours / 24.0
    day_angle = 2.0 * np.pi * days / 7.0

    columns = [
        log_return_1h,
        returns[2],
        returns[3],
        returns[6],
        returns[12],
        returns[24],
        returns[168],
        realized_vol[6],
        realized_vol[24],
        realized_vol[168],
        candle_body_pct,
        high_low_pct,
        close_location,
        log_volume,
        volume_z_24,
        log_quote_volume,
        quote_z_24,
        np.log1p(trade_counts),
        taker_ratio,
        ema_gap[10],
        ema_gap[20],
        ema_gap[50],
        ema_gap[100],
        ema_gap[200],
        ema20_slope_3h,
        ema50_slope_6h,
        rsi14,
        macd_norm,
        macd_signal_norm,
        macd_hist_norm,
        atr14_norm,
        bb_position,
        bb_width,
        np.sin(hour_angle),
        np.cos(hour_angle),
        np.sin(day_angle),
        np.cos(day_angle),
    ]
    feature_names = list(BASE_FEATURE_NAMES)

    if include_sentiment:
        fear_greed = np.asarray(
            [(_float(row.get("fear_greed_value"), 50.0) - 50.0) / 50.0 for row in ordered],
            dtype=np.float64,
        )
        news_count = np.asarray(
            [max(_float(row.get("news_count"), 0.0), 0.0) for row in ordered], dtype=np.float64
        )
        overall_sent = np.asarray(
            [_float(row.get("news_overall_sentiment_mean"), 0.0) for row in ordered], dtype=np.float64
        )
        btc_sent = np.asarray(
            [_float(row.get("news_btc_sentiment_mean"), 0.0) for row in ordered], dtype=np.float64
        )
        relevance = np.asarray(
            [_float(row.get("news_btc_relevance_mean"), 0.0) for row in ordered], dtype=np.float64
        )
        columns.extend([fear_greed, np.log1p(news_count), overall_sent, btc_sent, relevance])
        feature_names.extend(SENTIMENT_FEATURE_NAMES)

    X = np.column_stack(columns)
    simple_target = np.asarray(
        [_float(row.get("target_return_1h")) for row in ordered], dtype=np.float64
    )
    log_target = np.full(len(simple_target), np.nan, dtype=np.float64)
    target_mask = np.isfinite(simple_target) & (simple_target > -1.0)
    log_target[target_mask] = np.log1p(simple_target[target_mask])

    valid = np.all(np.isfinite(X), axis=1)
    valid &= np.isfinite(log_target)
    valid &= np.isfinite(closes)
    valid &= closes > 0.0

    if not np.any(valid):
        raise ValueError("no usable rows remain after causal feature warm-up")

    return FeatureDataset(
        timestamps=timestamps[valid],
        X=X[valid].astype(np.float32, copy=False),
        y_log_return=log_target[valid].astype(np.float32, copy=False),
        y_simple_return=simple_target[valid].astype(np.float64, copy=False),
        closes=closes[valid].astype(np.float64, copy=False),
        ema20=ema[20][valid].astype(np.float64, copy=False),
        ema50=ema[50][valid].astype(np.float64, copy=False),
        ema200=ema[200][valid].astype(np.float64, copy=False),
        feature_names=feature_names,
    )
