from __future__ import annotations

import math
from typing import Mapping

import numpy as np

from .contracts import ExpertForecast


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - float(np.max(values))
    exp = np.exp(shifted)
    return exp / float(exp.sum())


def _forecast(
    *,
    name: str,
    score: float,
    sigma: float,
    quality: float,
    timestamp: int,
    signals: Mapping[str, float],
    explanation: tuple[str, ...],
    available: bool = True,
) -> ExpertForecast:
    score = float(np.clip(score, -1.0, 1.0))
    sigma = float(max(sigma, 1e-6))
    quality = float(np.clip(quality, 0.0, 1.0))
    probs = _softmax(np.asarray([-2.2 * score, 1.15 * (1.0 - abs(score)), 2.2 * score], dtype=np.float64))
    confidence = float(np.max(probs)) * (0.5 + 0.5 * quality)
    return ExpertForecast(
        name=name,
        p_down=float(probs[0]),
        p_flat=float(probs[1]),
        p_up=float(probs[2]),
        expected_return=score * sigma,
        sigma=sigma,
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        quality=quality,
        available=bool(available and quality > 0.0),
        timestamp=int(timestamp),
        signals={str(k): float(v) for k, v in signals.items() if np.isfinite(v)},
        explanation=explanation,
    )


def technical_expert(signals: Mapping[str, float], *, realized_vol_24h: float, timestamp: int) -> ExpertForecast:
    t1 = math.tanh(float(signals.get("T1_trend_fast_slow", 0.0)) / 2.0)
    t2 = math.tanh(float(signals.get("T2_long_trend_location", 0.0)) / 3.0)
    t3 = math.tanh(float(signals.get("T3_breakout_strength", 0.0)))
    t4 = float(np.clip((float(signals.get("T4_range_position", 0.5)) - 0.5) * 2.0, -1.0, 1.0))
    rsi_context = float(np.clip((float(signals.get("T5_rsi14", 50.0)) - 50.0) / 50.0, -1.0, 1.0))
    t6 = math.tanh(float(signals.get("T6_macd_atr", 0.0)) * 3.0)
    t7 = math.tanh(float(signals.get("T7_bollinger_z", 0.0)) / 2.0)
    t8 = math.tanh(float(signals.get("T8_vol_expansion", 0.0)))
    t9 = math.tanh(float(signals.get("T9_volume_z", 0.0)) / 2.0)
    t10 = float(np.clip(float(signals.get("T10_trend_efficiency", 0.0)), -1.0, 1.0))
    breakout_confirmation = t3 * max(0.0, t9) * max(0.0, t8 + 0.25)
    score = (
        0.24 * t1 + 0.18 * t2 + 0.16 * t6 + 0.18 * t10 +
        0.08 * t4 + 0.08 * t7 + 0.12 * breakout_confirmation -
        0.06 * rsi_context * abs(rsi_context)
    )
    return _forecast(
        name="technical",
        score=score,
        sigma=realized_vol_24h,
        quality=1.0,
        timestamp=timestamp,
        signals=signals,
        explanation=(
            "T1-T10 quantify trend, breakout, range, momentum, volatility and volume.",
            "RSI is context, not a permanent contrarian buy/sell rule.",
        ),
    )


def macro_expert(signals: Mapping[str, float], *, realized_vol_24h: float, timestamp: int) -> ExpertForecast:
    keys = [f"M{idx}" for idx in range(1, 10)]
    finite = {key: float(signals[key]) for key in keys if key in signals and np.isfinite(signals[key])}
    quality = len(finite) / len(keys)
    if not finite:
        return _forecast(
            name="macro", score=0.0, sigma=realized_vol_24h, quality=0.0, timestamp=timestamp,
            signals={}, explanation=("No point-in-time macro feed is available; expert masked.",), available=False,
        )
    terms: list[tuple[float, float]] = []
    # M1 equity risk momentum, M3 dollar impulse and M6 real-yield impulse are signed in the spec.
    for key, weight in (("M1", 1.0), ("M3", 0.8), ("M4", -0.7), ("M6", 0.7), ("M9", 0.7)):
        if key in finite:
            terms.append((weight, math.tanh(finite[key])))
    # Gold is explicitly contextual: never encode gold-up => BTC-up permanently.
    if "M2" in finite and "M8" in finite:
        terms.append((0.55, math.tanh(finite["M2"]) * float(np.clip(finite["M8"], -1.0, 1.0))))
    if "M7" in finite and "M1" in finite:
        terms.append((0.45, math.tanh(finite["M1"]) * math.tanh(finite["M7"])))
    denominator = sum(abs(weight) for weight, _ in terms)
    score = sum(weight * value for weight, value in terms) / max(denominator, 1e-9) if terms else 0.0
    return _forecast(
        name="macro", score=score, sigma=realized_vol_24h, quality=quality, timestamp=timestamp,
        signals=finite,
        explanation=(f"{len(finite)}/9 macro signals available.", "Gold direction is conditional on current BTC/gold correlation."),
        available=len(finite) >= 2,
    )


def derivatives_expert(signals: Mapping[str, float], *, realized_vol_24h: float, timestamp: int) -> ExpertForecast:
    keys = [f"D{idx}" for idx in range(1, 11)]
    finite = {key: float(signals[key]) for key in keys if key in signals and np.isfinite(signals[key])}
    quality = len(finite) / len(keys)
    if not finite:
        return _forecast(
            name="derivatives", score=0.0, sigma=realized_vol_24h, quality=0.0, timestamp=timestamp,
            signals={}, explanation=("No derivatives/positioning snapshot is available; expert masked.",), available=False,
        )
    terms: list[tuple[float, float]] = []
    if "D5" in finite: terms.append((0.9, -math.tanh(finite["D5"])))
    if "D6" in finite: terms.append((1.0, math.tanh(finite["D6"] * 2.0)))
    if "D7" in finite: terms.append((0.65, math.tanh(finite["D7"])))
    if "D8" in finite: terms.append((0.8, math.tanh(finite["D8"] * 2.0)))
    if "D9" in finite: terms.append((0.55, -math.tanh(finite["D9"])))
    if "D2" in finite: terms.append((0.25, math.tanh(finite["D2"] * 20.0)))
    if "D1" in finite and "D4" in finite:
        terms.append((0.55, -math.tanh(finite["D1"]) * math.tanh(finite["D4"])))
    denominator = sum(abs(weight) for weight, _ in terms)
    score = sum(weight * value for weight, value in terms) / max(denominator, 1e-9) if terms else 0.0
    return _forecast(
        name="derivatives", score=score, sigma=realized_vol_24h, quality=quality, timestamp=timestamp,
        signals=finite,
        explanation=(f"{len(finite)}/10 derivatives signals available.", "Funding alone is never interpreted as an automatic reversal."),
        available=len(finite) >= 2,
    )


def news_expert(signals: Mapping[str, float], *, realized_vol_24h: float, timestamp: int) -> ExpertForecast:
    keys = [f"N{idx}" for idx in range(1, 8)]
    finite = {key: float(signals[key]) for key in keys if key in signals and np.isfinite(signals[key])}
    quality = len(finite) / len(keys)
    if not finite:
        return _forecast(
            name="news", score=0.0, sigma=realized_vol_24h, quality=0.0, timestamp=timestamp,
            signals={}, explanation=("No timestamp-valid news/event snapshot is available; expert masked.",), available=False,
        )
    terms: list[tuple[float, float]] = []
    for key, weight in (("N1", 1.0), ("N2", 0.7), ("N4", 0.9), ("N6", 0.35)):
        if key in finite:
            terms.append((weight, math.tanh(finite[key])))
    denominator = sum(abs(weight) for weight, _ in terms)
    score = sum(weight * value for weight, value in terms) / max(denominator, 1e-9) if terms else 0.0
    disagreement = abs(finite.get("N5", 0.0))
    intensity = max(0.0, finite.get("N3", 0.0))
    adjusted_quality = quality / (1.0 + 0.25 * disagreement)
    return _forecast(
        name="news", score=score, sigma=realized_vol_24h, quality=adjusted_quality, timestamp=timestamp,
        signals=finite,
        explanation=(f"{len(finite)}/7 timestamp-valid news/event signals available.", "Disagreement lowers confidence instead of forcing direction."),
        available=len(finite) >= 2,
    )


def ai_probability_expert(
    name: str,
    *,
    p_up: float,
    expected_return: float,
    realized_vol_24h: float,
    timestamp: int,
    quality: float = 1.0,
) -> ExpertForecast:
    probability = float(np.clip(p_up, 1e-6, 1.0 - 1e-6))
    flat = float(np.clip(1.0 - abs(probability - 0.5) * 2.0, 0.05, 0.45))
    directional = 1.0 - flat
    up = directional * probability
    down = directional * (1.0 - probability)
    return ExpertForecast(
        name=name,
        p_down=float(down), p_flat=float(flat), p_up=float(up),
        expected_return=float(expected_return), sigma=float(max(realized_vol_24h, 1e-6)),
        confidence=float(np.clip(max(up, down, flat) * (0.5 + 0.5 * quality), 0.0, 1.0)),
        quality=float(np.clip(quality, 0.0, 1.0)), available=True, timestamp=int(timestamp),
        signals={"legacy_p_up": probability},
        explanation=("Existing frozen AI forecast wrapped in the common expert contract.",),
    )
