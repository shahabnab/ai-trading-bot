from __future__ import annotations

from dataclasses import asdict, dataclass

from .features import ShortTermFeatures


@dataclass(frozen=True)
class ShortTermDecision:
    action: str  # ENTER_LONG, EXIT, HOLD
    confidence: float
    edge_proxy_bps: float
    reason: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def decide_momentum(
    f: ShortTermFeatures,
    *,
    long_open: bool,
    hold_minutes: float,
    unrealized_return: float,
    min_edge_bps: float,
    max_hold_minutes: float = 120.0,
) -> ShortTermDecision:
    """Transparent 15-minute momentum baseline.

    It intentionally requires a move proxy large enough to clear estimated
    round-trip costs. This is a benchmark, not a claim of calibrated alpha.
    """
    if long_open:
        if unrealized_return <= -0.008:
            return ShortTermDecision("EXIT", 1.0, abs(unrealized_return) * 10_000.0, "Protective exit: paper position fell 0.8% from entry.")
        if hold_minutes >= max_hold_minutes:
            return ShortTermDecision("EXIT", 0.8, abs(f.return_1h) * 10_000.0, f"Time exit after {hold_minutes:.0f} minutes.")
        if f.return_15m < -0.0015 and f.ema_fast_slope_bps < 0:
            return ShortTermDecision("EXIT", 0.75, abs(f.return_15m) * 10_000.0, "Momentum reversed on the latest 15-minute bar.")
        return ShortTermDecision("HOLD", 0.6, abs(f.return_1h) * 10_000.0, "Existing momentum position remains inside exit limits.")

    components = [
        f.return_15m > 0.0,
        f.return_30m > 0.0,
        f.return_1h > 0.0015,
        f.ema_gap_bps > 5.0,
        f.ema_fast_slope_bps > 0.0,
        50.0 <= f.rsi_14 <= 74.0,
        f.volume_z > -0.25,
        f.trade_imbalance > 0.03 if f.microstructure_coverage >= 0.35 else True,
        f.book_imbalance > -0.10 if f.book_imbalance is not None else True,
        f.spread_bps < 12.0 if f.spread_bps is not None else True,
    ]
    strength = sum(1 for item in components if item) / len(components)
    move_proxy_bps = max(
        f.return_1h * 10_000.0,
        f.return_2h * 5_000.0,
        f.atr_bps * 1.15,
        0.0,
    )
    confidence = _clamp01(0.35 + 0.65 * strength)
    if strength >= 0.72 and move_proxy_bps >= min_edge_bps and f.quality >= 0.65:
        return ShortTermDecision(
            "ENTER_LONG",
            confidence,
            move_proxy_bps,
            f"Momentum confirmation {strength:.0%}; move proxy {move_proxy_bps:.1f} bps clears {min_edge_bps:.1f} bps cost hurdle.",
        )
    return ShortTermDecision(
        "HOLD",
        confidence,
        move_proxy_bps,
        f"No momentum entry: confirmation={strength:.0%}, move_proxy={move_proxy_bps:.1f} bps, required={min_edge_bps:.1f} bps, quality={f.quality:.2f}.",
    )


def decide_mean_reversion(
    f: ShortTermFeatures,
    *,
    long_open: bool,
    hold_minutes: float,
    unrealized_return: float,
    min_edge_bps: float,
    max_hold_minutes: float = 120.0,
) -> ShortTermDecision:
    """Cost-aware oversold mean-reversion benchmark on completed 15m data."""
    if long_open:
        if unrealized_return <= -0.009:
            return ShortTermDecision("EXIT", 1.0, abs(unrealized_return) * 10_000.0, "Protective exit: mean-reversion position fell 0.9% from entry.")
        if hold_minutes >= max_hold_minutes:
            return ShortTermDecision("EXIT", 0.8, abs(f.vwap_distance_bps), f"Time exit after {hold_minutes:.0f} minutes.")
        if f.bollinger_z >= -0.10 or f.rsi_14 >= 54.0 or f.vwap_distance_bps >= -5.0:
            return ShortTermDecision("EXIT", 0.75, abs(f.vwap_distance_bps), "Price mean-reverted toward its 20-bar center/VWAP.")
        return ShortTermDecision("HOLD", 0.6, abs(f.vwap_distance_bps), "Oversold position has not yet reached its mean-reversion exit.")

    components = [
        f.bollinger_z <= -1.35,
        f.rsi_14 <= 38.0,
        f.return_15m < 0.0,
        f.vwap_distance_bps <= -25.0,
        f.volume_z > -0.75,
        f.trade_imbalance > -0.30 if f.microstructure_coverage >= 0.35 else True,
        f.book_imbalance > -0.30 if f.book_imbalance is not None else True,
        f.spread_bps < 12.0 if f.spread_bps is not None else True,
    ]
    strength = sum(1 for item in components if item) / len(components)
    displacement_bps = max(abs(f.vwap_distance_bps), abs(f.bollinger_z) * f.atr_bps, 0.0)
    confidence = _clamp01(0.35 + 0.65 * strength)
    if strength >= 0.75 and displacement_bps >= min_edge_bps and f.quality >= 0.65:
        return ShortTermDecision(
            "ENTER_LONG",
            confidence,
            displacement_bps,
            f"Oversold confirmation {strength:.0%}; displacement {displacement_bps:.1f} bps clears {min_edge_bps:.1f} bps cost hurdle.",
        )
    return ShortTermDecision(
        "HOLD",
        confidence,
        displacement_bps,
        f"No mean-reversion entry: confirmation={strength:.0%}, displacement={displacement_bps:.1f} bps, required={min_edge_bps:.1f} bps, quality={f.quality:.2f}.",
    )
