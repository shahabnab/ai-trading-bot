from __future__ import annotations

from dataclasses import asdict, dataclass

from .features import ShortTermFeatures


@dataclass(frozen=True)
class ShortTermDecision:
    action: str  # ENTER_LONG, EXIT, HOLD
    confidence: float
    edge_proxy_bps: float
    reason: str
    # Diagnostic metadata. ``confidence`` is kept for the existing RiskManager
    # contract; it is a transformed checklist score, not a calibrated win
    # probability. ``confirmation_score`` exposes the underlying checklist
    # fraction directly for audit/calibration.
    confirmation_score: float = 0.0
    setup_ready: bool = False
    microstructure_ready: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _microstructure_ready(f: ShortTermFeatures, *, min_coverage: float = 0.50) -> bool:
    """Require enough point-in-time microstructure for a new entry.

    Missing flow/depth data must never make an entry easier. Existing positions
    are still allowed to exit even when the collector is degraded.
    """
    return (
        f.microstructure_coverage >= min_coverage
        and f.spread_bps is not None
        and f.book_imbalance is not None
    )


def decide_momentum(
    f: ShortTermFeatures,
    *,
    long_open: bool,
    hold_minutes: float,
    unrealized_return: float,
    min_edge_bps: float,
    round_trip_cost_bps: float = 0.0,
    max_hold_minutes: float = 120.0,
) -> ShortTermDecision:
    """Transparent 15-minute momentum baseline.

    New entries require sufficiently complete microstructure and a directional
    price move large enough to clear the configured economic hurdle. ATR is a
    dispersion measure and is deliberately not treated as directional edge.
    """
    del round_trip_cost_bps  # reserved for symmetric strategy API / future exits

    if long_open:
        if unrealized_return <= -0.008:
            return ShortTermDecision(
                "EXIT",
                1.0,
                abs(unrealized_return) * 10_000.0,
                "Protective exit: paper position fell 0.8% from entry.",
            )
        if hold_minutes >= max_hold_minutes:
            return ShortTermDecision(
                "EXIT",
                0.8,
                abs(f.return_1h) * 10_000.0,
                f"Time exit after {hold_minutes:.0f} minutes.",
            )
        if f.return_15m < -0.0015 and f.ema_fast_slope_bps < 0:
            return ShortTermDecision(
                "EXIT",
                0.75,
                abs(f.return_15m) * 10_000.0,
                "Momentum reversed on the latest 15-minute bar.",
            )
        return ShortTermDecision(
            "HOLD",
            0.6,
            abs(f.return_1h) * 10_000.0,
            "Existing momentum position remains inside exit limits.",
        )

    micro_ready = _microstructure_ready(f)
    components = [
        f.return_15m > 0.0,
        f.return_30m > 0.0,
        f.return_1h > 0.0015,
        f.ema_gap_bps > 5.0,
        f.ema_fast_slope_bps > 0.0,
        50.0 <= f.rsi_14 <= 74.0,
        f.volume_z > -0.25,
        micro_ready and f.trade_imbalance > 0.03,
        micro_ready and f.book_imbalance is not None and f.book_imbalance > -0.10,
        micro_ready and f.spread_bps is not None and f.spread_bps < 12.0,
    ]
    strength = sum(1 for item in components if item) / len(components)
    move_proxy_bps = max(
        f.return_1h * 10_000.0,
        f.return_2h * 5_000.0,
        0.0,
    )
    confidence = _clamp01(0.35 + 0.65 * strength)
    setup_ready = micro_ready and strength >= 0.72
    if setup_ready and move_proxy_bps >= min_edge_bps:
        return ShortTermDecision(
            "ENTER_LONG",
            confidence,
            move_proxy_bps,
            f"Momentum confirmation {strength:.0%}; move proxy {move_proxy_bps:.1f} bps clears {min_edge_bps:.1f} bps cost hurdle.",
            confirmation_score=strength,
            setup_ready=True,
            microstructure_ready=micro_ready,
        )
    if not micro_ready:
        return ShortTermDecision(
            "HOLD",
            confidence,
            move_proxy_bps,
            f"No momentum entry: microstructure incomplete (coverage={f.microstructure_coverage:.0%}, spread={'ok' if f.spread_bps is not None else 'missing'}, book={'ok' if f.book_imbalance is not None else 'missing'}).",
            confirmation_score=strength,
            setup_ready=False,
            microstructure_ready=False,
        )
    return ShortTermDecision(
        "HOLD",
        confidence,
        move_proxy_bps,
        f"No momentum entry: confirmation={strength:.0%}, move_proxy={move_proxy_bps:.1f} bps, required={min_edge_bps:.1f} bps.",
        confirmation_score=strength,
        setup_ready=setup_ready,
        microstructure_ready=micro_ready,
    )


def decide_mean_reversion(
    f: ShortTermFeatures,
    *,
    long_open: bool,
    hold_minutes: float,
    unrealized_return: float,
    min_edge_bps: float,
    round_trip_cost_bps: float = 0.0,
    max_hold_minutes: float = 120.0,
) -> ShortTermDecision:
    """Cost-aware oversold mean-reversion benchmark on completed 15m data."""
    if long_open:
        if unrealized_return <= -0.009:
            return ShortTermDecision(
                "EXIT",
                1.0,
                abs(unrealized_return) * 10_000.0,
                "Protective exit: mean-reversion position fell 0.9% from entry.",
            )
        if hold_minutes >= max_hold_minutes:
            return ShortTermDecision(
                "EXIT",
                0.8,
                abs(f.vwap_distance_bps),
                f"Time exit after {hold_minutes:.0f} minutes.",
            )

        target_reached = f.bollinger_z >= -0.10 or f.rsi_14 >= 54.0 or f.vwap_distance_bps >= -5.0
        gross_unrealized_bps = unrealized_return * 10_000.0
        if target_reached and gross_unrealized_bps > round_trip_cost_bps:
            return ShortTermDecision(
                "EXIT",
                0.75,
                max(gross_unrealized_bps, 0.0),
                f"Price mean-reverted and gross gain {gross_unrealized_bps:.1f} bps clears {round_trip_cost_bps:.1f} bps round-trip cost.",
            )
        if target_reached:
            return ShortTermDecision(
                "HOLD",
                0.6,
                max(gross_unrealized_bps, 0.0),
                f"Mean-reversion target touched, but gross gain {gross_unrealized_bps:.1f} bps does not yet clear {round_trip_cost_bps:.1f} bps round-trip cost.",
            )
        return ShortTermDecision(
            "HOLD",
            0.6,
            abs(f.vwap_distance_bps),
            "Oversold position has not yet reached its mean-reversion exit.",
        )

    micro_ready = _microstructure_ready(f)
    components = [
        f.bollinger_z <= -1.35,
        f.rsi_14 <= 38.0,
        f.return_15m < 0.0,
        f.vwap_distance_bps <= -25.0,
        f.volume_z > -0.75,
        micro_ready and f.trade_imbalance > -0.30,
        micro_ready and f.book_imbalance is not None and f.book_imbalance > -0.30,
        micro_ready and f.spread_bps is not None and f.spread_bps < 12.0,
    ]
    strength = sum(1 for item in components if item) / len(components)
    displacement_bps = max(
        -f.vwap_distance_bps,
        -f.bollinger_z * f.atr_bps,
        0.0,
    )
    confidence = _clamp01(0.35 + 0.65 * strength)
    setup_ready = micro_ready and strength >= 0.75
    if setup_ready and displacement_bps >= min_edge_bps:
        return ShortTermDecision(
            "ENTER_LONG",
            confidence,
            displacement_bps,
            f"Oversold confirmation {strength:.0%}; displacement {displacement_bps:.1f} bps clears {min_edge_bps:.1f} bps cost hurdle.",
            confirmation_score=strength,
            setup_ready=True,
            microstructure_ready=micro_ready,
        )
    if not micro_ready:
        return ShortTermDecision(
            "HOLD",
            confidence,
            displacement_bps,
            f"No mean-reversion entry: microstructure incomplete (coverage={f.microstructure_coverage:.0%}, spread={'ok' if f.spread_bps is not None else 'missing'}, book={'ok' if f.book_imbalance is not None else 'missing'}).",
            confirmation_score=strength,
            setup_ready=False,
            microstructure_ready=False,
        )
    return ShortTermDecision(
        "HOLD",
        confidence,
        displacement_bps,
        f"No mean-reversion entry: confirmation={strength:.0%}, displacement={displacement_bps:.1f} bps, required={min_edge_bps:.1f} bps.",
        confirmation_score=strength,
        setup_ready=setup_ready,
        microstructure_ready=micro_ready,
    )
