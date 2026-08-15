from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import GateResult, RegimePosterior, TraderDecision


@dataclass(frozen=True)
class DecisionConfig:
    min_edge_bps: float = 5.0
    min_direction_probability: float = 0.58
    max_regime_entropy: float = 0.92
    uncertainty_penalty: float = 0.25
    max_position_fraction: float = 0.10


def decide(
    gate: GateResult,
    regime: RegimePosterior,
    *,
    one_way_cost_rate: float,
    data_quality_ok: bool,
    config: DecisionConfig = DecisionConfig(),
) -> TraderDecision:
    if not data_quality_ok:
        return TraderDecision("NO_TRADE", gate.expected_return, one_way_cost_rate, 0.0, -one_way_cost_rate, gate.confidence, 0.0, "Core data quality/availability check failed.")
    uncertainty = max(gate.sigma, 0.0) * max(config.uncertainty_penalty, 0.0)
    net_edge = abs(gate.expected_return) - max(one_way_cost_rate, 0.0) - uncertainty
    min_edge = max(config.min_edge_bps, 0.0) / 10_000.0
    if regime.entropy > config.max_regime_entropy:
        return TraderDecision("NO_TRADE", gate.expected_return, one_way_cost_rate, uncertainty, net_edge, gate.confidence, 0.0, f"Regime uncertainty {regime.entropy:.3f} exceeds limit {config.max_regime_entropy:.3f}.")
    if net_edge < min_edge:
        return TraderDecision("NO_TRADE", gate.expected_return, one_way_cost_rate, uncertainty, net_edge, gate.confidence, 0.0, f"Net edge {net_edge:.4%} does not clear the cost/uncertainty hurdle.")
    if gate.expected_return > 0.0 and gate.p_up >= config.min_direction_probability:
        confidence = float(np.clip((gate.p_up - 0.5) * 2.0, 0.0, 1.0))
        size = min(config.max_position_fraction, config.max_position_fraction * max(confidence, 0.25))
        return TraderDecision("LONG", gate.expected_return, one_way_cost_rate, uncertainty, net_edge, gate.p_up, size, "Positive net edge and directional probability cleared the entry hurdle.")
    if gate.expected_return < 0.0 and gate.p_down >= config.min_direction_probability:
        return TraderDecision("SHORT", gate.expected_return, one_way_cost_rate, uncertainty, net_edge, gate.p_down, 0.0, "Negative net edge and directional probability cleared the bearish hurdle.")
    return TraderDecision("NO_TRADE", gate.expected_return, one_way_cost_rate, uncertainty, net_edge, gate.confidence, 0.0, "Directional probability did not clear the trading hurdle.")
