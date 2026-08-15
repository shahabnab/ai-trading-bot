from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from backend.trader_brain.bandit import LinUCBBandit
from backend.trader_brain.contracts import GateResult, RegimePosterior
from backend.trader_brain.decision import DecisionConfig, decide
from backend.trader_brain.experience import TraderExperienceStore
from backend.trader_brain.experts import macro_expert, technical_expert
from backend.trader_brain.features import build_trader_feature_history
from backend.trader_brain.gate import ReliabilityWeightedGate
from backend.trader_brain.regime import PersistentGaussianRegimeDetector


def candles(count: int = 360) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    price = 50_000.0
    for idx in range(count):
        drift = 0.0005 + 0.0015 * math.sin(idx / 17.0)
        price *= 1.0 + drift
        rows.append({
            "created_at": (idx + 1) * 3_600_000,
            "open": str(price * 0.998), "high": str(price * 1.004), "low": str(price * 0.996),
            "close": str(price), "volume": str(1000.0 + 50.0 * math.sin(idx / 9.0)),
        })
    return rows


def test_features_and_regime_are_probabilistic() -> None:
    history = build_trader_feature_history(candles())
    latest = history[-1]
    assert len(latest.technical) == 10
    matrix = np.asarray([row.regime_vector for row in history], dtype=np.float64)
    posterior = PersistentGaussianRegimeDetector(n_regimes=4).fit_predict(matrix, timestamp=latest.timestamp)
    assert len(posterior.probabilities) == 4
    assert abs(sum(posterior.probabilities) - 1.0) < 1e-9
    assert 0.0 <= posterior.entropy <= 1.0


def test_missing_macro_is_masked_not_neutral_fabricated() -> None:
    expert = macro_expert({}, realized_vol_24h=0.01, timestamp=123)
    assert expert.available is False
    assert expert.quality == 0.0
    assert expert.p_flat > expert.p_up
    assert expert.p_flat > expert.p_down


def test_high_regime_entropy_forces_no_trade() -> None:
    regime = RegimePosterior((0.2, 0.2, 0.2, 0.2, 0.2), 1.0, ("a","b","c","d","e"), "test", 1.0, 1)
    gate = GateResult(0.05, 0.05, 0.90, 0.02, 0.002, {"technical": 1.0}, "test", (1.0,))
    decision = decide(gate, regime, one_way_cost_rate=0.001, data_quality_ok=True, config=DecisionConfig(max_regime_entropy=0.8))
    assert decision.action == "NO_TRADE"


def test_experience_shadow_reward_trains_bandit(tmp_path: Path) -> None:
    history = build_trader_feature_history(candles())
    latest = history[-1]
    technical = technical_expert(latest.technical, realized_vol_24h=latest.realized_vol_24h, timestamp=latest.timestamp)
    regime = RegimePosterior((0.05,0.10,0.60,0.15,0.10), 0.65, ("s0","s1","s2","s3","s4"), "test", 1.0, latest.timestamp)
    gate = ReliabilityWeightedGate().combine(regime, [technical])
    vector = tuple(gate.feature_vector) + (regime.entropy, 0.0)
    store = TraderExperienceStore(tmp_path / "paper.sqlite3")
    store.record(model_id="rl", feature_timestamp=100, target_timestamp=200, reference_price=100.0, position_before="FLAT", action="LONG", gate_vector=gate.feature_vector, bandit_vector=vector, gate=gate, experts=[technical], regime=regime, estimated_one_way_cost=0.001)
    assert store.resolve_due(model_id="rl", current_timestamp=200, current_price=102.0) == 1
    samples = store.bandit_shadow_samples("rl")
    assert len(samples) == 1
    bandit = LinUCBBandit(len(vector), alpha=0.0).fit_shadow_samples(samples)
    choice = bandit.choose(vector, valid_actions=("NO_TRADE","LONG"), min_samples=1, fallback_action="NO_TRADE")
    assert choice.action == "LONG"
