import math

import numpy as np

from run_v3_payoff_ablation import (
    _regime,
    required_probability,
    select_conditioned_policy,
)
from backend.ml.calibration import estimate_volatility_conditioned_payoffs


def test_required_probability_matches_entry_hurdle_equation() -> None:
    probability = required_probability(
        0.02,
        -0.01,
        one_way_cost_rate=0.0025,
        entry_margin=0.001,
    )
    # Solve p*0.02 + (1-p)*(-0.01) = 0.006.
    assert math.isclose(probability, 16.0 / 30.0, rel_tol=1e-12)


def test_regime_uses_calibration_cutoffs() -> None:
    returns = np.asarray([-0.01] * 30 + [0.02] * 30, dtype=float)
    volatility = np.linspace(0.001, 0.060, len(returns))
    payoff = estimate_volatility_conditioned_payoffs(
        returns,
        volatility,
        event_hurdle_bps=50.0,
        trim_fraction=0.0,
        shrinkage_samples=25.0,
    )
    assert _regime(payoff, payoff.low_vol_cutoff - 1e-6) == "LOW"
    assert _regime(payoff, (payoff.low_vol_cutoff + payoff.high_vol_cutoff) / 2.0) == "NORMAL"
    assert _regime(payoff, payoff.high_vol_cutoff + 1e-6) == "HIGH"
    assert _regime(payoff, float("nan")) == "GLOBAL_FALLBACK"


def test_conditioned_policy_selection_has_no_test_inputs() -> None:
    # The selector receives only calibration data and policy-validation data.
    # Test/OOS arrays are intentionally absent from the function signature.
    calibration_returns = np.asarray(
        [-0.012] * 30 + [-0.004] * 30 + [0.008] * 30 + [0.025] * 30,
        dtype=float,
    )
    calibration_volatility = np.linspace(0.002, 0.050, len(calibration_returns))
    policy_probability = np.asarray([0.20, 0.80, 0.75, 0.25] * 20, dtype=float)
    policy_actual_1h = np.asarray([-0.002, 0.003, 0.002, -0.001] * 20, dtype=float)
    policy_volatility = np.linspace(0.003, 0.045, len(policy_probability))

    payoff, shrinkage, margin, rows = select_conditioned_policy(
        calibration_returns,
        calibration_volatility,
        policy_probability,
        policy_actual_1h,
        policy_volatility,
        event_hurdle_bps=50.0,
        trim_fraction=0.0,
        cost_rate=0.0025,
        horizon_hours=1,
        margin_grid_bps=[0.0, 5.0],
        shrinkage_grid=[0.0, 25.0],
        min_trades=1,
        volatility_feature_name="realized_vol_24h",
    )

    assert shrinkage in {0.0, 25.0}
    assert margin in {0.0, 5.0}
    assert len(rows) == 4
    assert payoff.volatility_feature_name == "realized_vol_24h"
    assert all("shrinkage_samples" in row and "entry_margin_bps" in row for row in rows)
