import numpy as np

from backend.ml.calibration import (
    estimate_payoffs,
    estimate_volatility_conditioned_payoffs,
    ev_commitment_backtest,
    fit_platt_scaler,
    payoff_trim_sensitivity,
    trimmed_mean,
)


def test_platt_scaler_preserves_probability_order():
    raw = np.asarray([0.10, 0.20, 0.30, 0.70, 0.80, 0.90] * 20, dtype=float)
    labels = np.asarray([0, 0, 0, 1, 1, 1] * 20, dtype=float)
    calibrator = fit_platt_scaler(raw, labels)
    calibrated = calibrator.transform(raw)
    assert calibrated.shape == raw.shape
    assert np.all((calibrated > 0) & (calibrated < 1))
    assert calibrated[-1] > calibrated[0]


def test_trimmed_mean_limits_outlier_effect():
    values = np.asarray([1.0] * 95 + [1000.0] * 5)
    assert trimmed_mean(values, trim_fraction=0.05) == 1.0


def test_payoff_estimate_uses_fixed_hurdle():
    returns = np.asarray([-0.01] * 20 + [0.006] * 20)
    payoff = estimate_payoffs(returns, event_hurdle_bps=50.0)
    assert payoff.event_count == 20
    assert payoff.non_event_count == 20
    assert payoff.mean_event_return > 0.005


def test_payoff_trim_sensitivity_reports_without_selecting():
    returns = np.asarray([-0.01] * 40 + [0.006] * 38 + [0.03, 0.08])
    rows = payoff_trim_sensitivity(
        returns,
        event_hurdle_bps=50.0,
        trim_fractions=(0.0, 0.05),
    )
    assert [row["trim_fraction"] for row in rows] == [0.0, 0.05]
    assert rows[0]["event_count"] == rows[1]["event_count"]
    assert rows[0]["mean_event_return"] != rows[1]["mean_event_return"]


def _volatility_payoff_fixture(shrinkage_samples: float = 0.0):
    volatility = np.linspace(0.01, 0.09, 90)
    returns = np.asarray(
        [-0.004] * 15 + [0.008] * 15
        + [-0.004] * 15 + [0.020] * 15
        + [-0.004] * 15 + [0.060] * 15,
        dtype=float,
    )
    payoff = estimate_volatility_conditioned_payoffs(
        returns,
        volatility,
        event_hurdle_bps=50.0,
        trim_fraction=0.0,
        shrinkage_samples=shrinkage_samples,
    )
    return payoff


def test_volatility_conditioned_payoff_changes_magnitude_by_causal_state():
    payoff = _volatility_payoff_fixture(shrinkage_samples=0.0)
    probability = np.asarray([0.10, 0.10], dtype=float)
    state = np.asarray([0.015, 0.085], dtype=float)
    gross = payoff.expected_gross_return(probability, state)

    assert payoff.low_vol_cutoff < payoff.high_vol_cutoff
    assert payoff.low.sample_count > 0
    assert payoff.normal.sample_count > 0
    assert payoff.high.sample_count > 0
    assert gross[0] < 0.0
    assert gross[1] > 0.0
    assert gross[1] > gross[0]


def test_volatility_conditioned_payoff_shrinks_local_means_toward_global():
    raw = _volatility_payoff_fixture(shrinkage_samples=0.0)
    shrunk = _volatility_payoff_fixture(shrinkage_samples=100.0)

    raw_distance = abs(raw.high.mean_event_return - raw.global_payoff.mean_event_return)
    shrunk_distance = abs(shrunk.high.mean_event_return - shrunk.global_payoff.mean_event_return)
    assert shrunk_distance < raw_distance
    assert 0.0 < shrunk.high.event_shrinkage_weight < 1.0


def test_ev_backtest_uses_volatility_state_for_conditioned_payoff():
    payoff = _volatility_payoff_fixture(shrinkage_samples=0.0)
    probability = np.asarray([0.10, 0.10], dtype=float)
    actual = np.zeros_like(probability)
    state = np.asarray([0.015, 0.085], dtype=float)

    _, _, positions, turnovers, ev = ev_commitment_backtest(
        probability,
        actual,
        payoff=payoff,
        payoff_state=state,
        one_way_cost_rate=0.0,
        horizon_hours=1,
        entry_margin=0.0,
        force_flat_at_end=False,
    )

    assert positions.tolist() == [0.0, 1.0]
    assert turnovers.tolist() == [0.0, 1.0]
    assert ev[0] < 0.0
    assert ev[1] > 0.0


def test_ev_backtest_redecides_only_at_horizon_boundaries():
    probability = np.asarray([0.95, 0.01, 0.01, 0.95, 0.01, 0.01], dtype=float)
    actual = np.zeros_like(probability)
    payoff = estimate_payoffs(np.asarray([-0.01] * 20 + [0.02] * 20), event_hurdle_bps=50.0)
    metrics, _, positions, turnovers, ev = ev_commitment_backtest(
        probability,
        actual,
        payoff=payoff,
        one_way_cost_rate=0.0,
        horizon_hours=3,
        entry_margin=0.0,
    )
    assert positions[0] == 1.0
    assert positions[1] == 1.0
    assert positions[2] == 1.0
    assert np.isfinite(ev[0])
    assert np.isnan(ev[1]) and np.isnan(ev[2])
    assert np.isfinite(ev[3])
    assert metrics["trade_count"] >= 1
    assert np.sum(turnovers) >= 1.0


def test_ev_backtest_does_not_exit_for_small_negative_edge_below_exit_cost():
    # Payoffs are +2% for the event and -1% otherwise. At p=0.30 the gross
    # EV is -0.10%, which is negative but still better than paying a 0.25%
    # exit cost immediately. At p=0.00 the gross EV is -1%, so exit is justified.
    probability = np.asarray([0.95, 0.30, 0.00], dtype=float)
    actual = np.zeros_like(probability)
    payoff = estimate_payoffs(
        np.asarray([-0.01] * 20 + [0.02] * 20),
        event_hurdle_bps=50.0,
        trim_fraction=0.0,
    )
    _, _, positions, turnovers, ev = ev_commitment_backtest(
        probability,
        actual,
        payoff=payoff,
        one_way_cost_rate=0.0025,
        horizon_hours=1,
        entry_margin=0.0,
        exit_ev_threshold=0.0,
        force_flat_at_end=False,
    )
    assert positions.tolist() == [1.0, 1.0, 0.0]
    assert turnovers.tolist() == [1.0, 0.0, 1.0]
    assert ev[1] > 0.0
    assert ev[2] < 0.0
