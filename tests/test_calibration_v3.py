import numpy as np

from backend.ml.calibration import (
    estimate_payoffs,
    ev_commitment_backtest,
    fit_platt_scaler,
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
