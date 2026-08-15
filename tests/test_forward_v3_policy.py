import numpy as np

from backend.forward_v3 import _paper_signal_for_policy
from backend.ml.calibration import PayoffEstimate


def _payoff() -> PayoffEstimate:
    return PayoffEstimate(
        event_hurdle_bps=25.0,
        trim_fraction=0.05,
        event_count=100,
        non_event_count=100,
        event_rate=0.5,
        mean_event_return=0.02,
        mean_non_event_return=-0.005,
        median_event_return=0.015,
        median_non_event_return=-0.003,
    )


def test_flat_buys_only_when_round_trip_ev_beats_margin() -> None:
    signal, gross, decision = _paper_signal_for_policy(
        position_is_open=False,
        calibrated_probability=0.8,
        payoff=_payoff(),
        one_way_cost_rate=0.0025,
        entry_margin=0.001,
    )
    assert np.isclose(gross, 0.015)
    assert np.isclose(decision, 0.01)
    assert signal == "BUY"

    signal, _, decision = _paper_signal_for_policy(
        position_is_open=False,
        calibrated_probability=0.2,
        payoff=_payoff(),
        one_way_cost_rate=0.0025,
        entry_margin=0.001,
    )
    assert decision < 0.001
    assert signal == "HOLD"


def test_long_exits_when_gross_ev_is_not_positive() -> None:
    signal, gross, decision = _paper_signal_for_policy(
        position_is_open=True,
        calibrated_probability=0.1,
        payoff=_payoff(),
        one_way_cost_rate=0.0025,
        entry_margin=0.0,
    )
    assert gross < 0.0
    assert decision == gross
    assert signal == "SELL"

    signal, gross, _ = _paper_signal_for_policy(
        position_is_open=True,
        calibrated_probability=0.8,
        payoff=_payoff(),
        one_way_cost_rate=0.0025,
        entry_margin=0.0,
    )
    assert gross > 0.0
    assert signal == "HOLD"
