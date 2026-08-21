from __future__ import annotations

from backend.short_term.runtime import (
    MEAN_REVERSION_EXPLORE_ID,
    MOMENTUM_EXPLORE_ID,
    _policy,
)


def test_exploration_hurdle_stops_at_modeled_round_trip_cost():
    family, setup_floor, hurdle, version, exploration = _policy(MOMENTUM_EXPLORE_ID, 50.0)
    assert family == "momentum"
    assert setup_floor == 0.60
    assert hurdle == 50.0
    assert version == "short-term-v1-explore"
    assert exploration is True


def test_mean_reversion_exploration_keeps_same_exit_family_with_looser_entry_floor():
    family, setup_floor, hurdle, version, exploration = _policy(MEAN_REVERSION_EXPLORE_ID, 50.0)
    assert family == "mean_reversion"
    assert setup_floor == 0.625
    assert hurdle == 50.0
    assert version == "short-term-v1-explore"
    assert exploration is True


def test_official_policy_stays_more_conservative_than_exploration():
    _, official_floor, official_hurdle, _, official_exploration = _policy("short-momentum-15m", 50.0)
    _, explore_floor, explore_hurdle, _, explore_exploration = _policy(MOMENTUM_EXPLORE_ID, 50.0)
    assert official_floor == 0.72
    assert official_hurdle == 65.0
    assert official_exploration is False
    assert explore_floor < official_floor
    assert explore_hurdle < official_hurdle
    assert explore_exploration is True
