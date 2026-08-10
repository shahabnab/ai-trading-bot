from __future__ import annotations

import math

import numpy as np
import pytest

from backend.ml.evaluation import (
    DAY_MS,
    WalkForwardConfig,
    long_only_cost_aware_backtest,
    make_walk_forward_folds,
)


def test_walk_forward_is_chronological_and_non_overlapping() -> None:
    timestamps = np.arange(0, 210 * DAY_MS, 60 * 60 * 1000, dtype=np.int64)
    folds = make_walk_forward_folds(
        timestamps,
        WalkForwardConfig(train_days=90, validation_days=30, test_days=30, step_days=30),
    )

    assert len(folds) == 3
    for fold in folds:
        assert timestamps[fold.train_indices[-1]] < timestamps[fold.validation_indices[0]]
        assert timestamps[fold.validation_indices[-1]] < timestamps[fold.test_indices[0]]

    test_sets = [set(fold.test_indices.tolist()) for fold in folds]
    assert test_sets[0].isdisjoint(test_sets[1])
    assert test_sets[1].isdisjoint(test_sets[2])


def test_walk_forward_rejects_overlapping_test_windows() -> None:
    with pytest.raises(ValueError, match="step_days"):
        WalkForwardConfig(train_days=90, validation_days=30, test_days=30, step_days=7).validate()


def test_walk_forward_rejects_duplicate_timestamps() -> None:
    timestamps = np.asarray([0, DAY_MS, DAY_MS, 2 * DAY_MS], dtype=np.int64)

    with pytest.raises(ValueError, match="strictly increasing and unique"):
        make_walk_forward_folds(
            timestamps,
            WalkForwardConfig(train_days=1, validation_days=1, test_days=1, step_days=1),
        )


def test_cost_hurdle_suppresses_weak_position_changes() -> None:
    predicted = np.asarray([0.0005, 0.0030, -0.0005, -0.0030])
    actual = np.asarray([0.01, 0.01, -0.01, -0.01])

    metrics, _, positions, turnover = long_only_cost_aware_backtest(
        predicted,
        actual,
        cost_rate=0.001,
        execution_lambda=2.0,
    )

    assert positions.tolist() == [0.0, 1.0, 1.0, 0.0]
    assert turnover.tolist() == [0.0, 1.0, 0.0, 1.0]
    assert metrics["trade_count"] == 1
    assert metrics["round_trip_count"] == 1
    assert metrics["position_change_count"] == 2


def test_first_period_loss_counts_in_max_drawdown() -> None:
    metrics, _, _, _ = long_only_cost_aware_backtest(
        np.asarray([1.0]),
        np.asarray([-0.10]),
        cost_rate=0.0,
        execution_lambda=0.0,
    )

    assert math.isclose(metrics["max_drawdown"], -0.10, abs_tol=1e-12)


def test_sortino_uses_zero_mar_downside_deviation() -> None:
    predicted = np.asarray([1.0, 1.0, 1.0, 1.0])
    actual = np.asarray([0.01, -0.02, 0.03, -0.01])

    metrics, returns, _, _ = long_only_cost_aware_backtest(
        predicted,
        actual,
        cost_rate=0.0,
        execution_lambda=0.0,
    )

    mean_return = float(np.mean(returns))
    downside = np.minimum(returns, 0.0)
    downside_deviation = float(np.sqrt(np.mean(downside * downside)))
    expected = mean_return / downside_deviation * math.sqrt(24.0 * 365.0)

    assert math.isclose(metrics["sortino"], expected, rel_tol=1e-12)
