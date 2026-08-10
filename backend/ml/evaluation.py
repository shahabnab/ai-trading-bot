from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


HOUR_MS = 60 * 60 * 1000
DAY_MS = 24 * HOUR_MS
HOURS_PER_YEAR = 24.0 * 365.0


@dataclass(frozen=True)
class WalkForwardConfig:
    train_days: int = 90
    validation_days: int = 30
    test_days: int = 30
    step_days: int = 30
    # Bars removed from the end of the train and validation windows so that
    # targets looking `embargo_hours` into the future cannot overlap the next
    # split. With a 1-hour prediction horizon the last training bar's target is
    # the first validation hour's return; the embargo removes that overlap.
    embargo_hours: int = 1

    def validate(self) -> None:
        for name, value in (
            ("train_days", self.train_days),
            ("validation_days", self.validation_days),
            ("test_days", self.test_days),
            ("step_days", self.step_days),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.step_days < self.test_days:
            raise ValueError("step_days must be >= test_days so OOS test windows do not overlap")
        if self.embargo_hours < 0:
            raise ValueError("embargo_hours must be non-negative")
        if self.embargo_hours * HOUR_MS >= min(self.train_days, self.validation_days) * DAY_MS:
            raise ValueError("embargo_hours must be smaller than the train and validation windows")


@dataclass(frozen=True)
class WalkForwardFold:
    fold: int
    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray

    @property
    def sizes(self) -> dict[str, int]:
        return {
            "train": int(len(self.train_indices)),
            "validation": int(len(self.validation_indices)),
            "test": int(len(self.test_indices)),
        }


def make_walk_forward_folds(
    timestamps_ms: np.ndarray,
    config: WalkForwardConfig,
) -> list[WalkForwardFold]:
    config.validate()
    if len(timestamps_ms) == 0:
        return []
    if np.any(np.diff(timestamps_ms) <= 0):
        raise ValueError("timestamps must be strictly increasing and unique")

    first = int(timestamps_ms[0])
    last = int(timestamps_ms[-1])
    folds: list[WalkForwardFold] = []
    fold_id = 1
    train_start = first

    while True:
        train_end = train_start + config.train_days * DAY_MS
        validation_end = train_end + config.validation_days * DAY_MS
        test_end = validation_end + config.test_days * DAY_MS
        if test_end - HOUR_MS > last:
            break

        embargo_ms = config.embargo_hours * HOUR_MS
        train_idx = np.flatnonzero(
            (timestamps_ms >= train_start) & (timestamps_ms < train_end - embargo_ms)
        )
        val_idx = np.flatnonzero(
            (timestamps_ms >= train_end) & (timestamps_ms < validation_end - embargo_ms)
        )
        test_idx = np.flatnonzero((timestamps_ms >= validation_end) & (timestamps_ms < test_end))

        if len(train_idx) and len(val_idx) and len(test_idx):
            folds.append(
                WalkForwardFold(
                    fold=fold_id,
                    train_indices=train_idx,
                    validation_indices=val_idx,
                    test_indices=test_idx,
                )
            )
            fold_id += 1

        train_start += config.step_days * DAY_MS

    return folds


def regression_metrics(actual_log_return: np.ndarray, predicted_log_return: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual_log_return, dtype=np.float64)
    predicted = np.asarray(predicted_log_return, dtype=np.float64)
    if actual.shape != predicted.shape or actual.size == 0:
        raise ValueError("actual and predicted must be non-empty arrays with identical shapes")

    error = predicted - actual
    rmse = float(np.sqrt(np.mean(error * error)))
    mae = float(np.mean(np.abs(error)))
    direction_accuracy = float(np.mean((predicted > 0.0) == (actual > 0.0)))

    if float(np.std(actual)) > 0.0 and float(np.std(predicted)) > 0.0:
        correlation = float(np.corrcoef(actual, predicted)[0, 1])
    else:
        correlation = 0.0

    return {
        "rmse_log_return": rmse,
        "mae_log_return": mae,
        "direction_accuracy": direction_accuracy,
        "correlation": correlation,
    }


def _performance_metrics(returns: np.ndarray, positions: np.ndarray, turnovers: np.ndarray) -> dict[str, float]:
    returns = np.asarray(returns, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    turnovers = np.asarray(turnovers, dtype=np.float64)
    if returns.size == 0:
        raise ValueError("strategy return series is empty")
    if not (returns.shape == positions.shape == turnovers.shape):
        raise ValueError("returns, positions, and turnovers must have identical shapes")

    # Include the initial capital point so a first-period loss is measured as
    # drawdown from 1.0 instead of incorrectly becoming the first equity peak.
    equity_after = np.cumprod(1.0 + returns)
    equity = np.concatenate(([1.0], equity_after))
    cumulative_return = float(equity_after[-1] - 1.0)
    periods = float(len(returns))
    if equity_after[-1] > 0.0:
        annualized_return = float(equity_after[-1] ** (HOURS_PER_YEAR / periods) - 1.0)
    else:
        annualized_return = -1.0

    mean_return = float(np.mean(returns))
    std_return = float(np.std(returns, ddof=0))
    annualized_volatility = std_return * math.sqrt(HOURS_PER_YEAR)
    sharpe = mean_return / std_return * math.sqrt(HOURS_PER_YEAR) if std_return > 0 else 0.0

    running_max = np.maximum.accumulate(equity)
    drawdown = equity / running_max - 1.0
    max_drawdown = float(np.min(drawdown))

    # Sortino downside deviation versus MAR=0. Non-negative observations remain
    # in the denominator as zeros instead of being discarded before std().
    downside = np.minimum(returns, 0.0)
    downside_deviation = float(np.sqrt(np.mean(downside * downside)))
    sortino = (
        mean_return / downside_deviation * math.sqrt(HOURS_PER_YEAR)
        if downside_deviation > 0
        else 0.0
    )

    previous_positions = np.concatenate(([0.0], positions[:-1]))
    entries = (positions > previous_positions) & (turnovers > 0.0)
    exits = (positions < previous_positions) & (turnovers > 0.0)
    position_changes = turnovers > 0.0

    return {
        "cumulative_return": cumulative_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": max_drawdown,
        "exposure": float(np.mean(positions)),
        "turnover": float(np.sum(turnovers)),
        "trade_count": int(np.count_nonzero(entries)),
        "round_trip_count": int(np.count_nonzero(exits)),
        "position_change_count": int(np.count_nonzero(position_changes)),
        "positive_bar_rate": float(np.mean(returns > 0.0)),
    }


def long_only_cost_aware_backtest(
    predicted_log_return: np.ndarray,
    actual_simple_return: np.ndarray,
    *,
    cost_rate: float = 0.001,
    execution_lambda: float = 2.0,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    predicted = np.asarray(predicted_log_return, dtype=np.float64)
    actual = np.asarray(actual_simple_return, dtype=np.float64)
    if predicted.shape != actual.shape:
        raise ValueError("predicted and actual arrays must have identical shapes")
    if cost_rate < 0.0 or execution_lambda < 0.0:
        raise ValueError("cost_rate and execution_lambda must be non-negative")

    positions = np.zeros(len(predicted), dtype=np.float64)
    turnovers = np.zeros(len(predicted), dtype=np.float64)
    strategy_returns = np.zeros(len(predicted), dtype=np.float64)
    previous_position = 0.0

    for idx, forecast in enumerate(predicted):
        desired = 1.0 if forecast > 0.0 else 0.0
        requested_turnover = abs(desired - previous_position)
        position = previous_position

        if requested_turnover > 0.0:
            hurdle = execution_lambda * cost_rate * requested_turnover
            if abs(forecast) > hurdle:
                position = desired

        turnover = abs(position - previous_position)
        strategy_return = position * actual[idx] - cost_rate * turnover

        positions[idx] = position
        turnovers[idx] = turnover
        strategy_returns[idx] = strategy_return
        previous_position = position

    return _performance_metrics(strategy_returns, positions, turnovers), strategy_returns, positions, turnovers


def moving_average_baseline(
    actual_simple_return: np.ndarray,
    closes: np.ndarray,
    ema20: np.ndarray,
    ema50: np.ndarray,
    ema200: np.ndarray,
    *,
    cost_rate: float = 0.001,
) -> dict[str, float]:
    actual = np.asarray(actual_simple_return, dtype=np.float64)
    desired = ((ema20 > ema50) & (closes > ema200)).astype(np.float64)
    previous = np.concatenate(([0.0], desired[:-1]))
    turnover = np.abs(desired - previous)
    strategy_return = desired * actual - cost_rate * turnover
    return _performance_metrics(strategy_return, desired, turnover)


def buy_and_hold_baseline(actual_simple_return: np.ndarray, *, cost_rate: float = 0.001) -> dict[str, float]:
    actual = np.asarray(actual_simple_return, dtype=np.float64)
    positions = np.ones(len(actual), dtype=np.float64)
    turnover = np.zeros(len(actual), dtype=np.float64)
    if len(actual):
        turnover[0] = 1.0
    returns = actual.copy()
    if len(actual):
        returns[0] -= cost_rate
    return _performance_metrics(returns, positions, turnover)


def classification_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    """Direction-classification quality: accuracy, base rate, Brier score, AUC."""
    y = np.asarray(labels, dtype=np.float64)
    p = np.asarray(probabilities, dtype=np.float64)
    if y.shape != p.shape or y.size == 0:
        raise ValueError("labels and probabilities must be non-empty arrays with identical shapes")
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("probabilities must lie in [0, 1]")

    accuracy = float(np.mean((p >= 0.5) == (y > 0.5)))
    base_rate = float(np.mean(y))
    brier = float(np.mean((p - y) ** 2))

    positives = p[y > 0.5]
    negatives = p[y <= 0.5]
    if len(positives) and len(negatives):
        # Rank-based AUC (Mann-Whitney U) without external dependencies.
        order = np.argsort(np.concatenate([negatives, positives]), kind="mergesort")
        ranks = np.empty(len(order), dtype=np.float64)
        ranks[order] = np.arange(1, len(order) + 1, dtype=np.float64)
        # Average ranks for ties.
        combined = np.concatenate([negatives, positives])
        sorted_vals = combined[order]
        idx = 0
        while idx < len(sorted_vals):
            j = idx
            while j + 1 < len(sorted_vals) and sorted_vals[j + 1] == sorted_vals[idx]:
                j += 1
            if j > idx:
                ranks[order[idx : j + 1]] = float(np.mean(ranks[order[idx : j + 1]]))
            idx = j + 1
        rank_sum_pos = float(np.sum(ranks[len(negatives):]))
        auc = (rank_sum_pos - len(positives) * (len(positives) + 1) / 2.0) / (
            len(positives) * len(negatives)
        )
    else:
        auc = 0.5

    return {
        "direction_accuracy_prob": accuracy,
        "direction_base_rate": base_rate,
        "brier_score": brier,
        "auc": float(auc),
    }


def probability_gated_backtest(
    probability_up: np.ndarray,
    actual_simple_return: np.ndarray,
    *,
    cost_rate: float = 0.001,
    entry_threshold: float = 0.55,
    exit_threshold: float = 0.45,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """Long-only strategy driven by the direction-probability head.

    Hysteresis (enter above entry_threshold, exit below exit_threshold) keeps
    the position through low-confidence bars instead of flip-flopping, which
    is what usually kills hourly strategies once costs are charged.
    """
    prob = np.asarray(probability_up, dtype=np.float64)
    actual = np.asarray(actual_simple_return, dtype=np.float64)
    if prob.shape != actual.shape:
        raise ValueError("probability and actual arrays must have identical shapes")
    if not 0.0 <= exit_threshold <= entry_threshold <= 1.0:
        raise ValueError("thresholds must satisfy 0 <= exit_threshold <= entry_threshold <= 1")
    if cost_rate < 0.0:
        raise ValueError("cost_rate must be non-negative")

    positions = np.zeros(len(prob), dtype=np.float64)
    turnovers = np.zeros(len(prob), dtype=np.float64)
    strategy_returns = np.zeros(len(prob), dtype=np.float64)
    previous_position = 0.0

    for idx, p in enumerate(prob):
        if previous_position == 0.0:
            position = 1.0 if p > entry_threshold else 0.0
        else:
            position = 0.0 if p < exit_threshold else 1.0

        turnover = abs(position - previous_position)
        strategy_returns[idx] = position * actual[idx] - cost_rate * turnover
        positions[idx] = position
        turnovers[idx] = turnover
        previous_position = position

    return _performance_metrics(strategy_returns, positions, turnovers), strategy_returns, positions, turnovers
