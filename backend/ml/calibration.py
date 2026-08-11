from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np


@dataclass(frozen=True)
class PlattCalibrator:
    """Two-parameter logistic calibrator fitted to model probabilities.

    The base model probability is first converted to a logit.  A slope and
    intercept are then fitted by regularized Newton updates.  Keeping this
    implementation dependency-free avoids pulling scikit-learn onto the GPU
    server just for calibration.
    """

    slope: float
    intercept: float
    clip_eps: float = 1e-6

    def transform(self, probability: np.ndarray) -> np.ndarray:
        p = np.asarray(probability, dtype=np.float64)
        p = np.clip(p, self.clip_eps, 1.0 - self.clip_eps)
        x = np.log(p / (1.0 - p))
        z = np.clip(self.slope * x + self.intercept, -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-z))

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(np.asarray(z, dtype=np.float64), -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z))


def fit_platt_scaler(
    probability: np.ndarray,
    labels: np.ndarray,
    *,
    l2: float = 1e-3,
    max_iter: int = 100,
    tolerance: float = 1e-9,
    clip_eps: float = 1e-6,
) -> PlattCalibrator:
    """Fit Platt scaling using a stable 2x2 Newton solve.

    A single-class calibration slice cannot identify a logistic calibration
    curve.  In that case the function returns a constant-probability model by
    using slope=0 and a Laplace-smoothed class-rate intercept.
    """
    p = np.asarray(probability, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    if p.shape != y.shape or p.size == 0:
        raise ValueError("probability and labels must be non-empty arrays with identical shapes")
    if np.any(~np.isfinite(p)) or np.any(~np.isfinite(y)):
        raise ValueError("calibration inputs must be finite")
    if np.any((y != 0.0) & (y != 1.0)):
        raise ValueError("labels must be binary 0/1")
    if l2 < 0.0:
        raise ValueError("l2 must be non-negative")

    clipped = np.clip(p, clip_eps, 1.0 - clip_eps)
    x = np.log(clipped / (1.0 - clipped))
    positives = float(np.sum(y))
    negatives = float(len(y) - positives)

    if positives == 0.0 or negatives == 0.0:
        rate = (positives + 1.0) / (len(y) + 2.0)
        return PlattCalibrator(0.0, float(math.log(rate / (1.0 - rate))), clip_eps)

    slope = 1.0
    base_rate = (positives + 1.0) / (len(y) + 2.0)
    intercept = float(math.log(base_rate / (1.0 - base_rate)))

    for _ in range(max_iter):
        z = slope * x + intercept
        q = _sigmoid(z)
        residual = q - y
        weight = np.maximum(q * (1.0 - q), 1e-10)

        grad_slope = float(np.sum(residual * x) + l2 * slope)
        grad_intercept = float(np.sum(residual))
        h_ss = float(np.sum(weight * x * x) + l2)
        h_si = float(np.sum(weight * x))
        h_ii = float(np.sum(weight))

        det = h_ss * h_ii - h_si * h_si
        if not math.isfinite(det) or abs(det) < 1e-18:
            break
        step_slope = (h_ii * grad_slope - h_si * grad_intercept) / det
        step_intercept = (-h_si * grad_slope + h_ss * grad_intercept) / det

        slope_next = slope - step_slope
        intercept_next = intercept - step_intercept
        if max(abs(step_slope), abs(step_intercept)) < tolerance:
            slope, intercept = slope_next, intercept_next
            break
        slope, intercept = slope_next, intercept_next

    return PlattCalibrator(float(slope), float(intercept), clip_eps)


def brier_score(labels: np.ndarray, probability: np.ndarray) -> float:
    y = np.asarray(labels, dtype=np.float64)
    p = np.asarray(probability, dtype=np.float64)
    if y.shape != p.shape or y.size == 0:
        raise ValueError("labels and probability must have identical non-empty shapes")
    return float(np.mean((p - y) ** 2))


def trimmed_mean(values: np.ndarray, *, trim_fraction: float = 0.05) -> float:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if array.size == 0:
        raise ValueError("cannot estimate payoff from an empty array")
    if not 0.0 <= trim_fraction < 0.5:
        raise ValueError("trim_fraction must be in [0, 0.5)")
    ordered = np.sort(array)
    trim = int(math.floor(len(ordered) * trim_fraction))
    if trim > 0 and 2 * trim < len(ordered):
        ordered = ordered[trim:-trim]
    return float(np.mean(ordered))


@dataclass(frozen=True)
class PayoffEstimate:
    event_hurdle_bps: float
    trim_fraction: float
    event_count: int
    non_event_count: int
    event_rate: float
    mean_event_return: float
    mean_non_event_return: float
    median_event_return: float
    median_non_event_return: float

    def expected_gross_return(self, calibrated_probability: np.ndarray) -> np.ndarray:
        p = np.asarray(calibrated_probability, dtype=np.float64)
        return p * self.mean_event_return + (1.0 - p) * self.mean_non_event_return

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def estimate_payoffs(
    horizon_simple_return: np.ndarray,
    *,
    event_hurdle_bps: float,
    trim_fraction: float = 0.05,
) -> PayoffEstimate:
    returns = np.asarray(horizon_simple_return, dtype=np.float64).reshape(-1)
    returns = returns[np.isfinite(returns)]
    if returns.size == 0:
        raise ValueError("horizon returns are empty")
    hurdle = float(event_hurdle_bps) / 10_000.0
    event = returns > hurdle
    winners = returns[event]
    others = returns[~event]
    if winners.size < 5 or others.size < 5:
        raise ValueError("calibration slice has too few event/non-event observations for payoff estimation")
    return PayoffEstimate(
        event_hurdle_bps=float(event_hurdle_bps),
        trim_fraction=float(trim_fraction),
        event_count=int(winners.size),
        non_event_count=int(others.size),
        event_rate=float(np.mean(event)),
        mean_event_return=trimmed_mean(winners, trim_fraction=trim_fraction),
        mean_non_event_return=trimmed_mean(others, trim_fraction=trim_fraction),
        median_event_return=float(np.median(winners)),
        median_non_event_return=float(np.median(others)),
    )


def expected_value(
    calibrated_probability: np.ndarray,
    payoff: PayoffEstimate,
    *,
    position: float,
    one_way_cost_rate: float,
) -> np.ndarray:
    """Expected net return for a commitment decision."""
    if one_way_cost_rate < 0.0:
        raise ValueError("one_way_cost_rate must be non-negative")
    gross = payoff.expected_gross_return(calibrated_probability)
    if float(position) <= 0.0:
        return gross - 2.0 * one_way_cost_rate
    return gross


def ev_commitment_backtest(
    calibrated_probability: np.ndarray,
    actual_simple_return_1h: np.ndarray,
    *,
    payoff: PayoffEstimate,
    one_way_cost_rate: float,
    horizon_hours: int,
    entry_margin: float = 0.0,
    exit_ev_threshold: float = 0.0,
    force_flat_at_end: bool = True,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Long/flat EV strategy with horizon-matched commitment boundaries.

    A decision made at t is held for ``horizon_hours`` bars. The next EV check
    occurs only after that commitment has elapsed, which keeps h-hour payoff
    estimates consistent with the decision cadence. Realized P&L is accumulated
    from the actual next-hour return series and cost is charged only when the
    realized position changes.
    """
    from backend.ml.evaluation import _performance_metrics

    probability = np.asarray(calibrated_probability, dtype=np.float64).reshape(-1)
    actual = np.asarray(actual_simple_return_1h, dtype=np.float64).reshape(-1)
    if probability.shape != actual.shape or probability.size == 0:
        raise ValueError("probability and actual returns must have identical non-empty shapes")
    if horizon_hours <= 0:
        raise ValueError("horizon_hours must be positive")
    if one_way_cost_rate < 0.0:
        raise ValueError("one_way_cost_rate must be non-negative")

    positions = np.zeros(len(probability), dtype=np.float64)
    turnovers = np.zeros(len(probability), dtype=np.float64)
    strategy_returns = np.zeros(len(probability), dtype=np.float64)
    ev_values = np.full(len(probability), np.nan, dtype=np.float64)

    position = 0.0
    bars_until_decision = 0
    for idx, p in enumerate(probability):
        next_position = position
        if bars_until_decision <= 0:
            gross_ev = float(payoff.expected_gross_return(np.asarray([p]))[0])
            if position <= 0.0:
                decision_ev = gross_ev - 2.0 * one_way_cost_rate
                if decision_ev > entry_margin:
                    next_position = 1.0
            else:
                decision_ev = gross_ev
                if decision_ev <= exit_ev_threshold:
                    next_position = 0.0
            ev_values[idx] = decision_ev
            bars_until_decision = horizon_hours

        turnover = abs(next_position - position)
        strategy_returns[idx] = next_position * actual[idx] - one_way_cost_rate * turnover
        positions[idx] = next_position
        turnovers[idx] = turnover
        position = next_position
        bars_until_decision -= 1

    forced_exit = False
    if force_flat_at_end and position > 0.0 and len(strategy_returns):
        strategy_returns[-1] -= one_way_cost_rate
        turnovers[-1] += 1.0
        forced_exit = True

    metrics = _performance_metrics(strategy_returns, positions, turnovers)
    if forced_exit:
        metrics["round_trip_count"] = int(metrics["round_trip_count"]) + 1
        metrics["position_change_count"] = int(metrics["position_change_count"]) + 1
    return metrics, strategy_returns, positions, turnovers, ev_values
