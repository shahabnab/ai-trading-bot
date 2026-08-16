from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class PlattCalibrator:
    """Two-parameter logistic calibrator fitted to model probabilities.

    The base model probability is first converted to a logit. A slope and
    intercept are then fitted by regularized Newton updates. Keeping this
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
    curve. In that case the function returns a constant-probability model by
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


@dataclass(frozen=True)
class ConditionalPayoffRegime:
    """Shrunk event/non-event payoff magnitudes for one causal state bucket."""

    name: str
    sample_count: int
    event_count: int
    non_event_count: int
    mean_event_return: float
    mean_non_event_return: float
    event_shrinkage_weight: float
    non_event_shrinkage_weight: float

    def to_dict(self) -> dict[str, str | float | int]:
        return asdict(self)


@dataclass(frozen=True)
class VolatilityConditionedPayoffEstimate:
    """Calibration-only payoff model conditioned on current realized volatility.

    The event probability remains the calibrated classifier output. Only the
    conditional payoff magnitude changes by a LOW/NORMAL/HIGH volatility state.
    Volatility cutoffs are fitted on the calibration slice, and each local mean
    is shrunk toward the global payoff mean so sparse buckets cannot dominate.
    """

    event_hurdle_bps: float
    trim_fraction: float
    volatility_feature_name: str
    low_vol_cutoff: float
    high_vol_cutoff: float
    shrinkage_samples: float
    global_payoff: PayoffEstimate
    low: ConditionalPayoffRegime
    normal: ConditionalPayoffRegime
    high: ConditionalPayoffRegime

    def expected_gross_return(
        self,
        calibrated_probability: np.ndarray,
        realized_volatility: np.ndarray,
    ) -> np.ndarray:
        p, vol = np.broadcast_arrays(
            np.asarray(calibrated_probability, dtype=np.float64),
            np.asarray(realized_volatility, dtype=np.float64),
        )
        event_mean = np.full(p.shape, self.global_payoff.mean_event_return, dtype=np.float64)
        non_event_mean = np.full(p.shape, self.global_payoff.mean_non_event_return, dtype=np.float64)

        finite = np.isfinite(vol)
        low_mask = finite & (vol <= self.low_vol_cutoff)
        normal_mask = finite & (vol > self.low_vol_cutoff) & (vol <= self.high_vol_cutoff)
        high_mask = finite & (vol > self.high_vol_cutoff)

        for mask, regime in (
            (low_mask, self.low),
            (normal_mask, self.normal),
            (high_mask, self.high),
        ):
            event_mean[mask] = regime.mean_event_return
            non_event_mean[mask] = regime.mean_non_event_return

        return p * event_mean + (1.0 - p) * non_event_mean

    def to_dict(self) -> dict[str, object]:
        return {
            "event_hurdle_bps": self.event_hurdle_bps,
            "trim_fraction": self.trim_fraction,
            "volatility_feature_name": self.volatility_feature_name,
            "low_vol_cutoff": self.low_vol_cutoff,
            "high_vol_cutoff": self.high_vol_cutoff,
            "shrinkage_samples": self.shrinkage_samples,
            "global_payoff": self.global_payoff.to_dict(),
            "low": self.low.to_dict(),
            "normal": self.normal.to_dict(),
            "high": self.high.to_dict(),
        }


PayoffModel = PayoffEstimate | VolatilityConditionedPayoffEstimate


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


def _shrink_local_mean(
    values: np.ndarray,
    *,
    global_mean: float,
    trim_fraction: float,
    shrinkage_samples: float,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float(global_mean), 0.0
    local = trimmed_mean(values, trim_fraction=trim_fraction)
    if shrinkage_samples == 0.0:
        weight = 1.0
    else:
        weight = float(values.size) / (float(values.size) + shrinkage_samples)
    shrunk = float(global_mean) + weight * (float(local) - float(global_mean))
    return float(shrunk), float(weight)


def _conditional_regime(
    name: str,
    returns: np.ndarray,
    *,
    event_hurdle_bps: float,
    trim_fraction: float,
    shrinkage_samples: float,
    global_payoff: PayoffEstimate,
) -> ConditionalPayoffRegime:
    values = np.asarray(returns, dtype=np.float64).reshape(-1)
    hurdle = float(event_hurdle_bps) / 10_000.0
    event_mask = values > hurdle
    winners = values[event_mask]
    others = values[~event_mask]
    event_mean, event_weight = _shrink_local_mean(
        winners,
        global_mean=global_payoff.mean_event_return,
        trim_fraction=trim_fraction,
        shrinkage_samples=shrinkage_samples,
    )
    non_event_mean, non_event_weight = _shrink_local_mean(
        others,
        global_mean=global_payoff.mean_non_event_return,
        trim_fraction=trim_fraction,
        shrinkage_samples=shrinkage_samples,
    )
    return ConditionalPayoffRegime(
        name=name,
        sample_count=int(values.size),
        event_count=int(winners.size),
        non_event_count=int(others.size),
        mean_event_return=event_mean,
        mean_non_event_return=non_event_mean,
        event_shrinkage_weight=event_weight,
        non_event_shrinkage_weight=non_event_weight,
    )


def estimate_volatility_conditioned_payoffs(
    horizon_simple_return: np.ndarray,
    realized_volatility: np.ndarray,
    *,
    event_hurdle_bps: float,
    trim_fraction: float = 0.05,
    low_quantile: float = 1.0 / 3.0,
    high_quantile: float = 2.0 / 3.0,
    shrinkage_samples: float = 30.0,
    volatility_feature_name: str = "realized_vol_24h",
) -> VolatilityConditionedPayoffEstimate:
    """Fit a LOW/NORMAL/HIGH volatility payoff model on calibration data only.

    This helper is intentionally state-conditional only in the payoff magnitudes;
    it does not refit the classifier or event probabilities. The state variable
    must be known at decision time (for example trailing 24h realized volatility),
    never a future volatility target.
    """
    returns = np.asarray(horizon_simple_return, dtype=np.float64).reshape(-1)
    volatility = np.asarray(realized_volatility, dtype=np.float64).reshape(-1)
    if returns.shape != volatility.shape or returns.size == 0:
        raise ValueError("returns and realized_volatility must have identical non-empty shapes")
    if not 0.0 < low_quantile < high_quantile < 1.0:
        raise ValueError("volatility quantiles must satisfy 0 < low < high < 1")
    if shrinkage_samples < 0.0:
        raise ValueError("shrinkage_samples must be non-negative")

    finite = np.isfinite(returns) & np.isfinite(volatility)
    paired_returns = returns[finite]
    paired_volatility = volatility[finite]
    if paired_returns.size < 15:
        raise ValueError("too few finite calibration rows for volatility-conditioned payoff estimation")

    global_payoff = estimate_payoffs(
        paired_returns,
        event_hurdle_bps=event_hurdle_bps,
        trim_fraction=trim_fraction,
    )
    low_cutoff, high_cutoff = np.quantile(paired_volatility, [low_quantile, high_quantile])
    low_cutoff = float(low_cutoff)
    high_cutoff = float(high_cutoff)
    if not math.isfinite(low_cutoff) or not math.isfinite(high_cutoff) or low_cutoff >= high_cutoff:
        raise ValueError("calibration volatility does not support distinct LOW/NORMAL/HIGH cutoffs")

    low_mask = paired_volatility <= low_cutoff
    normal_mask = (paired_volatility > low_cutoff) & (paired_volatility <= high_cutoff)
    high_mask = paired_volatility > high_cutoff

    return VolatilityConditionedPayoffEstimate(
        event_hurdle_bps=float(event_hurdle_bps),
        trim_fraction=float(trim_fraction),
        volatility_feature_name=str(volatility_feature_name),
        low_vol_cutoff=low_cutoff,
        high_vol_cutoff=high_cutoff,
        shrinkage_samples=float(shrinkage_samples),
        global_payoff=global_payoff,
        low=_conditional_regime(
            "LOW",
            paired_returns[low_mask],
            event_hurdle_bps=event_hurdle_bps,
            trim_fraction=trim_fraction,
            shrinkage_samples=shrinkage_samples,
            global_payoff=global_payoff,
        ),
        normal=_conditional_regime(
            "NORMAL",
            paired_returns[normal_mask],
            event_hurdle_bps=event_hurdle_bps,
            trim_fraction=trim_fraction,
            shrinkage_samples=shrinkage_samples,
            global_payoff=global_payoff,
        ),
        high=_conditional_regime(
            "HIGH",
            paired_returns[high_mask],
            event_hurdle_bps=event_hurdle_bps,
            trim_fraction=trim_fraction,
            shrinkage_samples=shrinkage_samples,
            global_payoff=global_payoff,
        ),
    )


def payoff_trim_sensitivity(
    horizon_simple_return: np.ndarray,
    *,
    event_hurdle_bps: float,
    trim_fractions: Iterable[float] = (0.0, 0.025, 0.05),
) -> list[dict[str, float | int]]:
    """Return comparable payoff estimates without selecting a preferred trim.

    This is a diagnostic helper for the V3 research reports. It deliberately
    does not choose the trim fraction from OOS/test performance; callers should
    report the sensitivity and keep policy/model selection on validation data.
    """
    rows: list[dict[str, float | int]] = []
    for trim_fraction in trim_fractions:
        payoff = estimate_payoffs(
            horizon_simple_return,
            event_hurdle_bps=event_hurdle_bps,
            trim_fraction=float(trim_fraction),
        )
        rows.append(payoff.to_dict())
    return rows


def _gross_return(
    calibrated_probability: np.ndarray,
    payoff: PayoffModel,
    *,
    payoff_state: np.ndarray | None = None,
) -> np.ndarray:
    if isinstance(payoff, VolatilityConditionedPayoffEstimate):
        if payoff_state is None:
            raise ValueError("payoff_state is required for a volatility-conditioned payoff")
        return payoff.expected_gross_return(calibrated_probability, payoff_state)
    return payoff.expected_gross_return(calibrated_probability)


def expected_value(
    calibrated_probability: np.ndarray,
    payoff: PayoffModel,
    *,
    position: float,
    one_way_cost_rate: float,
    payoff_state: np.ndarray | None = None,
) -> np.ndarray:
    """Expected net return for a commitment decision."""
    if one_way_cost_rate < 0.0:
        raise ValueError("one_way_cost_rate must be non-negative")
    gross = _gross_return(calibrated_probability, payoff, payoff_state=payoff_state)
    if float(position) <= 0.0:
        return gross - 2.0 * one_way_cost_rate
    return gross


def ev_commitment_backtest(
    calibrated_probability: np.ndarray,
    actual_simple_return_1h: np.ndarray,
    *,
    payoff: PayoffModel,
    one_way_cost_rate: float,
    horizon_hours: int,
    entry_margin: float = 0.0,
    exit_ev_threshold: float = 0.0,
    force_flat_at_end: bool = True,
    payoff_state: np.ndarray | None = None,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Long/flat EV strategy with horizon-matched commitment boundaries.

    A decision made at t is held for ``horizon_hours`` bars. The next EV check
    occurs only after that commitment has elapsed, which keeps h-hour payoff
    estimates consistent with the decision cadence. Realized P&L is accumulated
    from the actual next-hour return series and cost is charged only when the
    realized position changes.

    ``exit_ev_threshold`` is an *additional margin* around the economically
    neutral HOLD-vs-EXIT boundary. While already long, exiting costs one-way
    execution cost immediately, so the baseline exit condition is
    ``gross_ev <= -one_way_cost_rate``. The historical value ``0.0`` therefore
    means "exit only when holding is worse than paying the exit cost", not
    "exit at any negative gross EV".

    For a ``VolatilityConditionedPayoffEstimate``, ``payoff_state`` must contain
    the causal realized-volatility value aligned with each probability row.
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

    state: np.ndarray | None = None
    if payoff_state is not None:
        state = np.asarray(payoff_state, dtype=np.float64).reshape(-1)
        if state.shape != probability.shape:
            raise ValueError("payoff_state must align with probability")
    if isinstance(payoff, VolatilityConditionedPayoffEstimate) and state is None:
        raise ValueError("payoff_state is required for a volatility-conditioned payoff")

    positions = np.zeros(len(probability), dtype=np.float64)
    turnovers = np.zeros(len(probability), dtype=np.float64)
    strategy_returns = np.zeros(len(probability), dtype=np.float64)
    ev_values = np.full(len(probability), np.nan, dtype=np.float64)

    position = 0.0
    bars_until_decision = 0
    for idx, p in enumerate(probability):
        next_position = position
        if bars_until_decision <= 0:
            local_state = None if state is None else np.asarray([state[idx]], dtype=np.float64)
            gross_ev = float(
                _gross_return(
                    np.asarray([p], dtype=np.float64),
                    payoff,
                    payoff_state=local_state,
                )[0]
            )
            if position <= 0.0:
                decision_ev = gross_ev - 2.0 * one_way_cost_rate
                if decision_ev > entry_margin:
                    next_position = 1.0
            else:
                # Compare holding the position with exiting now. Exiting has an
                # immediate one-way execution cost, so a mildly negative gross
                # EV can still be economically preferable to paying that cost.
                decision_ev = gross_ev + one_way_cost_rate
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
