#!/usr/bin/env python3
"""Multi-horizon research runner for the BTC trading models.

The existing ``train_all_server.py`` remains the immutable 1-hour baseline.
This runner performs a more demanding experiment suite:

* prediction horizons: 1h / 3h / 6h / 12h by default;
* XGBoost and LSTM hyperparameter random search using validation data only;
* model validation and execution-policy validation are separated;
* horizon-aware purging prevents labels from crossing validation boundaries;
* execution lambda is constrained to >= 1 by default;
* minimum holding periods are selected on policy-validation data;
* trading P&L is always calculated from next-hour realized returns, even when
  the model predicts a multi-hour return, avoiding overlapping-return P&L;
* multiple transaction-cost scenarios are evaluated without touching test data;
* every outer test window remains untouched until all model/policy choices for
  that fold have been made.

Protocol per fold:

    TRAIN -> MODEL VALIDATION -> POLICY VALIDATION -> UNTOUCHED TEST

Run from the repository root.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import random
import statistics
import subprocess
import time
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from backend.ml.evaluation import (
    HOUR_MS,
    WalkForwardConfig,
    _performance_metrics,
    make_walk_forward_folds,
    regression_metrics,
)
from backend.ml.features import FEATURE_VERSION, build_feature_dataset, read_jsonl
from backend.ml.sequences import (
    fit_standardizer,
    inverse_standardize,
    make_sequence_batch,
    standardize,
)

DEFAULT_DATASET = Path("data/processed/training/btc_hourly.jsonl")
DEFAULT_OUTPUT = Path("artifacts/ml/research_experiments")


@dataclass(frozen=True)
class ResearchDataset:
    horizon_hours: int
    timestamps: np.ndarray
    X: np.ndarray
    target_log_return: np.ndarray
    target_simple_return: np.ndarray
    execution_simple_return_1h: np.ndarray
    closes: np.ndarray
    ema20: np.ndarray
    ema50: np.ndarray
    ema200: np.ndarray
    feature_names: list[str]

    @property
    def size(self) -> int:
        return int(self.X.shape[0])


@dataclass(frozen=True)
class PolicyChoice:
    execution_lambda: float
    hold_hours: int


def _jsonable(value: object) -> object:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, tuple)):
        return list(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_jsonable),
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=_jsonable) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def parse_int_grid(value: str) -> list[int]:
    try:
        values = sorted({int(x.strip()) for x in value.split(",") if x.strip()})
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not values or values[0] <= 0:
        raise argparse.ArgumentTypeError("values must be positive integers")
    return values


def parse_float_grid(value: str) -> list[float]:
    try:
        values = sorted({float(x.strip()) for x in value.split(",") if x.strip()})
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not values or values[0] < 0.0:
        raise argparse.ArgumentTypeError("values must be non-negative")
    return values


def _raw_close_map(rows: list[dict[str, Any]]) -> tuple[dict[int, float], set[int]]:
    closes: dict[int, float] = {}
    for row in rows:
        try:
            ts = int(row["timestamp"])
            close = float(row["close"])
        except (KeyError, TypeError, ValueError):
            continue
        if ts > 0 and math.isfinite(close) and close > 0.0:
            closes[ts] = close
    return closes, set(closes)


def build_research_dataset(
    raw_rows: list[dict[str, Any]],
    *,
    horizon_hours: int,
    include_sentiment: bool = False,
) -> ResearchDataset:
    """Create a gap-safe future-return target while reusing causal features.

    A horizon target is kept only when every hourly timestamp from t+1 through
    t+h exists. This prevents a return from silently jumping across an exchange
    outage. The 1-hour realized return is retained separately for P&L.
    """
    if horizon_hours <= 0:
        raise ValueError("horizon_hours must be positive")

    base = build_feature_dataset(raw_rows, include_sentiment=include_sentiment)
    close_by_ts, available = _raw_close_map(raw_rows)

    keep: list[int] = []
    target_simple: list[float] = []
    target_log: list[float] = []
    execution_1h: list[float] = []

    for idx, timestamp_raw in enumerate(base.timestamps):
        timestamp = int(timestamp_raw)
        current_close = float(base.closes[idx])
        needed = [timestamp + offset * HOUR_MS for offset in range(1, horizon_hours + 1)]
        if any(ts not in available for ts in needed):
            continue
        future_close = close_by_ts[needed[-1]]
        next_close = close_by_ts[needed[0]]
        h_return = future_close / current_close - 1.0
        one_hour_return = next_close / current_close - 1.0
        if h_return <= -1.0 or not (math.isfinite(h_return) and math.isfinite(one_hour_return)):
            continue
        keep.append(idx)
        target_simple.append(h_return)
        target_log.append(math.log1p(h_return))
        execution_1h.append(one_hour_return)

    if not keep:
        raise ValueError(f"no usable rows for horizon={horizon_hours}h")

    selected = np.asarray(keep, dtype=np.int64)
    return ResearchDataset(
        horizon_hours=horizon_hours,
        timestamps=base.timestamps[selected],
        X=base.X[selected],
        target_log_return=np.asarray(target_log, dtype=np.float32),
        target_simple_return=np.asarray(target_simple, dtype=np.float64),
        execution_simple_return_1h=np.asarray(execution_1h, dtype=np.float64),
        closes=base.closes[selected],
        ema20=base.ema20[selected],
        ema50=base.ema50[selected],
        ema200=base.ema200[selected],
        feature_names=base.feature_names,
    )


def split_model_policy_validation(
    validation_indices: np.ndarray,
    fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.asarray(validation_indices, dtype=np.int64)
    if len(indices) < 8:
        raise ValueError("validation window is too small to split safely")
    cut = int(round(len(indices) * fraction))
    cut = max(2, min(cut, len(indices) - 2))
    return indices[:cut], indices[cut:]


def purge_label_boundary(
    indices: np.ndarray,
    timestamps: np.ndarray,
    *,
    horizon_hours: int,
    boundary_timestamp: int,
) -> np.ndarray:
    """Keep labels whose future endpoint is strictly before a boundary."""
    idx = np.asarray(indices, dtype=np.int64)
    ends = timestamps[idx] + int(horizon_hours) * HOUR_MS
    return idx[ends < int(boundary_timestamp)]


def purge_execution_boundary(
    indices: np.ndarray,
    timestamps: np.ndarray,
    *,
    boundary_timestamp: int,
) -> np.ndarray:
    """Policy validation uses next-hour P&L, so purge its last realized bar."""
    idx = np.asarray(indices, dtype=np.int64)
    ends = timestamps[idx] + HOUR_MS
    return idx[ends < int(boundary_timestamp)]


def held_cost_aware_backtest(
    predicted_horizon_log_return: np.ndarray,
    actual_simple_return_1h: np.ndarray,
    *,
    cost_rate: float,
    execution_lambda: float,
    hold_hours: int,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """Long/flat execution using hourly realized P&L and a minimum hold.

    ``predicted_horizon_log_return`` can describe 1h, 3h, 6h, ... ahead.
    P&L is deliberately based on the next-hour realized return at each decision
    time so overlapping multi-hour targets are never counted as independent P&L.
    """
    predicted = np.asarray(predicted_horizon_log_return, dtype=np.float64)
    actual = np.asarray(actual_simple_return_1h, dtype=np.float64)
    if predicted.shape != actual.shape or predicted.ndim != 1:
        raise ValueError("predicted and actual returns must be 1-D arrays with identical shapes")
    if cost_rate < 0.0 or execution_lambda < 0.0 or hold_hours <= 0:
        raise ValueError("invalid execution parameters")

    positions = np.zeros(len(predicted), dtype=np.float64)
    turnovers = np.zeros(len(predicted), dtype=np.float64)
    returns = np.zeros(len(predicted), dtype=np.float64)
    position = 0.0
    bars_since_change = hold_hours

    for idx, forecast in enumerate(predicted):
        desired = 1.0 if forecast > 0.0 else 0.0
        requested_turnover = abs(desired - position)
        next_position = position

        if requested_turnover > 0.0 and bars_since_change >= hold_hours:
            hurdle = execution_lambda * cost_rate * requested_turnover
            if abs(float(forecast)) > hurdle:
                next_position = desired

        turnover = abs(next_position - position)
        returns[idx] = next_position * actual[idx] - cost_rate * turnover
        positions[idx] = next_position
        turnovers[idx] = turnover

        if turnover > 0.0:
            bars_since_change = 1
        else:
            bars_since_change += 1
        position = next_position

    return _performance_metrics(returns, positions, turnovers), returns, positions, turnovers


def choose_execution_policy(
    predicted: np.ndarray,
    actual_execution_return_1h: np.ndarray,
    *,
    cost_rate: float,
    lambda_grid: list[float],
    holding_grid: list[int],
    min_trades: int,
    max_drawdown_abs: float,
) -> tuple[PolicyChoice, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for hold_hours in holding_grid:
        for execution_lambda in lambda_grid:
            metrics, _, _, _ = held_cost_aware_backtest(
                predicted,
                actual_execution_return_1h,
                cost_rate=cost_rate,
                execution_lambda=execution_lambda,
                hold_hours=hold_hours,
            )
            rows.append(
                {
                    "execution_lambda": execution_lambda,
                    "hold_hours": hold_hours,
                    **metrics,
                }
            )

    strong = [
        row
        for row in rows
        if int(row["trade_count"]) >= min_trades
        and float(row["sharpe"]) > 0.0
        and float(row["max_drawdown"]) >= -max_drawdown_abs
        and float(row["cumulative_return"]) > 0.0
    ]
    trade_ok = [row for row in rows if int(row["trade_count"]) >= min_trades]
    pool = strong or trade_ok or rows
    best = max(
        pool,
        key=lambda row: (
            float(row["sharpe"]),
            float(row["sortino"]),
            float(row["max_drawdown"]),
            float(row["cumulative_return"]),
            -float(row["turnover"]),
            -int(row["hold_hours"]),
            float(row["execution_lambda"]),
        ),
    )
    return (
        PolicyChoice(float(best["execution_lambda"]), int(best["hold_hours"])),
        rows,
    )


def aggregate_fold_strategies(
    prediction_rows: list[dict[str, Any]],
    *,
    cost_key: str,
    fold_metrics: list[dict[str, Any]],
) -> dict[str, float]:
    ordered = sorted(prediction_rows, key=lambda row: int(row["timestamp"]))
    metrics = _performance_metrics(
        np.asarray([row[f"{cost_key}_return"] for row in ordered], dtype=np.float64),
        np.asarray([row[f"{cost_key}_position"] for row in ordered], dtype=np.float64),
        np.asarray([row[f"{cost_key}_turnover"] for row in ordered], dtype=np.float64),
    )
    for name in ("trade_count", "round_trip_count", "position_change_count"):
        metrics[name] = int(sum(int(row[name]) for row in fold_metrics))
    metrics["turnover"] = float(sum(float(row["turnover"]) for row in fold_metrics))
    return metrics


def xgb_candidate_pool(trials: int, seed: int) -> list[dict[str, Any]]:
    from backend.ml.xgboost_core import DEFAULT_XGB_PARAMS

    if trials <= 0:
        raise ValueError("xgb_trials must be positive")
    base = dict(DEFAULT_XGB_PARAMS)
    candidates = [base]
    rng = random.Random(seed)
    seen = {json.dumps(base, sort_keys=True)}
    while len(candidates) < trials:
        candidate = dict(base)
        candidate.update(
            {
                "max_depth": rng.choice([2, 3, 4, 5, 6]),
                "eta": rng.choice([0.005, 0.01, 0.02, 0.05, 0.1]),
                "min_child_weight": rng.choice([5.0, 10.0, 20.0, 40.0, 80.0]),
                "subsample": rng.choice([0.6, 0.75, 0.9, 1.0]),
                "colsample_bytree": rng.choice([0.6, 0.75, 0.9, 1.0]),
                "lambda": rng.choice([1.0, 5.0, 10.0, 20.0, 50.0]),
                "alpha": rng.choice([0.0, 0.001, 0.01, 0.1, 1.0]),
                "gamma": rng.choice([0.0, 0.001, 0.01, 0.1]),
            }
        )
        key = json.dumps(candidate, sort_keys=True)
        if key not in seen:
            seen.add(key)
            candidates.append(candidate)
    return candidates


def lstm_candidate_pool(trials: int, seed: int) -> list[dict[str, Any]]:
    if trials <= 0:
        raise ValueError("lstm_trials must be positive")
    base = {
        "sequence_length": 48,
        "lstm_units": 64,
        "dense_units": 32,
        "dropout": 0.2,
        "learning_rate": 1e-3,
        "batch_size": 64,
        "clipnorm": 1.0,
    }
    candidates = [base]
    rng = random.Random(seed)
    seen = {json.dumps(base, sort_keys=True)}
    while len(candidates) < trials:
        candidate = {
            "sequence_length": rng.choice([24, 48, 72, 168]),
            "lstm_units": rng.choice([32, 64, 96, 128]),
            "dense_units": rng.choice([16, 32, 64]),
            "dropout": rng.choice([0.0, 0.1, 0.2, 0.3, 0.4]),
            "learning_rate": rng.choice([3e-4, 5e-4, 1e-3, 2e-3]),
            "batch_size": rng.choice([32, 64, 128]),
            "clipnorm": rng.choice([0.5, 1.0, 2.0]),
        }
        key = json.dumps(candidate, sort_keys=True)
        if key not in seen:
            seen.add(key)
            candidates.append(candidate)
    return candidates


def _mode_int(values: list[int]) -> int:
    if not values:
        raise ValueError("cannot take mode of empty values")
    counts = Counter(values)
    return min(counts, key=lambda value: (-counts[value], value))


def _policy_manifest(selections: dict[float, list[PolicyChoice]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for cost_bps, choices in sorted(selections.items()):
        payload[f"{cost_bps:g}"] = {
            "execution_lambda": float(statistics.median(c.execution_lambda for c in choices)),
            "hold_hours": _mode_int([c.hold_hours for c in choices]),
            "fold_lambdas": [c.execution_lambda for c in choices],
            "fold_hold_hours": [c.hold_hours for c in choices],
        }
    return payload


def train_xgboost_horizon(
    dataset: ResearchDataset,
    folds: list[Any],
    out: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    import xgboost as xgb

    candidates = xgb_candidate_pool(args.xgb_trials, args.seed + dataset.horizon_hours * 101)
    model_dir = out / "fold_models"
    model_dir.mkdir(parents=True, exist_ok=True)

    prediction_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    policy_rows: list[dict[str, Any]] = []
    policy_choices: dict[float, list[PolicyChoice]] = defaultdict(list)
    selected_candidate_ids: list[int] = []
    best_rounds: list[int] = []
    per_cost_fold_metrics: dict[float, list[dict[str, Any]]] = defaultdict(list)

    for fold in folds:
        model_idx_raw, policy_idx_raw = split_model_policy_validation(
            fold.validation_indices, args.model_validation_fraction
        )
        model_start = int(dataset.timestamps[model_idx_raw[0]])
        policy_start = int(dataset.timestamps[policy_idx_raw[0]])
        test_start = int(dataset.timestamps[fold.test_indices[0]])

        train_idx = purge_label_boundary(
            fold.train_indices,
            dataset.timestamps,
            horizon_hours=dataset.horizon_hours,
            boundary_timestamp=model_start,
        )
        model_idx = purge_label_boundary(
            model_idx_raw,
            dataset.timestamps,
            horizon_hours=dataset.horizon_hours,
            boundary_timestamp=policy_start,
        )
        policy_idx = purge_execution_boundary(
            policy_idx_raw,
            dataset.timestamps,
            boundary_timestamp=test_start,
        )
        test_idx = fold.test_indices
        if min(len(train_idx), len(model_idx), len(policy_idx), len(test_idx)) == 0:
            raise ValueError(f"empty purged split in XGBoost fold {fold.fold}")

        dtrain = xgb.DMatrix(
            dataset.X[train_idx],
            label=dataset.target_log_return[train_idx],
            feature_names=dataset.feature_names,
        )
        dmodel = xgb.DMatrix(
            dataset.X[model_idx],
            label=dataset.target_log_return[model_idx],
            feature_names=dataset.feature_names,
        )
        dpolicy = xgb.DMatrix(dataset.X[policy_idx], feature_names=dataset.feature_names)
        dtest = xgb.DMatrix(dataset.X[test_idx], feature_names=dataset.feature_names)

        best_booster = None
        best_candidate_id = -1
        best_model_score = math.inf
        best_round = 0
        for candidate_id, base_params in enumerate(candidates):
            params = dict(base_params)
            params["seed"] = args.seed + fold.fold + candidate_id * 1000
            if args.xgb_device == "cuda":
                params["device"] = "cuda"
            booster = xgb.train(
                params,
                dtrain,
                num_boost_round=args.xgb_rounds,
                evals=[(dmodel, "model_validation")],
                callbacks=[
                    xgb.callback.EarlyStopping(
                        rounds=args.xgb_early_stopping,
                        save_best=True,
                    )
                ],
                verbose_eval=False,
            )
            pred = booster.predict(dmodel).astype(np.float64)
            val_metrics = regression_metrics(dataset.target_log_return[model_idx], pred)
            score = float(val_metrics["rmse_log_return"])
            rounds = int(booster.best_iteration) + 1
            selection_rows.append(
                {
                    "model": "xgboost",
                    "horizon_hours": dataset.horizon_hours,
                    "fold": fold.fold,
                    "candidate_id": candidate_id,
                    "best_rounds": rounds,
                    **val_metrics,
                    "params_json": json.dumps(params, sort_keys=True),
                }
            )
            if score < best_model_score:
                best_model_score = score
                best_booster = booster
                best_candidate_id = candidate_id
                best_round = rounds

        assert best_booster is not None
        selected_candidate_ids.append(best_candidate_id)
        best_rounds.append(best_round)
        best_booster.save_model(model_dir / f"fold_{fold.fold:02d}.json")

        policy_pred = best_booster.predict(dpolicy).astype(np.float64)
        test_pred = best_booster.predict(dtest).astype(np.float64)
        forecast = regression_metrics(dataset.target_log_return[test_idx], test_pred)

        fold_row: dict[str, Any] = {
            "model": "xgboost",
            "horizon_hours": dataset.horizon_hours,
            "fold": fold.fold,
            "selected_candidate_id": best_candidate_id,
            "best_rounds": best_round,
            "model_validation_rmse": best_model_score,
            "train_rows_after_purge": len(train_idx),
            "model_validation_rows_after_purge": len(model_idx),
            "policy_validation_rows_after_purge": len(policy_idx),
            **{f"forecast_{k}": v for k, v in forecast.items()},
        }

        cost_outputs: dict[float, tuple[PolicyChoice, dict[str, float], np.ndarray, np.ndarray, np.ndarray]] = {}
        for cost_bps in args.cost_bps:
            cost_rate = cost_bps / 10_000.0
            choice, sweep = choose_execution_policy(
                policy_pred,
                dataset.execution_simple_return_1h[policy_idx],
                cost_rate=cost_rate,
                lambda_grid=args.threshold_grid,
                holding_grid=args.holding_grid,
                min_trades=args.min_policy_trades,
                max_drawdown_abs=args.max_validation_drawdown,
            )
            policy_choices[cost_bps].append(choice)
            policy_rows.extend(
                {
                    "model": "xgboost",
                    "horizon_hours": dataset.horizon_hours,
                    "fold": fold.fold,
                    "cost_bps": cost_bps,
                    **row,
                }
                for row in sweep
            )
            metrics, returns, positions, turnovers = held_cost_aware_backtest(
                test_pred,
                dataset.execution_simple_return_1h[test_idx],
                cost_rate=cost_rate,
                execution_lambda=choice.execution_lambda,
                hold_hours=choice.hold_hours,
            )
            per_cost_fold_metrics[cost_bps].append(metrics)
            cost_outputs[cost_bps] = (choice, metrics, returns, positions, turnovers)
            prefix = f"cost_{cost_bps:g}bps"
            fold_row[f"{prefix}_lambda"] = choice.execution_lambda
            fold_row[f"{prefix}_hold_hours"] = choice.hold_hours
            for key, value in metrics.items():
                fold_row[f"{prefix}_{key}"] = value

        fold_rows.append(fold_row)
        primary = cost_outputs[args.primary_cost_bps]
        print(
            f"[Research XGB h={dataset.horizon_hours:02d}] "
            f"fold {fold.fold:02d}/{len(folds):02d} candidate={best_candidate_id} "
            f"rounds={best_round} lambda={primary[0].execution_lambda:g} "
            f"hold={primary[0].hold_hours}h test_sharpe={primary[1]['sharpe']:.3f}"
        )

        for local, dataset_idx in enumerate(test_idx):
            row: dict[str, Any] = {
                "model": "xgboost",
                "horizon_hours": dataset.horizon_hours,
                "fold": fold.fold,
                "timestamp": int(dataset.timestamps[dataset_idx]),
                "actual_target_log_return": float(dataset.target_log_return[dataset_idx]),
                "actual_target_simple_return": float(dataset.target_simple_return[dataset_idx]),
                "actual_execution_return_1h": float(dataset.execution_simple_return_1h[dataset_idx]),
                "predicted_target_log_return": float(test_pred[local]),
            }
            for cost_bps, (choice, _, returns, positions, turnovers) in cost_outputs.items():
                prefix = f"cost_{cost_bps:g}bps"
                row[f"{prefix}_lambda"] = choice.execution_lambda
                row[f"{prefix}_hold_hours"] = choice.hold_hours
                row[f"{prefix}_return"] = float(returns[local])
                row[f"{prefix}_position"] = float(positions[local])
                row[f"{prefix}_turnover"] = float(turnovers[local])
            prediction_rows.append(row)

    prediction_rows.sort(key=lambda row: int(row["timestamp"]))
    forecast = regression_metrics(
        np.asarray([row["actual_target_log_return"] for row in prediction_rows], dtype=np.float64),
        np.asarray([row["predicted_target_log_return"] for row in prediction_rows], dtype=np.float64),
    )
    strategies: dict[str, Any] = {}
    for cost_bps in args.cost_bps:
        prefix = f"cost_{cost_bps:g}bps"
        strategies[f"{cost_bps:g}"] = aggregate_fold_strategies(
            prediction_rows,
            cost_key=prefix,
            fold_metrics=per_cost_fold_metrics[cost_bps],
        )

    primary_choices = policy_choices[args.primary_cost_bps]
    no_cost_rows: list[dict[str, Any]] = []
    no_cost_folds: list[dict[str, Any]] = []
    by_fold: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in prediction_rows:
        by_fold[int(row["fold"])].append(row)
    for fold_id, choice in zip(sorted(by_fold), primary_choices, strict=True):
        rows = sorted(by_fold[fold_id], key=lambda row: int(row["timestamp"]))
        pred = np.asarray([row["predicted_target_log_return"] for row in rows], dtype=np.float64)
        actual = np.asarray([row["actual_execution_return_1h"] for row in rows], dtype=np.float64)
        metrics, returns, positions, turnovers = held_cost_aware_backtest(
            pred,
            actual,
            cost_rate=0.0,
            execution_lambda=0.0,
            hold_hours=choice.hold_hours,
        )
        no_cost_folds.append(metrics)
        for row, ret, pos, turn in zip(rows, returns, positions, turnovers, strict=True):
            row["no_cost_return"] = float(ret)
            row["no_cost_position"] = float(pos)
            row["no_cost_turnover"] = float(turn)
            no_cost_rows.append(row)
    no_cost = aggregate_fold_strategies(
        no_cost_rows,
        cost_key="no_cost",
        fold_metrics=no_cost_folds,
    )

    final_candidate_id = _mode_int(selected_candidate_ids)
    final_params = dict(candidates[final_candidate_id])
    final_params["seed"] = args.seed
    if args.xgb_device == "cuda":
        final_params["device"] = "cuda"
    selected_rounds = [
        rounds
        for cid, rounds in zip(selected_candidate_ids, best_rounds, strict=True)
        if cid == final_candidate_id
    ] or best_rounds
    final_rounds = max(1, int(round(statistics.median(selected_rounds))))

    deployment = out / "deployment"
    deployment.mkdir(parents=True, exist_ok=True)
    dall = xgb.DMatrix(
        dataset.X,
        label=dataset.target_log_return,
        feature_names=dataset.feature_names,
    )
    final_model = xgb.train(final_params, dall, num_boost_round=final_rounds, verbose_eval=False)
    final_model.save_model(deployment / "model.json")
    write_json(
        deployment / "manifest.json",
        {
            "model_family": "xgboost",
            "horizon_hours": dataset.horizon_hours,
            "feature_version": FEATURE_VERSION,
            "feature_names": dataset.feature_names,
            "final_training_rows": dataset.size,
            "selected_candidate_id": final_candidate_id,
            "final_num_boost_round": final_rounds,
            "xgboost_params": final_params,
            "recommended_policy_by_cost_bps": _policy_manifest(policy_choices),
            "policy_note": "Policy selected on policy-validation data only; test windows remain untouched.",
            "execution_note": "Prediction target is multi-hour; realized strategy P&L is next-hour to avoid overlapping-return double counting.",
        },
    )

    write_jsonl(out / "predictions.jsonl", prediction_rows)
    write_csv(out / "folds.csv", fold_rows)
    write_csv(out / "model_selection.csv", selection_rows)
    write_csv(out / "policy_search.csv", policy_rows)

    return {
        "model_family": "xgboost",
        "horizon_hours": dataset.horizon_hours,
        "forecast_metrics": forecast,
        "strategies_by_cost_bps": strategies,
        "no_cost_diagnostic": no_cost,
        "recommended_policy_by_cost_bps": _policy_manifest(policy_choices),
        "selected_candidate_ids": selected_candidate_ids,
        "final_candidate_id": final_candidate_id,
        "final_model": str(deployment / "model.json"),
    }


def _scaler_payload(
    feature_names: list[str],
    sequence_length: int,
    x_stats: Any,
    y_stats: Any,
) -> dict[str, Any]:
    return {
        "feature_names": feature_names,
        "sequence_length": sequence_length,
        "feature_mean": x_stats.mean,
        "feature_scale": x_stats.scale,
        "target_mean": y_stats.mean,
        "target_scale": y_stats.scale,
    }


def train_lstm_horizon(
    dataset: ResearchDataset,
    folds: list[Any],
    out: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    import tensorflow as tf
    from backend.ml.lstm_core import _build_model

    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

    candidates = lstm_candidate_pool(args.lstm_trials, args.seed + dataset.horizon_hours * 211)
    model_dir = out / "fold_models"
    scaler_dir = out / "fold_scalers"
    model_dir.mkdir(parents=True, exist_ok=True)
    scaler_dir.mkdir(parents=True, exist_ok=True)

    prediction_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    policy_rows: list[dict[str, Any]] = []
    policy_choices: dict[float, list[PolicyChoice]] = defaultdict(list)
    selected_candidate_ids: list[int] = []
    selected_best_epochs: list[int] = []
    per_cost_fold_metrics: dict[float, list[dict[str, Any]]] = defaultdict(list)

    for fold in folds:
        model_idx_raw, policy_idx_raw = split_model_policy_validation(
            fold.validation_indices, args.model_validation_fraction
        )
        model_start = int(dataset.timestamps[model_idx_raw[0]])
        policy_start = int(dataset.timestamps[policy_idx_raw[0]])
        test_start = int(dataset.timestamps[fold.test_indices[0]])
        train_idx = purge_label_boundary(
            fold.train_indices,
            dataset.timestamps,
            horizon_hours=dataset.horizon_hours,
            boundary_timestamp=model_start,
        )
        model_idx = purge_label_boundary(
            model_idx_raw,
            dataset.timestamps,
            horizon_hours=dataset.horizon_hours,
            boundary_timestamp=policy_start,
        )
        policy_idx = purge_execution_boundary(
            policy_idx_raw,
            dataset.timestamps,
            boundary_timestamp=test_start,
        )
        test_idx = fold.test_indices
        if min(len(train_idx), len(model_idx), len(policy_idx), len(test_idx)) == 0:
            raise ValueError(f"empty purged split in LSTM fold {fold.fold}")

        x_stats = fit_standardizer(dataset.X[train_idx])
        y_stats = fit_standardizer(dataset.target_log_return[train_idx])
        x_scaled = standardize(dataset.X, x_stats)
        y_scaled = standardize(dataset.target_log_return, y_stats)

        best_candidate_id = -1
        best_val_loss = math.inf
        best_epoch = 0
        best_weights: list[np.ndarray] | None = None

        for candidate_id, config in enumerate(candidates):
            seq_len = int(config["sequence_length"])
            train_x, train_y, _ = make_sequence_batch(
                x_scaled,
                y_scaled,
                dataset.timestamps,
                train_idx,
                sequence_length=seq_len,
                min_context_index=int(train_idx[0]),
            )
            model_x, model_y, _ = make_sequence_batch(
                x_scaled,
                y_scaled,
                dataset.timestamps,
                model_idx,
                sequence_length=seq_len,
            )
            if min(len(train_x), len(model_x)) == 0:
                selection_rows.append(
                    {
                        "model": "lstm",
                        "horizon_hours": dataset.horizon_hours,
                        "fold": fold.fold,
                        "candidate_id": candidate_id,
                        "status": "skipped_empty_sequence",
                        "config_json": json.dumps(config, sort_keys=True),
                    }
                )
                continue

            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(args.seed + fold.fold + candidate_id * 1000)
            model = _build_model(
                tf,
                sequence_length=seq_len,
                feature_count=len(dataset.feature_names),
                lstm_units=int(config["lstm_units"]),
                dense_units=int(config["dense_units"]),
                dropout=float(config["dropout"]),
                learning_rate=float(config["learning_rate"]),
                clipnorm=float(config["clipnorm"]),
            )
            history = model.fit(
                train_x,
                train_y,
                validation_data=(model_x, model_y),
                epochs=args.lstm_epochs,
                batch_size=int(config["batch_size"]),
                shuffle=False,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor="val_loss",
                        mode="min",
                        patience=args.lstm_early_stopping_patience,
                        restore_best_weights=True,
                        verbose=0,
                    )
                ],
                verbose=0,
            )
            val_losses = [float(x) for x in history.history.get("val_loss", [])]
            candidate_best_epoch = int(np.argmin(val_losses) + 1) if val_losses else len(history.epoch)
            candidate_val_loss = min(val_losses) if val_losses else math.inf
            selection_rows.append(
                {
                    "model": "lstm",
                    "horizon_hours": dataset.horizon_hours,
                    "fold": fold.fold,
                    "candidate_id": candidate_id,
                    "status": "ok",
                    "best_epoch": candidate_best_epoch,
                    "model_validation_scaled_mse": candidate_val_loss,
                    "config_json": json.dumps(config, sort_keys=True),
                }
            )
            if candidate_val_loss < best_val_loss:
                best_val_loss = candidate_val_loss
                best_candidate_id = candidate_id
                best_epoch = candidate_best_epoch
                best_weights = [np.array(weight, copy=True) for weight in model.get_weights()]

        if best_candidate_id < 0 or best_weights is None:
            raise ValueError(f"no valid LSTM candidate in fold {fold.fold}")
        selected_candidate_ids.append(best_candidate_id)
        selected_best_epochs.append(best_epoch)
        best_config = candidates[best_candidate_id]
        seq_len = int(best_config["sequence_length"])

        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(args.seed + fold.fold)
        model = _build_model(
            tf,
            sequence_length=seq_len,
            feature_count=len(dataset.feature_names),
            lstm_units=int(best_config["lstm_units"]),
            dense_units=int(best_config["dense_units"]),
            dropout=float(best_config["dropout"]),
            learning_rate=float(best_config["learning_rate"]),
            clipnorm=float(best_config["clipnorm"]),
        )
        model.set_weights(best_weights)
        model.save(model_dir / f"fold_{fold.fold:02d}.keras")
        write_json(
            scaler_dir / f"fold_{fold.fold:02d}.json",
            _scaler_payload(dataset.feature_names, seq_len, x_stats, y_stats),
        )

        policy_x, _, policy_targets = make_sequence_batch(
            x_scaled,
            y_scaled,
            dataset.timestamps,
            policy_idx,
            sequence_length=seq_len,
        )
        test_x, _, test_targets = make_sequence_batch(
            x_scaled,
            y_scaled,
            dataset.timestamps,
            test_idx,
            sequence_length=seq_len,
        )
        if min(len(policy_x), len(test_x)) == 0:
            raise ValueError(f"empty selected LSTM policy/test sequence in fold {fold.fold}")

        policy_scaled = model.predict(
            policy_x, batch_size=int(best_config["batch_size"]), verbose=0
        ).reshape(-1)
        policy_pred = inverse_standardize(policy_scaled, y_stats).astype(np.float64)
        test_scaled = model.predict(
            test_x, batch_size=int(best_config["batch_size"]), verbose=0
        ).reshape(-1)
        test_pred = inverse_standardize(test_scaled, y_stats).astype(np.float64)
        forecast = regression_metrics(dataset.target_log_return[test_targets], test_pred)

        fold_row: dict[str, Any] = {
            "model": "lstm",
            "horizon_hours": dataset.horizon_hours,
            "fold": fold.fold,
            "selected_candidate_id": best_candidate_id,
            "best_epoch": best_epoch,
            "model_validation_scaled_mse": best_val_loss,
            "sequence_length": seq_len,
            "train_rows_after_purge": len(train_idx),
            "model_validation_rows_after_purge": len(model_idx),
            "policy_validation_rows_after_purge": len(policy_idx),
            **{f"forecast_{k}": v for k, v in forecast.items()},
        }

        cost_outputs: dict[float, tuple[PolicyChoice, dict[str, float], np.ndarray, np.ndarray, np.ndarray]] = {}
        for cost_bps in args.cost_bps:
            cost_rate = cost_bps / 10_000.0
            choice, sweep = choose_execution_policy(
                policy_pred,
                dataset.execution_simple_return_1h[policy_targets],
                cost_rate=cost_rate,
                lambda_grid=args.threshold_grid,
                holding_grid=args.holding_grid,
                min_trades=args.min_policy_trades,
                max_drawdown_abs=args.max_validation_drawdown,
            )
            policy_choices[cost_bps].append(choice)
            policy_rows.extend(
                {
                    "model": "lstm",
                    "horizon_hours": dataset.horizon_hours,
                    "fold": fold.fold,
                    "cost_bps": cost_bps,
                    **row,
                }
                for row in sweep
            )
            metrics, returns, positions, turnovers = held_cost_aware_backtest(
                test_pred,
                dataset.execution_simple_return_1h[test_targets],
                cost_rate=cost_rate,
                execution_lambda=choice.execution_lambda,
                hold_hours=choice.hold_hours,
            )
            per_cost_fold_metrics[cost_bps].append(metrics)
            cost_outputs[cost_bps] = (choice, metrics, returns, positions, turnovers)
            prefix = f"cost_{cost_bps:g}bps"
            fold_row[f"{prefix}_lambda"] = choice.execution_lambda
            fold_row[f"{prefix}_hold_hours"] = choice.hold_hours
            for key, value in metrics.items():
                fold_row[f"{prefix}_{key}"] = value

        fold_rows.append(fold_row)
        primary = cost_outputs[args.primary_cost_bps]
        print(
            f"[Research LSTM h={dataset.horizon_hours:02d}] "
            f"fold {fold.fold:02d}/{len(folds):02d} candidate={best_candidate_id} "
            f"epoch={best_epoch} seq={seq_len} lambda={primary[0].execution_lambda:g} "
            f"hold={primary[0].hold_hours}h test_sharpe={primary[1]['sharpe']:.3f}"
        )

        for local, dataset_idx in enumerate(test_targets):
            row: dict[str, Any] = {
                "model": "lstm",
                "horizon_hours": dataset.horizon_hours,
                "fold": fold.fold,
                "timestamp": int(dataset.timestamps[dataset_idx]),
                "actual_target_log_return": float(dataset.target_log_return[dataset_idx]),
                "actual_target_simple_return": float(dataset.target_simple_return[dataset_idx]),
                "actual_execution_return_1h": float(dataset.execution_simple_return_1h[dataset_idx]),
                "predicted_target_log_return": float(test_pred[local]),
            }
            for cost_bps, (choice, _, returns, positions, turnovers) in cost_outputs.items():
                prefix = f"cost_{cost_bps:g}bps"
                row[f"{prefix}_lambda"] = choice.execution_lambda
                row[f"{prefix}_hold_hours"] = choice.hold_hours
                row[f"{prefix}_return"] = float(returns[local])
                row[f"{prefix}_position"] = float(positions[local])
                row[f"{prefix}_turnover"] = float(turnovers[local])
            prediction_rows.append(row)

    prediction_rows.sort(key=lambda row: int(row["timestamp"]))
    forecast = regression_metrics(
        np.asarray([row["actual_target_log_return"] for row in prediction_rows], dtype=np.float64),
        np.asarray([row["predicted_target_log_return"] for row in prediction_rows], dtype=np.float64),
    )
    strategies: dict[str, Any] = {}
    for cost_bps in args.cost_bps:
        prefix = f"cost_{cost_bps:g}bps"
        strategies[f"{cost_bps:g}"] = aggregate_fold_strategies(
            prediction_rows,
            cost_key=prefix,
            fold_metrics=per_cost_fold_metrics[cost_bps],
        )

    primary_choices = policy_choices[args.primary_cost_bps]
    no_cost_rows: list[dict[str, Any]] = []
    no_cost_folds: list[dict[str, Any]] = []
    by_fold: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in prediction_rows:
        by_fold[int(row["fold"])].append(row)
    for fold_id, choice in zip(sorted(by_fold), primary_choices, strict=True):
        rows = sorted(by_fold[fold_id], key=lambda row: int(row["timestamp"]))
        pred = np.asarray([row["predicted_target_log_return"] for row in rows], dtype=np.float64)
        actual = np.asarray([row["actual_execution_return_1h"] for row in rows], dtype=np.float64)
        metrics, returns, positions, turnovers = held_cost_aware_backtest(
            pred,
            actual,
            cost_rate=0.0,
            execution_lambda=0.0,
            hold_hours=choice.hold_hours,
        )
        no_cost_folds.append(metrics)
        for row, ret, pos, turn in zip(rows, returns, positions, turnovers, strict=True):
            row["no_cost_return"] = float(ret)
            row["no_cost_position"] = float(pos)
            row["no_cost_turnover"] = float(turn)
            no_cost_rows.append(row)
    no_cost = aggregate_fold_strategies(
        no_cost_rows,
        cost_key="no_cost",
        fold_metrics=no_cost_folds,
    )

    final_candidate_id = _mode_int(selected_candidate_ids)
    final_config = candidates[final_candidate_id]
    final_epochs_values = [
        epoch
        for cid, epoch in zip(selected_candidate_ids, selected_best_epochs, strict=True)
        if cid == final_candidate_id
    ] or selected_best_epochs
    final_epochs = max(1, int(round(statistics.median(final_epochs_values))))
    seq_len = int(final_config["sequence_length"])

    deployment = out / "deployment"
    deployment.mkdir(parents=True, exist_ok=True)
    x_stats = fit_standardizer(dataset.X)
    y_stats = fit_standardizer(dataset.target_log_return)
    x_scaled = standardize(dataset.X, x_stats)
    y_scaled = standardize(dataset.target_log_return, y_stats)
    all_idx = np.arange(dataset.size, dtype=np.int64)
    all_x, all_y, _ = make_sequence_batch(
        x_scaled,
        y_scaled,
        dataset.timestamps,
        all_idx,
        sequence_length=seq_len,
        min_context_index=0,
    )
    if len(all_x) == 0:
        raise ValueError("no final LSTM sequences")
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(args.seed)
    final_model = _build_model(
        tf,
        sequence_length=seq_len,
        feature_count=len(dataset.feature_names),
        lstm_units=int(final_config["lstm_units"]),
        dense_units=int(final_config["dense_units"]),
        dropout=float(final_config["dropout"]),
        learning_rate=float(final_config["learning_rate"]),
        clipnorm=float(final_config["clipnorm"]),
    )
    final_model.fit(
        all_x,
        all_y,
        epochs=final_epochs,
        batch_size=int(final_config["batch_size"]),
        shuffle=False,
        verbose=0,
    )
    final_model.save(deployment / "model.keras")
    write_json(
        deployment / "scaler.json",
        _scaler_payload(dataset.feature_names, seq_len, x_stats, y_stats),
    )
    write_json(
        deployment / "manifest.json",
        {
            "model_family": "lstm",
            "horizon_hours": dataset.horizon_hours,
            "feature_version": FEATURE_VERSION,
            "feature_names": dataset.feature_names,
            "final_training_rows": dataset.size,
            "selected_candidate_id": final_candidate_id,
            "final_config": final_config,
            "final_epochs": final_epochs,
            "recommended_policy_by_cost_bps": _policy_manifest(policy_choices),
            "tensorflow_version": tf.__version__,
            "tensorflow_gpu_count": len(tf.config.list_physical_devices("GPU")),
            "policy_note": "Policy selected on policy-validation data only; test windows remain untouched.",
            "execution_note": "Prediction target is multi-hour; realized strategy P&L is next-hour to avoid overlapping-return double counting.",
        },
    )

    write_jsonl(out / "predictions.jsonl", prediction_rows)
    write_csv(out / "folds.csv", fold_rows)
    write_csv(out / "model_selection.csv", selection_rows)
    write_csv(out / "policy_search.csv", policy_rows)

    return {
        "model_family": "lstm",
        "horizon_hours": dataset.horizon_hours,
        "forecast_metrics": forecast,
        "strategies_by_cost_bps": strategies,
        "no_cost_diagnostic": no_cost,
        "recommended_policy_by_cost_bps": _policy_manifest(policy_choices),
        "selected_candidate_ids": selected_candidate_ids,
        "final_candidate_id": final_candidate_id,
        "final_model": str(deployment / "model.keras"),
    }


def moving_average_test_baseline(
    dataset: ResearchDataset,
    folds: list[Any],
    *,
    cost_rate: float,
) -> dict[str, float]:
    fold_metrics: list[dict[str, Any]] = []
    combined_rows: list[dict[str, Any]] = []
    for fold in folds:
        idx = fold.test_indices
        desired = (
            (dataset.ema20[idx] > dataset.ema50[idx])
            & (dataset.closes[idx] > dataset.ema200[idx])
        ).astype(np.float64)
        previous = np.concatenate(([0.0], desired[:-1]))
        turnover = np.abs(desired - previous)
        returns = desired * dataset.execution_simple_return_1h[idx] - cost_rate * turnover
        metrics = _performance_metrics(returns, desired, turnover)
        fold_metrics.append(metrics)
        for local, dataset_idx in enumerate(idx):
            combined_rows.append(
                {
                    "timestamp": int(dataset.timestamps[dataset_idx]),
                    "ma_return": float(returns[local]),
                    "ma_position": float(desired[local]),
                    "ma_turnover": float(turnover[local]),
                }
            )
    return aggregate_fold_strategies(combined_rows, cost_key="ma", fold_metrics=fold_metrics)


def buy_hold_test_baseline(
    dataset: ResearchDataset,
    folds: list[Any],
    *,
    cost_rate: float,
) -> dict[str, float]:
    fold_metrics: list[dict[str, Any]] = []
    combined_rows: list[dict[str, Any]] = []
    for fold in folds:
        idx = fold.test_indices
        positions = np.ones(len(idx), dtype=np.float64)
        turnover = np.zeros(len(idx), dtype=np.float64)
        if len(idx):
            turnover[0] = 1.0
        returns = dataset.execution_simple_return_1h[idx].astype(np.float64).copy()
        if len(idx):
            returns[0] -= cost_rate
        metrics = _performance_metrics(returns, positions, turnover)
        fold_metrics.append(metrics)
        for local, dataset_idx in enumerate(idx):
            combined_rows.append(
                {
                    "timestamp": int(dataset.timestamps[dataset_idx]),
                    "bh_return": float(returns[local]),
                    "bh_position": float(positions[local]),
                    "bh_turnover": float(turnover[local]),
                }
            )
    return aggregate_fold_strategies(combined_rows, cost_key="bh", fold_metrics=fold_metrics)


def comparison_row(result: dict[str, Any], primary_cost_bps: float) -> dict[str, Any]:
    forecast = result["forecast_metrics"]
    strategy = result["strategies_by_cost_bps"][f"{primary_cost_bps:g}"]
    policy = result["recommended_policy_by_cost_bps"][f"{primary_cost_bps:g}"]
    no_cost = result["no_cost_diagnostic"]
    return {
        "model": result["model_family"],
        "horizon_hours": result["horizon_hours"],
        "direction_accuracy": forecast["direction_accuracy"],
        "correlation": forecast["correlation"],
        "rmse_log_return": forecast["rmse_log_return"],
        "mae_log_return": forecast["mae_log_return"],
        "primary_cost_bps": primary_cost_bps,
        "test_sharpe": strategy["sharpe"],
        "test_sortino": strategy["sortino"],
        "test_max_drawdown": strategy["max_drawdown"],
        "test_cumulative_return": strategy["cumulative_return"],
        "test_trade_count": strategy["trade_count"],
        "test_turnover": strategy["turnover"],
        "no_cost_sharpe": no_cost["sharpe"],
        "no_cost_cumulative_return": no_cost["cumulative_return"],
        "recommended_lambda": policy["execution_lambda"],
        "recommended_hold_hours": policy["hold_hours"],
        "final_model": result["final_model"],
    }


def markdown_report(
    meta: dict[str, Any],
    comparison: list[dict[str, Any]],
    baselines: dict[str, Any],
) -> str:
    lines = [
        "# Multi-Horizon AI Trading Research Report",
        "",
        f"Run: `{meta['run_id']}`  ",
        f"Git: `{meta['git_commit']}`  ",
        f"Dataset SHA256: `{meta['dataset_sha256']}`  ",
        f"Horizons: **{', '.join(str(x) + 'h' for x in meta['horizons'])}**  ",
        f"Primary transaction cost: **{meta['primary_cost_bps']:.2f} bps per position change**  ",
        f"Cost scenarios: **{', '.join(str(x) for x in meta['cost_bps'])} bps**  ",
        "",
        "## Methodology",
        "`TRAIN -> MODEL VALIDATION -> POLICY VALIDATION -> UNTOUCHED TEST`",
        "",
        "- Multi-hour labels are purged at split boundaries.",
        "- Hyperparameters use model-validation data only.",
        "- Lambda and minimum holding period use separate policy-validation data only.",
        "- Test windows are never used for model or policy selection.",
        "- Multi-hour forecasts are executed against next-hour realized P&L, preventing overlapping-target returns from being double counted.",
        "- The default threshold grid excludes lambda < 1.",
        "",
        "## Primary-cost results",
        "",
        "| Model | Horizon | Direction | Corr | Sharpe | No-cost Sharpe | Max DD | Cum Return | Trades | Lambda | Hold |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison:
        lines.append(
            f"| {row['model']} | {int(row['horizon_hours'])}h | {row['direction_accuracy']:.4f} | "
            f"{row['correlation']:.4f} | {row['test_sharpe']:.3f} | {row['no_cost_sharpe']:.3f} | "
            f"{row['test_max_drawdown']:.2%} | {row['test_cumulative_return']:.2%} | "
            f"{int(row['test_trade_count'])} | {row['recommended_lambda']:.3g} | "
            f"{int(row['recommended_hold_hours'])}h |"
        )
    lines += ["", "## Baselines at primary cost"]
    for horizon, values in sorted(baselines.items(), key=lambda item: int(item[0])):
        lines.append(
            f"- {horizon}h test windows: moving-average Sharpe **{values['moving_average']['sharpe']:.3f}**, "
            f"buy-and-hold Sharpe **{values['buy_and_hold']['sharpe']:.3f}**"
        )
    lines += [
        "",
        "## Interpretation rule",
        "A candidate is not deployment-ready merely because it ranks first. Positive performance should be stable across folds, costs, and market regimes, with acceptable drawdown and turnover. The no-cost diagnostic helps distinguish weak prediction from transaction-cost/turnover problems.",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-horizon BTC ML research experiments.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--models", nargs="+", choices=["xgboost", "lstm"], default=["xgboost", "lstm"])
    parser.add_argument("--horizons", type=parse_int_grid, default=parse_int_grid("1,3,6,12"))
    parser.add_argument("--train-days", type=int, default=365)
    parser.add_argument("--validation-days", type=int, default=60)
    parser.add_argument("--test-days", type=int, default=60)
    parser.add_argument("--step-days", type=int, default=60)
    parser.add_argument("--model-validation-fraction", type=float, default=0.5)
    parser.add_argument("--cost-bps", type=parse_float_grid, default=parse_float_grid("10,15,20,25"))
    parser.add_argument("--primary-cost-bps", type=float, default=25.0)
    parser.add_argument("--threshold-grid", type=parse_float_grid, default=parse_float_grid("1,1.25,1.5,2,2.5,3,4,5,6,8,10"))
    parser.add_argument("--holding-grid", type=parse_int_grid, default=parse_int_grid("1,3,6,12"))
    parser.add_argument("--min-policy-trades", type=int, default=5)
    parser.add_argument("--max-validation-drawdown", type=float, default=0.35)
    parser.add_argument("--xgb-trials", type=int, default=8)
    parser.add_argument("--xgb-rounds", type=int, default=1500)
    parser.add_argument("--xgb-early-stopping", type=int, default=40)
    parser.add_argument("--xgb-device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--lstm-trials", type=int, default=4)
    parser.add_argument("--lstm-epochs", type=int, default=35)
    parser.add_argument("--lstm-early-stopping-patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.dataset.is_file():
        raise FileNotFoundError(args.dataset)
    if not 0.1 <= args.model_validation_fraction <= 0.9:
        raise ValueError("model_validation_fraction must be between 0.1 and 0.9")
    if args.primary_cost_bps not in args.cost_bps:
        raise ValueError("primary_cost_bps must be included in cost_bps")
    if min(args.threshold_grid) < 1.0:
        raise ValueError("research threshold grid must keep execution_lambda >= 1")
    if args.min_policy_trades < 0:
        raise ValueError("min_policy_trades must be non-negative")
    if not 0.0 < args.max_validation_drawdown <= 1.0:
        raise ValueError("max_validation_drawdown must be in (0, 1]")
    if args.xgb_trials <= 0 or args.lstm_trials <= 0:
        raise ValueError("hyperparameter trial counts must be positive")


def main() -> int:
    args = parse_args()
    validate_args(args)
    started = time.time()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    raw_rows = read_jsonl(args.dataset)
    walk_forward = WalkForwardConfig(
        train_days=args.train_days,
        validation_days=args.validation_days,
        test_days=args.test_days,
        step_days=args.step_days,
    )
    meta: dict[str, Any] = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_head(),
        "dataset_path": str(args.dataset),
        "dataset_sha256": sha256_file(args.dataset),
        "raw_jsonl_rows": len(raw_rows),
        "feature_version": FEATURE_VERSION,
        "horizons": args.horizons,
        "models": args.models,
        "walk_forward": asdict(walk_forward),
        "model_validation_fraction": args.model_validation_fraction,
        "cost_bps": args.cost_bps,
        "primary_cost_bps": args.primary_cost_bps,
        "threshold_grid": args.threshold_grid,
        "holding_grid": args.holding_grid,
        "min_policy_trades": args.min_policy_trades,
        "max_validation_drawdown": args.max_validation_drawdown,
        "xgb_trials": args.xgb_trials,
        "lstm_trials": args.lstm_trials,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    write_json(run_dir / "metadata.json", meta)

    all_results: dict[str, Any] = {}
    comparison: list[dict[str, Any]] = []
    baselines: dict[str, Any] = {}

    for horizon in args.horizons:
        dataset = build_research_dataset(raw_rows, horizon_hours=horizon)
        folds = make_walk_forward_folds(dataset.timestamps, walk_forward)
        if not folds:
            raise ValueError(f"no walk-forward folds for horizon={horizon}h")
        print(
            f"=== HORIZON {horizon}h: usable_rows={dataset.size:,} folds={len(folds)} ==="
        )
        horizon_dir = run_dir / f"h{horizon:02d}"
        horizon_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            horizon_dir / "dataset_info.json",
            {
                "horizon_hours": horizon,
                "usable_rows": dataset.size,
                "feature_names": dataset.feature_names,
                "fold_count": len(folds),
            },
        )
        primary_cost_rate = args.primary_cost_bps / 10_000.0
        baselines[str(horizon)] = {
            "moving_average": moving_average_test_baseline(
                dataset, folds, cost_rate=primary_cost_rate
            ),
            "buy_and_hold": buy_hold_test_baseline(
                dataset, folds, cost_rate=primary_cost_rate
            ),
        }

        for model_name in args.models:
            out = horizon_dir / model_name
            out.mkdir(parents=True, exist_ok=True)
            if model_name == "xgboost":
                result = train_xgboost_horizon(dataset, folds, out, args)
            else:
                result = train_lstm_horizon(dataset, folds, out, args)
            key = f"{model_name}_h{horizon}"
            all_results[key] = result
            write_json(out / "summary.json", result)
            comparison.append(comparison_row(result, args.primary_cost_bps))

    comparison.sort(
        key=lambda row: (
            float(row["test_sharpe"]),
            float(row["no_cost_sharpe"]),
            float(row["direction_accuracy"]),
            float(row["correlation"]),
        ),
        reverse=True,
    )
    write_json(run_dir / "all_results.json", all_results)
    write_json(run_dir / "baselines.json", baselines)
    write_csv(run_dir / "model_comparison.csv", comparison)
    (run_dir / "REPORT.md").write_text(
        markdown_report(meta, comparison, baselines), encoding="utf-8"
    )

    meta["duration_seconds"] = time.time() - started
    meta["candidate_count"] = len(comparison)
    write_json(run_dir / "metadata.json", meta)

    research_zip = args.output_root / f"{run_id}_research_report.zip"
    with zipfile.ZipFile(research_zip, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in run_dir.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(run_dir.parent))

    deployment_zip = args.output_root / f"{run_id}_research_deployment.zip"
    with zipfile.ZipFile(deployment_zip, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in ("REPORT.md", "model_comparison.csv", "metadata.json", "baselines.json", "all_results.json"):
            path = run_dir / name
            if path.is_file():
                archive.write(path, path.relative_to(run_dir))
        for deployment in run_dir.glob("h*/**/deployment"):
            if deployment.is_dir():
                for path in deployment.rglob("*"):
                    if path.is_file():
                        archive.write(path, path.relative_to(run_dir))

    print("RESEARCH COMPLETE")
    print(f"report={run_dir / 'REPORT.md'}")
    print(f"research_bundle={research_zip}")
    print(f"deployment_bundle={deployment_zip}")
    if comparison:
        best = comparison[0]
        print(
            "highest historical WFO primary-cost candidate (not proof of future profitability): "
            f"{best['model']} horizon={int(best['horizon_hours'])}h "
            f"Sharpe={best['test_sharpe']:.3f} no_cost={best['no_cost_sharpe']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
