#!/usr/bin/env python3
"""Multi-horizon research runner with LSTM V2 dual-head training.

This is the V2-compatible companion to ``run_research_experiments.py``.
It reuses the time-series-safe dataset, fold, XGBoost, and execution helpers
from that module, while replacing the LSTM path with the stacked Huber +
direction-probability architecture from ``backend.ml.lstm_core``.

Protocol per outer fold:
    TRAIN -> MODEL VALIDATION -> POLICY VALIDATION -> UNTOUCHED TEST

The model-validation block selects network hyperparameters. The separate
policy-validation block selects trading rules. The outer test block is never
used for either decision.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import random
import statistics
import time
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from backend.ml.evaluation import classification_metrics, regression_metrics
from backend.ml.features import FEATURE_VERSION, read_jsonl
from backend.ml.sequences import (
    fit_standardizer,
    inverse_standardize,
    make_sequence_batch,
    standardize,
)
from run_research_experiments import (
    DEFAULT_DATASET,
    DEFAULT_OUTPUT,
    PolicyChoice,
    ResearchDataset,
    aggregate_fold_strategies,
    build_research_dataset,
    buy_hold_test_baseline,
    choose_execution_policy,
    comparison_row as xgb_comparison_row,
    git_head,
    make_walk_forward_folds,
    moving_average_test_baseline,
    parse_float_grid,
    parse_int_grid,
    purge_execution_boundary,
    purge_label_boundary,
    sha256_file,
    split_model_policy_validation,
    train_xgboost_horizon,
    write_csv,
    write_json,
    write_jsonl,
    WalkForwardConfig,
    held_cost_aware_backtest,
)


@dataclass(frozen=True)
class ProbabilityPolicyChoice:
    entry_threshold: float
    exit_threshold: float
    hold_hours: int


def _mode(values: list[int]) -> int:
    if not values:
        raise ValueError("cannot take mode of empty list")
    counts = Counter(values)
    return min(counts, key=lambda value: (-counts[value], value))


def _fmt_ts(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000.0, timezone.utc).strftime("%Y-%m-%d")


def _find_log(logs: dict[str, Any], *fragments: str) -> float | None:
    for key, value in logs.items():
        if all(fragment in key for fragment in fragments):
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


def _progress_callback(
    tf: Any,
    *,
    horizon: int,
    fold: int,
    fold_count: int,
    candidate: int,
    candidate_count: int,
    epochs: int,
    patience: int,
    target_scale: float,
    enabled: bool,
) -> Any:
    class Progress(tf.keras.callbacks.Callback):
        def __init__(self) -> None:
            super().__init__()
            self.best = float("inf")
            self.best_epoch = 0

        def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
            if not enabled:
                return
            logs = logs or {}
            val_loss = _find_log(logs, "val_loss")
            if val_loss is not None and val_loss < self.best:
                self.best = val_loss
                self.best_epoch = epoch + 1
            mae_scaled = _find_log(logs, "val_", "log_return", "mae")
            direction_acc = _find_log(logs, "val_", "direction", "acc")
            mae_bps = mae_scaled * target_scale * 10_000.0 if mae_scaled is not None else float("nan")
            since_best = epoch + 1 - self.best_epoch
            remaining = max(patience - since_best, 0)
            marker = "*" if since_best == 0 else f"early stop in {remaining}"
            print(
                f"[LSTM-V2 h={horizon:02d} fold {fold:02d}/{fold_count:02d} cand {candidate + 1}/{candidate_count}] "
                f"epoch {epoch + 1:02d}/{epochs:02d} | val error {mae_bps:6.1f} bps | val direction "
                + (f"{direction_acc:5.1%}" if direction_acc is not None else " n/a ")
                + f" | {marker}",
                flush=True,
            )

    return Progress()


def held_probability_backtest(
    probability: np.ndarray,
    actual_simple_return_1h: np.ndarray,
    *,
    cost_rate: float,
    entry_threshold: float,
    exit_threshold: float,
    hold_hours: int,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    from backend.ml.evaluation import _performance_metrics

    prob = np.asarray(probability, dtype=np.float64)
    actual = np.asarray(actual_simple_return_1h, dtype=np.float64)
    if prob.shape != actual.shape or prob.ndim != 1:
        raise ValueError("probability and actual return arrays must be identical 1-D shapes")
    if not 0.0 <= exit_threshold < entry_threshold <= 1.0:
        raise ValueError("probability policy requires 0 <= exit < entry <= 1")
    if cost_rate < 0.0 or hold_hours <= 0:
        raise ValueError("invalid probability execution parameters")

    positions = np.zeros(len(prob), dtype=np.float64)
    turnovers = np.zeros(len(prob), dtype=np.float64)
    returns = np.zeros(len(prob), dtype=np.float64)
    position = 0.0
    bars_since_change = hold_hours

    for idx, p in enumerate(prob):
        desired = position
        if position == 0.0 and p >= entry_threshold:
            desired = 1.0
        elif position == 1.0 and p <= exit_threshold:
            desired = 0.0

        next_position = position
        if desired != position and bars_since_change >= hold_hours:
            next_position = desired

        turnover = abs(next_position - position)
        returns[idx] = next_position * actual[idx] - cost_rate * turnover
        positions[idx] = next_position
        turnovers[idx] = turnover
        bars_since_change = 1 if turnover > 0.0 else bars_since_change + 1
        position = next_position

    return _performance_metrics(returns, positions, turnovers), returns, positions, turnovers


def choose_probability_policy(
    probability: np.ndarray,
    actual_simple_return_1h: np.ndarray,
    *,
    cost_rate: float,
    entry_grid: list[float],
    exit_grid: list[float],
    holding_grid: list[int],
    min_trades: int,
    max_drawdown_abs: float,
) -> tuple[ProbabilityPolicyChoice, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for hold in holding_grid:
        for entry in entry_grid:
            for exit_ in exit_grid:
                if exit_ >= entry:
                    continue
                metrics, _, _, _ = held_probability_backtest(
                    probability,
                    actual_simple_return_1h,
                    cost_rate=cost_rate,
                    entry_threshold=entry,
                    exit_threshold=exit_,
                    hold_hours=hold,
                )
                rows.append({"entry_threshold": entry, "exit_threshold": exit_, "hold_hours": hold, **metrics})
    if not rows:
        raise ValueError("probability policy grid produced no valid candidates")

    strong = [
        row for row in rows
        if int(row["trade_count"]) >= min_trades
        and float(row["sharpe"]) > 0.0
        and float(row["cumulative_return"]) > 0.0
        and float(row["max_drawdown"]) >= -max_drawdown_abs
    ]
    trade_ok = [row for row in rows if int(row["trade_count"]) >= min_trades]
    pool = strong or trade_ok or rows
    best = max(
        pool,
        key=lambda row: (
            float(row["sharpe"]), float(row["sortino"]), float(row["max_drawdown"]),
            float(row["cumulative_return"]), -float(row["turnover"]), -int(row["hold_hours"]),
            float(row["entry_threshold"]), -float(row["exit_threshold"]),
        ),
    )
    return ProbabilityPolicyChoice(float(best["entry_threshold"]), float(best["exit_threshold"]), int(best["hold_hours"])), rows


def lstm_v2_candidate_pool(trials: int, seed: int) -> list[dict[str, Any]]:
    if trials <= 0:
        raise ValueError("lstm_trials must be positive")
    base = {
        "sequence_length": 48, "lstm_units": 64, "lstm_layers": 2, "dense_units": 32,
        "dropout": 0.20, "learning_rate": 1e-3, "batch_size": 64, "clipnorm": 1.0,
        "huber_delta": 1.0, "direction_loss_weight": 0.30, "direction_threshold_bps": 0.0,
    }
    candidates = [base]
    rng = random.Random(seed)
    seen = {json.dumps(base, sort_keys=True)}
    while len(candidates) < trials:
        candidate = {
            "sequence_length": rng.choice([24, 48, 72, 168]),
            "lstm_units": rng.choice([32, 64, 96, 128]),
            "lstm_layers": rng.choice([1, 2]),
            "dense_units": rng.choice([16, 32, 64]),
            "dropout": rng.choice([0.10, 0.20, 0.30, 0.40]),
            "learning_rate": rng.choice([3e-4, 5e-4, 1e-3, 2e-3]),
            "batch_size": rng.choice([32, 64, 128]),
            "clipnorm": rng.choice([0.5, 1.0, 2.0]),
            "huber_delta": rng.choice([0.5, 1.0, 1.5]),
            "direction_loss_weight": rng.choice([0.20, 0.30, 0.50, 0.75]),
            "direction_threshold_bps": rng.choice([0.0, 10.0, 25.0]),
        }
        key = json.dumps(candidate, sort_keys=True)
        if key not in seen:
            seen.add(key)
            candidates.append(candidate)
    return candidates


def _probability_manifest(choices: dict[float, list[ProbabilityPolicyChoice]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for cost_bps, fold_choices in sorted(choices.items()):
        out[f"{cost_bps:g}"] = {
            "entry_threshold": float(statistics.median(c.entry_threshold for c in fold_choices)),
            "exit_threshold": float(statistics.median(c.exit_threshold for c in fold_choices)),
            "hold_hours": _mode([c.hold_hours for c in fold_choices]),
            "fold_entry_thresholds": [c.entry_threshold for c in fold_choices],
            "fold_exit_thresholds": [c.exit_threshold for c in fold_choices],
            "fold_hold_hours": [c.hold_hours for c in fold_choices],
        }
    return out


def _regression_policy_manifest(choices: dict[float, list[PolicyChoice]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for cost_bps, fold_choices in sorted(choices.items()):
        out[f"{cost_bps:g}"] = {
            "execution_lambda": float(statistics.median(c.execution_lambda for c in fold_choices)),
            "hold_hours": _mode([c.hold_hours for c in fold_choices]),
            "fold_lambdas": [c.execution_lambda for c in fold_choices],
            "fold_hold_hours": [c.hold_hours for c in fold_choices],
        }
    return out


def train_lstm_v2_horizon(dataset: ResearchDataset, folds: list[Any], out: Path, args: argparse.Namespace) -> dict[str, Any]:
    import tensorflow as tf
    from backend.ml.lstm_core import _build_model

    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

    candidates = lstm_v2_candidate_pool(args.lstm_trials, args.seed + dataset.horizon_hours * 211)
    model_dir = out / "fold_models"
    scaler_dir = out / "fold_scalers"
    model_dir.mkdir(parents=True, exist_ok=True)
    scaler_dir.mkdir(parents=True, exist_ok=True)

    prediction_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    reg_policy_rows: list[dict[str, Any]] = []
    prob_policy_rows: list[dict[str, Any]] = []
    reg_choices: dict[float, list[PolicyChoice]] = defaultdict(list)
    prob_choices: dict[float, list[ProbabilityPolicyChoice]] = defaultdict(list)
    reg_fold_metrics: dict[float, list[dict[str, Any]]] = defaultdict(list)
    prob_fold_metrics: dict[float, list[dict[str, Any]]] = defaultdict(list)
    selected_candidate_ids: list[int] = []
    selected_best_epochs: list[int] = []

    for fold in folds:
        model_idx_raw, policy_idx_raw = split_model_policy_validation(fold.validation_indices, args.model_validation_fraction)
        model_start = int(dataset.timestamps[model_idx_raw[0]])
        policy_start = int(dataset.timestamps[policy_idx_raw[0]])
        test_start = int(dataset.timestamps[fold.test_indices[0]])
        train_idx = purge_label_boundary(fold.train_indices, dataset.timestamps, horizon_hours=dataset.horizon_hours, boundary_timestamp=model_start)
        model_idx = purge_label_boundary(model_idx_raw, dataset.timestamps, horizon_hours=dataset.horizon_hours, boundary_timestamp=policy_start)
        policy_idx = purge_execution_boundary(policy_idx_raw, dataset.timestamps, boundary_timestamp=test_start)
        test_idx = fold.test_indices
        if min(len(train_idx), len(model_idx), len(policy_idx), len(test_idx)) == 0:
            raise ValueError(f"empty purged split in LSTM-V2 fold {fold.fold}")

        print(
            f"\n=== LSTM-V2 {dataset.horizon_hours}h fold {fold.fold}/{len(folds)} | "
            f"train {_fmt_ts(int(dataset.timestamps[train_idx[0]]))} to {_fmt_ts(int(dataset.timestamps[train_idx[-1]]))} | "
            f"model-val -> {_fmt_ts(int(dataset.timestamps[model_idx[-1]]))} | "
            f"policy-val -> {_fmt_ts(int(dataset.timestamps[policy_idx[-1]]))} | "
            f"test -> {_fmt_ts(int(dataset.timestamps[test_idx[-1]]))} ===",
            flush=True,
        )

        x_stats = fit_standardizer(dataset.X[train_idx])
        y_stats = fit_standardizer(dataset.target_log_return[train_idx])
        x_scaled = standardize(dataset.X, x_stats)
        y_scaled = standardize(dataset.target_log_return, y_stats)
        best_candidate_id = -1
        best_candidate_key: tuple[float, float, float, float] | None = None
        best_epoch = 0
        best_weights: list[np.ndarray] | None = None

        for candidate_id, config in enumerate(candidates):
            seq_len = int(config["sequence_length"])
            train_x, train_y, train_targets = make_sequence_batch(x_scaled, y_scaled, dataset.timestamps, train_idx, sequence_length=seq_len, min_context_index=int(train_idx[0]))
            model_x, model_y, model_targets = make_sequence_batch(x_scaled, y_scaled, dataset.timestamps, model_idx, sequence_length=seq_len)
            if min(len(train_x), len(model_x)) == 0:
                selection_rows.append({"model": "lstm_v2", "horizon_hours": dataset.horizon_hours, "fold": fold.fold, "candidate_id": candidate_id, "status": "skipped_empty_sequence", "config_json": json.dumps(config, sort_keys=True)})
                continue

            direction_threshold = float(config["direction_threshold_bps"]) / 10_000.0
            train_dir = (dataset.target_log_return[train_targets] > direction_threshold).astype(np.float32)
            model_dir_labels = (dataset.target_log_return[model_targets] > direction_threshold).astype(np.float32)
            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(args.seed + dataset.horizon_hours * 100_000 + fold.fold * 1_000 + candidate_id)
            model = _build_model(
                tf, sequence_length=seq_len, feature_count=len(dataset.feature_names),
                lstm_units=int(config["lstm_units"]), lstm_layers=int(config["lstm_layers"]),
                dense_units=int(config["dense_units"]), dropout=float(config["dropout"]),
                learning_rate=float(config["learning_rate"]), clipnorm=float(config["clipnorm"]),
                huber_delta=float(config["huber_delta"]), direction_loss_weight=float(config["direction_loss_weight"]),
            )
            history = model.fit(
                train_x,
                {"next_hour_log_return_scaled": train_y, "direction_up_prob": train_dir},
                validation_data=(model_x, {"next_hour_log_return_scaled": model_y, "direction_up_prob": model_dir_labels}),
                epochs=args.lstm_epochs, batch_size=int(config["batch_size"]), shuffle=False,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=args.lstm_early_stopping_patience, restore_best_weights=True, verbose=0),
                    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", mode="min", factor=0.5, patience=max(args.lstm_early_stopping_patience // 2, 1), min_lr=1e-5, verbose=0),
                    _progress_callback(tf, horizon=dataset.horizon_hours, fold=fold.fold, fold_count=len(folds), candidate=candidate_id, candidate_count=len(candidates), epochs=args.lstm_epochs, patience=args.lstm_early_stopping_patience, target_scale=float(np.mean(np.atleast_1d(y_stats.scale))), enabled=args.epoch_verbose),
                ], verbose=0,
            )
            outputs = model.predict(model_x, batch_size=int(config["batch_size"]), verbose=0)
            val_pred = inverse_standardize(np.asarray(outputs[0]).reshape(-1), y_stats).astype(np.float64)
            val_prob = np.clip(np.asarray(outputs[1]).reshape(-1), 0.0, 1.0)
            reg = regression_metrics(dataset.target_log_return[model_targets], val_pred)
            cls = classification_metrics(model_dir_labels.astype(np.float64), val_prob)
            val_losses = [float(x) for x in history.history.get("val_loss", [])]
            candidate_best_epoch = int(np.argmin(val_losses) + 1) if val_losses else len(history.epoch)
            candidate_val_loss = min(val_losses) if val_losses else math.inf
            selection_rows.append({"model": "lstm_v2", "horizon_hours": dataset.horizon_hours, "fold": fold.fold, "candidate_id": candidate_id, "status": "ok", "best_epoch": candidate_best_epoch, "model_validation_total_loss": candidate_val_loss, **{f"reg_{k}": v for k, v in reg.items()}, **{f"direction_{k}": v for k, v in cls.items()}, "config_json": json.dumps(config, sort_keys=True)})
            print(f"  candidate {candidate_id + 1}/{len(candidates)} done | epoch={candidate_best_epoch} | skill={reg['skill_vs_zero']:+.2%} | direction={cls['direction_accuracy_prob']:.1%} | AUC={cls['auc']:.3f} | Brier={cls['brier_score']:.3f}", flush=True)
            key = (float(cls["auc"]), -float(cls["brier_score"]), float(reg["skill_vs_zero"]), -float(candidate_val_loss))
            if best_candidate_key is None or key > best_candidate_key:
                best_candidate_key = key
                best_candidate_id = candidate_id
                best_epoch = candidate_best_epoch
                best_weights = [np.array(weight, copy=True) for weight in model.get_weights()]

        if best_candidate_id < 0 or best_weights is None:
            raise ValueError(f"no valid LSTM-V2 candidate in fold {fold.fold}")
        selected_candidate_ids.append(best_candidate_id)
        selected_best_epochs.append(best_epoch)
        best_config = candidates[best_candidate_id]
        seq_len = int(best_config["sequence_length"])
        direction_threshold = float(best_config["direction_threshold_bps"]) / 10_000.0
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(args.seed + fold.fold)
        model = _build_model(
            tf, sequence_length=seq_len, feature_count=len(dataset.feature_names),
            lstm_units=int(best_config["lstm_units"]), lstm_layers=int(best_config["lstm_layers"]),
            dense_units=int(best_config["dense_units"]), dropout=float(best_config["dropout"]),
            learning_rate=float(best_config["learning_rate"]), clipnorm=float(best_config["clipnorm"]),
            huber_delta=float(best_config["huber_delta"]), direction_loss_weight=float(best_config["direction_loss_weight"]),
        )
        model.set_weights(best_weights)
        model.save(model_dir / f"fold_{fold.fold:02d}.keras")
        write_json(scaler_dir / f"fold_{fold.fold:02d}.json", {"feature_names": dataset.feature_names, "sequence_length": seq_len, "feature_mean": x_stats.mean, "feature_scale": x_stats.scale, "target_mean": y_stats.mean, "target_scale": y_stats.scale, "lstm_layers": int(best_config["lstm_layers"]), "direction_threshold_bps": float(best_config["direction_threshold_bps"])})

        policy_x, _, policy_targets = make_sequence_batch(x_scaled, y_scaled, dataset.timestamps, policy_idx, sequence_length=seq_len)
        test_x, _, test_targets = make_sequence_batch(x_scaled, y_scaled, dataset.timestamps, test_idx, sequence_length=seq_len)
        if min(len(policy_x), len(test_x)) == 0:
            raise ValueError(f"empty selected LSTM-V2 policy/test block in fold {fold.fold}")
        policy_outputs = model.predict(policy_x, batch_size=int(best_config["batch_size"]), verbose=0)
        test_outputs = model.predict(test_x, batch_size=int(best_config["batch_size"]), verbose=0)
        policy_pred = inverse_standardize(np.asarray(policy_outputs[0]).reshape(-1), y_stats).astype(np.float64)
        policy_prob = np.clip(np.asarray(policy_outputs[1]).reshape(-1), 0.0, 1.0)
        test_pred = inverse_standardize(np.asarray(test_outputs[0]).reshape(-1), y_stats).astype(np.float64)
        test_prob = np.clip(np.asarray(test_outputs[1]).reshape(-1), 0.0, 1.0)
        test_reg = regression_metrics(dataset.target_log_return[test_targets], test_pred)
        test_labels = (dataset.target_log_return[test_targets] > direction_threshold).astype(np.float64)
        test_cls = classification_metrics(test_labels, test_prob)
        fold_row: dict[str, Any] = {"model": "lstm_v2", "horizon_hours": dataset.horizon_hours, "fold": fold.fold, "selected_candidate_id": best_candidate_id, "best_epoch": best_epoch, "sequence_length": seq_len, "direction_threshold_bps": float(best_config["direction_threshold_bps"]), **{f"forecast_{k}": v for k, v in test_reg.items()}, **{f"direction_{k}": v for k, v in test_cls.items()}}

        cost_outputs = {}
        for cost_bps in args.cost_bps:
            cost_rate = cost_bps / 10_000.0
            reg_choice, reg_sweep = choose_execution_policy(policy_pred, dataset.execution_simple_return_1h[policy_targets], cost_rate=cost_rate, lambda_grid=args.threshold_grid, holding_grid=args.holding_grid, min_trades=args.min_policy_trades, max_drawdown_abs=args.max_validation_drawdown)
            prob_choice, prob_sweep = choose_probability_policy(policy_prob, dataset.execution_simple_return_1h[policy_targets], cost_rate=cost_rate, entry_grid=args.prob_entry_grid, exit_grid=args.prob_exit_grid, holding_grid=args.holding_grid, min_trades=args.min_policy_trades, max_drawdown_abs=args.max_validation_drawdown)
            reg_choices[cost_bps].append(reg_choice)
            prob_choices[cost_bps].append(prob_choice)
            reg_policy_rows.extend({"model": "lstm_v2_regression", "horizon_hours": dataset.horizon_hours, "fold": fold.fold, "cost_bps": cost_bps, **row} for row in reg_sweep)
            prob_policy_rows.extend({"model": "lstm_v2_probability", "horizon_hours": dataset.horizon_hours, "fold": fold.fold, "cost_bps": cost_bps, **row} for row in prob_sweep)
            reg_metrics, reg_ret, reg_pos, reg_turn = held_cost_aware_backtest(test_pred, dataset.execution_simple_return_1h[test_targets], cost_rate=cost_rate, execution_lambda=reg_choice.execution_lambda, hold_hours=reg_choice.hold_hours)
            prob_metrics, prob_ret, prob_pos, prob_turn = held_probability_backtest(test_prob, dataset.execution_simple_return_1h[test_targets], cost_rate=cost_rate, entry_threshold=prob_choice.entry_threshold, exit_threshold=prob_choice.exit_threshold, hold_hours=prob_choice.hold_hours)
            reg_fold_metrics[cost_bps].append(reg_metrics)
            prob_fold_metrics[cost_bps].append(prob_metrics)
            cost_outputs[cost_bps] = (reg_choice, reg_metrics, reg_ret, reg_pos, reg_turn, prob_choice, prob_metrics, prob_ret, prob_pos, prob_turn)
            prefix = f"cost_{cost_bps:g}bps"
            fold_row[f"{prefix}_reg_sharpe"] = reg_metrics["sharpe"]
            fold_row[f"{prefix}_reg_trade_count"] = reg_metrics["trade_count"]
            fold_row[f"{prefix}_prob_sharpe"] = prob_metrics["sharpe"]
            fold_row[f"{prefix}_prob_trade_count"] = prob_metrics["trade_count"]
        fold_rows.append(fold_row)
        primary = cost_outputs[args.primary_cost_bps]
        print(f"LSTM-V2 fold {fold.fold:02d} TEST | skill={test_reg['skill_vs_zero']:+.2%} | direction={test_cls['direction_accuracy_prob']:.1%} AUC={test_cls['auc']:.3f} | REG Sharpe={primary[1]['sharpe']:+.3f} trades={int(primary[1]['trade_count'])} | PROB Sharpe={primary[6]['sharpe']:+.3f} trades={int(primary[6]['trade_count'])}", flush=True)

        for local, dataset_idx in enumerate(test_targets):
            row: dict[str, Any] = {"model": "lstm_v2", "horizon_hours": dataset.horizon_hours, "fold": fold.fold, "timestamp": int(dataset.timestamps[dataset_idx]), "actual_target_log_return": float(dataset.target_log_return[dataset_idx]), "actual_execution_return_1h": float(dataset.execution_simple_return_1h[dataset_idx]), "predicted_target_log_return": float(test_pred[local]), "predicted_direction_probability": float(test_prob[local]), "direction_threshold_bps": float(best_config["direction_threshold_bps"])}
            for cost_bps, values in cost_outputs.items():
                prefix = f"cost_{cost_bps:g}bps"
                reg_choice, _, reg_ret, reg_pos, reg_turn, prob_choice, _, prob_ret, prob_pos, prob_turn = values
                row[f"reg_{prefix}_lambda"] = reg_choice.execution_lambda
                row[f"reg_{prefix}_hold_hours"] = reg_choice.hold_hours
                row[f"reg_{prefix}_return"] = float(reg_ret[local])
                row[f"reg_{prefix}_position"] = float(reg_pos[local])
                row[f"reg_{prefix}_turnover"] = float(reg_turn[local])
                row[f"prob_{prefix}_entry"] = prob_choice.entry_threshold
                row[f"prob_{prefix}_exit"] = prob_choice.exit_threshold
                row[f"prob_{prefix}_hold_hours"] = prob_choice.hold_hours
                row[f"prob_{prefix}_return"] = float(prob_ret[local])
                row[f"prob_{prefix}_position"] = float(prob_pos[local])
                row[f"prob_{prefix}_turnover"] = float(prob_turn[local])
            prediction_rows.append(row)

    prediction_rows.sort(key=lambda row: int(row["timestamp"]))
    forecast = regression_metrics(np.asarray([row["actual_target_log_return"] for row in prediction_rows]), np.asarray([row["predicted_target_log_return"] for row in prediction_rows]))
    overall_labels = np.asarray([float(row["actual_target_log_return"] > row["direction_threshold_bps"] / 10_000.0) for row in prediction_rows], dtype=np.float64)
    overall_prob = np.asarray([row["predicted_direction_probability"] for row in prediction_rows], dtype=np.float64)
    direction = classification_metrics(overall_labels, overall_prob)
    reg_strategies: dict[str, Any] = {}
    prob_strategies: dict[str, Any] = {}
    for cost_bps in args.cost_bps:
        cost_key = f"cost_{cost_bps:g}bps"
        reg_strategies[f"{cost_bps:g}"] = aggregate_fold_strategies(prediction_rows, cost_key=f"reg_{cost_key}", fold_metrics=reg_fold_metrics[cost_bps])
        prob_strategies[f"{cost_bps:g}"] = aggregate_fold_strategies(prediction_rows, cost_key=f"prob_{cost_key}", fold_metrics=prob_fold_metrics[cost_bps])

    final_candidate_id = _mode(selected_candidate_ids)
    final_config = dict(candidates[final_candidate_id])
    final_epochs_pool = [epoch for cid, epoch in zip(selected_candidate_ids, selected_best_epochs, strict=True) if cid == final_candidate_id] or selected_best_epochs
    final_epochs = max(1, int(round(statistics.median(final_epochs_pool))))
    seq_len = int(final_config["sequence_length"])
    direction_threshold = float(final_config["direction_threshold_bps"]) / 10_000.0
    x_stats = fit_standardizer(dataset.X)
    y_stats = fit_standardizer(dataset.target_log_return)
    x_scaled = standardize(dataset.X, x_stats)
    y_scaled = standardize(dataset.target_log_return, y_stats)
    all_idx = np.arange(dataset.size, dtype=np.int64)
    all_x, all_y, all_targets = make_sequence_batch(x_scaled, y_scaled, dataset.timestamps, all_idx, sequence_length=seq_len, min_context_index=0)
    all_dir = (dataset.target_log_return[all_targets] > direction_threshold).astype(np.float32)
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(args.seed)
    final_model = _build_model(tf, sequence_length=seq_len, feature_count=len(dataset.feature_names), lstm_units=int(final_config["lstm_units"]), lstm_layers=int(final_config["lstm_layers"]), dense_units=int(final_config["dense_units"]), dropout=float(final_config["dropout"]), learning_rate=float(final_config["learning_rate"]), clipnorm=float(final_config["clipnorm"]), huber_delta=float(final_config["huber_delta"]), direction_loss_weight=float(final_config["direction_loss_weight"]))
    final_model.fit(all_x, {"next_hour_log_return_scaled": all_y, "direction_up_prob": all_dir}, epochs=final_epochs, batch_size=int(final_config["batch_size"]), shuffle=False, verbose=0)
    deployment = out / "deployment"
    deployment.mkdir(parents=True, exist_ok=True)
    final_model.save(deployment / "model.keras")
    write_json(deployment / "scaler.json", {"feature_names": dataset.feature_names, "sequence_length": seq_len, "feature_mean": x_stats.mean, "feature_scale": x_stats.scale, "target_mean": y_stats.mean, "target_scale": y_stats.scale})
    write_json(deployment / "manifest.json", {"model_family": "lstm_v2", "horizon_hours": dataset.horizon_hours, "feature_version": FEATURE_VERSION, "feature_names": dataset.feature_names, "selected_candidate_id": final_candidate_id, "final_config": final_config, "final_epochs": final_epochs, "recommended_regression_policy_by_cost_bps": _regression_policy_manifest(reg_choices), "recommended_probability_policy_by_cost_bps": _probability_manifest(prob_choices), "policy_selection": "policy-validation only; outer test never used for thresholds"})
    write_jsonl(out / "predictions.jsonl", prediction_rows)
    write_csv(out / "folds.csv", fold_rows)
    write_csv(out / "model_selection.csv", selection_rows)
    write_csv(out / "regression_policy_search.csv", reg_policy_rows)
    write_csv(out / "probability_policy_search.csv", prob_policy_rows)
    return {"model_family": "lstm_v2", "horizon_hours": dataset.horizon_hours, "forecast_metrics": forecast, "direction_classification": direction, "regression_strategies_by_cost_bps": reg_strategies, "probability_strategies_by_cost_bps": prob_strategies, "recommended_regression_policy_by_cost_bps": _regression_policy_manifest(reg_choices), "recommended_probability_policy_by_cost_bps": _probability_manifest(prob_choices), "selected_candidate_ids": selected_candidate_ids, "final_candidate_id": final_candidate_id, "final_model": str(deployment / "model.keras")}


def comparison_rows_lstm(result: dict[str, Any], primary_cost_bps: float) -> list[dict[str, Any]]:
    forecast = result["forecast_metrics"]
    direction = result["direction_classification"]
    key = f"{primary_cost_bps:g}"
    reg = result["regression_strategies_by_cost_bps"][key]
    prob = result["probability_strategies_by_cost_bps"][key]
    reg_policy = result["recommended_regression_policy_by_cost_bps"][key]
    prob_policy = result["recommended_probability_policy_by_cost_bps"][key]
    common = {"horizon_hours": result["horizon_hours"], "direction_accuracy": direction["direction_accuracy_prob"], "auc": direction["auc"], "brier_score": direction["brier_score"], "skill_vs_zero": forecast["skill_vs_zero"], "correlation": forecast["correlation"], "primary_cost_bps": primary_cost_bps, "final_model": result["final_model"]}
    return [
        {"model": "lstm_v2_regression", **common, "test_sharpe": reg["sharpe"], "test_sortino": reg["sortino"], "test_max_drawdown": reg["max_drawdown"], "test_cumulative_return": reg["cumulative_return"], "test_trade_count": reg["trade_count"], "test_turnover": reg["turnover"], "policy": f"lambda={reg_policy['execution_lambda']:.3g}, hold={reg_policy['hold_hours']}h"},
        {"model": "lstm_v2_probability", **common, "test_sharpe": prob["sharpe"], "test_sortino": prob["sortino"], "test_max_drawdown": prob["max_drawdown"], "test_cumulative_return": prob["cumulative_return"], "test_trade_count": prob["trade_count"], "test_turnover": prob["turnover"], "policy": f"entry={prob_policy['entry_threshold']:.2f}, exit={prob_policy['exit_threshold']:.2f}, hold={prob_policy['hold_hours']}h"},
    ]


def comparison_row_xgb(result: dict[str, Any], primary_cost_bps: float) -> dict[str, Any]:
    base = xgb_comparison_row(result, primary_cost_bps)
    forecast = result["forecast_metrics"]
    return {"model": "xgboost", "horizon_hours": base["horizon_hours"], "direction_accuracy": base["direction_accuracy"], "auc": float("nan"), "brier_score": float("nan"), "skill_vs_zero": forecast.get("skill_vs_zero", float("nan")), "correlation": base["correlation"], "primary_cost_bps": primary_cost_bps, "test_sharpe": base["test_sharpe"], "test_sortino": base["test_sortino"], "test_max_drawdown": base["test_max_drawdown"], "test_cumulative_return": base["test_cumulative_return"], "test_trade_count": base["test_trade_count"], "test_turnover": base["test_turnover"], "policy": f"lambda={base['recommended_lambda']:.3g}, hold={int(base['recommended_hold_hours'])}h", "final_model": base["final_model"]}


def _verdict(row: dict[str, Any], min_total_trades: int) -> str:
    trades = int(row["test_trade_count"])
    sharpe = float(row["test_sharpe"])
    if trades < min_total_trades:
        return f"INSUFFICIENT TRADES ({trades} < {min_total_trades})"
    if sharpe <= 0.0:
        return "NOT PROFITABLE AFTER COSTS"
    if float(row["test_max_drawdown"]) < -0.35:
        return "POSITIVE BUT DRAWDOWN TOO LARGE"
    return "RESEARCH CANDIDATE - FORWARD TEST REQUIRED"


def markdown_report(meta: dict[str, Any], comparison: list[dict[str, Any]], baselines: dict[str, Any]) -> str:
    lines = [
        "# Multi-Horizon AI Trading Research V2", "",
        f"Run: `{meta['run_id']}`  ", f"Git: `{meta['git_commit']}`  ",
        f"Dataset SHA256: `{meta['dataset_sha256']}`  ", f"Feature version: `{meta['feature_version']}`  ",
        f"Horizons: **{', '.join(str(x) + 'h' for x in meta['horizons'])}**  ",
        f"Primary cost: **{meta['primary_cost_bps']:.1f} bps per position change**", "",
        "## Protocol", "`TRAIN -> MODEL VALIDATION -> POLICY VALIDATION -> UNTOUCHED TEST`", "",
        "LSTM V2 uses stacked LSTM layers, Huber return regression, and a separate direction-probability head. Probability entry/exit thresholds and holding periods are selected only on policy validation data.", "",
        "## Results", "",
        "| Model | Horizon | Skill vs zero | Direction | AUC | Sharpe | Return | Max DD | Trades | Policy | Verdict |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in comparison:
        auc = row["auc"]
        auc_text = "-" if not math.isfinite(float(auc)) else f"{float(auc):.3f}"
        lines.append(f"| {row['model']} | {int(row['horizon_hours'])}h | {float(row['skill_vs_zero']):+.2%} | {float(row['direction_accuracy']):.2%} | {auc_text} | {float(row['test_sharpe']):+.3f} | {float(row['test_cumulative_return']):+.2%} | {float(row['test_max_drawdown']):.2%} | {int(row['test_trade_count'])} | {row['policy']} | {_verdict(row, meta['min_total_test_trades'])} |")
    lines += ["", "## Baselines"]
    for horizon, values in sorted(baselines.items(), key=lambda item: int(item[0])):
        lines.append(f"- {horizon}h windows: EMA Sharpe **{values['moving_average']['sharpe']:+.3f}**, buy-and-hold Sharpe **{values['buy_and_hold']['sharpe']:+.3f}**")
    lines += ["", "## Interpretation", "A positive Sharpe with only a handful of trades is explicitly marked insufficient. This report is for research selection and later forward paper trading, not a profitability claim."]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-horizon BTC ML research with LSTM V2.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--models", nargs="+", choices=["xgboost", "lstm_v2"], default=["xgboost", "lstm_v2"])
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
    parser.add_argument("--prob-entry-grid", type=parse_float_grid, default=parse_float_grid("0.52,0.55,0.58,0.60,0.65,0.70"))
    parser.add_argument("--prob-exit-grid", type=parse_float_grid, default=parse_float_grid("0.30,0.35,0.40,0.45,0.48,0.50"))
    parser.add_argument("--min-policy-trades", type=int, default=5)
    parser.add_argument("--min-total-test-trades", type=int, default=20)
    parser.add_argument("--max-validation-drawdown", type=float, default=0.35)
    parser.add_argument("--xgb-trials", type=int, default=8)
    parser.add_argument("--xgb-rounds", type=int, default=1500)
    parser.add_argument("--xgb-early-stopping", type=int, default=40)
    parser.add_argument("--xgb-device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--lstm-trials", type=int, default=4)
    parser.add_argument("--lstm-epochs", type=int, default=35)
    parser.add_argument("--lstm-early-stopping-patience", type=int, default=5)
    parser.add_argument("--epoch-verbose", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.dataset.is_file():
        raise FileNotFoundError(args.dataset)
    if args.primary_cost_bps not in args.cost_bps:
        raise ValueError("primary_cost_bps must be included in cost_bps")
    if min(args.threshold_grid) < 1.0:
        raise ValueError("execution lambda grid must remain >= 1")
    if not 0.1 <= args.model_validation_fraction <= 0.9:
        raise ValueError("model_validation_fraction must be between 0.1 and 0.9")
    if any(not 0.5 <= x < 1.0 for x in args.prob_entry_grid):
        raise ValueError("probability entry thresholds must be in [0.5, 1)")
    if any(not 0.0 <= x <= 0.5 for x in args.prob_exit_grid):
        raise ValueError("probability exit thresholds must be in [0, 0.5]")


def main() -> int:
    args = parse_args()
    validate_args(args)
    started = time.time()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    raw_rows = read_jsonl(args.dataset)
    walk_forward = WalkForwardConfig(train_days=args.train_days, validation_days=args.validation_days, test_days=args.test_days, step_days=args.step_days, embargo_hours=1)
    meta: dict[str, Any] = {"run_id": run_id, "created_at": datetime.now(timezone.utc).isoformat(), "git_commit": git_head(), "dataset_path": str(args.dataset), "dataset_sha256": sha256_file(args.dataset), "raw_jsonl_rows": len(raw_rows), "feature_version": FEATURE_VERSION, "horizons": args.horizons, "models": args.models, "walk_forward": asdict(walk_forward), "cost_bps": args.cost_bps, "primary_cost_bps": args.primary_cost_bps, "min_total_test_trades": args.min_total_test_trades, "lstm_trials": args.lstm_trials, "xgb_trials": args.xgb_trials, "python": platform.python_version(), "platform": platform.platform()}
    write_json(run_dir / "metadata.json", meta)
    results: dict[str, Any] = {}
    comparison: list[dict[str, Any]] = []
    baselines: dict[str, Any] = {}

    for horizon in args.horizons:
        dataset = build_research_dataset(raw_rows, horizon_hours=horizon)
        folds = make_walk_forward_folds(dataset.timestamps, walk_forward)
        if not folds:
            raise ValueError(f"no folds for horizon={horizon}h")
        print(f"\n######## HORIZON {horizon}h | rows={dataset.size:,} | folds={len(folds)} ########")
        horizon_dir = run_dir / f"h{horizon:02d}"
        horizon_dir.mkdir(parents=True, exist_ok=True)
        primary_cost_rate = args.primary_cost_bps / 10_000.0
        baselines[str(horizon)] = {"moving_average": moving_average_test_baseline(dataset, folds, cost_rate=primary_cost_rate), "buy_and_hold": buy_hold_test_baseline(dataset, folds, cost_rate=primary_cost_rate)}
        if "xgboost" in args.models:
            out = horizon_dir / "xgboost"
            out.mkdir(parents=True, exist_ok=True)
            result = train_xgboost_horizon(dataset, folds, out, args)
            results[f"xgboost_h{horizon}"] = result
            write_json(out / "summary.json", result)
            comparison.append(comparison_row_xgb(result, args.primary_cost_bps))
        if "lstm_v2" in args.models:
            out = horizon_dir / "lstm_v2"
            out.mkdir(parents=True, exist_ok=True)
            result = train_lstm_v2_horizon(dataset, folds, out, args)
            results[f"lstm_v2_h{horizon}"] = result
            write_json(out / "summary.json", result)
            comparison.extend(comparison_rows_lstm(result, args.primary_cost_bps))

    comparison.sort(key=lambda row: (int(row["test_trade_count"]) >= args.min_total_test_trades, float(row["test_sharpe"]), float(row["direction_accuracy"])), reverse=True)
    write_json(run_dir / "all_results.json", results)
    write_json(run_dir / "baselines.json", baselines)
    write_csv(run_dir / "model_comparison.csv", comparison)
    (run_dir / "REPORT.md").write_text(markdown_report(meta, comparison, baselines), encoding="utf-8")
    meta["duration_seconds"] = time.time() - started
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
            for path in deployment.rglob("*"):
                if path.is_file():
                    archive.write(path, path.relative_to(run_dir))
    print("\n================ RESEARCH V2 COMPLETE ================")
    for row in comparison:
        print(f"{row['model']:24s} h={int(row['horizon_hours']):2d} | skill={float(row['skill_vs_zero']):+6.2%} | dir={float(row['direction_accuracy']):5.1%} | Sharpe={float(row['test_sharpe']):+6.3f} | return={float(row['test_cumulative_return']):+7.2%} | trades={int(row['test_trade_count']):4d} | {_verdict(row, args.min_total_test_trades)}")
    print(f"report={run_dir / 'REPORT.md'}")
    print(f"research_bundle={research_zip}")
    print(f"deployment_bundle={deployment_zip}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
