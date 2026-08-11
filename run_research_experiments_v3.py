#!/usr/bin/env python3
"""V3 research runner: richer market context + calibrated economic-event trading.

Primary research question:
    Does the LSTM V2 representation contain stable information about whether
    the future h-hour BTC return clears an economically meaningful hurdle after
    calibration and realistic execution costs?

Protocol per outer fold:
    TRAIN -> MODEL VALIDATION -> CALIBRATION -> POLICY VALIDATION -> UNTOUCHED TEST

The newest 30 days of the dataset are excluded from all research/model selection
and are opened only once for a final EUR 1,000-equivalent shadow simulation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import random
import statistics as py_statistics
import subprocess
import time
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from backend.ml.calibration import (
    PayoffEstimate,
    brier_score,
    estimate_payoffs,
    ev_commitment_backtest,
    fit_platt_scaler,
)
from backend.ml.context_features import build_context_feature_matrix
from backend.ml.evaluation import (
    DAY_MS,
    HOUR_MS,
    WalkForwardConfig,
    _performance_metrics,
    buy_and_hold_baseline,
    classification_metrics,
    make_walk_forward_folds,
)
from backend.ml.features import FEATURE_VERSION, read_jsonl
from backend.ml.sequences import fit_standardizer, inverse_standardize, make_sequence_batch, standardize
from backend.ml.statistics import block_bootstrap_auc_ci
from run_research_experiments import ResearchDataset, build_research_dataset, purge_label_boundary

DEFAULT_DATASET = Path("data/processed/training/btc_hourly_v3.jsonl")
DEFAULT_OUTPUT = Path("artifacts/ml/research_v3")


@dataclass(frozen=True)
class ExperimentKey:
    target_hurdle_bps: float
    feature_set: str
    horizon_hours: int

    @property
    def slug(self) -> str:
        hurdle = str(int(self.target_hurdle_bps)) if float(self.target_hurdle_bps).is_integer() else str(self.target_hurdle_bps).replace(".", "p")
        return f"target{hurdle}bps_{self.feature_set}_h{self.horizon_hours:02d}"


@dataclass(frozen=True)
class FoldSplit:
    train: np.ndarray
    model_validation: np.ndarray
    calibration: np.ndarray
    policy_validation: np.ndarray
    test: np.ndarray


@dataclass(frozen=True)
class ModelCandidate:
    sequence_length: int
    lstm_units: int
    lstm_layers: int
    dense_units: int
    dropout: float
    learning_rate: float
    batch_size: int
    clipnorm: float
    huber_delta: float
    direction_loss_weight: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_jsonable), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
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
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _fmt_ts(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000.0, timezone.utc).strftime("%Y-%m-%d")


def parse_int_grid(value: str) -> list[int]:
    values = sorted({int(x.strip()) for x in value.split(",") if x.strip()})
    if not values or values[0] <= 0:
        raise argparse.ArgumentTypeError("grid must contain positive integers")
    return values


def parse_float_grid(value: str) -> list[float]:
    values = sorted({float(x.strip()) for x in value.split(",") if x.strip()})
    if not values:
        raise argparse.ArgumentTypeError("grid cannot be empty")
    return values


def _mode(values: list[int]) -> int:
    counts = Counter(values)
    return min(counts, key=lambda v: (-counts[v], v))


def _subset_dataset(dataset: ResearchDataset, indices: np.ndarray) -> ResearchDataset:
    idx = np.asarray(indices, dtype=np.int64)
    return ResearchDataset(
        horizon_hours=dataset.horizon_hours,
        timestamps=dataset.timestamps[idx],
        X=dataset.X[idx],
        target_log_return=dataset.target_log_return[idx],
        target_simple_return=dataset.target_simple_return[idx],
        execution_simple_return_1h=dataset.execution_simple_return_1h[idx],
        closes=dataset.closes[idx],
        ema20=dataset.ema20[idx],
        ema50=dataset.ema50[idx],
        ema200=dataset.ema200[idx],
        feature_names=list(dataset.feature_names),
    )


def enrich_dataset(dataset: ResearchDataset, feature_set: str) -> ResearchDataset:
    context = build_context_feature_matrix(dataset.timestamps, dataset.closes, feature_set=feature_set)
    if context.X.shape[1] == 0:
        return dataset
    return replace(
        dataset,
        X=np.column_stack([dataset.X, context.X]).astype(np.float32, copy=False),
        feature_names=list(dataset.feature_names) + context.feature_names,
    )


def candidate_pool(trials: int, seed: int) -> list[ModelCandidate]:
    if trials <= 0:
        raise ValueError("trial count must be positive")
    base = ModelCandidate(48, 64, 2, 32, 0.20, 1e-3, 64, 1.0, 1.0, 0.30)
    candidates = [base]
    rng = random.Random(seed)
    seen = {json.dumps(base.to_dict(), sort_keys=True)}
    while len(candidates) < trials:
        candidate = ModelCandidate(
            sequence_length=rng.choice([24, 48, 72, 96, 168]),
            lstm_units=rng.choice([48, 64, 96, 128]),
            lstm_layers=rng.choice([1, 2]),
            dense_units=rng.choice([24, 32, 64]),
            dropout=rng.choice([0.10, 0.20, 0.30, 0.40]),
            learning_rate=rng.choice([3e-4, 5e-4, 8e-4, 1e-3, 1.5e-3]),
            batch_size=rng.choice([32, 64, 128]),
            clipnorm=rng.choice([0.5, 1.0, 2.0]),
            huber_delta=rng.choice([0.5, 1.0, 1.5]),
            direction_loss_weight=rng.choice([0.20, 0.30, 0.50, 0.75]),
        )
        key = json.dumps(candidate.to_dict(), sort_keys=True)
        if key not in seen:
            seen.add(key)
            candidates.append(candidate)
    return candidates


def _split_validation_by_time(validation_indices: np.ndarray, timestamps: np.ndarray, *, model_days: int, calibration_days: int, policy_days: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.asarray(validation_indices, dtype=np.int64)
    if idx.size == 0:
        raise ValueError("empty validation block")
    start = int(timestamps[idx[0]])
    model_end = start + model_days * DAY_MS
    calibration_end = model_end + calibration_days * DAY_MS
    policy_end = calibration_end + policy_days * DAY_MS
    model = idx[(timestamps[idx] >= start) & (timestamps[idx] < model_end)]
    calibration = idx[(timestamps[idx] >= model_end) & (timestamps[idx] < calibration_end)]
    policy = idx[(timestamps[idx] >= calibration_end) & (timestamps[idx] < policy_end)]
    if min(len(model), len(calibration), len(policy)) == 0:
        raise ValueError("one validation sub-block is empty")
    return model, calibration, policy


def make_fold_split(dataset: ResearchDataset, fold: Any, args: argparse.Namespace) -> FoldSplit:
    model_raw, calibration_raw, policy_raw = _split_validation_by_time(
        fold.validation_indices,
        dataset.timestamps,
        model_days=args.model_validation_days,
        calibration_days=args.calibration_days,
        policy_days=args.policy_validation_days,
    )
    model_start = int(dataset.timestamps[model_raw[0]])
    calibration_start = int(dataset.timestamps[calibration_raw[0]])
    policy_start = int(dataset.timestamps[policy_raw[0]])
    test_start = int(dataset.timestamps[fold.test_indices[0]])
    test_end = int(dataset.timestamps[fold.test_indices[-1]]) + HOUR_MS

    train = purge_label_boundary(fold.train_indices, dataset.timestamps, horizon_hours=dataset.horizon_hours, boundary_timestamp=model_start)
    model = purge_label_boundary(model_raw, dataset.timestamps, horizon_hours=dataset.horizon_hours, boundary_timestamp=calibration_start)
    calibration = purge_label_boundary(calibration_raw, dataset.timestamps, horizon_hours=dataset.horizon_hours, boundary_timestamp=policy_start)
    policy = purge_label_boundary(policy_raw, dataset.timestamps, horizon_hours=dataset.horizon_hours, boundary_timestamp=test_start)
    test = purge_label_boundary(fold.test_indices, dataset.timestamps, horizon_hours=dataset.horizon_hours, boundary_timestamp=test_end)
    if min(len(train), len(model), len(calibration), len(policy), len(test)) == 0:
        raise ValueError(f"empty V3 purged split in fold {fold.fold}")
    return FoldSplit(train, model, calibration, policy, test)


def _progress_callback(tf: Any, *, key: ExperimentKey, fold: int, folds: int, candidate: int, candidates: int, epochs: int, patience: int, target_scale: float, enabled: bool) -> Any:
    class Progress(tf.keras.callbacks.Callback):
        def __init__(self) -> None:
            super().__init__()
            self.best = float("inf")
            self.best_epoch = 0

        def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
            if not enabled:
                return
            logs = logs or {}
            val_loss = float(logs.get("val_loss", math.inf))
            if val_loss < self.best:
                self.best = val_loss
                self.best_epoch = epoch + 1
            mae = None
            direction = None
            for name, value in logs.items():
                if "val_" in name and "log_return" in name and "mae" in name:
                    mae = float(value)
                if "val_" in name and "direction" in name and "acc" in name:
                    direction = float(value)
            mae_bps = mae * target_scale * 10_000.0 if mae is not None else float("nan")
            since = epoch + 1 - self.best_epoch
            marker = "*" if since == 0 else f"early stop in {max(patience - since, 0)}"
            print(
                f"[V3 {key.feature_set} target={key.target_hurdle_bps:g}bps h={key.horizon_hours:02d} fold {fold:02d}/{folds:02d} cand {candidate + 1}/{candidates}] "
                f"epoch {epoch + 1:02d}/{epochs:02d} | val MAE {mae_bps:6.1f} bps | val direction {(direction if direction is not None else float('nan')):5.1%} | {marker}",
                flush=True,
            )
    return Progress()


def _train_model_candidate(tf: Any, dataset: ResearchDataset, split: FoldSplit, candidate: ModelCandidate, *, event_hurdle_bps: float, seed: int, epochs: int, patience: int, progress: Any) -> tuple[Any, Any, Any, int, dict[str, float]]:
    x_stats = fit_standardizer(dataset.X[split.train])
    y_stats = fit_standardizer(dataset.target_log_return[split.train])
    x_scaled = standardize(dataset.X, x_stats)
    y_scaled = standardize(dataset.target_log_return, y_stats)
    seq = candidate.sequence_length
    train_x, train_y, train_targets = make_sequence_batch(x_scaled, y_scaled, dataset.timestamps, split.train, sequence_length=seq, min_context_index=int(split.train[0]))
    val_x, val_y, val_targets = make_sequence_batch(x_scaled, y_scaled, dataset.timestamps, split.model_validation, sequence_length=seq)
    if min(len(train_x), len(val_x)) == 0:
        raise ValueError("empty sequence batch")
    hurdle = event_hurdle_bps / 10_000.0
    train_dir = (dataset.target_simple_return[train_targets] > hurdle).astype(np.float32)
    val_dir = (dataset.target_simple_return[val_targets] > hurdle).astype(np.float32)

    from backend.ml.lstm_core import _build_model
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    model = _build_model(
        tf,
        sequence_length=seq,
        feature_count=len(dataset.feature_names),
        lstm_units=candidate.lstm_units,
        lstm_layers=candidate.lstm_layers,
        dense_units=candidate.dense_units,
        dropout=candidate.dropout,
        learning_rate=candidate.learning_rate,
        clipnorm=candidate.clipnorm,
        huber_delta=candidate.huber_delta,
        direction_loss_weight=candidate.direction_loss_weight,
    )
    history = model.fit(
        train_x,
        {"next_hour_log_return_scaled": train_y, "direction_up_prob": train_dir},
        validation_data=(val_x, {"next_hour_log_return_scaled": val_y, "direction_up_prob": val_dir}),
        epochs=epochs,
        batch_size=candidate.batch_size,
        shuffle=False,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=patience, restore_best_weights=True, verbose=0),
            progress,
        ],
        verbose=0,
    )
    output = model.predict(val_x, verbose=0)
    raw_prob = np.clip(np.asarray(output[1]).reshape(-1), 0.0, 1.0)
    cls = classification_metrics(val_dir, raw_prob)
    losses = [float(v) for v in history.history.get("val_loss", [])]
    best_epoch = int(np.argmin(losses) + 1) if losses else max(1, len(history.epoch))
    return model, x_stats, y_stats, best_epoch, cls


def choose_margin(probability: np.ndarray, actual_1h: np.ndarray, payoff: PayoffEstimate, *, cost_rate: float, horizon_hours: int, margin_grid_bps: list[float], min_trades: int) -> tuple[float, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for margin_bps in margin_grid_bps:
        metrics, _, _, _, _ = ev_commitment_backtest(
            probability,
            actual_1h,
            payoff=payoff,
            one_way_cost_rate=cost_rate,
            horizon_hours=horizon_hours,
            entry_margin=margin_bps / 10_000.0,
            exit_ev_threshold=0.0,
        )
        rows.append({"entry_margin_bps": margin_bps, **metrics})
    trade_ok = [row for row in rows if int(row["trade_count"]) >= min_trades]
    positive = [row for row in trade_ok if float(row["sharpe"]) > 0.0 and float(row["cumulative_return"]) > 0.0]
    pool = positive or trade_ok or rows
    best = max(pool, key=lambda row: (float(row["sharpe"]), float(row["cumulative_return"]), float(row["max_drawdown"]), -float(row["turnover"])))
    return float(best["entry_margin_bps"]), rows


def _aggregate_fold_series(folds: list[dict[str, Any]], cost_bps: float) -> dict[str, float]:
    returns = np.concatenate([np.asarray(row[f"returns_{cost_bps:g}"], dtype=np.float64) for row in folds])
    positions = np.concatenate([np.asarray(row[f"positions_{cost_bps:g}"], dtype=np.float64) for row in folds])
    turnovers = np.concatenate([np.asarray(row[f"turnovers_{cost_bps:g}"], dtype=np.float64) for row in folds])
    metrics = _performance_metrics(returns, positions, turnovers)
    for name in ("trade_count", "round_trip_count", "position_change_count"):
        metrics[name] = int(sum(int(row[f"metrics_{cost_bps:g}"][name]) for row in folds))
    metrics["turnover"] = float(sum(float(row[f"metrics_{cost_bps:g}"]["turnover"]) for row in folds))
    return metrics


def _buy_hold_over_tests(dataset: ResearchDataset, fold_target_indices: list[np.ndarray], cost_rate: float) -> dict[str, float]:
    fold_returns: list[np.ndarray] = []
    fold_positions: list[np.ndarray] = []
    fold_turnovers: list[np.ndarray] = []
    fold_metrics: list[dict[str, float]] = []
    for idx in fold_target_indices:
        actual = dataset.execution_simple_return_1h[idx]
        metrics = buy_and_hold_baseline(actual, cost_rate=cost_rate)
        positions = np.ones(len(idx), dtype=np.float64)
        turnover = np.zeros(len(idx), dtype=np.float64)
        if len(idx):
            turnover[0] = 1.0
        returns = actual.copy()
        if len(idx):
            returns[0] -= cost_rate
        fold_returns.append(returns)
        fold_positions.append(positions)
        fold_turnovers.append(turnover)
        fold_metrics.append(metrics)
    metrics = _performance_metrics(np.concatenate(fold_returns), np.concatenate(fold_positions), np.concatenate(fold_turnovers))
    metrics["trade_count"] = int(sum(int(m["trade_count"]) for m in fold_metrics))
    metrics["round_trip_count"] = int(sum(int(m["round_trip_count"]) for m in fold_metrics))
    metrics["position_change_count"] = int(sum(int(m["position_change_count"]) for m in fold_metrics))
    return metrics


def _gate(summary: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    primary = summary["strategy_by_cost_bps"]["25"]
    c30 = summary["strategy_by_cost_bps"]["30"]
    if int(primary["trade_count"]) < 30:
        reasons.append("trades < 30")
    if float(primary["cumulative_return"]) <= 0.0:
        reasons.append("cumulative return <= 0")
    if float(primary["sharpe"]) < 0.25:
        reasons.append("Sharpe < 0.25")
    if float(primary["sharpe"]) <= float(summary["buy_and_hold"]["sharpe"]):
        reasons.append("Sharpe does not beat buy-and-hold")
    if float(primary["max_drawdown"]) < -0.30:
        reasons.append("max drawdown worse than -30%")
    if float(summary["median_fold_sharpe"]) <= 0.0:
        reasons.append("median fold Sharpe <= 0")
    if int(summary["profitable_folds"]) * 2 <= int(summary["fold_count"]):
        reasons.append("profitable folds are not a majority")
    if float(summary["median_fold_auc"]) <= 0.50:
        reasons.append("median fold AUC <= 0.50")
    if float(summary["auc_ci_168h"]["lower"]) <= 0.50:
        reasons.append("168h block-bootstrap AUC lower bound <= 0.50")
    if float(c30["sharpe"]) <= 0.0 or float(c30["cumulative_return"]) <= 0.0:
        reasons.append("not robust at 30 bps")
    if not math.isfinite(float(primary["sharpe"])):
        reasons.append("non-finite primary Sharpe")
    return not reasons, reasons


def run_experiment(dataset: ResearchDataset, key: ExperimentKey, args: argparse.Namespace, out: Path, *, trial_count: int) -> dict[str, Any]:
    import tensorflow as tf
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

    context_dataset = enrich_dataset(dataset, key.feature_set)
    shadow_start = int(context_dataset.timestamps[-1]) - args.shadow_days * DAY_MS
    pre_mask = (context_dataset.timestamps + key.horizon_hours * HOUR_MS) < shadow_start
    pre_indices = np.flatnonzero(pre_mask)
    pre = _subset_dataset(context_dataset, pre_indices)

    validation_days = args.model_validation_days + args.calibration_days + args.policy_validation_days
    wf = WalkForwardConfig(train_days=args.train_days, validation_days=validation_days, test_days=args.test_days, step_days=args.step_days, embargo_hours=0)
    folds = make_walk_forward_folds(pre.timestamps, wf)
    if not folds:
        raise ValueError(f"no V3 folds for {key.slug}")

    candidates = candidate_pool(trial_count, args.seed + key.horizon_hours * 1009 + int(key.target_hurdle_bps) * 17 + len(key.feature_set))
    fold_payloads: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    policy_rows: list[dict[str, Any]] = []
    selected_candidate_ids: list[int] = []
    test_target_blocks: list[np.ndarray] = []

    print(f"\n{'#' * 88}\nV3 EXPERIMENT {key.slug} | features={len(pre.feature_names)} | folds={len(folds)} | candidates={len(candidates)}\n{'#' * 88}", flush=True)

    for fold in folds:
        split = make_fold_split(pre, fold, args)
        print(
            f"\n=== {key.slug} fold {fold.fold}/{len(folds)} | train {_fmt_ts(int(pre.timestamps[split.train[0]]))}..{_fmt_ts(int(pre.timestamps[split.train[-1]]))} | "
            f"model {_fmt_ts(int(pre.timestamps[split.model_validation[0]]))}..{_fmt_ts(int(pre.timestamps[split.model_validation[-1]]))} | "
            f"cal {_fmt_ts(int(pre.timestamps[split.calibration[0]]))}..{_fmt_ts(int(pre.timestamps[split.calibration[-1]]))} | "
            f"policy {_fmt_ts(int(pre.timestamps[split.policy_validation[0]]))}..{_fmt_ts(int(pre.timestamps[split.policy_validation[-1]]))} | "
            f"test {_fmt_ts(int(pre.timestamps[split.test[0]]))}..{_fmt_ts(int(pre.timestamps[split.test[-1]]))} ===",
            flush=True,
        )

        best_id = -1
        best_key: tuple[float, float, float] | None = None
        best_weights: list[np.ndarray] | None = None
        best_epoch = 0
        best_x_stats = None
        best_y_stats = None

        for cid, candidate in enumerate(candidates):
            target_scale = float(np.asarray(fit_standardizer(pre.target_log_return[split.train]).scale).reshape(-1)[0])
            progress = _progress_callback(
                tf,
                key=key,
                fold=fold.fold,
                folds=len(folds),
                candidate=cid,
                candidates=len(candidates),
                epochs=args.epochs,
                patience=args.early_stopping_patience,
                target_scale=target_scale,
                enabled=args.epoch_verbose,
            )
            try:
                model, x_stats, y_stats, best_candidate_epoch, cls = _train_model_candidate(
                    tf,
                    pre,
                    split,
                    candidate,
                    event_hurdle_bps=key.target_hurdle_bps,
                    seed=args.seed + fold.fold * 10_000 + cid,
                    epochs=args.epochs,
                    patience=args.early_stopping_patience,
                    progress=progress,
                )
            except ValueError as exc:
                selection_rows.append({"fold": fold.fold, "candidate_id": cid, "status": f"skipped:{exc}", **candidate.to_dict()})
                continue
            selection_rows.append({"fold": fold.fold, "candidate_id": cid, "status": "ok", "best_epoch": best_candidate_epoch, **cls, **candidate.to_dict()})
            print(f"candidate {cid + 1}/{len(candidates)} | epoch={best_candidate_epoch} | model-val AUC={cls['auc']:.3f} | accuracy={cls['direction_accuracy_prob']:.1%} | Brier={cls['brier_score']:.4f}", flush=True)
            rank = (float(cls["auc"]), -float(cls["brier_score"]), float(cls["direction_accuracy_prob"]))
            if best_key is None or rank > best_key:
                best_key = rank
                best_id = cid
                best_epoch = best_candidate_epoch
                best_weights = [np.array(w, copy=True) for w in model.get_weights()]
                best_x_stats = x_stats
                best_y_stats = y_stats

        if best_id < 0 or best_weights is None or best_x_stats is None or best_y_stats is None:
            raise ValueError(f"no valid model candidate in fold {fold.fold}")
        selected_candidate_ids.append(best_id)
        candidate = candidates[best_id]
        x_scaled = standardize(pre.X, best_x_stats)
        y_scaled = standardize(pre.target_log_return, best_y_stats)

        from backend.ml.lstm_core import _build_model
        tf.keras.backend.clear_session()
        model = _build_model(
            tf,
            sequence_length=candidate.sequence_length,
            feature_count=len(pre.feature_names),
            lstm_units=candidate.lstm_units,
            lstm_layers=candidate.lstm_layers,
            dense_units=candidate.dense_units,
            dropout=candidate.dropout,
            learning_rate=candidate.learning_rate,
            clipnorm=candidate.clipnorm,
            huber_delta=candidate.huber_delta,
            direction_loss_weight=candidate.direction_loss_weight,
        )
        model.set_weights(best_weights)

        def predict_block(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            xx, _, targets = make_sequence_batch(x_scaled, y_scaled, pre.timestamps, indices, sequence_length=candidate.sequence_length)
            if len(xx) == 0:
                raise ValueError("empty prediction sequence block")
            output = model.predict(xx, verbose=0)
            regression = inverse_standardize(np.asarray(output[0]).reshape(-1), best_y_stats).astype(np.float64)
            probability = np.clip(np.asarray(output[1]).reshape(-1), 1e-6, 1.0 - 1e-6)
            return regression, probability, targets

        _, cal_raw, cal_targets = predict_block(split.calibration)
        hurdle = key.target_hurdle_bps / 10_000.0
        cal_labels = (pre.target_simple_return[cal_targets] > hurdle).astype(np.float64)
        calibrator = fit_platt_scaler(cal_raw, cal_labels)
        cal_prob = calibrator.transform(cal_raw)
        payoff = estimate_payoffs(pre.target_simple_return[cal_targets], event_hurdle_bps=key.target_hurdle_bps, trim_fraction=args.trim_fraction)

        _, policy_raw, policy_targets = predict_block(split.policy_validation)
        policy_prob = calibrator.transform(policy_raw)
        margin_bps, sweep = choose_margin(
            policy_prob,
            pre.execution_simple_return_1h[policy_targets],
            payoff,
            cost_rate=args.primary_cost_bps / 10_000.0,
            horizon_hours=key.horizon_hours,
            margin_grid_bps=args.ev_margin_grid_bps,
            min_trades=args.min_policy_trades,
        )
        policy_rows.extend({"fold": fold.fold, **row} for row in sweep)

        _, test_raw, test_targets = predict_block(split.test)
        test_prob = calibrator.transform(test_raw)
        test_labels = (pre.target_simple_return[test_targets] > hurdle).astype(np.float64)
        test_cls = classification_metrics(test_labels, test_prob)
        test_target_blocks.append(test_targets)

        payload: dict[str, Any] = {
            "fold": fold.fold,
            "selected_candidate_id": best_id,
            "best_epoch": best_epoch,
            "margin_bps": margin_bps,
            "auc": float(test_cls["auc"]),
            "brier": float(test_cls["brier_score"]),
            "direction_accuracy": float(test_cls["direction_accuracy_prob"]),
            "calibration_brier_before": brier_score(cal_labels, cal_raw),
            "calibration_brier_after": brier_score(cal_labels, cal_prob),
            "payoff": payoff.to_dict(),
            "calibrator": calibrator.to_dict(),
        }
        for cost_bps in args.cost_bps:
            metrics, returns, positions, turnovers, ev = ev_commitment_backtest(
                test_prob,
                pre.execution_simple_return_1h[test_targets],
                payoff=payoff,
                one_way_cost_rate=cost_bps / 10_000.0,
                horizon_hours=key.horizon_hours,
                entry_margin=margin_bps / 10_000.0,
                exit_ev_threshold=0.0,
            )
            payload[f"metrics_{cost_bps:g}"] = metrics
            payload[f"returns_{cost_bps:g}"] = returns.tolist()
            payload[f"positions_{cost_bps:g}"] = positions.tolist()
            payload[f"turnovers_{cost_bps:g}"] = turnovers.tolist()
            payload[f"ev_{cost_bps:g}"] = ev.tolist()

        primary = payload[f"metrics_{args.primary_cost_bps:g}"]
        print(
            f"FOLD {fold.fold:02d} TEST | AUC={test_cls['auc']:.3f} | Brier={test_cls['brier_score']:.4f} | "
            f"Sharpe={primary['sharpe']:+.3f} | return={primary['cumulative_return']:+.2%} | trades={int(primary['trade_count'])} | "
            f"margin={margin_bps:g}bps | cal Brier {payload['calibration_brier_before']:.4f}->{payload['calibration_brier_after']:.4f}",
            flush=True,
        )

        for local, target_idx in enumerate(test_targets):
            prediction_rows.append({
                "fold": fold.fold,
                "timestamp": int(pre.timestamps[target_idx]),
                "label": float(test_labels[local]),
                "calibrated_probability": float(test_prob[local]),
                "actual_horizon_return": float(pre.target_simple_return[target_idx]),
                "actual_1h_return": float(pre.execution_simple_return_1h[target_idx]),
            })
        fold_payloads.append(payload)

    labels = np.asarray([row["label"] for row in prediction_rows], dtype=np.float64)
    probs = np.asarray([row["calibrated_probability"] for row in prediction_rows], dtype=np.float64)
    overall_cls = classification_metrics(labels, probs)
    ci24 = block_bootstrap_auc_ci(labels, probs, block_length=24, samples=args.bootstrap_samples, seed=args.seed + 24)
    ci168 = block_bootstrap_auc_ci(labels, probs, block_length=168, samples=args.bootstrap_samples, seed=args.seed + 168)
    strategies = {f"{cost:g}": _aggregate_fold_series(fold_payloads, cost) for cost in args.cost_bps}
    buy_hold = _buy_hold_over_tests(pre, test_target_blocks, args.primary_cost_bps / 10_000.0)
    fold_sharpes = [float(row[f"metrics_{args.primary_cost_bps:g}"]["sharpe"]) for row in fold_payloads]
    fold_returns = [float(row[f"metrics_{args.primary_cost_bps:g}"]["cumulative_return"]) for row in fold_payloads]
    fold_aucs = [float(row["auc"]) for row in fold_payloads]

    summary: dict[str, Any] = {
        "experiment": asdict(key),
        "feature_count": len(pre.feature_names),
        "feature_names": pre.feature_names,
        "fold_count": len(folds),
        "overall_classification": overall_cls,
        "median_fold_auc": float(py_statistics.median(fold_aucs)),
        "mean_fold_auc": float(np.mean(fold_aucs)),
        "fold_auc_values": fold_aucs,
        "folds_auc_gt_050": int(sum(v > 0.50 for v in fold_aucs)),
        "folds_auc_gt_055": int(sum(v > 0.55 for v in fold_aucs)),
        "auc_ci_24h": ci24.to_dict(),
        "auc_ci_168h": ci168.to_dict(),
        "median_fold_sharpe": float(py_statistics.median(fold_sharpes)),
        "mean_fold_sharpe": float(np.mean(fold_sharpes)),
        "profitable_folds": int(sum(v > 0.0 for v in fold_returns)),
        "strategy_by_cost_bps": strategies,
        "buy_and_hold": buy_hold,
        "selected_candidate_ids": selected_candidate_ids,
        "modal_candidate_id": _mode(selected_candidate_ids),
        "candidate_pool": [candidate.to_dict() for candidate in candidates],
    }
    passed, reasons = _gate(summary)
    summary["gate_passed"] = passed
    summary["gate_reasons"] = reasons

    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "summary.json", summary)
    write_json(out / "folds.json", fold_payloads)
    write_jsonl(out / "predictions.jsonl", prediction_rows)
    write_csv(out / "model_selection.csv", selection_rows)
    write_csv(out / "policy_search.csv", policy_rows)
    return summary


def _capital_curve(starting: float, actual: np.ndarray, positions: np.ndarray, turnovers: np.ndarray, cost_rate: float) -> tuple[np.ndarray, float, float]:
    capital = float(starting)
    curve = np.zeros(len(actual), dtype=np.float64)
    fees = 0.0
    for i, (ret, pos, turn) in enumerate(zip(actual, positions, turnovers, strict=True)):
        fee = capital * cost_rate * float(turn)
        fees += fee
        capital *= 1.0 + float(pos) * float(ret) - cost_rate * float(turn)
        curve[i] = capital
    return curve, capital, fees


def _completed_trade_returns(strategy_returns: np.ndarray, positions: np.ndarray, turnovers: np.ndarray) -> list[float]:
    trades: list[float] = []
    factor: float | None = None
    previous = 0.0
    for ret, pos, turn in zip(strategy_returns, positions, turnovers, strict=True):
        if previous == 0.0 and pos > 0.0 and turn > 0.0:
            factor = 1.0 + float(ret)
        elif factor is not None:
            factor *= 1.0 + float(ret)
        if factor is not None and previous > 0.0 and turn > 0.0 and pos <= 0.0:
            trades.append(factor - 1.0)
            factor = None
        previous = float(pos)
    if factor is not None:
        trades.append(factor - 1.0)
    return trades


def _final_window_indices(dataset: ResearchDataset, shadow_start: int, args: argparse.Namespace) -> FoldSplit:
    policy_start = shadow_start - args.policy_validation_days * DAY_MS
    calibration_start = policy_start - args.calibration_days * DAY_MS
    model_start = calibration_start - args.model_validation_days * DAY_MS
    train_start = model_start - args.train_days * DAY_MS
    shadow_end = int(dataset.timestamps[-1]) + HOUR_MS

    def between(start: int, end: int) -> np.ndarray:
        return np.flatnonzero((dataset.timestamps >= start) & (dataset.timestamps < end))

    train_raw = between(train_start, model_start)
    model_raw = between(model_start, calibration_start)
    calibration_raw = between(calibration_start, policy_start)
    policy_raw = between(policy_start, shadow_start)
    shadow_raw = between(shadow_start, shadow_end)
    train = purge_label_boundary(train_raw, dataset.timestamps, horizon_hours=dataset.horizon_hours, boundary_timestamp=model_start)
    model = purge_label_boundary(model_raw, dataset.timestamps, horizon_hours=dataset.horizon_hours, boundary_timestamp=calibration_start)
    calibration = purge_label_boundary(calibration_raw, dataset.timestamps, horizon_hours=dataset.horizon_hours, boundary_timestamp=policy_start)
    policy = purge_label_boundary(policy_raw, dataset.timestamps, horizon_hours=dataset.horizon_hours, boundary_timestamp=shadow_start)
    shadow = purge_label_boundary(shadow_raw, dataset.timestamps, horizon_hours=dataset.horizon_hours, boundary_timestamp=shadow_end)
    if min(len(train), len(model), len(calibration), len(policy), len(shadow)) == 0:
        raise ValueError("final shadow split contains an empty block")
    return FoldSplit(train, model, calibration, policy, shadow)


def run_final_shadow(base_dataset: ResearchDataset, winner: dict[str, Any], args: argparse.Namespace, out: Path) -> dict[str, Any]:
    import tensorflow as tf
    exp = winner["experiment"]
    key = ExperimentKey(float(exp["target_hurdle_bps"]), str(exp["feature_set"]), int(exp["horizon_hours"]))
    dataset = enrich_dataset(base_dataset, key.feature_set)
    shadow_start = int(dataset.timestamps[-1]) - args.shadow_days * DAY_MS
    split = _final_window_indices(dataset, shadow_start, args)
    candidate_id = int(winner["modal_candidate_id"])
    candidate = ModelCandidate(**winner["candidate_pool"][candidate_id])

    x_stats = fit_standardizer(dataset.X[split.train])
    y_stats = fit_standardizer(dataset.target_log_return[split.train])
    x_scaled = standardize(dataset.X, x_stats)
    y_scaled = standardize(dataset.target_log_return, y_stats)
    seq = candidate.sequence_length
    train_x, train_y, train_targets = make_sequence_batch(x_scaled, y_scaled, dataset.timestamps, split.train, sequence_length=seq, min_context_index=int(split.train[0]))
    model_x, model_y, model_targets = make_sequence_batch(x_scaled, y_scaled, dataset.timestamps, split.model_validation, sequence_length=seq)
    hurdle = key.target_hurdle_bps / 10_000.0
    train_labels = (dataset.target_simple_return[train_targets] > hurdle).astype(np.float32)
    model_labels = (dataset.target_simple_return[model_targets] > hurdle).astype(np.float32)

    from backend.ml.lstm_core import _build_model
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(args.seed + 999999)
    model = _build_model(
        tf,
        sequence_length=seq,
        feature_count=len(dataset.feature_names),
        lstm_units=candidate.lstm_units,
        lstm_layers=candidate.lstm_layers,
        dense_units=candidate.dense_units,
        dropout=candidate.dropout,
        learning_rate=candidate.learning_rate,
        clipnorm=candidate.clipnorm,
        huber_delta=candidate.huber_delta,
        direction_loss_weight=candidate.direction_loss_weight,
    )
    history = model.fit(
        train_x,
        {"next_hour_log_return_scaled": train_y, "direction_up_prob": train_labels},
        validation_data=(model_x, {"next_hour_log_return_scaled": model_y, "direction_up_prob": model_labels}),
        epochs=args.final_epochs,
        batch_size=candidate.batch_size,
        shuffle=False,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.early_stopping_patience, restore_best_weights=True, verbose=1)],
        verbose=1,
    )

    def predict(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xx, _, targets = make_sequence_batch(x_scaled, y_scaled, dataset.timestamps, indices, sequence_length=seq)
        output = model.predict(xx, verbose=0)
        return np.clip(np.asarray(output[1]).reshape(-1), 1e-6, 1.0 - 1e-6), targets

    cal_raw, cal_targets = predict(split.calibration)
    cal_labels = (dataset.target_simple_return[cal_targets] > hurdle).astype(np.float64)
    calibrator = fit_platt_scaler(cal_raw, cal_labels)
    payoff = estimate_payoffs(dataset.target_simple_return[cal_targets], event_hurdle_bps=key.target_hurdle_bps, trim_fraction=args.trim_fraction)
    policy_raw, policy_targets = predict(split.policy_validation)
    policy_prob = calibrator.transform(policy_raw)
    margin_bps, _ = choose_margin(policy_prob, dataset.execution_simple_return_1h[policy_targets], payoff, cost_rate=args.primary_cost_bps / 10_000.0, horizon_hours=key.horizon_hours, margin_grid_bps=args.ev_margin_grid_bps, min_trades=args.min_policy_trades)

    shadow_raw, shadow_targets = predict(split.test)
    shadow_prob = calibrator.transform(shadow_raw)
    actual = dataset.execution_simple_return_1h[shadow_targets]
    timestamp = dataset.timestamps[shadow_targets]
    shadow_labels = (dataset.target_simple_return[shadow_targets] > hurdle).astype(np.float64)
    cls = classification_metrics(shadow_labels, shadow_prob)

    cost_results: dict[str, Any] = {}
    primary_arrays = None
    for cost_bps in args.cost_bps:
        metrics, returns, positions, turnovers, ev = ev_commitment_backtest(
            shadow_prob,
            actual,
            payoff=payoff,
            one_way_cost_rate=cost_bps / 10_000.0,
            horizon_hours=key.horizon_hours,
            entry_margin=margin_bps / 10_000.0,
            exit_ev_threshold=0.0,
        )
        curve, ending, fees = _capital_curve(args.starting_capital_eur, actual, positions, turnovers, cost_bps / 10_000.0)
        cost_results[f"{cost_bps:g}"] = {"metrics": metrics, "ending_capital_eur": ending, "profit_eur": ending - args.starting_capital_eur, "fees_eur": fees}
        if cost_bps == args.primary_cost_bps:
            primary_arrays = (returns, positions, turnovers, ev, curve)

    assert primary_arrays is not None
    returns, positions, turnovers, ev, curve = primary_arrays
    _, gross_ending, _ = _capital_curve(args.starting_capital_eur, actual, positions, np.zeros_like(turnovers), 0.0)
    trade_returns = _completed_trade_returns(returns, positions, turnovers)
    wins = sum(v > 0.0 for v in trade_returns)
    losses = sum(v <= 0.0 for v in trade_returns)

    bh_positions = np.ones(len(actual), dtype=np.float64)
    bh_turnover = np.zeros(len(actual), dtype=np.float64)
    if len(actual):
        bh_turnover[0] = 1.0
        bh_turnover[-1] += 1.0
    _, bh_ending, bh_fees = _capital_curve(args.starting_capital_eur, actual, bh_positions, bh_turnover, args.primary_cost_bps / 10_000.0)

    ema_positions = ((dataset.ema20[shadow_targets] > dataset.ema50[shadow_targets]) & (dataset.closes[shadow_targets] > dataset.ema200[shadow_targets])).astype(np.float64)
    ema_prev = np.concatenate(([0.0], ema_positions[:-1]))
    ema_turn = np.abs(ema_positions - ema_prev)
    if len(ema_positions) and ema_positions[-1] > 0.0:
        ema_turn[-1] += 1.0
    _, ema_ending, ema_fees = _capital_curve(args.starting_capital_eur, actual, ema_positions, ema_turn, args.primary_cost_bps / 10_000.0)

    primary = cost_results[f"{args.primary_cost_bps:g}"]
    equity_rows: list[dict[str, Any]] = []
    for i in range(len(timestamp)):
        equity_rows.append({
            "timestamp": int(timestamp[i]),
            "date_utc": datetime.fromtimestamp(int(timestamp[i]) / 1000.0, timezone.utc).isoformat(),
            "capital_eur": float(curve[i]),
            "position": float(positions[i]),
            "turnover": float(turnovers[i]),
            "actual_1h_return": float(actual[i]),
            "calibrated_probability": float(shadow_prob[i]),
            "decision_ev": None if not np.isfinite(ev[i]) else float(ev[i]),
        })

    result = {
        "winner": winner["experiment"],
        "winner_research_gate_passed": bool(winner["gate_passed"]),
        "winner_research_gate_reasons": winner["gate_reasons"],
        "shadow_start": int(timestamp[0]),
        "shadow_end": int(timestamp[-1]),
        "shadow_days_requested": args.shadow_days,
        "starting_capital_eur": args.starting_capital_eur,
        "primary_cost_bps": args.primary_cost_bps,
        "entry_margin_bps": margin_bps,
        "classification": cls,
        "payoff": payoff.to_dict(),
        "calibrator": calibrator.to_dict(),
        "cost_sensitivity": cost_results,
        "primary": primary,
        "gross_ending_capital_same_trades_eur": gross_ending,
        "gross_profit_same_trades_eur": gross_ending - args.starting_capital_eur,
        "completed_trades": len(trade_returns),
        "winning_trades": wins,
        "losing_trades": losses,
        "win_rate": wins / len(trade_returns) if trade_returns else 0.0,
        "buy_and_hold_ending_eur": bh_ending,
        "buy_and_hold_profit_eur": bh_ending - args.starting_capital_eur,
        "buy_and_hold_fees_eur": bh_fees,
        "ema_ending_eur": ema_ending,
        "ema_profit_eur": ema_ending - args.starting_capital_eur,
        "ema_fees_eur": ema_fees,
        "cash_ending_eur": args.starting_capital_eur,
        "eur_note": "EUR 1,000-equivalent notional applies BTC percentage returns; EUR/USDT FX variation is not modeled.",
        "final_training_best_epoch": int(np.argmin(history.history.get("val_loss", [0.0])) + 1),
    }
    out.mkdir(parents=True, exist_ok=True)
    model.save(out / "model.keras")
    write_json(out / "manifest.json", {
        "experiment": asdict(key),
        "feature_version": FEATURE_VERSION,
        "feature_names": dataset.feature_names,
        "candidate": candidate.to_dict(),
        "calibrator": calibrator.to_dict(),
        "payoff": payoff.to_dict(),
        "entry_margin_bps": margin_bps,
        "horizon_commitment_hours": key.horizon_hours,
    })
    write_json(out / "final_1000_eur.json", result)
    write_csv(out / "final_30d_equity.csv", equity_rows)
    return result


def _print_final_eur_report(result: dict[str, Any]) -> str:
    primary = result["primary"]
    metrics = primary["metrics"]
    start = result["starting_capital_eur"]
    lines = [
        "",
        "=" * 72,
        "FINAL UNTOUCHED 30-DAY EUR 1,000-EQUIVALENT SIMULATION",
        "=" * 72,
        f"Period: {_fmt_ts(result['shadow_start'])} -> {_fmt_ts(result['shadow_end'])}",
        f"Selected experiment: {result['winner']}",
        f"Research gate before shadow: {'PASS' if result['winner_research_gate_passed'] else 'FAIL'}",
        "",
        f"Starting capital:              EUR {start:,.2f}",
        f"Ending capital ({result['primary_cost_bps']:g} bps): EUR {primary['ending_capital_eur']:,.2f}",
        f"NET PROFIT:                    EUR {primary['profit_eur']:+,.2f}",
        f"NET RETURN:                    {metrics['cumulative_return']:+.2%}",
        f"Gross ending, same trades:     EUR {result['gross_ending_capital_same_trades_eur']:,.2f}",
        f"Estimated transaction costs:   EUR {primary['fees_eur']:,.2f}",
        "",
        f"Completed trades:              {result['completed_trades']}",
        f"Winning / losing:              {result['winning_trades']} / {result['losing_trades']}",
        f"Win rate:                      {result['win_rate']:.1%}",
        f"Sharpe:                        {metrics['sharpe']:+.3f}",
        f"Max drawdown:                  {metrics['max_drawdown']:.2%}",
        f"Shadow AUC:                    {result['classification']['auc']:.3f}",
        "",
        f"Buy & hold ending:             EUR {result['buy_and_hold_ending_eur']:,.2f} ({result['buy_and_hold_profit_eur']:+,.2f})",
        f"EMA baseline ending:           EUR {result['ema_ending_eur']:,.2f} ({result['ema_profit_eur']:+,.2f})",
        f"Cash ending:                   EUR {result['cash_ending_eur']:,.2f}",
        f"AI excess profit vs BTC:       EUR {primary['profit_eur'] - result['buy_and_hold_profit_eur']:+,.2f}",
        "",
        "Cost sensitivity:",
    ]
    for cost, values in sorted(result["cost_sensitivity"].items(), key=lambda item: float(item[0])):
        lines.append(f"  {float(cost):4.0f} bps -> EUR {values['ending_capital_eur']:,.2f} | profit {values['profit_eur']:+,.2f} | Sharpe {values['metrics']['sharpe']:+.3f}")
    lines += ["", result["eur_note"], "=" * 72]
    text = "\n".join(lines)
    print(text, flush=True)
    return text


def markdown_report(meta: dict[str, Any], summaries: list[dict[str, Any]], final_shadow: dict[str, Any]) -> str:
    lines = [
        "# AI Trading V3 — Calibrated Economic-Event Research",
        "",
        f"Run: `{meta['run_id']}`  ",
        f"Git: `{meta['git_commit']}`  ",
        f"Dataset SHA256: `{meta['dataset_sha256']}`  ",
        f"Feature version: `{meta['feature_version']}`  ",
        "",
        "Protocol: `TRAIN -> MODEL VALIDATION -> CALIBRATION -> POLICY VALIDATION -> UNTOUCHED TEST`, followed by one reserved final 30-day shadow month.",
        "",
        "## Research results (primary target first)",
        "",
        "| Target | Features | H | AUC | Median fold AUC | 168h CI | Sharpe@25 | Return@25 | Trades | Sharpe@30 | Gate |",
        "|---:|---|---:|---:|---:|---|---:|---:|---:|---:|---|",
    ]
    for s in summaries:
        p25 = s["strategy_by_cost_bps"]["25"]
        p30 = s["strategy_by_cost_bps"]["30"]
        ci = s["auc_ci_168h"]
        e = s["experiment"]
        lines.append(
            f"| {e['target_hurdle_bps']:.0f} bps | {e['feature_set']} | {e['horizon_hours']}h | "
            f"{s['overall_classification']['auc']:.3f} | {s['median_fold_auc']:.3f} | [{ci['lower']:.3f}, {ci['upper']:.3f}] | "
            f"{p25['sharpe']:+.3f} | {p25['cumulative_return']:+.2%} | {int(p25['trade_count'])} | {p30['sharpe']:+.3f} | {'PASS' if s['gate_passed'] else 'FAIL'} |"
        )
    lines += [
        "",
        "## Final untouched 30-day EUR 1,000-equivalent shadow",
        "",
        f"Starting capital: **EUR {final_shadow['starting_capital_eur']:,.2f}**  ",
        f"Ending capital: **EUR {final_shadow['primary']['ending_capital_eur']:,.2f}**  ",
        f"Net profit: **EUR {final_shadow['primary']['profit_eur']:+,.2f}**  ",
        f"Buy-and-hold ending: **EUR {final_shadow['buy_and_hold_ending_eur']:,.2f}**  ",
        f"EMA ending: **EUR {final_shadow['ema_ending_eur']:,.2f}**  ",
        "",
        final_shadow["eur_note"],
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V3 calibrated economic-event BTC research.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--horizons", type=parse_int_grid, default=parse_int_grid("1,3,6,12"))
    parser.add_argument("--feature-sets", nargs="+", choices=["technical", "technical_micro", "full_context"], default=["technical", "technical_micro", "full_context"])
    parser.add_argument("--primary-target-bps", type=float, default=50.0)
    parser.add_argument("--sensitivity-target-bps", type=float, default=25.0)
    parser.add_argument("--train-days", type=int, default=365)
    parser.add_argument("--model-validation-days", type=int, default=45)
    parser.add_argument("--calibration-days", type=int, default=60)
    parser.add_argument("--policy-validation-days", type=int, default=45)
    parser.add_argument("--test-days", type=int, default=60)
    parser.add_argument("--step-days", type=int, default=60)
    parser.add_argument("--shadow-days", type=int, default=30)
    parser.add_argument("--cost-bps", type=parse_float_grid, default=parse_float_grid("20,25,30,40"))
    parser.add_argument("--primary-cost-bps", type=float, default=25.0)
    parser.add_argument("--ev-margin-grid-bps", type=parse_float_grid, default=parse_float_grid("0,2.5,5,7.5,10,15,20,25"))
    parser.add_argument("--min-policy-trades", type=int, default=3)
    parser.add_argument("--trim-fraction", type=float, default=0.05)
    parser.add_argument("--ablation-trials", type=int, default=2)
    parser.add_argument("--full-trials", type=int, default=8)
    parser.add_argument("--sensitivity-trials", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--final-epochs", type=int, default=100)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--starting-capital-eur", type=float, default=1000.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epoch-verbose", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.dataset.is_file():
        raise FileNotFoundError(args.dataset)
    if args.primary_cost_bps not in args.cost_bps:
        raise ValueError("primary cost must be included in cost-bps")
    if 30.0 not in args.cost_bps:
        raise ValueError("30 bps must be included for robustness gate")
    if args.primary_target_bps <= 0.0 or args.sensitivity_target_bps <= 0.0:
        raise ValueError("event hurdles must be positive")
    if not 0.0 <= args.trim_fraction < 0.5:
        raise ValueError("trim-fraction must be in [0,0.5)")
    if args.starting_capital_eur <= 0.0:
        raise ValueError("starting capital must be positive")


def main() -> int:
    args = parse_args()
    validate_args(args)
    started = time.time()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    raw_rows = read_jsonl(args.dataset)
    if not raw_rows:
        raise ValueError("dataset is empty")
    first_date = datetime.fromtimestamp(int(raw_rows[0]["timestamp"]) / 1000.0, timezone.utc).date()
    if first_date.year > 2020:
        print(f"WARNING: dataset starts {first_date}; V3 is designed for a 2020-present regime-diverse rebuild.", flush=True)

    meta = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_head(),
        "dataset": str(args.dataset),
        "dataset_sha256": sha256_file(args.dataset),
        "raw_rows": len(raw_rows),
        "feature_version": FEATURE_VERSION,
        "horizons": args.horizons,
        "feature_sets": args.feature_sets,
        "primary_target_bps": args.primary_target_bps,
        "sensitivity_target_bps": args.sensitivity_target_bps,
        "cost_bps": args.cost_bps,
        "split_days": {
            "train": args.train_days,
            "model_validation": args.model_validation_days,
            "calibration": args.calibration_days,
            "policy_validation": args.policy_validation_days,
            "test": args.test_days,
            "shadow": args.shadow_days,
        },
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    write_json(run_dir / "metadata.json", meta)

    summaries: list[dict[str, Any]] = []
    base_by_horizon: dict[int, ResearchDataset] = {}
    for horizon in args.horizons:
        base_by_horizon[horizon] = build_research_dataset(raw_rows, horizon_hours=horizon)

    for horizon in args.horizons:
        for feature_set in args.feature_sets:
            key = ExperimentKey(args.primary_target_bps, feature_set, horizon)
            trials = args.full_trials if feature_set == "full_context" else args.ablation_trials
            summary = run_experiment(base_by_horizon[horizon], key, args, run_dir / key.slug, trial_count=trials)
            summaries.append(summary)

    for horizon in args.horizons:
        key = ExperimentKey(args.sensitivity_target_bps, "full_context", horizon)
        summary = run_experiment(base_by_horizon[horizon], key, args, run_dir / key.slug, trial_count=args.sensitivity_trials)
        summaries.append(summary)

    primary_summaries = [s for s in summaries if float(s["experiment"]["target_hurdle_bps"]) == args.primary_target_bps]
    primary_summaries.sort(
        key=lambda s: (
            bool(s["gate_passed"]),
            float(s["strategy_by_cost_bps"][f"{args.primary_cost_bps:g}"]["sharpe"]),
            float(s["auc_ci_168h"]["lower"]),
            float(s["overall_classification"]["auc"]),
            float(s["strategy_by_cost_bps"][f"{args.primary_cost_bps:g}"]["cumulative_return"]),
        ),
        reverse=True,
    )
    winner = primary_summaries[0]
    winner_h = int(winner["experiment"]["horizon_hours"])
    final_shadow = run_final_shadow(base_by_horizon[winner_h], winner, args, run_dir / "final_shadow")
    final_text = _print_final_eur_report(final_shadow)
    (run_dir / "FINAL_1000_EUR_REPORT.txt").write_text(final_text + "\n", encoding="utf-8")

    summaries.sort(key=lambda s: (float(s["experiment"]["target_hurdle_bps"]) != args.primary_target_bps, -float(s["strategy_by_cost_bps"][f"{args.primary_cost_bps:g}"]["sharpe"])))
    write_json(run_dir / "all_results.json", summaries)
    write_json(run_dir / "final_shadow.json", final_shadow)
    (run_dir / "REPORT.md").write_text(markdown_report(meta, summaries, final_shadow), encoding="utf-8")

    comparison_rows: list[dict[str, Any]] = []
    for s in summaries:
        e = s["experiment"]
        p25 = s["strategy_by_cost_bps"]["25"]
        p30 = s["strategy_by_cost_bps"]["30"]
        comparison_rows.append({
            **e,
            "feature_count": s["feature_count"],
            "auc": s["overall_classification"]["auc"],
            "median_fold_auc": s["median_fold_auc"],
            "auc_24h_lower": s["auc_ci_24h"]["lower"],
            "auc_168h_lower": s["auc_ci_168h"]["lower"],
            "sharpe_25bps": p25["sharpe"],
            "return_25bps": p25["cumulative_return"],
            "trades_25bps": p25["trade_count"],
            "max_dd_25bps": p25["max_drawdown"],
            "sharpe_30bps": p30["sharpe"],
            "return_30bps": p30["cumulative_return"],
            "gate_passed": s["gate_passed"],
            "gate_reasons": "; ".join(s["gate_reasons"]),
        })
    write_csv(run_dir / "model_comparison.csv", comparison_rows)

    meta["duration_seconds"] = time.time() - started
    meta["winner"] = winner["experiment"]
    meta["winner_gate_passed"] = winner["gate_passed"]
    write_json(run_dir / "metadata.json", meta)

    report_zip = args.output_root / f"{run_id}_research_report.zip"
    with zipfile.ZipFile(report_zip, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in run_dir.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(run_dir.parent))
    deployment_zip = args.output_root / f"{run_id}_research_deployment.zip"
    with zipfile.ZipFile(deployment_zip, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in (run_dir / "final_shadow").rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(run_dir))
        for name in ("REPORT.md", "FINAL_1000_EUR_REPORT.txt", "model_comparison.csv", "metadata.json", "final_shadow.json"):
            path = run_dir / name
            if path.is_file():
                archive.write(path, path.relative_to(run_dir))

    print("\nV3 RESEARCH COMPLETE", flush=True)
    print(f"report={run_dir / 'REPORT.md'}", flush=True)
    print(f"eur1000_report={run_dir / 'FINAL_1000_EUR_REPORT.txt'}", flush=True)
    print(f"research_bundle={report_zip}", flush=True)
    print(f"deployment_bundle={deployment_zip}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
