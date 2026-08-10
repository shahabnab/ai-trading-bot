#!/usr/bin/env python3
"""Train all current ML approaches with time-series-safe validation.

Run from the repository root.

Protocol per fold:
    TRAIN -> EARLY-STOP VALIDATION -> THRESHOLD VALIDATION -> UNTOUCHED TEST

Random K-fold is intentionally not used for market time series.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import statistics
import subprocess
import time
import zipfile
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from backend.ml.evaluation import (
    WalkForwardConfig,
    _performance_metrics,
    buy_and_hold_baseline,
    long_only_cost_aware_backtest,
    make_walk_forward_folds,
    moving_average_baseline,
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
DEFAULT_OUTPUT = Path("artifacts/ml/server_training")


def _jsonable(value: object) -> object:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_jsonable),
        encoding="utf-8",
    )


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
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def parse_grid(value: str) -> list[float]:
    values = sorted({float(x.strip()) for x in value.split(",") if x.strip()})
    if not values or values[0] < 0:
        raise argparse.ArgumentTypeError("threshold grid must be non-negative")
    return values


def split_validation(indices: np.ndarray, early_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) < 4:
        raise ValueError("validation window too small to split")
    cut = max(1, min(int(round(len(indices) * early_fraction)), len(indices) - 1))
    return indices[:cut], indices[cut:]


def choose_lambda(
    predicted: np.ndarray,
    actual_simple: np.ndarray,
    *,
    cost_rate: float,
    grid: list[float],
    min_trades: int,
) -> tuple[float, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for execution_lambda in grid:
        metrics, _, _, _ = long_only_cost_aware_backtest(
            predicted,
            actual_simple,
            cost_rate=cost_rate,
            execution_lambda=execution_lambda,
        )
        rows.append({"execution_lambda": execution_lambda, **metrics})

    eligible = [row for row in rows if int(row["trade_count"]) >= min_trades]
    pool = eligible or rows
    best = max(
        pool,
        key=lambda row: (
            float(row["sharpe"]),
            float(row["sortino"]),
            float(row["max_drawdown"]),
            -float(row["turnover"]),
        ),
    )
    return float(best["execution_lambda"]), rows


def aggregate_strategy(rows: list[dict[str, Any]], prefix: str) -> dict[str, float]:
    return _performance_metrics(
        np.asarray([row[f"{prefix}_return"] for row in rows], dtype=np.float64),
        np.asarray([row[f"{prefix}_position"] for row in rows], dtype=np.float64),
        np.asarray([row[f"{prefix}_turnover"] for row in rows], dtype=np.float64),
    )


def train_xgboost(
    dataset: Any,
    folds: list[Any],
    out: Path,
    args: argparse.Namespace,
    cost_rate: float,
) -> dict[str, Any]:
    import xgboost as xgb
    from backend.ml.xgboost_core import DEFAULT_XGB_PARAMS

    params = dict(DEFAULT_XGB_PARAMS)
    params["seed"] = args.seed
    if args.xgb_device == "cuda":
        params["device"] = "cuda"

    model_dir = out / "fold_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    predictions: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    best_rounds: list[int] = []
    selected_lambdas: list[float] = []

    for fold in folds:
        early_idx, threshold_idx = split_validation(
            fold.validation_indices, args.early_stop_fraction
        )
        train_idx = fold.train_indices
        test_idx = fold.test_indices

        dtrain = xgb.DMatrix(
            dataset.X[train_idx],
            label=dataset.y_log_return[train_idx],
            feature_names=dataset.feature_names,
        )
        dearly = xgb.DMatrix(
            dataset.X[early_idx],
            label=dataset.y_log_return[early_idx],
            feature_names=dataset.feature_names,
        )
        dthreshold = xgb.DMatrix(
            dataset.X[threshold_idx], feature_names=dataset.feature_names
        )
        dtest = xgb.DMatrix(dataset.X[test_idx], feature_names=dataset.feature_names)

        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=args.xgb_rounds,
            evals=[(dearly, "early_stop")],
            callbacks=[
                xgb.callback.EarlyStopping(
                    rounds=args.xgb_early_stopping, save_best=True
                )
            ],
            verbose_eval=False,
        )
        booster.save_model(model_dir / f"fold_{fold.fold:02d}.json")
        best_round = int(booster.best_iteration) + 1
        best_rounds.append(best_round)

        validation_pred = booster.predict(dthreshold).astype(np.float64)
        selected_lambda, sweep = choose_lambda(
            validation_pred,
            dataset.y_simple_return[threshold_idx],
            cost_rate=cost_rate,
            grid=args.threshold_grid,
            min_trades=args.min_threshold_trades,
        )
        selected_lambdas.append(selected_lambda)
        threshold_rows.extend(
            {"model": "xgboost", "fold": fold.fold, **row} for row in sweep
        )

        test_pred = booster.predict(dtest).astype(np.float64)
        actual_simple = dataset.y_simple_return[test_idx].astype(np.float64)
        actual_log = dataset.y_log_return[test_idx].astype(np.float64)
        tuned, tuned_ret, tuned_pos, tuned_turn = long_only_cost_aware_backtest(
            test_pred,
            actual_simple,
            cost_rate=cost_rate,
            execution_lambda=selected_lambda,
        )
        default, default_ret, default_pos, default_turn = long_only_cost_aware_backtest(
            test_pred,
            actual_simple,
            cost_rate=cost_rate,
            execution_lambda=2.0,
        )
        forecast = regression_metrics(actual_log, test_pred)

        fold_rows.append(
            {
                "model": "xgboost",
                "fold": fold.fold,
                "best_rounds": best_round,
                "selected_execution_lambda": selected_lambda,
                **{f"forecast_{k}": v for k, v in forecast.items()},
                **{f"test_tuned_{k}": v for k, v in tuned.items()},
                **{f"test_lambda2_{k}": v for k, v in default.items()},
            }
        )
        for i, dataset_idx in enumerate(test_idx):
            predictions.append(
                {
                    "model": "xgboost",
                    "fold": fold.fold,
                    "timestamp": int(dataset.timestamps[dataset_idx]),
                    "actual_log_return_1h": float(actual_log[i]),
                    "actual_simple_return_1h": float(actual_simple[i]),
                    "predicted_log_return_1h": float(test_pred[i]),
                    "selected_execution_lambda": selected_lambda,
                    "tuned_return": float(tuned_ret[i]),
                    "tuned_position": float(tuned_pos[i]),
                    "tuned_turnover": float(tuned_turn[i]),
                    "lambda2_return": float(default_ret[i]),
                    "lambda2_position": float(default_pos[i]),
                    "lambda2_turnover": float(default_turn[i]),
                }
            )
        print(
            f"[XGBoost] fold {fold.fold:02d}/{len(folds):02d} "
            f"rounds={best_round} lambda={selected_lambda:g} "
            f"test_sharpe={tuned['sharpe']:.3f}"
        )

    predictions.sort(key=lambda row: int(row["timestamp"]))
    forecast = regression_metrics(
        np.asarray([row["actual_log_return_1h"] for row in predictions]),
        np.asarray([row["predicted_log_return_1h"] for row in predictions]),
    )
    tuned = aggregate_strategy(predictions, "tuned")
    default = aggregate_strategy(predictions, "lambda2")

    final_rounds = max(1, int(round(statistics.median(best_rounds))))
    final_lambda = float(statistics.median(selected_lambdas))
    deployment = out / "deployment"
    deployment.mkdir(parents=True, exist_ok=True)
    dall = xgb.DMatrix(
        dataset.X,
        label=dataset.y_log_return,
        feature_names=dataset.feature_names,
    )
    final_model = xgb.train(params, dall, num_boost_round=final_rounds, verbose_eval=False)
    final_model.save_model(deployment / "model.json")
    write_json(
        deployment / "manifest.json",
        {
            "model_family": "xgboost",
            "feature_version": FEATURE_VERSION,
            "feature_names": dataset.feature_names,
            "final_training_rows": dataset.size,
            "final_num_boost_round": final_rounds,
            "recommended_execution_lambda": final_lambda,
            "threshold_selection": "median of validation-only fold selections",
            "xgboost_params": params,
        },
    )
    write_jsonl(out / "predictions.jsonl", predictions)
    write_csv(out / "folds.csv", fold_rows)
    write_csv(out / "threshold_search.csv", threshold_rows)
    return {
        "model_family": "xgboost",
        "forecast_metrics": forecast,
        "tuned_strategy": tuned,
        "lambda2_strategy": default,
        "recommended_execution_lambda": final_lambda,
        "fold_selected_lambdas": selected_lambdas,
        "final_model": str(deployment / "model.json"),
    }


def scaler_payload(feature_names: list[str], sequence_length: int, x_stats: Any, y_stats: Any) -> dict[str, Any]:
    return {
        "feature_names": feature_names,
        "sequence_length": sequence_length,
        "feature_mean": x_stats.mean,
        "feature_scale": x_stats.scale,
        "target_mean": y_stats.mean,
        "target_scale": y_stats.scale,
    }


def train_lstm(
    dataset: Any,
    folds: list[Any],
    out: Path,
    args: argparse.Namespace,
    cost_rate: float,
) -> dict[str, Any]:
    import tensorflow as tf
    from backend.ml.lstm_core import _build_model

    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

    model_dir = out / "fold_models"
    scaler_dir = out / "fold_scalers"
    model_dir.mkdir(parents=True, exist_ok=True)
    scaler_dir.mkdir(parents=True, exist_ok=True)
    predictions: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    best_epochs: list[int] = []
    selected_lambdas: list[float] = []

    for fold in folds:
        early_idx, threshold_idx = split_validation(
            fold.validation_indices, args.early_stop_fraction
        )
        train_idx = fold.train_indices
        test_idx = fold.test_indices
        x_stats = fit_standardizer(dataset.X[train_idx])
        y_stats = fit_standardizer(dataset.y_log_return[train_idx])
        x_scaled = standardize(dataset.X, x_stats)
        y_scaled = standardize(dataset.y_log_return, y_stats)

        train_x, train_y, _ = make_sequence_batch(
            x_scaled,
            y_scaled,
            dataset.timestamps,
            train_idx,
            sequence_length=args.sequence_length,
            min_context_index=int(train_idx[0]),
        )
        early_x, early_y, _ = make_sequence_batch(
            x_scaled,
            y_scaled,
            dataset.timestamps,
            early_idx,
            sequence_length=args.sequence_length,
        )
        threshold_x, _, threshold_targets = make_sequence_batch(
            x_scaled,
            y_scaled,
            dataset.timestamps,
            threshold_idx,
            sequence_length=args.sequence_length,
        )
        test_x, _, test_targets = make_sequence_batch(
            x_scaled,
            y_scaled,
            dataset.timestamps,
            test_idx,
            sequence_length=args.sequence_length,
        )
        if min(len(train_x), len(early_x), len(threshold_x), len(test_x)) == 0:
            raise ValueError(f"empty LSTM sequence block in fold {fold.fold}")

        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(args.seed + fold.fold)
        model = _build_model(
            tf,
            sequence_length=args.sequence_length,
            feature_count=len(dataset.feature_names),
            lstm_units=args.lstm_units,
            dense_units=args.dense_units,
            dropout=args.dropout,
            learning_rate=args.learning_rate,
            clipnorm=args.clipnorm,
        )
        history = model.fit(
            train_x,
            train_y,
            validation_data=(early_x, early_y),
            epochs=args.epochs,
            batch_size=args.batch_size,
            shuffle=False,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    mode="min",
                    patience=args.early_stopping_patience,
                    restore_best_weights=True,
                    verbose=0,
                )
            ],
            verbose=0,
        )
        val_losses = [float(x) for x in history.history.get("val_loss", [])]
        best_epoch = int(np.argmin(val_losses) + 1) if val_losses else len(history.epoch)
        best_epochs.append(best_epoch)
        model.save(model_dir / f"fold_{fold.fold:02d}.keras")
        write_json(
            scaler_dir / f"fold_{fold.fold:02d}.json",
            scaler_payload(dataset.feature_names, args.sequence_length, x_stats, y_stats),
        )

        validation_scaled = model.predict(
            threshold_x, batch_size=args.batch_size, verbose=0
        ).reshape(-1)
        validation_pred = inverse_standardize(validation_scaled, y_stats).astype(np.float64)
        selected_lambda, sweep = choose_lambda(
            validation_pred,
            dataset.y_simple_return[threshold_targets],
            cost_rate=cost_rate,
            grid=args.threshold_grid,
            min_trades=args.min_threshold_trades,
        )
        selected_lambdas.append(selected_lambda)
        threshold_rows.extend(
            {"model": "lstm", "fold": fold.fold, **row} for row in sweep
        )

        test_scaled = model.predict(test_x, batch_size=args.batch_size, verbose=0).reshape(-1)
        test_pred = inverse_standardize(test_scaled, y_stats).astype(np.float64)
        actual_simple = dataset.y_simple_return[test_targets].astype(np.float64)
        actual_log = dataset.y_log_return[test_targets].astype(np.float64)
        tuned, tuned_ret, tuned_pos, tuned_turn = long_only_cost_aware_backtest(
            test_pred,
            actual_simple,
            cost_rate=cost_rate,
            execution_lambda=selected_lambda,
        )
        default, default_ret, default_pos, default_turn = long_only_cost_aware_backtest(
            test_pred,
            actual_simple,
            cost_rate=cost_rate,
            execution_lambda=2.0,
        )
        forecast = regression_metrics(actual_log, test_pred)

        fold_rows.append(
            {
                "model": "lstm",
                "fold": fold.fold,
                "best_epoch": best_epoch,
                "selected_execution_lambda": selected_lambda,
                **{f"forecast_{k}": v for k, v in forecast.items()},
                **{f"test_tuned_{k}": v for k, v in tuned.items()},
                **{f"test_lambda2_{k}": v for k, v in default.items()},
            }
        )
        for i, dataset_idx in enumerate(test_targets):
            predictions.append(
                {
                    "model": "lstm",
                    "fold": fold.fold,
                    "timestamp": int(dataset.timestamps[dataset_idx]),
                    "actual_log_return_1h": float(actual_log[i]),
                    "actual_simple_return_1h": float(actual_simple[i]),
                    "predicted_log_return_1h": float(test_pred[i]),
                    "selected_execution_lambda": selected_lambda,
                    "tuned_return": float(tuned_ret[i]),
                    "tuned_position": float(tuned_pos[i]),
                    "tuned_turnover": float(tuned_turn[i]),
                    "lambda2_return": float(default_ret[i]),
                    "lambda2_position": float(default_pos[i]),
                    "lambda2_turnover": float(default_turn[i]),
                }
            )
        print(
            f"[LSTM] fold {fold.fold:02d}/{len(folds):02d} "
            f"best_epoch={best_epoch} lambda={selected_lambda:g} "
            f"test_sharpe={tuned['sharpe']:.3f}"
        )

    predictions.sort(key=lambda row: int(row["timestamp"]))
    forecast = regression_metrics(
        np.asarray([row["actual_log_return_1h"] for row in predictions]),
        np.asarray([row["predicted_log_return_1h"] for row in predictions]),
    )
    tuned = aggregate_strategy(predictions, "tuned")
    default = aggregate_strategy(predictions, "lambda2")

    final_epochs = max(1, int(round(statistics.median(best_epochs))))
    final_lambda = float(statistics.median(selected_lambdas))
    deployment = out / "deployment"
    deployment.mkdir(parents=True, exist_ok=True)
    x_stats = fit_standardizer(dataset.X)
    y_stats = fit_standardizer(dataset.y_log_return)
    x_scaled = standardize(dataset.X, x_stats)
    y_scaled = standardize(dataset.y_log_return, y_stats)
    all_idx = np.arange(dataset.size, dtype=np.int64)
    all_x, all_y, _ = make_sequence_batch(
        x_scaled,
        y_scaled,
        dataset.timestamps,
        all_idx,
        sequence_length=args.sequence_length,
        min_context_index=0,
    )
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(args.seed)
    final_model = _build_model(
        tf,
        sequence_length=args.sequence_length,
        feature_count=len(dataset.feature_names),
        lstm_units=args.lstm_units,
        dense_units=args.dense_units,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        clipnorm=args.clipnorm,
    )
    final_model.fit(
        all_x,
        all_y,
        epochs=final_epochs,
        batch_size=args.batch_size,
        shuffle=False,
        verbose=0,
    )
    final_model.save(deployment / "model.keras")
    write_json(
        deployment / "scaler.json",
        scaler_payload(dataset.feature_names, args.sequence_length, x_stats, y_stats),
    )
    write_json(
        deployment / "manifest.json",
        {
            "model_family": "lstm",
            "feature_version": FEATURE_VERSION,
            "feature_names": dataset.feature_names,
            "sequence_length": args.sequence_length,
            "lstm_units": args.lstm_units,
            "dense_units": args.dense_units,
            "dropout": args.dropout,
            "final_epochs": final_epochs,
            "recommended_execution_lambda": final_lambda,
            "threshold_selection": "median of validation-only fold selections",
            "tensorflow_version": tf.__version__,
            "tensorflow_gpu_count": len(tf.config.list_physical_devices("GPU")),
        },
    )
    write_jsonl(out / "predictions.jsonl", predictions)
    write_csv(out / "folds.csv", fold_rows)
    write_csv(out / "threshold_search.csv", threshold_rows)
    return {
        "model_family": "lstm",
        "forecast_metrics": forecast,
        "tuned_strategy": tuned,
        "lambda2_strategy": default,
        "recommended_execution_lambda": final_lambda,
        "fold_selected_lambdas": selected_lambdas,
        "final_model": str(deployment / "model.keras"),
    }


def comparison_row(feature_set: str, result: dict[str, Any]) -> dict[str, Any]:
    forecast = result["forecast_metrics"]
    strategy = result["tuned_strategy"]
    return {
        "model": result["model_family"],
        "feature_set": feature_set,
        "direction_accuracy": forecast["direction_accuracy"],
        "rmse_log_return": forecast["rmse_log_return"],
        "mae_log_return": forecast["mae_log_return"],
        "correlation": forecast["correlation"],
        "tuned_sharpe": strategy["sharpe"],
        "tuned_sortino": strategy["sortino"],
        "tuned_max_drawdown": strategy["max_drawdown"],
        "tuned_cumulative_return": strategy["cumulative_return"],
        "tuned_trade_count": strategy["trade_count"],
        "lambda2_sharpe": result["lambda2_strategy"]["sharpe"],
        "recommended_execution_lambda": result["recommended_execution_lambda"],
        "final_model": result["final_model"],
    }


def markdown_report(meta: dict[str, Any], rows: list[dict[str, Any]], baselines: dict[str, Any]) -> str:
    lines = [
        "# AI Trading Server Training Report",
        "",
        f"Run: `{meta['run_id']}`  ",
        f"Git: `{meta['git_commit']}`  ",
        f"Dataset SHA256: `{meta['dataset_sha256']}`  ",
        f"Walk-forward folds: **{meta['fold_count']}**  ",
        f"Trading-cost assumption: **{meta['total_cost_bps']:.2f} bps** per position change",
        "",
        "| Model | Features | Direction | Tuned Sharpe | Sortino | Max DD | Cum Return | Trades | Lambda |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['feature_set']} | {row['direction_accuracy']:.4f} | "
            f"{row['tuned_sharpe']:.3f} | {row['tuned_sortino']:.3f} | "
            f"{row['tuned_max_drawdown']:.2%} | {row['tuned_cumulative_return']:.2%} | "
            f"{int(row['tuned_trade_count'])} | {row['recommended_execution_lambda']:.3g} |"
        )
    lines += [
        "",
        "## Baselines",
        f"- Moving-average Sharpe: **{baselines['moving_average']['sharpe']:.3f}**",
        f"- Buy-and-hold Sharpe: **{baselines['buy_and_hold']['sharpe']:.3f}**",
        "",
        "## Validation protocol",
        "`TRAIN -> EARLY STOP -> THRESHOLD TUNE -> UNTOUCHED TEST`",
        "",
        "The execution threshold is chosen only on validation data. `PAPER_MIN_CONFIDENCE` is not tuned because the current regressors do not yet output calibrated confidence probabilities.",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--models", nargs="+", choices=["xgboost", "lstm"], default=["xgboost", "lstm"])
    parser.add_argument("--feature-sets", nargs="+", choices=["base", "sentiment"], default=["base"])
    parser.add_argument("--train-days", type=int, default=365)
    parser.add_argument("--validation-days", type=int, default=60)
    parser.add_argument("--test-days", type=int, default=60)
    parser.add_argument("--step-days", type=int, default=60)
    parser.add_argument("--early-stop-fraction", type=float, default=0.5)
    parser.add_argument("--fee-bps", type=float, default=20.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--spread-bps", type=float, default=0.0)
    parser.add_argument("--threshold-grid", type=parse_grid, default=parse_grid("0,0.25,0.5,0.75,1,1.25,1.5,2,2.5,3,4,5,6,8,10"))
    parser.add_argument("--min-threshold-trades", type=int, default=5)
    parser.add_argument("--xgb-device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--xgb-rounds", type=int, default=2000)
    parser.add_argument("--xgb-early-stopping", type=int, default=50)
    parser.add_argument("--sequence-length", type=int, default=48)
    parser.add_argument("--lstm-units", type=int, default=64)
    parser.add_argument("--dense-units", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--clipnorm", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--early-stopping-patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.dataset.is_file():
        raise FileNotFoundError(args.dataset)
    started = time.time()
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    raw_rows = read_jsonl(args.dataset)
    base = build_feature_dataset(raw_rows, include_sentiment=False)
    config = WalkForwardConfig(
        train_days=args.train_days,
        validation_days=args.validation_days,
        test_days=args.test_days,
        step_days=args.step_days,
    )
    folds = make_walk_forward_folds(base.timestamps, config)
    if not folds:
        raise ValueError("no walk-forward folds produced")

    cost_rate = (args.fee_bps + args.slippage_bps + args.spread_bps) / 10000.0
    test_idx = np.concatenate([fold.test_indices for fold in folds])
    baselines = {
        "moving_average": moving_average_baseline(
            base.y_simple_return[test_idx],
            base.closes[test_idx],
            base.ema20[test_idx],
            base.ema50[test_idx],
            base.ema200[test_idx],
            cost_rate=cost_rate,
        ),
        "buy_and_hold": buy_and_hold_baseline(base.y_simple_return[test_idx], cost_rate=cost_rate),
    }
    meta = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": git_head(),
        "dataset_path": str(args.dataset),
        "dataset_sha256": sha256_file(args.dataset),
        "raw_jsonl_rows": len(raw_rows),
        "base_usable_rows": base.size,
        "feature_version": FEATURE_VERSION,
        "walk_forward": asdict(config),
        "fold_count": len(folds),
        "fee_bps": args.fee_bps,
        "slippage_bps": args.slippage_bps,
        "spread_bps": args.spread_bps,
        "total_cost_bps": args.fee_bps + args.slippage_bps + args.spread_bps,
        "threshold_grid": args.threshold_grid,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    write_json(run_dir / "metadata.json", meta)
    write_json(run_dir / "baselines.json", baselines)

    results: dict[str, Any] = {}
    comparison: list[dict[str, Any]] = []
    for feature_set in args.feature_sets:
        dataset = build_feature_dataset(raw_rows, include_sentiment=(feature_set == "sentiment"))
        if not np.array_equal(dataset.timestamps, base.timestamps):
            raise RuntimeError("feature variants produced different timestamps")
        for model_name in args.models:
            out = run_dir / f"{model_name}_{feature_set}"
            out.mkdir(parents=True, exist_ok=True)
            write_json(out / "feature_names.json", {"feature_names": dataset.feature_names})
            if model_name == "xgboost":
                result = train_xgboost(dataset, folds, out, args, cost_rate)
            else:
                result = train_lstm(dataset, folds, out, args, cost_rate)
            results[f"{model_name}_{feature_set}"] = result
            write_json(out / "summary.json", result)
            comparison.append(comparison_row(feature_set, result))

    comparison.sort(key=lambda row: float(row["tuned_sharpe"]), reverse=True)
    write_json(run_dir / "all_results.json", results)
    write_csv(run_dir / "model_comparison.csv", comparison)
    (run_dir / "REPORT.md").write_text(
        markdown_report(meta, comparison, baselines), encoding="utf-8"
    )
    meta["duration_seconds"] = time.time() - started
    write_json(run_dir / "metadata.json", meta)

    training_zip = args.output_root / f"{run_id}_training_report.zip"
    with zipfile.ZipFile(training_zip, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in run_dir.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(run_dir.parent))

    deployment_zip = args.output_root / f"{run_id}_deployment_models.zip"
    with zipfile.ZipFile(deployment_zip, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in ("REPORT.md", "model_comparison.csv", "metadata.json", "baselines.json", "all_results.json"):
            path = run_dir / name
            if path.is_file():
                archive.write(path, path.relative_to(run_dir))
        for variant in run_dir.iterdir():
            deployment = variant / "deployment"
            if deployment.is_dir():
                for path in deployment.rglob("*"):
                    if path.is_file():
                        archive.write(path, path.relative_to(run_dir))

    print("TRAINING COMPLETE")
    print(f"report={run_dir / 'REPORT.md'}")
    print(f"training_bundle={training_zip}")
    print(f"deployment_bundle={deployment_zip}")
    if comparison:
        best = comparison[0]
        print(
            "highest historical WFO test Sharpe candidate (not proof of future profitability): "
            f"{best['model']}/{best['feature_set']} Sharpe={best['tuned_sharpe']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
