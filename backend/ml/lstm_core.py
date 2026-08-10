from __future__ import annotations

import argparse
import json
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from backend.ml.evaluation import (
    WalkForwardConfig,
    buy_and_hold_baseline,
    long_only_cost_aware_backtest,
    make_walk_forward_folds,
    moving_average_baseline,
    regression_metrics,
)
from backend.ml.features import FEATURE_VERSION, build_feature_dataset, read_jsonl
from backend.ml.persistence import RunLogger
from backend.ml.sequences import (
    fit_standardizer,
    inverse_standardize,
    make_sequence_batch,
    standardize,
)


DEFAULT_DATASET = Path("data/processed/training/btc_hourly.jsonl")
DEFAULT_RUN_ROOT = Path("artifacts/ml/runs")
DATASET_VERSION = "btc-hourly-jsonl-v1"


def _jsonable(value: object) -> object:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_jsonable),
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=_jsonable) + "\n")


def _build_model(
    tf: Any,
    *,
    sequence_length: int,
    feature_count: int,
    lstm_units: int,
    dense_units: int,
    dropout: float,
    learning_rate: float,
    clipnorm: float,
) -> Any:
    inputs = tf.keras.Input(shape=(sequence_length, feature_count), name="features")
    # Keep recurrent dropout at zero so TensorFlow can use its optimized LSTM
    # implementation when compatible hardware is available. Regularization is
    # applied after the recurrent layer instead.
    x = tf.keras.layers.LSTM(lstm_units, name="lstm")(inputs)
    if dropout > 0.0:
        x = tf.keras.layers.Dropout(dropout, name="lstm_dropout")(x)
    x = tf.keras.layers.Dense(dense_units, activation="relu", name="dense")(x)
    if dropout > 0.0:
        x = tf.keras.layers.Dropout(dropout, name="dense_dropout")(x)
    outputs = tf.keras.layers.Dense(1, name="next_hour_log_return_scaled")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="btc_lstm_return_forecaster")
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=clipnorm)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    return model


def train_lstm_walk_forward(
    *,
    dataset_path: Path = DEFAULT_DATASET,
    run_root: Path = DEFAULT_RUN_ROOT,
    walk_forward: WalkForwardConfig = WalkForwardConfig(),
    fee_bps: float = 5.0,
    slippage_bps: float = 3.0,
    spread_bps: float = 2.0,
    execution_lambda: float = 2.0,
    include_sentiment: bool = False,
    sequence_length: int = 48,
    lstm_units: int = 64,
    dense_units: int = 32,
    dropout: float = 0.20,
    learning_rate: float = 1e-3,
    clipnorm: float = 1.0,
    epochs: int = 50,
    batch_size: int = 64,
    early_stopping_patience: int = 7,
    seed: int = 42,
) -> dict[str, Any]:
    import tensorflow as tf

    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Training dataset not found: {dataset_path}. "
            "Run `python -m backend.data_collection.historical_dataset --months 6 --skip-news` first."
        )
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if lstm_units <= 0 or dense_units <= 0:
        raise ValueError("lstm_units and dense_units must be positive")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    if learning_rate <= 0.0 or clipnorm <= 0.0:
        raise ValueError("learning_rate and clipnorm must be positive")
    if epochs <= 0 or batch_size <= 0 or early_stopping_patience < 0:
        raise ValueError("epochs and batch_size must be positive; patience must be non-negative")

    rows = read_jsonl(dataset_path)
    dataset = build_feature_dataset(rows, include_sentiment=include_sentiment)
    folds = make_walk_forward_folds(dataset.timestamps, walk_forward)
    if not folds:
        raise ValueError(
            "Not enough usable history for the requested walk-forward windows. "
            "For roughly six months of hourly data, start with "
            "--train-days 90 --validation-days 30 --test-days 30 --step-days 30."
        )

    devices = tf.config.list_physical_devices()
    gpu_devices = tf.config.list_physical_devices("GPU")
    logger = RunLogger(run_root)
    run = logger.start_run(
        purpose="walk-forward BTC next-hour return forecasting",
        model_family="lstm",
        dataset_version=DATASET_VERSION,
        feature_version=FEATURE_VERSION,
        config={
            "dataset_path": str(dataset_path),
            "walk_forward": walk_forward.__dict__,
            "fee_bps": fee_bps,
            "slippage_bps": slippage_bps,
            "spread_bps": spread_bps,
            "total_cost_bps": fee_bps + slippage_bps + spread_bps,
            "execution_lambda": execution_lambda,
            "include_sentiment": include_sentiment,
            "sequence_length": sequence_length,
            "lstm_units": lstm_units,
            "dense_units": dense_units,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "clipnorm": clipnorm,
            "epochs": epochs,
            "batch_size": batch_size,
            "early_stopping_patience": early_stopping_patience,
            "seed": seed,
        },
        machine={
            "python": platform.python_version(),
            "platform": platform.platform(),
            "tensorflow": tf.__version__,
            "tensorflow_devices": [device.device_type for device in devices],
            "tensorflow_gpu_count": len(gpu_devices),
        },
    )
    run_id = str(run["run_id"])
    run_dir = run_root / run_id
    model_dir = run_dir / "models"
    scaler_dir = run_dir / "scalers"
    model_dir.mkdir(parents=True, exist_ok=True)
    scaler_dir.mkdir(parents=True, exist_ok=True)

    prediction_rows: list[dict[str, Any]] = []
    fold_summaries: list[dict[str, Any]] = []

    try:
        for fold in folds:
            train_idx = fold.train_indices
            val_idx = fold.validation_indices
            test_idx = fold.test_indices

            # Fit all normalization parameters strictly on this fold's training
            # region. Validation and test statistics never affect preprocessing.
            x_stats = fit_standardizer(dataset.X[train_idx])
            y_stats = fit_standardizer(dataset.y_log_return[train_idx])
            X_scaled = standardize(dataset.X, x_stats)
            y_scaled = standardize(dataset.y_log_return, y_stats)

            train_X, train_y, train_targets = make_sequence_batch(
                X_scaled,
                y_scaled,
                dataset.timestamps,
                train_idx,
                sequence_length=sequence_length,
                min_context_index=int(train_idx[0]),
            )
            val_X, val_y, val_targets = make_sequence_batch(
                X_scaled,
                y_scaled,
                dataset.timestamps,
                val_idx,
                sequence_length=sequence_length,
            )
            test_X, _, test_targets = make_sequence_batch(
                X_scaled,
                y_scaled,
                dataset.timestamps,
                test_idx,
                sequence_length=sequence_length,
            )

            if len(train_X) == 0 or len(val_X) == 0 or len(test_X) == 0:
                raise ValueError(
                    f"fold {fold.fold} has no usable LSTM sequences; "
                    "reduce --sequence-length or provide more continuous hourly data"
                )

            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(seed + fold.fold)
            model = _build_model(
                tf,
                sequence_length=sequence_length,
                feature_count=len(dataset.feature_names),
                lstm_units=lstm_units,
                dense_units=dense_units,
                dropout=dropout,
                learning_rate=learning_rate,
                clipnorm=clipnorm,
            )
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    mode="min",
                    patience=early_stopping_patience,
                    restore_best_weights=True,
                    verbose=0,
                )
            ]
            history = model.fit(
                train_X,
                train_y,
                validation_data=(val_X, val_y),
                epochs=epochs,
                batch_size=batch_size,
                shuffle=False,
                callbacks=callbacks,
                verbose=0,
            )

            predicted_scaled = model.predict(test_X, batch_size=batch_size, verbose=0).reshape(-1)
            predictions = inverse_standardize(predicted_scaled, y_stats).astype(np.float64)
            actual_log = dataset.y_log_return[test_targets].astype(np.float64)
            fold_metrics = regression_metrics(actual_log, predictions)

            val_losses = [float(value) for value in history.history.get("val_loss", [])]
            best_epoch = int(np.argmin(val_losses) + 1) if val_losses else len(history.epoch)
            best_val_loss = float(np.min(val_losses)) if val_losses else float("nan")

            model_path = model_dir / f"fold_{fold.fold:02d}.keras"
            model.save(model_path)
            scaler_payload = {
                "feature_names": dataset.feature_names,
                "sequence_length": sequence_length,
                "feature_mean": x_stats.mean,
                "feature_scale": x_stats.scale,
                "target_mean": y_stats.mean,
                "target_scale": y_stats.scale,
            }
            _write_json(scaler_dir / f"fold_{fold.fold:02d}.json", scaler_payload)

            fold_summary = {
                "fold": fold.fold,
                "sizes": fold.sizes,
                "sequence_sizes": {
                    "train": int(len(train_targets)),
                    "validation": int(len(val_targets)),
                    "test": int(len(test_targets)),
                },
                "train_start": int(dataset.timestamps[train_idx[0]]),
                "train_end": int(dataset.timestamps[train_idx[-1]]),
                "validation_start": int(dataset.timestamps[val_idx[0]]),
                "validation_end": int(dataset.timestamps[val_idx[-1]]),
                "test_start": int(dataset.timestamps[test_idx[0]]),
                "test_end": int(dataset.timestamps[test_idx[-1]]),
                "epochs_trained": int(len(history.epoch)),
                "best_epoch": best_epoch,
                "best_validation_loss_scaled": best_val_loss,
                **fold_metrics,
            }
            fold_summaries.append(fold_summary)
            logger.log_event(run_id, "fold_completed", **fold_summary)

            for local_idx, dataset_idx in enumerate(test_targets):
                prediction_rows.append(
                    {
                        "timestamp": int(dataset.timestamps[dataset_idx]),
                        "fold": fold.fold,
                        "actual_simple_return_1h": float(dataset.y_simple_return[dataset_idx]),
                        "actual_log_return_1h": float(dataset.y_log_return[dataset_idx]),
                        "predicted_log_return_1h": float(predictions[local_idx]),
                        "close": float(dataset.closes[dataset_idx]),
                        "ema20": float(dataset.ema20[dataset_idx]),
                        "ema50": float(dataset.ema50[dataset_idx]),
                        "ema200": float(dataset.ema200[dataset_idx]),
                    }
                )

        prediction_rows.sort(key=lambda row: int(row["timestamp"]))
        predicted = np.asarray(
            [row["predicted_log_return_1h"] for row in prediction_rows], dtype=np.float64
        )
        actual_log = np.asarray(
            [row["actual_log_return_1h"] for row in prediction_rows], dtype=np.float64
        )
        actual_simple = np.asarray(
            [row["actual_simple_return_1h"] for row in prediction_rows], dtype=np.float64
        )
        closes = np.asarray([row["close"] for row in prediction_rows], dtype=np.float64)
        ema20 = np.asarray([row["ema20"] for row in prediction_rows], dtype=np.float64)
        ema50 = np.asarray([row["ema50"] for row in prediction_rows], dtype=np.float64)
        ema200 = np.asarray([row["ema200"] for row in prediction_rows], dtype=np.float64)

        forecast_metrics = regression_metrics(actual_log, predicted)
        total_cost_bps = fee_bps + slippage_bps + spread_bps
        cost_rate = total_cost_bps / 10_000.0
        lstm_strategy, _, positions, turnovers = long_only_cost_aware_backtest(
            predicted,
            actual_simple,
            cost_rate=cost_rate,
            execution_lambda=execution_lambda,
        )
        ma_strategy = moving_average_baseline(
            actual_simple,
            closes,
            ema20,
            ema50,
            ema200,
            cost_rate=cost_rate,
        )
        buy_hold = buy_and_hold_baseline(actual_simple, cost_rate=cost_rate)

        for row, position, turnover in zip(prediction_rows, positions, turnovers, strict=True):
            row["lstm_position"] = float(position)
            row["lstm_turnover"] = float(turnover)

        summary = {
            "run_id": run_id,
            "created_at": datetime.now(UTC).isoformat(),
            "dataset_path": str(dataset_path),
            "usable_rows": dataset.size,
            "feature_version": FEATURE_VERSION,
            "feature_count": len(dataset.feature_names),
            "feature_names": dataset.feature_names,
            "sequence_length": sequence_length,
            "fold_count": len(folds),
            "oos_prediction_count": len(prediction_rows),
            "forecast_metrics": forecast_metrics,
            "strategies": {
                "lstm_cost_aware_long_only": lstm_strategy,
                "ema20_ema50_price_above_ema200": ma_strategy,
                "buy_and_hold": buy_hold,
            },
            "folds": fold_summaries,
        }

        _write_json(run_dir / "summary.json", summary)
        _write_jsonl(run_dir / "predictions.jsonl", prediction_rows)
        logger.finish_run(run_id, summary=summary)
        return summary
    except Exception as exc:
        logger.log_event(run_id, "run_failed", error=repr(exc))
        logger.finish_run(run_id, status="failed", summary={"error": repr(exc)})
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate BTC AI Core Step 2: LSTM next-hour log-return forecasting."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--train-days", type=int, default=90)
    parser.add_argument("--validation-days", type=int, default=30)
    parser.add_argument("--test-days", type=int, default=30)
    parser.add_argument("--step-days", type=int, default=30)
    parser.add_argument("--fee-bps", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=3.0)
    parser.add_argument("--spread-bps", type=float, default=2.0)
    parser.add_argument("--execution-lambda", type=float, default=2.0)
    parser.add_argument("--include-sentiment", action="store_true")
    parser.add_argument("--sequence-length", type=int, default=48)
    parser.add_argument("--lstm-units", type=int, default=64)
    parser.add_argument("--dense-units", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--clipnorm", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--early-stopping-patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = train_lstm_walk_forward(
        dataset_path=args.dataset,
        run_root=args.run_root,
        walk_forward=WalkForwardConfig(
            train_days=args.train_days,
            validation_days=args.validation_days,
            test_days=args.test_days,
            step_days=args.step_days,
        ),
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        spread_bps=args.spread_bps,
        execution_lambda=args.execution_lambda,
        include_sentiment=args.include_sentiment,
        sequence_length=args.sequence_length,
        lstm_units=args.lstm_units,
        dense_units=args.dense_units,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        clipnorm=args.clipnorm,
        epochs=args.epochs,
        batch_size=args.batch_size,
        early_stopping_patience=args.early_stopping_patience,
        seed=args.seed,
    )

    print("LSTM walk-forward evaluation complete")
    print(f"Run: {summary['run_id']}")
    print(f"OOS rows: {summary['oos_prediction_count']:,}")
    print(f"Direction accuracy: {summary['forecast_metrics']['direction_accuracy']:.4f}")
    print(
        "Cost-aware Sharpe: "
        f"{summary['strategies']['lstm_cost_aware_long_only']['sharpe']:.3f}"
    )
    print(f"Artifacts: {args.run_root / summary['run_id']}")


if __name__ == "__main__":
    main()
