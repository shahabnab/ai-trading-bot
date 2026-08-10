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


DEFAULT_DATASET = Path("data/processed/training/btc_hourly.jsonl")
DEFAULT_RUN_ROOT = Path("artifacts/ml/runs")
DATASET_VERSION = "btc-hourly-jsonl-v1"

DEFAULT_XGB_PARAMS: dict[str, Any] = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",
    "max_depth": 3,
    "eta": 0.02,
    "min_child_weight": 20.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "lambda": 10.0,
    "alpha": 0.001,
    "seed": 42,
    "nthread": 0,
}


def _jsonable(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
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


def train_xgboost_walk_forward(
    *,
    dataset_path: Path = DEFAULT_DATASET,
    run_root: Path = DEFAULT_RUN_ROOT,
    walk_forward: WalkForwardConfig = WalkForwardConfig(),
    # Defaults mirror the paper engine (PAPER_FEE_RATE=0.002, PAPER_SLIPPAGE_BPS=5)
    # so backtest and paper-trading results are computed under the same costs.
    fee_bps: float = 20.0,
    slippage_bps: float = 5.0,
    spread_bps: float = 0.0,
    execution_lambda: float = 2.0,
    include_sentiment: bool = False,
    num_boost_round: int = 2000,
    early_stopping_rounds: int = 50,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import xgboost as xgb

    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Training dataset not found: {dataset_path}. "
            "Run `python -m backend.data_collection.historical_dataset --months 6 --skip-news` first."
        )

    rows = read_jsonl(dataset_path)
    dataset = build_feature_dataset(rows, include_sentiment=include_sentiment)
    folds = make_walk_forward_folds(dataset.timestamps, walk_forward)
    if not folds:
        raise ValueError(
            "Not enough usable history for the requested walk-forward windows. "
            "For roughly six months of hourly data, start with "
            "--train-days 90 --validation-days 30 --test-days 30 --step-days 30."
        )

    xgb_params = dict(DEFAULT_XGB_PARAMS)
    if params:
        xgb_params.update(params)

    logger = RunLogger(run_root)
    run = logger.start_run(
        purpose="walk-forward BTC next-hour return forecasting",
        model_family="xgboost",
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
            "num_boost_round": num_boost_round,
            "early_stopping_rounds": early_stopping_rounds,
            "xgboost_params": xgb_params,
        },
        machine={
            "python": platform.python_version(),
            "platform": platform.platform(),
            "xgboost": xgb.__version__,
        },
    )
    run_id = str(run["run_id"])
    run_dir = run_root / run_id
    model_dir = run_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    prediction_rows: list[dict[str, Any]] = []
    fold_summaries: list[dict[str, Any]] = []
    importance_gain: dict[str, float] = {name: 0.0 for name in dataset.feature_names}

    try:
        for fold in folds:
            train_idx = fold.train_indices
            val_idx = fold.validation_indices
            test_idx = fold.test_indices

            dtrain = xgb.DMatrix(
                dataset.X[train_idx],
                label=dataset.y_log_return[train_idx],
                feature_names=dataset.feature_names,
            )
            dval = xgb.DMatrix(
                dataset.X[val_idx],
                label=dataset.y_log_return[val_idx],
                feature_names=dataset.feature_names,
            )
            dtest = xgb.DMatrix(dataset.X[test_idx], feature_names=dataset.feature_names)

            # The default early_stopping_rounds API returns the last booster, while
            # OOS evaluation historically used only best_iteration. save_best=True
            # prunes the returned booster itself, so the model written to disk is
            # exactly the model whose OOS predictions are reported.
            early_stop = xgb.callback.EarlyStopping(
                rounds=early_stopping_rounds,
                save_best=True,
            )
            booster = xgb.train(
                xgb_params,
                dtrain,
                num_boost_round=num_boost_round,
                evals=[(dval, "validation")],
                callbacks=[early_stop],
                verbose_eval=False,
            )
            predictions = booster.predict(dtest)
            fold_metrics = regression_metrics(dataset.y_log_return[test_idx], predictions)

            model_path = model_dir / f"fold_{fold.fold:02d}.json"
            booster.save_model(model_path)

            for feature_name, gain in booster.get_score(importance_type="gain").items():
                importance_gain[feature_name] = importance_gain.get(feature_name, 0.0) + float(gain)

            fold_summary = {
                "fold": fold.fold,
                "sizes": fold.sizes,
                "train_start": int(dataset.timestamps[train_idx[0]]),
                "train_end": int(dataset.timestamps[train_idx[-1]]),
                "validation_start": int(dataset.timestamps[val_idx[0]]),
                "validation_end": int(dataset.timestamps[val_idx[-1]]),
                "test_start": int(dataset.timestamps[test_idx[0]]),
                "test_end": int(dataset.timestamps[test_idx[-1]]),
                "best_iteration": int(booster.best_iteration),
                "best_score": float(booster.best_score),
                **fold_metrics,
            }
            fold_summaries.append(fold_summary)
            logger.log_event(run_id, "fold_completed", **fold_summary)

            for local_idx, dataset_idx in enumerate(test_idx):
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
        xgb_strategy, _, positions, turnovers = long_only_cost_aware_backtest(
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
            row["xgb_position"] = float(position)
            row["xgb_turnover"] = float(turnover)

        normalized_importance = dict(
            sorted(importance_gain.items(), key=lambda item: item[1], reverse=True)
        )
        summary = {
            "run_id": run_id,
            "created_at": datetime.now(UTC).isoformat(),
            "dataset_path": str(dataset_path),
            "usable_rows": dataset.size,
            "feature_version": FEATURE_VERSION,
            "feature_count": len(dataset.feature_names),
            "feature_names": dataset.feature_names,
            "fold_count": len(folds),
            "oos_prediction_count": len(prediction_rows),
            "forecast_metrics": forecast_metrics,
            "strategies": {
                "xgboost_cost_aware_long_only": xgb_strategy,
                "ema20_ema50_price_above_ema200": ma_strategy,
                "buy_and_hold": buy_hold,
            },
            "folds": fold_summaries,
            "top_features_by_gain": list(normalized_importance.items())[:20],
        }

        _write_json(run_dir / "summary.json", summary)
        _write_json(run_dir / "feature_importance_gain.json", normalized_importance)
        _write_jsonl(run_dir / "predictions.jsonl", prediction_rows)
        logger.finish_run(run_id, summary=summary)
        return summary
    except Exception as exc:
        logger.log_event(run_id, "run_failed", error=repr(exc))
        logger.finish_run(run_id, status="failed", summary={"error": repr(exc)})
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the first BTC AI core: XGBoost on next-hour log returns."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--train-days", type=int, default=90)
    parser.add_argument("--validation-days", type=int, default=30)
    parser.add_argument("--test-days", type=int, default=30)
    parser.add_argument("--step-days", type=int, default=30)
    parser.add_argument("--fee-bps", type=float, default=20.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--spread-bps", type=float, default=0.0)
    parser.add_argument("--execution-lambda", type=float, default=2.0)
    parser.add_argument("--include-sentiment", action="store_true")
    parser.add_argument("--num-boost-round", type=int, default=2000)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = train_xgboost_walk_forward(
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
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
    )

    print("XGBoost walk-forward evaluation complete")
    print(f"Run: {summary['run_id']}")
    print(f"OOS rows: {summary['oos_prediction_count']:,}")
    print(f"Direction accuracy: {summary['forecast_metrics']['direction_accuracy']:.4f}")
    print(
        "Cost-aware Sharpe: "
        f"{summary['strategies']['xgboost_cost_aware_long_only']['sharpe']:.3f}"
    )
    print(f"Artifacts: {args.run_root / summary['run_id']}")


if __name__ == "__main__":
    main()
