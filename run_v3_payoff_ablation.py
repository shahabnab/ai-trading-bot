#!/usr/bin/env python3
"""Controlled V3 A/B test: global payoff vs volatility-conditioned payoff.

The runner deliberately trains each fold/model only once. The exact same raw and
Platt-calibrated classifier probabilities are then fed into two economic policy
variants:

A. GLOBAL payoff magnitude estimated on the calibration slice.
B. VOLATILITY-CONDITIONED payoff magnitude estimated on the same calibration
   slice using causal realized volatility (LOW/NORMAL/HIGH).

No test observation is used for payoff fitting, volatility cutoffs, shrinkage
selection, entry-margin selection, model selection, or calibration. Shrinkage
and entry margin for B are selected jointly on POLICY VALIDATION only.

Existing frozen V3 artifacts and paper ledgers are never modified.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import statistics as py_statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import run_research_experiments_v3 as v3
from backend.ml.calibration import (
    PayoffEstimate,
    VolatilityConditionedPayoffEstimate,
    brier_score,
    estimate_payoffs,
    estimate_volatility_conditioned_payoffs,
    ev_commitment_backtest,
    fit_platt_scaler,
)
from backend.ml.evaluation import DAY_MS, HOUR_MS, WalkForwardConfig, _performance_metrics, classification_metrics, make_walk_forward_folds
from backend.ml.features import FEATURE_VERSION, read_jsonl
from backend.ml.sequences import fit_standardizer, inverse_standardize, make_sequence_batch, standardize

DEFAULT_DATASET = Path("data/processed/training/btc_hourly_v3.jsonl")
DEFAULT_OUTPUT = Path("artifacts/ml/payoff_ablation_v3")


@dataclass(frozen=True)
class AblationConfig:
    target_hurdle_bps: float
    feature_set: str
    horizon_hours: int

    @property
    def slug(self) -> str:
        hurdle = str(int(self.target_hurdle_bps)) if float(self.target_hurdle_bps).is_integer() else str(self.target_hurdle_bps).replace(".", "p")
        return f"target{hurdle}bps_{self.feature_set}_h{self.horizon_hours:02d}"


def parse_config(value: str) -> AblationConfig:
    try:
        hurdle, feature_set, horizon = value.split(":", 2)
        config = AblationConfig(float(hurdle), feature_set.strip(), int(horizon))
    except Exception as exc:
        raise argparse.ArgumentTypeError("config must be HURDLE_BPS:FEATURE_SET:HORIZON_HOURS") from exc
    if config.target_hurdle_bps <= 0.0 or config.horizon_hours <= 0:
        raise argparse.ArgumentTypeError("hurdle and horizon must be positive")
    if config.feature_set not in {"technical", "technical_micro", "full_context"}:
        raise argparse.ArgumentTypeError("feature set must be technical, technical_micro, or full_context")
    return config


def parse_float_grid(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("grid cannot be empty")
    return values


def _policy_pool(rows: list[dict[str, Any]], min_trades: int) -> list[dict[str, Any]]:
    trade_ok = [row for row in rows if int(row["trade_count"]) >= min_trades]
    positive = [row for row in trade_ok if float(row["sharpe"]) > 0.0 and float(row["cumulative_return"]) > 0.0]
    return positive or trade_ok or rows


def _policy_rank(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(row["sharpe"]),
        float(row["cumulative_return"]),
        float(row["max_drawdown"]),
        -float(row["turnover"]),
    )


def select_conditioned_policy(
    calibration_returns: np.ndarray,
    calibration_volatility: np.ndarray,
    policy_probability: np.ndarray,
    policy_actual_1h: np.ndarray,
    policy_volatility: np.ndarray,
    *,
    event_hurdle_bps: float,
    trim_fraction: float,
    cost_rate: float,
    horizon_hours: int,
    margin_grid_bps: list[float],
    shrinkage_grid: list[float],
    min_trades: int,
    volatility_feature_name: str,
) -> tuple[VolatilityConditionedPayoffEstimate, float, float, list[dict[str, Any]]]:
    """Select shrinkage and margin using calibration + policy validation only."""
    rows: list[dict[str, Any]] = []
    fitted: dict[float, VolatilityConditionedPayoffEstimate] = {}
    for shrinkage in shrinkage_grid:
        payoff = estimate_volatility_conditioned_payoffs(
            calibration_returns,
            calibration_volatility,
            event_hurdle_bps=event_hurdle_bps,
            trim_fraction=trim_fraction,
            shrinkage_samples=float(shrinkage),
            volatility_feature_name=volatility_feature_name,
        )
        fitted[float(shrinkage)] = payoff
        for margin_bps in margin_grid_bps:
            metrics, _, _, _, _ = ev_commitment_backtest(
                policy_probability,
                policy_actual_1h,
                payoff=payoff,
                payoff_state=policy_volatility,
                one_way_cost_rate=cost_rate,
                horizon_hours=horizon_hours,
                entry_margin=float(margin_bps) / 10_000.0,
                exit_ev_threshold=0.0,
            )
            rows.append({
                "shrinkage_samples": float(shrinkage),
                "entry_margin_bps": float(margin_bps),
                **metrics,
            })
    best = max(_policy_pool(rows, min_trades), key=_policy_rank)
    shrinkage = float(best["shrinkage_samples"])
    return fitted[shrinkage], shrinkage, float(best["entry_margin_bps"]), rows


def _regime(payoff: VolatilityConditionedPayoffEstimate, volatility: float) -> str:
    if not math.isfinite(float(volatility)):
        return "GLOBAL_FALLBACK"
    if volatility <= payoff.low_vol_cutoff:
        return "LOW"
    if volatility <= payoff.high_vol_cutoff:
        return "NORMAL"
    return "HIGH"


def _state_means(payoff: VolatilityConditionedPayoffEstimate, volatility: float) -> tuple[float, float]:
    state = _regime(payoff, volatility)
    if state == "LOW":
        regime = payoff.low
    elif state == "NORMAL":
        regime = payoff.normal
    elif state == "HIGH":
        regime = payoff.high
    else:
        return payoff.global_payoff.mean_event_return, payoff.global_payoff.mean_non_event_return
    return regime.mean_event_return, regime.mean_non_event_return


def required_probability(
    event_mean: float,
    non_event_mean: float,
    *,
    one_way_cost_rate: float,
    entry_margin: float,
) -> float:
    """Probability required for gross EV to clear round-trip cost + margin."""
    denominator = float(event_mean) - float(non_event_mean)
    if denominator <= 0.0 or not math.isfinite(denominator):
        return math.nan
    target = 2.0 * float(one_way_cost_rate) + float(entry_margin)
    return (target - float(non_event_mean)) / denominator


def _action_series(positions: np.ndarray, turnovers: np.ndarray, decision_ev: np.ndarray) -> list[str]:
    actions: list[str] = []
    previous = 0.0
    for position, turnover, ev in zip(positions, turnovers, decision_ev, strict=True):
        position = float(position)
        turnover = float(turnover)
        if not math.isfinite(float(ev)):
            action = "COMMIT"
        elif turnover > 0.0 and position > previous:
            action = "ENTER"
        elif turnover > 0.0 and position < previous:
            action = "EXIT"
        elif position > 0.0:
            action = "HOLD_LONG"
        else:
            action = "HOLD_FLAT"
        actions.append(action)
        previous = position
    return actions


def _aggregate_method(folds: list[dict[str, Any]], method: str, cost_bps: float) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    returns = np.concatenate([np.asarray(fold[method][f"{cost_bps:g}"]["returns"], dtype=np.float64) for fold in folds])
    positions = np.concatenate([np.asarray(fold[method][f"{cost_bps:g}"]["positions"], dtype=np.float64) for fold in folds])
    turnovers = np.concatenate([np.asarray(fold[method][f"{cost_bps:g}"]["turnovers"], dtype=np.float64) for fold in folds])
    metrics = _performance_metrics(returns, positions, turnovers)
    for name in ("trade_count", "round_trip_count", "position_change_count"):
        metrics[name] = int(sum(int(fold[method][f"{cost_bps:g}"]["metrics"][name]) for fold in folds))
    metrics["turnover"] = float(sum(float(fold[method][f"{cost_bps:g}"]["metrics"]["turnover"]) for fold in folds))
    return metrics, returns, positions, turnovers


def _method_summary(folds: list[dict[str, Any]], method: str, cost_bps: float, starting_capital: float) -> dict[str, Any]:
    metrics, returns, positions, turnovers = _aggregate_method(folds, method, cost_bps)
    ending = float(starting_capital * np.prod(1.0 + returns))
    trades = v3._completed_trade_returns(returns, positions, turnovers)
    wins = [value for value in trades if value > 0.0]
    return {
        "metrics": metrics,
        "starting_capital_eur": float(starting_capital),
        "ending_capital_eur": ending,
        "profit_eur": ending - float(starting_capital),
        "invested_fraction": float(np.mean(positions > 0.0)) if len(positions) else 0.0,
        "completed_trades": len(trades),
        "win_rate": float(len(wins) / len(trades)) if trades else 0.0,
        "average_trade_return": float(np.mean(trades)) if trades else 0.0,
    }


def _state_breakdown(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for state in ("LOW", "NORMAL", "HIGH", "GLOBAL_FALLBACK"):
        block = [row for row in rows if row["volatility_regime"] == state]
        if not block:
            continue
        g_ret = np.asarray([row["global_strategy_return"] for row in block], dtype=np.float64)
        c_ret = np.asarray([row["conditioned_strategy_return"] for row in block], dtype=np.float64)
        result[state] = {
            "bar_count": len(block),
            "mean_probability": float(np.mean([row["calibrated_probability"] for row in block])),
            "mean_global_gross_ev": float(np.mean([row["global_gross_ev"] for row in block])),
            "mean_conditioned_gross_ev": float(np.mean([row["conditioned_gross_ev"] for row in block])),
            "global_compounded_bar_return": float(np.prod(1.0 + g_ret) - 1.0),
            "conditioned_compounded_bar_return": float(np.prod(1.0 + c_ret) - 1.0),
            "global_invested_fraction": float(np.mean([row["global_position"] > 0.0 for row in block])),
            "conditioned_invested_fraction": float(np.mean([row["conditioned_position"] > 0.0 for row in block])),
            "global_entries": int(sum(row["global_action"] == "ENTER" for row in block)),
            "conditioned_entries": int(sum(row["conditioned_action"] == "ENTER" for row in block)),
            "global_hold_flat": int(sum(row["global_action"] == "HOLD_FLAT" for row in block)),
            "conditioned_hold_flat": int(sum(row["conditioned_action"] == "HOLD_FLAT" for row in block)),
        }
    return result


def _delta(conditioned: dict[str, Any], global_: dict[str, Any]) -> dict[str, float]:
    cm = conditioned["metrics"]
    gm = global_["metrics"]
    return {
        "ending_capital_eur": float(conditioned["ending_capital_eur"] - global_["ending_capital_eur"]),
        "cumulative_return": float(cm["cumulative_return"] - gm["cumulative_return"]),
        "sharpe": float(cm["sharpe"] - gm["sharpe"]),
        "sortino": float(cm.get("sortino", 0.0) - gm.get("sortino", 0.0)),
        "max_drawdown": float(cm["max_drawdown"] - gm["max_drawdown"]),
        "trade_count": float(cm["trade_count"] - gm["trade_count"]),
        "turnover": float(cm["turnover"] - gm["turnover"]),
    }


def run_config(raw_rows: list[dict[str, Any]], config: AblationConfig, args: argparse.Namespace, out: Path) -> dict[str, Any]:
    import tensorflow as tf

    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

    base = v3.build_research_dataset(raw_rows, horizon_hours=config.horizon_hours)
    dataset = v3.enrich_dataset(base, config.feature_set)
    shadow_start = int(dataset.timestamps[-1]) - args.shadow_days * DAY_MS
    pre_indices = np.flatnonzero((dataset.timestamps + config.horizon_hours * HOUR_MS) < shadow_start)
    pre = v3._subset_dataset(dataset, pre_indices)

    if args.volatility_feature not in pre.feature_names:
        raise ValueError(f"volatility feature {args.volatility_feature!r} not present in feature set")
    vol_idx = pre.feature_names.index(args.volatility_feature)

    validation_days = args.model_validation_days + args.calibration_days + args.policy_validation_days
    wf = WalkForwardConfig(
        train_days=args.train_days,
        validation_days=validation_days,
        test_days=args.test_days,
        step_days=args.step_days,
        embargo_hours=0,
    )
    folds = make_walk_forward_folds(pre.timestamps, wf)
    if not folds:
        raise ValueError(f"no folds for {config.slug}")

    trial_count = args.full_context_trials if config.feature_set == "full_context" else args.ablation_trials
    candidates = v3.candidate_pool(
        trial_count,
        args.seed + config.horizon_hours * 1009 + int(config.target_hurdle_bps) * 17 + len(config.feature_set),
    )

    fold_results: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    policy_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []

    print(f"\n{'#' * 88}\nPAYOFF ABLATION {config.slug} | folds={len(folds)} | candidates={len(candidates)}\n{'#' * 88}", flush=True)

    for fold in folds:
        split = v3.make_fold_split(pre, fold, args)
        best_id = -1
        best_key: tuple[float, float, float] | None = None
        best_weights: list[np.ndarray] | None = None
        best_x_stats = None
        best_y_stats = None
        best_epoch = 0

        for cid, candidate in enumerate(candidates):
            target_scale = float(np.asarray(fit_standardizer(pre.target_log_return[split.train]).scale).reshape(-1)[0])
            progress = v3._progress_callback(
                tf,
                key=v3.ExperimentKey(config.target_hurdle_bps, config.feature_set, config.horizon_hours),
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
                model, x_stats, y_stats, candidate_epoch, cls = v3._train_model_candidate(
                    tf,
                    pre,
                    split,
                    candidate,
                    event_hurdle_bps=config.target_hurdle_bps,
                    seed=args.seed + fold.fold * 10_000 + cid,
                    epochs=args.epochs,
                    patience=args.early_stopping_patience,
                    progress=progress,
                )
            except ValueError as exc:
                model_rows.append({"fold": fold.fold, "candidate_id": cid, "status": f"skipped:{exc}", **candidate.to_dict()})
                continue
            model_rows.append({"fold": fold.fold, "candidate_id": cid, "status": "ok", "best_epoch": candidate_epoch, **cls, **candidate.to_dict()})
            rank = (float(cls["auc"]), -float(cls["brier_score"]), float(cls["direction_accuracy_prob"]))
            if best_key is None or rank > best_key:
                best_key = rank
                best_id = cid
                best_epoch = candidate_epoch
                best_weights = [np.array(weight, copy=True) for weight in model.get_weights()]
                best_x_stats = x_stats
                best_y_stats = y_stats

        if best_id < 0 or best_weights is None or best_x_stats is None or best_y_stats is None:
            raise ValueError(f"no valid model candidate in fold {fold.fold}")

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

        def predict(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            xx, _, targets = make_sequence_batch(
                x_scaled,
                y_scaled,
                pre.timestamps,
                indices,
                sequence_length=candidate.sequence_length,
            )
            if len(xx) == 0:
                raise ValueError("empty prediction block")
            output = model.predict(xx, verbose=0)
            probability = np.clip(np.asarray(output[1]).reshape(-1), 1e-6, 1.0 - 1e-6)
            return probability, targets

        hurdle = config.target_hurdle_bps / 10_000.0
        cal_raw, cal_targets = predict(split.calibration)
        cal_labels = (pre.target_simple_return[cal_targets] > hurdle).astype(np.float64)
        calibrator = fit_platt_scaler(cal_raw, cal_labels)
        cal_prob = calibrator.transform(cal_raw)
        calibration_returns = pre.target_simple_return[cal_targets]
        cal_vol = pre.X[cal_targets, vol_idx]
        global_payoff = estimate_payoffs(
            calibration_returns,
            event_hurdle_bps=config.target_hurdle_bps,
            trim_fraction=args.trim_fraction,
        )

        policy_raw, policy_targets = predict(split.policy_validation)
        policy_prob = calibrator.transform(policy_raw)
        policy_actual = pre.execution_simple_return_1h[policy_targets]
        policy_vol = pre.X[policy_targets, vol_idx]

        global_margin_bps, global_search = v3.choose_margin(
            policy_prob,
            policy_actual,
            global_payoff,
            cost_rate=args.primary_cost_bps / 10_000.0,
            horizon_hours=config.horizon_hours,
            margin_grid_bps=args.margin_grid_bps,
            min_trades=args.min_policy_trades,
        )
        for row in global_search:
            policy_rows.append({"fold": fold.fold, "method": "global", "shrinkage_samples": "", **row})

        conditioned_payoff, selected_shrinkage, conditioned_margin_bps, conditioned_search = select_conditioned_policy(
            calibration_returns,
            cal_vol,
            policy_prob,
            policy_actual,
            policy_vol,
            event_hurdle_bps=config.target_hurdle_bps,
            trim_fraction=args.trim_fraction,
            cost_rate=args.primary_cost_bps / 10_000.0,
            horizon_hours=config.horizon_hours,
            margin_grid_bps=args.margin_grid_bps,
            shrinkage_grid=args.shrinkage_grid,
            min_trades=args.min_policy_trades,
            volatility_feature_name=args.volatility_feature,
        )
        for row in conditioned_search:
            policy_rows.append({"fold": fold.fold, "method": "volatility_conditioned", **row})

        test_raw, test_targets = predict(split.test)
        test_prob = calibrator.transform(test_raw)
        test_labels = (pre.target_simple_return[test_targets] > hurdle).astype(np.float64)
        test_actual = pre.execution_simple_return_1h[test_targets]
        test_vol = pre.X[test_targets, vol_idx]
        cls = classification_metrics(test_labels, test_prob)

        fold_payload: dict[str, Any] = {
            "fold": fold.fold,
            "selected_candidate_id": best_id,
            "best_epoch": best_epoch,
            "classification": cls,
            "calibration_brier_before": brier_score(cal_labels, cal_raw),
            "calibration_brier_after": brier_score(cal_labels, cal_prob),
            "global_payoff": global_payoff.to_dict(),
            "conditioned_payoff": conditioned_payoff.to_dict(),
            "global_margin_bps": global_margin_bps,
            "conditioned_margin_bps": conditioned_margin_bps,
            "selected_shrinkage_samples": selected_shrinkage,
            "global": {},
            "conditioned": {},
        }

        primary_global = None
        primary_conditioned = None
        for cost_bps in args.cost_bps:
            global_eval = ev_commitment_backtest(
                test_prob,
                test_actual,
                payoff=global_payoff,
                one_way_cost_rate=cost_bps / 10_000.0,
                horizon_hours=config.horizon_hours,
                entry_margin=global_margin_bps / 10_000.0,
                exit_ev_threshold=0.0,
            )
            conditioned_eval = ev_commitment_backtest(
                test_prob,
                test_actual,
                payoff=conditioned_payoff,
                payoff_state=test_vol,
                one_way_cost_rate=cost_bps / 10_000.0,
                horizon_hours=config.horizon_hours,
                entry_margin=conditioned_margin_bps / 10_000.0,
                exit_ev_threshold=0.0,
            )
            gm, gr, gp, gt, ge = global_eval
            cm, cr, cp, ct, ce = conditioned_eval
            fold_payload["global"][f"{cost_bps:g}"] = {
                "metrics": gm,
                "returns": gr.tolist(),
                "positions": gp.tolist(),
                "turnovers": gt.tolist(),
            }
            fold_payload["conditioned"][f"{cost_bps:g}"] = {
                "metrics": cm,
                "returns": cr.tolist(),
                "positions": cp.tolist(),
                "turnovers": ct.tolist(),
            }
            if cost_bps == args.primary_cost_bps:
                primary_global = global_eval
                primary_conditioned = conditioned_eval

        assert primary_global is not None and primary_conditioned is not None
        gm, gr, gp, gt, ge = primary_global
        cm, cr, cp, ct, ce = primary_conditioned

        fixed_metrics, _, _, _, _ = ev_commitment_backtest(
            test_prob,
            test_actual,
            payoff=conditioned_payoff,
            payoff_state=test_vol,
            one_way_cost_rate=args.primary_cost_bps / 10_000.0,
            horizon_hours=config.horizon_hours,
            entry_margin=global_margin_bps / 10_000.0,
            exit_ev_threshold=0.0,
        )
        fold_payload["conditioned_at_global_margin_primary"] = fixed_metrics

        global_gross = global_payoff.expected_gross_return(test_prob)
        conditioned_gross = conditioned_payoff.expected_gross_return(test_prob, test_vol)
        global_required = required_probability(
            global_payoff.mean_event_return,
            global_payoff.mean_non_event_return,
            one_way_cost_rate=args.primary_cost_bps / 10_000.0,
            entry_margin=global_margin_bps / 10_000.0,
        )
        conditioned_required = np.asarray([
            required_probability(
                *_state_means(conditioned_payoff, float(vol)),
                one_way_cost_rate=args.primary_cost_bps / 10_000.0,
                entry_margin=conditioned_margin_bps / 10_000.0,
            )
            for vol in test_vol
        ])
        global_actions = _action_series(gp, gt, ge)
        conditioned_actions = _action_series(cp, ct, ce)

        for i, target in enumerate(test_targets):
            diagnostics.append({
                "config": config.slug,
                "fold": fold.fold,
                "timestamp": int(pre.timestamps[target]),
                "calibrated_probability": float(test_prob[i]),
                "actual_horizon_return": float(pre.target_simple_return[target]),
                "actual_1h_return": float(test_actual[i]),
                "realized_volatility": float(test_vol[i]),
                "volatility_regime": _regime(conditioned_payoff, float(test_vol[i])),
                "global_gross_ev": float(global_gross[i]),
                "conditioned_gross_ev": float(conditioned_gross[i]),
                "global_net_entry_ev": float(global_gross[i] - 2.0 * args.primary_cost_bps / 10_000.0),
                "conditioned_net_entry_ev": float(conditioned_gross[i] - 2.0 * args.primary_cost_bps / 10_000.0),
                "global_required_probability": float(global_required),
                "conditioned_required_probability": float(conditioned_required[i]),
                "global_decision_ev": None if not math.isfinite(float(ge[i])) else float(ge[i]),
                "conditioned_decision_ev": None if not math.isfinite(float(ce[i])) else float(ce[i]),
                "global_position": float(gp[i]),
                "conditioned_position": float(cp[i]),
                "global_turnover": float(gt[i]),
                "conditioned_turnover": float(ct[i]),
                "global_strategy_return": float(gr[i]),
                "conditioned_strategy_return": float(cr[i]),
                "global_action": global_actions[i],
                "conditioned_action": conditioned_actions[i],
            })

        fold_results.append(fold_payload)
        print(
            f"fold {fold.fold:02d} | AUC={cls['auc']:.3f} | global Sharpe={gm['sharpe']:+.3f} return={gm['cumulative_return']:+.2%} | "
            f"conditioned Sharpe={cm['sharpe']:+.3f} return={cm['cumulative_return']:+.2%} | shrink={selected_shrinkage:g} | margins={global_margin_bps:g}/{conditioned_margin_bps:g}bps",
            flush=True,
        )

    by_cost: dict[str, Any] = {}
    for cost_bps in args.cost_bps:
        global_summary = _method_summary(fold_results, "global", cost_bps, args.starting_capital_eur)
        conditioned_summary = _method_summary(fold_results, "conditioned", cost_bps, args.starting_capital_eur)
        by_cost[f"{cost_bps:g}"] = {
            "global": global_summary,
            "volatility_conditioned": conditioned_summary,
            "conditioned_minus_global": _delta(conditioned_summary, global_summary),
        }

    primary = by_cost[f"{args.primary_cost_bps:g}"]
    fold_global_sharpe = [float(fold["global"][f"{args.primary_cost_bps:g}"]["metrics"]["sharpe"]) for fold in fold_results]
    fold_conditioned_sharpe = [float(fold["conditioned"][f"{args.primary_cost_bps:g}"]["metrics"]["sharpe"]) for fold in fold_results]
    summary = {
        "config": asdict(config),
        "fold_count": len(fold_results),
        "volatility_feature": args.volatility_feature,
        "shrinkage_grid": args.shrinkage_grid,
        "primary_cost_bps_one_way": args.primary_cost_bps,
        "round_trip_cost_bps_primary": 2.0 * args.primary_cost_bps,
        "same_model_predictions_for_both_methods": True,
        "selection_protocol": "model validation selects model; calibration fits Platt/payoffs/cutoffs; policy validation selects margins and conditioned shrinkage; test is untouched",
        "by_cost_bps": by_cost,
        "median_fold_sharpe_global": float(py_statistics.median(fold_global_sharpe)),
        "median_fold_sharpe_conditioned": float(py_statistics.median(fold_conditioned_sharpe)),
        "conditioned_better_sharpe_folds": int(sum(c > g for c, g in zip(fold_conditioned_sharpe, fold_global_sharpe, strict=True))),
        "selected_shrinkage_by_fold": [float(fold["selected_shrinkage_samples"]) for fold in fold_results],
        "global_margin_bps_by_fold": [float(fold["global_margin_bps"]) for fold in fold_results],
        "conditioned_margin_bps_by_fold": [float(fold["conditioned_margin_bps"]) for fold in fold_results],
        "state_breakdown_primary": _state_breakdown(diagnostics),
        "action_counts_primary": {
            "global": {name: int(sum(row["global_action"] == name for row in diagnostics)) for name in ("ENTER", "EXIT", "HOLD_FLAT", "HOLD_LONG", "COMMIT")},
            "volatility_conditioned": {name: int(sum(row["conditioned_action"] == name for row in diagnostics)) for name in ("ENTER", "EXIT", "HOLD_FLAT", "HOLD_LONG", "COMMIT")},
        },
        "primary_comparison": primary,
    }

    out.mkdir(parents=True, exist_ok=True)
    v3.write_json(out / "summary.json", summary)
    v3.write_json(out / "folds.json", fold_results)
    v3.write_jsonl(out / "diagnostics.jsonl", diagnostics)
    v3.write_csv(out / "policy_search.csv", policy_rows)
    v3.write_csv(out / "model_selection.csv", model_rows)

    comparison_rows = []
    for cost, values in by_cost.items():
        for method_key, label in (("global", "global"), ("volatility_conditioned", "volatility_conditioned")):
            item = values[method_key]
            metrics = item["metrics"]
            comparison_rows.append({
                "cost_bps_one_way": float(cost),
                "round_trip_cost_bps": 2.0 * float(cost),
                "method": label,
                "ending_capital_eur": item["ending_capital_eur"],
                "profit_eur": item["profit_eur"],
                "cumulative_return": metrics["cumulative_return"],
                "sharpe": metrics["sharpe"],
                "sortino": metrics.get("sortino", 0.0),
                "max_drawdown": metrics["max_drawdown"],
                "trade_count": metrics["trade_count"],
                "turnover": metrics["turnover"],
                "win_rate": item["win_rate"],
                "average_trade_return": item["average_trade_return"],
                "invested_fraction": item["invested_fraction"],
            })
    v3.write_csv(out / "comparison.csv", comparison_rows)

    global_primary = primary["global"]
    conditioned_primary = primary["volatility_conditioned"]
    report = [
        f"# V3 Payoff Ablation — {config.slug}",
        "",
        "Same trained models, same calibrated probabilities, same folds, same costs and same commitment/exit logic are used for both methods.",
        "Conditioned shrinkage and entry margin are selected on policy validation only; the test windows remain untouched.",
        "",
        f"Primary cost: {args.primary_cost_bps:g} bps one-way / {2 * args.primary_cost_bps:g} bps round-trip.",
        "",
        "| Method | End EUR | Return | Sharpe | Sortino | Max DD | Trades | Turnover | Win rate | Invested |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, item in (("Global", global_primary), ("Vol-conditioned", conditioned_primary)):
        m = item["metrics"]
        report.append(
            f"| {label} | {item['ending_capital_eur']:.2f} | {m['cumulative_return']:+.2%} | {m['sharpe']:+.3f} | {m.get('sortino', 0.0):+.3f} | {m['max_drawdown']:.2%} | {int(m['trade_count'])} | {m['turnover']:.1f} | {item['win_rate']:.1%} | {item['invested_fraction']:.1%} |"
        )
    d = primary["conditioned_minus_global"]
    report += [
        "",
        f"Conditioned - Global: ending capital EUR {d['ending_capital_eur']:+.2f}, return {d['cumulative_return']:+.2%}, Sharpe {d['sharpe']:+.3f}.",
        "",
        "See `diagnostics.jsonl` for probability, gross/net EV, break-even probability, volatility regime, decision EV and action at every untouched test timestamp.",
    ]
    (out / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Controlled global-vs-volatility-conditioned V3 payoff A/B test.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--configs",
        nargs="+",
        type=parse_config,
        default=[parse_config("50:technical:3"), parse_config("25:full_context:12")],
        help="HURDLE_BPS:FEATURE_SET:HORIZON_HOURS; defaults match the frozen V3 3h/12h policies.",
    )
    parser.add_argument("--volatility-feature", default="realized_vol_24h")
    parser.add_argument("--shrinkage-grid", type=parse_float_grid, default=parse_float_grid("0,10,25,50,100"))
    parser.add_argument("--margin-grid-bps", type=parse_float_grid, default=parse_float_grid("0,2.5,5,7.5,10,15,20,25"))
    parser.add_argument("--cost-bps", type=parse_float_grid, default=parse_float_grid("20,25,30,40"))
    parser.add_argument("--primary-cost-bps", type=float, default=25.0)
    parser.add_argument("--train-days", type=int, default=365)
    parser.add_argument("--model-validation-days", type=int, default=45)
    parser.add_argument("--calibration-days", type=int, default=60)
    parser.add_argument("--policy-validation-days", type=int, default=45)
    parser.add_argument("--test-days", type=int, default=60)
    parser.add_argument("--step-days", type=int, default=60)
    parser.add_argument("--shadow-days", type=int, default=30)
    parser.add_argument("--min-policy-trades", type=int, default=3)
    parser.add_argument("--trim-fraction", type=float, default=0.05)
    parser.add_argument("--ablation-trials", type=int, default=2)
    parser.add_argument("--full-context-trials", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--starting-capital-eur", type=float, default=1000.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epoch-verbose", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.dataset.is_file():
        raise FileNotFoundError(args.dataset)
    if args.primary_cost_bps not in args.cost_bps:
        raise ValueError("primary cost must be included in cost-bps")
    if any(value < 0.0 for value in args.shrinkage_grid):
        raise ValueError("shrinkage grid must be non-negative")
    if any(value < 0.0 for value in args.margin_grid_bps):
        raise ValueError("margin grid must be non-negative")
    if not 0.0 <= args.trim_fraction < 0.5:
        raise ValueError("trim fraction must be in [0, 0.5)")
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

    meta = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": v3.git_head(),
        "dataset": str(args.dataset),
        "dataset_sha256": v3.sha256_file(args.dataset),
        "raw_rows": len(raw_rows),
        "feature_version": FEATURE_VERSION,
        "configs": [asdict(config) for config in args.configs],
        "volatility_feature": args.volatility_feature,
        "shrinkage_grid": args.shrinkage_grid,
        "margin_grid_bps": args.margin_grid_bps,
        "cost_bps_one_way": args.cost_bps,
        "primary_cost_bps_one_way": args.primary_cost_bps,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "protocol": "same model probabilities A/B; calibration-only payoff fit; policy-validation-only hyperparameter selection; untouched test evaluation",
    }
    v3.write_json(run_dir / "metadata.json", meta)

    summaries: list[dict[str, Any]] = []
    for config in args.configs:
        summaries.append(run_config(raw_rows, config, args, run_dir / config.slug))

    overview_rows: list[dict[str, Any]] = []
    for summary in summaries:
        config = summary["config"]
        primary = summary["primary_comparison"]
        for method in ("global", "volatility_conditioned"):
            item = primary[method]
            metrics = item["metrics"]
            overview_rows.append({
                **config,
                "method": method,
                "ending_capital_eur": item["ending_capital_eur"],
                "cumulative_return": metrics["cumulative_return"],
                "sharpe": metrics["sharpe"],
                "sortino": metrics.get("sortino", 0.0),
                "max_drawdown": metrics["max_drawdown"],
                "trade_count": metrics["trade_count"],
                "turnover": metrics["turnover"],
                "win_rate": item["win_rate"],
                "invested_fraction": item["invested_fraction"],
            })
    v3.write_json(run_dir / "all_results.json", summaries)
    v3.write_csv(run_dir / "overview.csv", overview_rows)
    meta["duration_seconds"] = time.time() - started
    v3.write_json(run_dir / "metadata.json", meta)

    print(f"\nPayoff ablation complete: {run_dir}", flush=True)
    for summary in summaries:
        p = summary["primary_comparison"]
        g = p["global"]
        c = p["volatility_conditioned"]
        print(
            f"{summary['config']} | Global EUR {g['ending_capital_eur']:.2f}, Sharpe {g['metrics']['sharpe']:+.3f} | "
            f"Conditioned EUR {c['ending_capital_eur']:.2f}, Sharpe {c['metrics']['sharpe']:+.3f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
