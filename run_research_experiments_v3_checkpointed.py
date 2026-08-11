#!/usr/bin/env python3
"""Checkpointed/resumable wrapper for V3 research.

This runner keeps the scientific protocol from ``run_research_experiments_v3``
but makes long GPU jobs robust to server interruptions and small disks:

* each completed experiment is checkpointed by ``summary.json``;
* ``--resume-run-dir`` skips experiments whose summaries already exist;
* heavy per-fold return/position/turnover/EV arrays are not written to
  ``folds.json`` (they are only needed transiently to compute aggregate metrics);
* the final 30-day EUR 1,000 shadow test is still run only after all research
  experiments are complete.

The original V3 implementation remains the source of truth for model training,
calibration, execution, statistics, and reporting.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import run_research_experiments_v3 as v3


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _compact_fold_payload(payload: Any) -> Any:
    """Drop large arrays that are redundant once aggregate metrics exist."""
    if not isinstance(payload, list):
        return payload
    compact: list[Any] = []
    heavy_prefixes = ("returns_", "positions_", "turnovers_", "ev_")
    for row in payload:
        if isinstance(row, dict):
            compact.append({k: value for k, value in row.items() if not k.startswith(heavy_prefixes)})
        else:
            compact.append(row)
    return compact


def _install_compact_writer() -> None:
    original_write_json = v3.write_json

    def compact_write_json(path: Path, payload: Any) -> None:
        if Path(path).name == "folds.json":
            payload = _compact_fold_payload(payload)
        original_write_json(path, payload)

    v3.write_json = compact_write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or resume checkpointed V3 research.")
    parser.add_argument("--resume-run-dir", type=Path, default=None)
    parser.add_argument("--dataset", type=Path, default=v3.DEFAULT_DATASET)
    parser.add_argument("--output-root", type=Path, default=v3.DEFAULT_OUTPUT)
    parser.add_argument("--horizons", type=v3.parse_int_grid, default=v3.parse_int_grid("1,3,6,12"))
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        choices=["technical", "technical_micro", "full_context"],
        default=["technical", "technical_micro", "full_context"],
    )
    parser.add_argument("--primary-target-bps", type=float, default=50.0)
    parser.add_argument("--sensitivity-target-bps", type=float, default=25.0)
    parser.add_argument("--train-days", type=int, default=365)
    parser.add_argument("--model-validation-days", type=int, default=45)
    parser.add_argument("--calibration-days", type=int, default=60)
    parser.add_argument("--policy-validation-days", type=int, default=45)
    parser.add_argument("--test-days", type=int, default=60)
    parser.add_argument("--step-days", type=int, default=60)
    parser.add_argument("--shadow-days", type=int, default=30)
    parser.add_argument("--cost-bps", type=v3.parse_float_grid, default=v3.parse_float_grid("20,25,30,40"))
    parser.add_argument("--primary-cost-bps", type=float, default=25.0)
    parser.add_argument(
        "--ev-margin-grid-bps",
        type=v3.parse_float_grid,
        default=v3.parse_float_grid("0,2.5,5,7.5,10,15,20,25"),
    )
    parser.add_argument("--min-policy-trades", type=int, default=3)
    parser.add_argument("--trim-fraction", type=float, default=0.05)
    parser.add_argument("--ablation-trials", type=int, default=2)
    parser.add_argument("--full-trials", type=int, default=8)
    parser.add_argument("--sensitivity-trials", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--final-epochs", type=int, default=100)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--bootstrap-samples", type=int, default=3000)
    parser.add_argument("--starting-capital-eur", type=float, default=1000.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epoch-verbose", action="store_true")
    parser.add_argument(
        "--no-report-zip",
        action="store_true",
        help="Do not create the final research/deployment ZIPs (useful on very small disks).",
    )
    return parser.parse_args()


def _validate_resume_compatibility(meta: dict[str, Any], args: argparse.Namespace) -> None:
    expected_sha = meta.get("dataset_sha256")
    actual_sha = v3.sha256_file(args.dataset)
    if expected_sha and expected_sha != actual_sha:
        raise ValueError(
            "resume dataset SHA256 does not match the original run: "
            f"expected {expected_sha}, got {actual_sha}"
        )
    comparisons = {
        "horizons": args.horizons,
        "feature_sets": args.feature_sets,
        "primary_target_bps": args.primary_target_bps,
        "sensitivity_target_bps": args.sensitivity_target_bps,
        "cost_bps": args.cost_bps,
    }
    for key, current in comparisons.items():
        if key in meta and meta[key] != current:
            raise ValueError(f"resume argument {key}={current!r} does not match original run {meta[key]!r}")
    split = meta.get("split_days", {})
    split_current = {
        "train": args.train_days,
        "model_validation": args.model_validation_days,
        "calibration": args.calibration_days,
        "policy_validation": args.policy_validation_days,
        "test": args.test_days,
        "shadow": args.shadow_days,
    }
    for key, current in split_current.items():
        if key in split and split[key] != current:
            raise ValueError(f"resume split {key}={current} does not match original run {split[key]}")


def _checkpoint_or_run(
    base: v3.ResearchDataset,
    key: v3.ExperimentKey,
    args: argparse.Namespace,
    run_dir: Path,
    *,
    trial_count: int,
) -> dict[str, Any]:
    experiment_dir = run_dir / key.slug
    summary_path = experiment_dir / "summary.json"
    if summary_path.is_file():
        try:
            summary = _load_json(summary_path)
            exp = summary.get("experiment", {})
            if (
                float(exp.get("target_hurdle_bps")) == float(key.target_hurdle_bps)
                and str(exp.get("feature_set")) == key.feature_set
                and int(exp.get("horizon_hours")) == key.horizon_hours
            ):
                print(f"[RESUME] {key.slug}: valid summary.json found -> skipping retraining", flush=True)
                return summary
        except Exception as exc:
            print(f"[RESUME] {key.slug}: invalid checkpoint ({exc}); retraining", flush=True)
    print(f"[RESUME] {key.slug}: no completed checkpoint -> training", flush=True)
    return v3.run_experiment(base, key, args, experiment_dir, trial_count=trial_count)


def _write_final_outputs(
    run_dir: Path,
    args: argparse.Namespace,
    meta: dict[str, Any],
    summaries: list[dict[str, Any]],
    final_shadow: dict[str, Any],
    started: float,
) -> tuple[Path | None, Path | None]:
    final_text = v3._print_final_eur_report(final_shadow)
    (run_dir / "FINAL_1000_EUR_REPORT.txt").write_text(final_text + "\n", encoding="utf-8")

    summaries.sort(
        key=lambda s: (
            float(s["experiment"]["target_hurdle_bps"]) != args.primary_target_bps,
            -float(s["strategy_by_cost_bps"][f"{args.primary_cost_bps:g}"]["sharpe"]),
        )
    )
    v3.write_json(run_dir / "all_results.json", summaries)
    v3.write_json(run_dir / "final_shadow.json", final_shadow)
    (run_dir / "REPORT.md").write_text(v3.markdown_report(meta, summaries, final_shadow), encoding="utf-8")

    comparison_rows: list[dict[str, Any]] = []
    for s in summaries:
        e = s["experiment"]
        p25 = s["strategy_by_cost_bps"]["25"]
        p30 = s["strategy_by_cost_bps"]["30"]
        comparison_rows.append(
            {
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
            }
        )
    v3.write_csv(run_dir / "model_comparison.csv", comparison_rows)

    meta["duration_seconds_last_process"] = time.time() - started
    meta["resumed_or_completed_at"] = datetime.now(timezone.utc).isoformat()
    meta["winner"] = final_shadow["winner"]
    meta["winner_gate_passed"] = final_shadow["winner_research_gate_passed"]
    v3.write_json(run_dir / "metadata.json", meta)

    if args.no_report_zip:
        return None, None

    output_root = run_dir.parent
    run_id = str(meta["run_id"])
    report_zip = output_root / f"{run_id}_research_report.zip"
    with zipfile.ZipFile(report_zip, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in run_dir.rglob("*"):
            if path.is_file() and path.name != "model.keras":
                archive.write(path, path.relative_to(output_root))

    deployment_zip = output_root / f"{run_id}_research_deployment.zip"
    with zipfile.ZipFile(deployment_zip, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in (run_dir / "final_shadow").rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(run_dir))
        for name in (
            "REPORT.md",
            "FINAL_1000_EUR_REPORT.txt",
            "model_comparison.csv",
            "metadata.json",
            "final_shadow.json",
        ):
            path = run_dir / name
            if path.is_file():
                archive.write(path, path.relative_to(run_dir))
    return report_zip, deployment_zip


def main() -> int:
    args = parse_args()
    v3.validate_args(args)
    _install_compact_writer()
    started = time.time()

    if args.resume_run_dir is not None:
        run_dir = args.resume_run_dir.resolve()
        metadata_path = run_dir / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"resume metadata not found: {metadata_path}")
        meta = _load_json(metadata_path)
        _validate_resume_compatibility(meta, args)
        print(f"RESUMING V3 RUN: {run_dir}", flush=True)
        print(f"Dataset SHA256 verified: {meta.get('dataset_sha256')}", flush=True)
    else:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = args.output_root / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        meta = {
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": v3.git_head(),
            "dataset": str(args.dataset),
            "dataset_sha256": v3.sha256_file(args.dataset),
            "feature_version": v3.FEATURE_VERSION,
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
        v3.write_json(run_dir / "metadata.json", meta)

    raw_rows = v3.read_jsonl(args.dataset)
    if not raw_rows:
        raise ValueError("dataset is empty")
    meta["raw_rows"] = len(raw_rows)

    base_by_horizon: dict[int, v3.ResearchDataset] = {
        horizon: v3.build_research_dataset(raw_rows, horizon_hours=horizon)
        for horizon in args.horizons
    }

    summaries: list[dict[str, Any]] = []
    for horizon in args.horizons:
        for feature_set in args.feature_sets:
            key = v3.ExperimentKey(args.primary_target_bps, feature_set, horizon)
            trials = args.full_trials if feature_set == "full_context" else args.ablation_trials
            summaries.append(
                _checkpoint_or_run(base_by_horizon[horizon], key, args, run_dir, trial_count=trials)
            )

    for horizon in args.horizons:
        key = v3.ExperimentKey(args.sensitivity_target_bps, "full_context", horizon)
        summaries.append(
            _checkpoint_or_run(
                base_by_horizon[horizon], key, args, run_dir, trial_count=args.sensitivity_trials
            )
        )

    primary_summaries = [
        s
        for s in summaries
        if float(s["experiment"]["target_hurdle_bps"]) == args.primary_target_bps
    ]
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

    final_shadow_path = run_dir / "final_shadow.json"
    if final_shadow_path.is_file():
        try:
            final_shadow = _load_json(final_shadow_path)
            print("[RESUME] valid final_shadow.json found -> skipping final shadow retraining", flush=True)
        except Exception:
            final_shadow = v3.run_final_shadow(
                base_by_horizon[winner_h], winner, args, run_dir / "final_shadow"
            )
    else:
        final_shadow = v3.run_final_shadow(
            base_by_horizon[winner_h], winner, args, run_dir / "final_shadow"
        )

    report_zip, deployment_zip = _write_final_outputs(
        run_dir, args, meta, summaries, final_shadow, started
    )

    print("\nV3 CHECKPOINTED RESEARCH COMPLETE", flush=True)
    print(f"report={run_dir / 'REPORT.md'}", flush=True)
    print(f"eur1000_report={run_dir / 'FINAL_1000_EUR_REPORT.txt'}", flush=True)
    if report_zip is not None:
        print(f"research_bundle={report_zip}", flush=True)
    if deployment_zip is not None:
        print(f"deployment_bundle={deployment_zip}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
