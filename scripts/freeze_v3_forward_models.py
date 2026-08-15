#!/usr/bin/env python3
"""Freeze the two pre-selected V3 models for prospective CPU paper trading.

This script is intentionally a *one-time final fit*, not online fine-tuning. It
reuses the model candidates selected by the completed V3 research run, fits the
same LSTM/calibration/policy pipeline close to the end of the historical data,
and exports everything required for deterministic forward inference:

* Keras model
* input/output standardization statistics
* Platt calibrator
* payoff estimate
* EV entry margin / horizon policy
* feature names and candidate hyperparameters
* dataset/model SHA256 and exact fit-window metadata

The default 1-day smoke holdout is only a deployment sanity check. It is not a
new research test. The prospective paper experiment begins after the dataset
cutoff recorded in the exported manifest.
"""
from __future__ import annotations

import argparse
import json
import zipfile
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import run_research_experiments_v3 as v3
from backend.ml.sequences import fit_standardizer


MODEL_SPECS = (
    {
        "model_id": "v3-25bps-fullcontext-12h",
        "display_name": "V3 12h Economic",
        "summary_slug": "target25bps_full_context_h12",
    },
    {
        "model_id": "v3-50bps-technical-3h",
        "display_name": "V3 3h Signal Control",
        "summary_slug": "target50bps_technical_h03",
    },
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _iso(ts_ms: int) -> str:
    return datetime.fromtimestamp(int(ts_ms) / 1000.0, timezone.utc).isoformat()


def _window(dataset: v3.ResearchDataset, indices: np.ndarray) -> dict[str, Any]:
    if len(indices) == 0:
        return {"rows": 0, "start": None, "end": None}
    return {
        "rows": int(len(indices)),
        "start": _iso(int(dataset.timestamps[int(indices[0])])),
        "end": _iso(int(dataset.timestamps[int(indices[-1])])),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze selected V3 models for forward paper trading")
    parser.add_argument(
        "--research-run-dir",
        type=Path,
        default=Path("artifacts/ml/research_v3/20260811T115429Z"),
        help="Completed V3 research run containing per-experiment summary.json files.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/processed/training/btc_hourly_v3.jsonl"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/ml/forward_deployment/v3-paper"),
    )
    parser.add_argument(
        "--smoke-holdout-days",
        type=int,
        default=1,
        help="Small deployment sanity holdout. Future paper data remains the real prospective test.",
    )
    parser.add_argument("--final-epochs", type=int, default=100)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _runtime_args(meta: dict[str, Any], cli: argparse.Namespace) -> Namespace:
    split = meta.get("split_days", {}) if isinstance(meta.get("split_days"), dict) else {}
    return Namespace(
        train_days=int(split.get("train", 365)),
        model_validation_days=int(split.get("model_validation", 45)),
        calibration_days=int(split.get("calibration", 60)),
        policy_validation_days=int(split.get("policy_validation", 45)),
        shadow_days=int(cli.smoke_holdout_days),
        final_epochs=int(cli.final_epochs),
        early_stopping_patience=int(cli.early_stopping_patience),
        seed=int(cli.seed),
        trim_fraction=0.05,
        primary_cost_bps=25.0,
        ev_margin_grid_bps=[0.0, 2.5, 5.0, 7.5, 10.0, 15.0, 20.0, 25.0],
        min_policy_trades=3,
        cost_bps=[20.0, 25.0, 30.0, 40.0],
        starting_capital_eur=1000.0,
    )


def main() -> int:
    cli = parse_args()
    if cli.smoke_holdout_days <= 0:
        raise ValueError("--smoke-holdout-days must be positive")
    if not cli.dataset.is_file():
        raise FileNotFoundError(cli.dataset)
    if not cli.research_run_dir.is_dir():
        raise FileNotFoundError(cli.research_run_dir)

    metadata_path = cli.research_run_dir / "metadata.json"
    meta = _load_json(metadata_path) if metadata_path.is_file() else {}
    args = _runtime_args(meta, cli)

    expected_sha = meta.get("dataset_sha256")
    dataset_sha = v3.sha256_file(cli.dataset)
    if expected_sha and expected_sha != dataset_sha:
        raise ValueError(
            "dataset SHA256 differs from the completed V3 research run: "
            f"expected {expected_sha}, got {dataset_sha}"
        )

    raw_rows = v3.read_jsonl(cli.dataset)
    if not raw_rows:
        raise ValueError("dataset is empty")

    cli.output_root.mkdir(parents=True, exist_ok=True)
    bundle_manifest: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "deployment_kind": "prospective_forward_paper",
        "source_research_run": str(cli.research_run_dir),
        "source_research_git_commit": meta.get("git_commit"),
        "dataset": str(cli.dataset),
        "dataset_sha256": dataset_sha,
        "smoke_holdout_days": cli.smoke_holdout_days,
        "fine_tuning_policy": "frozen_during_forward_test",
        "models": [],
    }

    for spec in MODEL_SPECS:
        summary_path = cli.research_run_dir / str(spec["summary_slug"]) / "summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"required research summary missing: {summary_path}")
        summary = _load_json(summary_path)
        exp = summary.get("experiment")
        if not isinstance(exp, dict):
            raise ValueError(f"summary missing experiment object: {summary_path}")

        horizon = int(exp["horizon_hours"])
        feature_set = str(exp["feature_set"])
        model_id = str(spec["model_id"])
        print(f"\n=== FREEZING {model_id} ({feature_set}, {horizon}h) ===", flush=True)

        base = v3.build_research_dataset(raw_rows, horizon_hours=horizon)
        enriched = v3.enrich_dataset(base, feature_set)
        shadow_start = int(enriched.timestamps[-1]) - args.shadow_days * v3.DAY_MS
        split = v3._final_window_indices(enriched, shadow_start, args)

        x_stats = fit_standardizer(enriched.X[split.train])
        y_stats = fit_standardizer(enriched.target_log_return[split.train])

        out = cli.output_root / model_id
        out.mkdir(parents=True, exist_ok=True)

        # Reuse the exact V3 final-fit implementation and the modal candidate
        # selected by the completed walk-forward research summary.
        smoke_result = v3.run_final_shadow(base, summary, args, out)

        standardizer = {
            "x_mean": np.asarray(x_stats.mean, dtype=np.float64).tolist(),
            "x_scale": np.asarray(x_stats.scale, dtype=np.float64).tolist(),
            "y_mean": np.asarray(y_stats.mean, dtype=np.float64).reshape(-1).tolist(),
            "y_scale": np.asarray(y_stats.scale, dtype=np.float64).reshape(-1).tolist(),
        }
        v3.write_json(out / "standardizer.json", standardizer)

        manifest_path = out / "manifest.json"
        manifest = _load_json(manifest_path)
        manifest.update(
            {
                "model_id": model_id,
                "display_name": spec["display_name"],
                "deployment_kind": "prospective_forward_paper",
                "source_research_run": str(cli.research_run_dir),
                "source_research_summary": str(summary_path),
                "source_research_summary_sha256": v3.sha256_file(summary_path),
                "dataset_sha256": dataset_sha,
                "dataset_last_feature_timestamp": _iso(int(enriched.timestamps[-1])),
                "standardizer_file": "standardizer.json",
                "fit_windows": {
                    "train": _window(enriched, split.train),
                    "model_validation": _window(enriched, split.model_validation),
                    "calibration": _window(enriched, split.calibration),
                    "policy_validation": _window(enriched, split.policy_validation),
                    "smoke_holdout": _window(enriched, split.test),
                },
                "smoke_holdout_days": args.shadow_days,
                "forward_test_rule": "Do not retrain or tune this artifact during the predefined forward paper period.",
            }
        )
        if (out / "model.keras").is_file():
            manifest["model_sha256"] = v3.sha256_file(out / "model.keras")
        v3.write_json(manifest_path, manifest)

        bundle_manifest["models"].append(
            {
                "model_id": model_id,
                "display_name": spec["display_name"],
                "experiment": exp,
                "research_gate_passed": bool(summary.get("gate_passed", False)),
                "research_auc": summary.get("overall_classification", {}).get("auc"),
                "research_median_fold_auc": summary.get("median_fold_auc"),
                "smoke_classification": smoke_result.get("classification"),
                "artifact_dir": model_id,
                "model_sha256": manifest.get("model_sha256"),
            }
        )

    v3.write_json(cli.output_root / "deployment_manifest.json", bundle_manifest)

    zip_path = cli.output_root.parent / f"{cli.output_root.name}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in cli.output_root.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(cli.output_root.parent))

    print("\nFORWARD MODEL FREEZE COMPLETE", flush=True)
    print(f"deployment_dir={cli.output_root}", flush=True)
    print(f"deployment_zip={zip_path}", flush=True)
    print(f"dataset_sha256={dataset_sha}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
