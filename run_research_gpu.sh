#!/usr/bin/env bash
set -euo pipefail

# Run from the ai-trading-bot repository root.
# Reuses the already-prepared training venv when present and publishes results.

DATASET="${DATASET:-data/processed/training/btc_hourly.jsonl}"
EXPECTED_DATASET_SHA256="${EXPECTED_DATASET_SHA256:-}"
PUBLISH_TO_GITHUB="${PUBLISH_TO_GITHUB:-1}"

if [[ -x .venv-training/bin/python ]]; then
  PYTHON_BIN="${PYTHON_BIN:-.venv-training/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi

if [[ ! -f "$DATASET" ]]; then
  echo "ERROR: dataset not found: $DATASET" >&2
  exit 10
fi

if [[ -n "$EXPECTED_DATASET_SHA256" ]]; then
  actual="$(sha256sum "$DATASET" | awk '{print $1}')"
  expected="$(printf '%s' "$EXPECTED_DATASET_SHA256" | tr '[:upper:]' '[:lower:]')"
  if [[ "$actual" != "$expected" ]]; then
    echo "ERROR: dataset SHA256 mismatch" >&2
    echo "expected: $expected" >&2
    echo "actual:   $actual" >&2
    exit 11
  fi
  echo "Dataset SHA256 verified: $actual"
else
  echo "Dataset SHA256: $(sha256sum "$DATASET" | awk '{print $1}')"
fi

echo '=== ENVIRONMENT CHECK ==='
"$PYTHON_BIN" --version
nvidia-smi || true
"$PYTHON_BIN" - <<'PY'
import tensorflow as tf
import xgboost as xgb
print('TensorFlow:', tf.__version__)
print('XGBoost:', xgb.__version__)
print('GPUs:', tf.config.list_physical_devices('GPU'))
PY

echo '=== START MULTI-HORIZON RESEARCH ==='
"$PYTHON_BIN" run_research_experiments.py \
  --dataset "$DATASET" \
  --models xgboost lstm \
  --horizons 1,3,6,12 \
  --train-days 365 \
  --validation-days 60 \
  --test-days 60 \
  --step-days 60 \
  --cost-bps 10,15,20,25 \
  --primary-cost-bps 25 \
  --threshold-grid 1,1.25,1.5,2,2.5,3,4,5,6,8,10 \
  --holding-grid 1,3,6,12 \
  --xgb-trials 8 \
  --lstm-trials 4 \
  --lstm-epochs 35

latest_zip="$(ls -1t artifacts/ml/research_experiments/*_research_report.zip | head -n1)"

echo
if [[ "$PUBLISH_TO_GITHUB" == "1" ]]; then
  echo '=== PERSIST RESEARCH ARTIFACTS TO GITHUB ==='
  bash publish_research_release.sh "$latest_zip"
else
  echo "PUBLISH_TO_GITHUB=0: GitHub publishing skipped."
  echo "Copy this file somewhere durable before deleting the GPU server:"
  echo "  $latest_zip"
fi
