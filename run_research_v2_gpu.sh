#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-data/processed/training/btc_hourly.jsonl}"
EXPECTED_DATASET_SHA256="${EXPECTED_DATASET_SHA256:-}"
PUBLISH_TO_GITHUB="${PUBLISH_TO_GITHUB:-1}"

if [[ -x .venv-training/bin/python ]]; then
  PYTHON_BIN="${PYTHON_BIN:-.venv-training/bin/python}"
elif [[ -x .venv311/bin/python ]]; then
  PYTHON_BIN="${PYTHON_BIN:-.venv311/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi

if [[ ! -f "$DATASET" ]]; then
  echo "ERROR: dataset not found: $DATASET" >&2
  exit 10
fi

actual="$(sha256sum "$DATASET" | awk '{print $1}')"
if [[ -n "$EXPECTED_DATASET_SHA256" ]]; then
  expected="$(printf '%s' "$EXPECTED_DATASET_SHA256" | tr '[:upper:]' '[:lower:]')"
  if [[ "$actual" != "$expected" ]]; then
    echo "ERROR: dataset SHA256 mismatch" >&2
    echo "expected: $expected" >&2
    echo "actual:   $actual" >&2
    exit 11
  fi
fi

echo "Dataset SHA256: $actual"
echo '=== ENVIRONMENT CHECK ==='
"$PYTHON_BIN" --version
nvidia-smi || true
"$PYTHON_BIN" - <<'PY'
import tensorflow as tf
import xgboost as xgb
from backend.ml.features import FEATURE_VERSION
print('TensorFlow:', tf.__version__)
print('XGBoost:', xgb.__version__)
print('Feature version:', FEATURE_VERSION)
print('GPUs:', tf.config.list_physical_devices('GPU'))
PY

echo '=== START MULTI-HORIZON RESEARCH V2 ==='
"$PYTHON_BIN" run_research_experiments_v2.py \
  --dataset "$DATASET" \
  --models xgboost lstm_v2 \
  --horizons 1,3,6,12 \
  --train-days 365 \
  --validation-days 60 \
  --test-days 60 \
  --step-days 60 \
  --cost-bps 10,15,20,25 \
  --primary-cost-bps 25 \
  --threshold-grid 1,1.25,1.5,2,2.5,3,4,5,6,8,10 \
  --holding-grid 1,3,6,12 \
  --prob-entry-grid 0.52,0.55,0.58,0.60,0.65,0.70 \
  --prob-exit-grid 0.30,0.35,0.40,0.45,0.48,0.50 \
  --min-policy-trades 5 \
  --min-total-test-trades 20 \
  --xgb-trials 8 \
  --lstm-trials 4 \
  --lstm-epochs 35 \
  --epoch-verbose

latest_zip="$(ls -1t artifacts/ml/research_experiments/*_research_report.zip | head -n1)"

echo
if [[ "$PUBLISH_TO_GITHUB" == "1" ]]; then
  echo '=== PERSIST RESEARCH V2 ARTIFACTS TO GITHUB ==='
  bash publish_research_release.sh "$latest_zip"
else
  echo "PUBLISH_TO_GITHUB=0: publishing skipped."
  echo "Research bundle: $latest_zip"
fi
