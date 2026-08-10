#!/usr/bin/env bash
set -euo pipefail

# Run from the ai-trading-bot repository root.
# By default, successful training is also published to a GitHub prerelease.

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV="${VENV:-.venv-training}"
PUBLISH_TO_GITHUB="${PUBLISH_TO_GITHUB:-1}"
EXPECTED_DATASET_SHA256="${EXPECTED_DATASET_SHA256:-}"
DATASET="${DATASET:-data/processed/training/btc_hourly.jsonl}"

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

[[ -d "$VENV" ]] || "$PYTHON_BIN" -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel setuptools
python -m pip install -r requirements.txt
python -m pip install 'tensorflow[and-cuda]>=2.21,<2.22'

echo '=== GPU CHECK ==='
nvidia-smi || true
python - <<'PY'
import tensorflow as tf
print('TensorFlow:', tf.__version__)
print('GPUs:', tf.config.list_physical_devices('GPU'))
PY

echo '=== START TRAINING ==='
python train_all_server.py \
  --dataset "$DATASET" \
  --models xgboost lstm \
  --feature-sets base \
  --train-days 365 \
  --validation-days 60 \
  --test-days 60 \
  --step-days 60 \
  --fee-bps 20 \
  --slippage-bps 5 \
  --spread-bps 0 \
  --epochs 50 \
  --batch-size 64

latest_training_zip="$(ls -1t artifacts/ml/server_training/*_training_report.zip | head -n1)"

echo
if [[ "$PUBLISH_TO_GITHUB" == "1" ]]; then
  echo '=== PERSIST TRAINING ARTIFACTS TO GITHUB ==='
  bash publish_training_release.sh "$latest_training_zip"
else
  echo "PUBLISH_TO_GITHUB=0: GitHub publishing skipped."
  echo "Copy this file somewhere durable before deleting the GPU server:"
  echo "  $latest_training_zip"
fi
