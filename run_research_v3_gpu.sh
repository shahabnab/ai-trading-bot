#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-data/processed/training/btc_hourly_v3.jsonl}"
PUBLISH_TO_GITHUB="${PUBLISH_TO_GITHUB:-1}"
PREPARE_DATA="${PREPARE_DATA:-0}"
FULL_TRIALS="${FULL_TRIALS:-8}"
ABLATION_TRIALS="${ABLATION_TRIALS:-2}"
SENSITIVITY_TRIALS="${SENSITIVITY_TRIALS:-4}"
EPOCHS="${EPOCHS:-80}"
FINAL_EPOCHS="${FINAL_EPOCHS:-100}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-3000}"

if [[ -x .venv-training/bin/python ]]; then
  PYTHON_BIN="${PYTHON_BIN:-.venv-training/bin/python}"
elif [[ -x .venv311/bin/python ]]; then
  PYTHON_BIN="${PYTHON_BIN:-.venv311/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi

if [[ "$PREPARE_DATA" == "1" ]]; then
  bash prepare_v3_data.sh
fi

required=(
  "$DATASET"
  data/processed/context/btc_spot_aggtrades_hourly.jsonl
  data/processed/context/btc_um_futures_hourly.jsonl
  data/processed/context/eth_spot_hourly.jsonl
)
for file in "${required[@]}"; do
  [[ -s "$file" ]] || { echo "ERROR: V3 input missing: $file" >&2; echo "Run: bash prepare_v3_data.sh" >&2; exit 10; }
done

echo "=== V3 ENVIRONMENT CHECK ==="
"$PYTHON_BIN" --version
nvidia-smi || true
"$PYTHON_BIN" - <<'PY'
import tensorflow as tf
from backend.ml.features import FEATURE_VERSION
print('TensorFlow:', tf.__version__)
print('Feature version:', FEATURE_VERSION)
print('GPUs:', tf.config.list_physical_devices('GPU'))
PY

echo
echo "Dataset: $DATASET"
sha256sum "$DATASET"
echo "Rows: $(wc -l < "$DATASET")"
echo

echo "=== RUN V3 MULTI-HOUR RESEARCH ==="
echo "This is intentionally a large research run. Actual duration depends on fold count and early stopping."
echo "Full-context trials=$FULL_TRIALS, ablation trials=$ABLATION_TRIALS, sensitivity trials=$SENSITIVITY_TRIALS, epochs=$EPOCHS"

"$PYTHON_BIN" run_research_experiments_v3.py \
  --dataset "$DATASET" \
  --horizons 1,3,6,12 \
  --feature-sets technical technical_micro full_context \
  --primary-target-bps 50 \
  --sensitivity-target-bps 25 \
  --train-days 365 \
  --model-validation-days 45 \
  --calibration-days 60 \
  --policy-validation-days 45 \
  --test-days 60 \
  --step-days 60 \
  --shadow-days 30 \
  --cost-bps 20,25,30,40 \
  --primary-cost-bps 25 \
  --ev-margin-grid-bps 0,2.5,5,7.5,10,15,20,25 \
  --min-policy-trades 3 \
  --trim-fraction 0.05 \
  --ablation-trials "$ABLATION_TRIALS" \
  --full-trials "$FULL_TRIALS" \
  --sensitivity-trials "$SENSITIVITY_TRIALS" \
  --epochs "$EPOCHS" \
  --final-epochs "$FINAL_EPOCHS" \
  --early-stopping-patience 10 \
  --bootstrap-samples "$BOOTSTRAP_SAMPLES" \
  --starting-capital-eur 1000 \
  --epoch-verbose

latest_zip="$(ls -1t artifacts/ml/research_v3/*_research_report.zip | head -n1)"

if [[ "$PUBLISH_TO_GITHUB" == "1" ]]; then
  echo
  echo "=== PERSIST V3 ARTIFACTS TO GITHUB ==="
  bash publish_research_v3_release.sh "$latest_zip"
else
  echo "PUBLISH_TO_GITHUB=0: publishing skipped."
  echo "Research bundle: $latest_zip"
fi
