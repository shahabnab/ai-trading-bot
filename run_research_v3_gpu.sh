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
RESUME_RUN_DIR="${RESUME_RUN_DIR:-}"
NO_REPORT_ZIP="${NO_REPORT_ZIP:-0}"

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

echo "=== RUN V3 MULTI-HOUR RESEARCH (CHECKPOINTED) ==="
echo "This is intentionally a large research run. Completed experiments are resumable via summary.json checkpoints."
echo "Full-context trials=$FULL_TRIALS, ablation trials=$ABLATION_TRIALS, sensitivity trials=$SENSITIVITY_TRIALS, epochs=$EPOCHS"

resume_args=()
if [[ -n "$RESUME_RUN_DIR" ]]; then
  resume_args+=(--resume-run-dir "$RESUME_RUN_DIR")
  echo "Resume directory: $RESUME_RUN_DIR"
fi
zip_args=()
if [[ "$NO_REPORT_ZIP" == "1" ]]; then
  zip_args+=(--no-report-zip)
fi

"$PYTHON_BIN" run_research_experiments_v3_checkpointed.py \
  "${resume_args[@]}" \
  "${zip_args[@]}" \
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

if [[ -n "$RESUME_RUN_DIR" ]]; then
  RUN_DIR="$RESUME_RUN_DIR"
else
  RUN_DIR="$(ls -1dt artifacts/ml/research_v3/20*/ 2>/dev/null | head -n1)"
fi

latest_zip=""
if [[ "$NO_REPORT_ZIP" != "1" ]]; then
  latest_zip="$(ls -1t artifacts/ml/research_v3/*_research_report.zip 2>/dev/null | head -n1 || true)"
fi

if [[ "$PUBLISH_TO_GITHUB" == "1" && -n "$latest_zip" ]]; then
  echo
  echo "=== PERSIST V3 ARTIFACTS TO GITHUB ==="
  bash publish_research_v3_release.sh "$latest_zip"
elif [[ "$PUBLISH_TO_GITHUB" == "1" ]]; then
  echo "No report ZIP exists (NO_REPORT_ZIP=1 or ZIP creation skipped); GitHub publishing skipped."
else
  echo "PUBLISH_TO_GITHUB=0: publishing skipped."
fi

echo "Run directory: $RUN_DIR"
