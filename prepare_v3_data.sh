#!/usr/bin/env bash
set -euo pipefail

START_DATE="${START_DATE:-2020-01-01}"
END_DATE="${END_DATE:-$(date -u -d 'yesterday' +%F)}"
KEEP_ARCHIVES="${KEEP_ARCHIVES:-0}"
FORCE_CONTEXT="${FORCE_CONTEXT:-0}"

if [[ -x .venv-training/bin/python ]]; then
  PYTHON_BIN="${PYTHON_BIN:-.venv-training/bin/python}"
elif [[ -x .venv311/bin/python ]]; then
  PYTHON_BIN="${PYTHON_BIN:-.venv311/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi

args=(
  -m backend.data_collection.binance_research_context
  --start "$START_DATE"
  --end "$END_DATE"
)
[[ "$KEEP_ARCHIVES" == "1" ]] && args+=(--keep-archives)
[[ "$FORCE_CONTEXT" == "1" ]] && args+=(--force)

echo "=== PREPARE V3 MARKET DATA ==="
echo "Range: $START_DATE -> $END_DATE"
echo "Python: $PYTHON_BIN"
echo "Raw ZIP retention: $KEEP_ARCHIVES"
echo
"$PYTHON_BIN" "${args[@]}"

echo
echo "=== V3 DATASET CHECK ==="
DATASET="data/processed/training/btc_hourly_v3.jsonl"
for file in \
  "$DATASET" \
  data/processed/context/btc_spot_aggtrades_hourly.jsonl \
  data/processed/context/btc_um_futures_hourly.jsonl \
  data/processed/context/eth_spot_hourly.jsonl; do
  test -s "$file" || { echo "ERROR: missing/empty $file" >&2; exit 20; }
  echo "$(wc -l < "$file") rows  $file"
done
sha256sum "$DATASET"

echo "V3 data preparation complete."
