#!/usr/bin/env bash
set -euo pipefail

# Persist a completed training run in GitHub Releases so an ephemeral GPU server
# can be safely deleted without losing trained models or reports.
#
# Authenticate with either:
#   export GH_TOKEN='...'
# or:
#   gh auth login

REPO_SLUG="${REPO_SLUG:-shahabnab/ai-trading-bot}"
TARGET_BRANCH="${TARGET_BRANCH:-step-2-coinex-readonly}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/ml/server_training}"

if ! command -v gh >/dev/null 2>&1; then
  echo "ERROR: GitHub CLI (gh) is not installed." >&2
  echo "Install gh, then authenticate with GH_TOKEN or 'gh auth login'." >&2
  exit 2
fi

if ! gh auth status >/dev/null 2>&1; then
  echo "ERROR: GitHub CLI is not authenticated." >&2
  echo "Set GH_TOKEN or run: gh auth login" >&2
  exit 3
fi

training_zip="${1:-}"
if [[ -z "$training_zip" ]]; then
  training_zip="$(ls -1t "$OUTPUT_ROOT"/*_training_report.zip 2>/dev/null | head -n1 || true)"
fi
if [[ -z "$training_zip" || ! -f "$training_zip" ]]; then
  echo "ERROR: training-report ZIP not found under $OUTPUT_ROOT" >&2
  exit 4
fi

base="$(basename "$training_zip")"
run_id="${base%_training_report.zip}"
run_dir="$OUTPUT_ROOT/$run_id"
deployment_zip="$OUTPUT_ROOT/${run_id}_deployment_models.zip"
report="$run_dir/REPORT.md"
comparison="$run_dir/model_comparison.csv"
metadata="$run_dir/metadata.json"
tag="training-${run_id}"
title="AI trading ML training ${run_id}"

assets=("$training_zip")
[[ -f "$deployment_zip" ]] && assets+=("$deployment_zip")
[[ -f "$comparison" ]] && assets+=("$comparison")
[[ -f "$metadata" ]] && assets+=("$metadata")

if gh release view "$tag" --repo "$REPO_SLUG" >/dev/null 2>&1; then
  echo "Release $tag already exists; replacing assets."
  gh release upload "$tag" "${assets[@]}" --repo "$REPO_SLUG" --clobber
else
  if [[ -f "$report" ]]; then
    gh release create "$tag" "${assets[@]}" \
      --repo "$REPO_SLUG" \
      --target "$TARGET_BRANCH" \
      --title "$title" \
      --notes-file "$report" \
      --prerelease
  else
    gh release create "$tag" "${assets[@]}" \
      --repo "$REPO_SLUG" \
      --target "$TARGET_BRANCH" \
      --title "$title" \
      --notes "Automated ML training archive for $run_id" \
      --prerelease
  fi
fi

echo
echo "Training artifacts persisted to GitHub Release: $tag"
echo "Repository: $REPO_SLUG"
echo "Verify the release assets before deleting the GPU server."
