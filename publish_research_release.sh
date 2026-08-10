#!/usr/bin/env bash
set -euo pipefail

REPO="${GITHUB_REPOSITORY:-shahabnab/ai-trading-bot}"
TARGET="${GITHUB_TARGET:-step-2-coinex-readonly}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/ml/research_experiments}"

if ! command -v gh >/dev/null 2>&1; then
  echo "ERROR: GitHub CLI (gh) is required." >&2
  exit 20
fi
if ! gh auth status >/dev/null 2>&1; then
  echo "ERROR: gh is not authenticated. Run: gh auth login" >&2
  exit 21
fi

research_zip="${1:-}"
if [[ -z "$research_zip" ]]; then
  research_zip="$(ls -1t "$OUTPUT_ROOT"/*_research_report.zip 2>/dev/null | head -n1 || true)"
fi
if [[ -z "$research_zip" || ! -f "$research_zip" ]]; then
  echo "ERROR: research report ZIP not found." >&2
  exit 22
fi

base="$(basename "$research_zip")"
run_id="${base%_research_report.zip}"
run_dir="$OUTPUT_ROOT/$run_id"
deployment_zip="$OUTPUT_ROOT/${run_id}_research_deployment.zip"
tag="research-${run_id}"

if [[ ! -f "$run_dir/REPORT.md" ]]; then
  echo "ERROR: report not found: $run_dir/REPORT.md" >&2
  exit 23
fi

assets=("$research_zip")
[[ -f "$deployment_zip" ]] && assets+=("$deployment_zip")
[[ -f "$run_dir/model_comparison.csv" ]] && assets+=("$run_dir/model_comparison.csv")
[[ -f "$run_dir/metadata.json" ]] && assets+=("$run_dir/metadata.json")
[[ -f "$run_dir/baselines.json" ]] && assets+=("$run_dir/baselines.json")

if gh release view "$tag" --repo "$REPO" >/dev/null 2>&1; then
  gh release upload "$tag" "${assets[@]}" --repo "$REPO" --clobber
else
  gh release create "$tag" "${assets[@]}" \
    --repo "$REPO" \
    --target "$TARGET" \
    --title "AI trading multi-horizon research ${run_id}" \
    --notes-file "$run_dir/REPORT.md" \
    --prerelease
fi

echo "Research artifacts persisted to GitHub Release: $tag"
echo "Repository: $REPO"
echo "Verify the release assets before deleting the GPU server."
