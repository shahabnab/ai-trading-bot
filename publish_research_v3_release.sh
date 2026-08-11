#!/usr/bin/env bash
set -euo pipefail

REPO="${GITHUB_REPOSITORY:-shahabnab/ai-trading-bot}"
TARGET="${GITHUB_TARGET:-step-4-ev-calibrated-v3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/ml/research_v3}"

command -v gh >/dev/null 2>&1 || { echo "ERROR: GitHub CLI (gh) is required." >&2; exit 20; }
gh auth status >/dev/null 2>&1 || { echo "ERROR: gh is not authenticated." >&2; exit 21; }

research_zip="${1:-}"
if [[ -z "$research_zip" ]]; then
  research_zip="$(ls -1t "$OUTPUT_ROOT"/*_research_report.zip 2>/dev/null | head -n1 || true)"
fi
[[ -f "$research_zip" ]] || { echo "ERROR: research report ZIP not found." >&2; exit 22; }

base="$(basename "$research_zip")"
run_id="${base%_research_report.zip}"
run_dir="$OUTPUT_ROOT/$run_id"
deployment_zip="$OUTPUT_ROOT/${run_id}_research_deployment.zip"
tag="research-v3-${run_id}"
[[ -f "$run_dir/REPORT.md" ]] || { echo "ERROR: report missing: $run_dir/REPORT.md" >&2; exit 23; }

assets=("$research_zip")
[[ -f "$deployment_zip" ]] && assets+=("$deployment_zip")
for name in REPORT.md FINAL_1000_EUR_REPORT.txt model_comparison.csv metadata.json final_shadow.json; do
  [[ -f "$run_dir/$name" ]] && assets+=("$run_dir/$name")
done
[[ -f "$run_dir/final_shadow/final_30d_equity.csv" ]] && assets+=("$run_dir/final_shadow/final_30d_equity.csv")

if gh release view "$tag" --repo "$REPO" >/dev/null 2>&1; then
  gh release upload "$tag" "${assets[@]}" --repo "$REPO" --clobber
else
  gh release create "$tag" "${assets[@]}" \
    --repo "$REPO" \
    --target "$TARGET" \
    --title "AI trading V3 calibrated EV research ${run_id}" \
    --notes-file "$run_dir/REPORT.md" \
    --prerelease
fi

echo "V3 research artifacts persisted to GitHub Release: $tag"
echo "Repository: $REPO"
echo "Verify the release assets before deleting the GPU server."
