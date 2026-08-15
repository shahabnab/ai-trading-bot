#!/usr/bin/env bash
set -euo pipefail

# Export paper-trading state from the running server and push it to a dedicated
# Git branch. Runtime code stays on step-5-live-paper-dashboard; audit snapshots
# are committed to paper-live-results so daily data does not clutter code history.

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RESULTS_BRANCH="${PAPER_RESULTS_BRANCH:-paper-live-results}"
RESULTS_WORKTREE="${PAPER_RESULTS_WORKTREE:-$(dirname "$REPO_ROOT")/ai-trading-bot-paper-results}"
API_BASE="${PAPER_API_BASE:-http://127.0.0.1:8000}"
DB_PATH="${PAPER_DB_PATH:-$REPO_ROOT/data/paper_trading.sqlite3}"
PYTHON_BIN="${PAPER_PYTHON_BIN:-$REPO_ROOT/.venv-training/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3 || command -v python)"
fi

cd "$REPO_ROOT"

# The dedicated results branch is intentionally manipulated only in a separate
# worktree so the running application checkout is never switched underneath
# FastAPI/Next.js.
if [[ ! -d "$RESULTS_WORKTREE/.git" && ! -f "$RESULTS_WORKTREE/.git" ]]; then
  rm -rf "$RESULTS_WORKTREE"
  git fetch origin "$RESULTS_BRANCH"
  git worktree add "$RESULTS_WORKTREE" "origin/$RESULTS_BRANCH"
fi

# Keep the audit worktree synchronized. This branch should only receive snapshot
# commits, so a fast-forward pull is the safe behavior.
git -C "$RESULTS_WORKTREE" fetch origin "$RESULTS_BRANCH"
git -C "$RESULTS_WORKTREE" checkout -B "$RESULTS_BRANCH" "origin/$RESULTS_BRANCH"

"$PYTHON_BIN" "$REPO_ROOT/scripts/export_daily_paper_snapshot.py" \
  --repo-root "$REPO_ROOT" \
  --db "$DB_PATH" \
  --out-dir "$RESULTS_WORKTREE/paper_snapshots" \
  --api-base "$API_BASE"

git -C "$RESULTS_WORKTREE" add paper_snapshots

if git -C "$RESULTS_WORKTREE" diff --cached --quiet; then
  echo "No paper snapshot changes to commit."
  exit 0
fi

DAY="$(date -u +%F)"
git -C "$RESULTS_WORKTREE" commit -m "paper: daily forward snapshot $DAY"
git -C "$RESULTS_WORKTREE" push origin "$RESULTS_BRANCH"

echo "Daily paper snapshot pushed to origin/$RESULTS_BRANCH"
