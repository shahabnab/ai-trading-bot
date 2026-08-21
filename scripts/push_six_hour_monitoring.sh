#!/usr/bin/env bash
set -euo pipefail

# Export a compact decision-vs-reality monitoring snapshot and push it to the
# dedicated paper-live-results branch. The running application checkout is
# never switched; all result commits happen in an isolated git worktree.

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RESULTS_BRANCH="${PAPER_RESULTS_BRANCH:-paper-live-results}"
RESULTS_WORKTREE="${PAPER_RESULTS_WORKTREE:-$(dirname "$REPO_ROOT")/ai-trading-bot-paper-results}"
API_BASE="${PAPER_API_BASE:-http://127.0.0.1:8000}"
DB_PATH="${PAPER_DB_PATH:-$REPO_ROOT/data/paper_trading.sqlite3}"
PYTHON_BIN="${PAPER_PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3 || command -v python)"
fi

cd "$REPO_ROOT"

if [[ ! -d "$RESULTS_WORKTREE/.git" && ! -f "$RESULTS_WORKTREE/.git" ]]; then
  rm -rf "$RESULTS_WORKTREE"
  git fetch origin "$RESULTS_BRANCH"
  git worktree add -B "$RESULTS_BRANCH" "$RESULTS_WORKTREE" "origin/$RESULTS_BRANCH"
else
  git -C "$RESULTS_WORKTREE" fetch origin "$RESULTS_BRANCH"
  git -C "$RESULTS_WORKTREE" checkout "$RESULTS_BRANCH"
  git -C "$RESULTS_WORKTREE" pull --ff-only origin "$RESULTS_BRANCH"
fi

"$PYTHON_BIN" "$REPO_ROOT/scripts/export_six_hour_monitoring.py" \
  --repo-root "$REPO_ROOT" \
  --db "$DB_PATH" \
  --out-dir "$RESULTS_WORKTREE/paper_monitoring" \
  --api-base "$API_BASE"

git -C "$RESULTS_WORKTREE" add paper_monitoring

if git -C "$RESULTS_WORKTREE" diff --cached --quiet; then
  echo "No six-hour monitoring changes to commit."
  exit 0
fi

STAMP="$(date -u +%FT%H:%MZ)"
git -C "$RESULTS_WORKTREE" commit -m "paper: six-hour monitoring snapshot $STAMP"
git -C "$RESULTS_WORKTREE" push origin "$RESULTS_BRANCH"

echo "Six-hour monitoring snapshot pushed to origin/$RESULTS_BRANCH"
