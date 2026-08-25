#!/usr/bin/env bash
set -euo pipefail

# Capture tracked production Git drift without modifying the working tree.
# This script deliberately does NOT run reset, checkout, clean, stash, pull,
# merge, rebase, or any other command that can change production files.

REPO_ROOT="${1:-.}"
OUT_ROOT="${2:-production_drift_capture}"

if ! git -C "$REPO_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Not a Git work tree: $REPO_ROOT" >&2
  exit 2
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="$OUT_ROOT/$STAMP"
mkdir -p "$OUT_DIR"

# Metadata useful for comparing VPS state with GitHub. None of these commands
# mutate the repository.
git -C "$REPO_ROOT" rev-parse HEAD > "$OUT_DIR/head.txt"
git -C "$REPO_ROOT" branch --show-current > "$OUT_DIR/branch.txt" || true
git -C "$REPO_ROOT" remote -v > "$OUT_DIR/remotes.txt" || true
git -C "$REPO_ROOT" status --short --branch > "$OUT_DIR/status.txt"
git -C "$REPO_ROOT" diff --stat > "$OUT_DIR/tracked_diff_stat.txt"
git -C "$REPO_ROOT" diff --cached --stat > "$OUT_DIR/staged_diff_stat.txt"

# Preserve tracked changes as patches. --binary keeps tracked binary deltas
# representable without copying the whole working tree.
git -C "$REPO_ROOT" diff --binary > "$OUT_DIR/tracked_worktree.patch"
git -C "$REPO_ROOT" diff --cached --binary > "$OUT_DIR/staged.patch"

# Record untracked paths only. We intentionally do not copy their contents
# because production untracked files may include secrets or runtime data.
git -C "$REPO_ROOT" ls-files --others --exclude-standard > "$OUT_DIR/untracked_paths.txt"

cat > "$OUT_DIR/README.txt" <<'EOF'
This directory is a read-only capture of production Git drift.

Safe next steps:
1. Review status.txt and the two diff-stat files.
2. Review tracked_worktree.patch and staged.patch for legitimate code changes.
3. Convert legitimate changes into reviewed Git commits on a branch.
4. Inspect untracked_paths.txt manually; do not publish secrets/runtime data.
5. Only after the legitimate changes are preserved and reviewed, restore a clean
   production checkout using your normal deployment process.

No destructive Git command was executed by the capture script.
EOF

printf 'Captured production Git drift at: %s\n' "$OUT_DIR"
printf 'HEAD: %s\n' "$(cat "$OUT_DIR/head.txt")"
printf 'Tracked changes: %s\n' "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=no | wc -l | tr -d ' ')"
printf 'Untracked paths: %s\n' "$(wc -l < "$OUT_DIR/untracked_paths.txt" | tr -d ' ')"
