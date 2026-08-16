#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BRANCH = "step-5-live-paper-dashboard"
DEFAULT_REMOTE = "origin"

ALLOWED_PATHS = {
    ".gitignore",
    "OPS_SUPERVISION.md",
    "scripts/ops_supervisor.py",
    "scripts/publish_ops_status.py",
    "scripts/install_ops_supervisor.sh",
    "scripts/safe_ops_auto_update.py",
    "tests/test_ops_supervisor.py",
}


def _run(args: list[str], *, cwd: Path, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args, cwd=cwd, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, timeout=timeout, check=False,
    )


def _git(*args: str, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return _run(["git", *args], cwd=ROOT, timeout=timeout)


def _tracked_dirty() -> bool:
    result = _git("status", "--porcelain", "--untracked-files=no")
    return result.returncode != 0 or bool(result.stdout.strip())


def _changed_files(remote_ref: str) -> list[str]:
    result = _git("diff", "--name-only", f"HEAD..{remote_ref}")
    if result.returncode != 0:
        raise RuntimeError(f"git diff failed: {result.stdout.strip()}")
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _validate_changes(paths: list[str]) -> tuple[bool, list[str]]:
    blocked = [path for path in paths if path not in ALLOWED_PATHS]
    return not blocked, blocked


def _run_tests_at_ref(remote_ref: str, python: Path) -> tuple[bool, str]:
    worktree_root = Path(tempfile.mkdtemp(prefix="ai-trading-ops-update-"))
    try:
        add = _git("worktree", "add", "--detach", str(worktree_root), remote_ref)
        if add.returncode != 0:
            return False, f"worktree add failed: {add.stdout.strip()}"
        env = os.environ.copy()
        env["PYTHONPATH"] = str(worktree_root)
        test = subprocess.run(
            [str(python), "-m", "pytest", "-q", "tests/test_ops_supervisor.py"],
            cwd=worktree_root, env=env, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, timeout=180, check=False,
        )
        return test.returncode == 0, test.stdout.strip()
    except subprocess.TimeoutExpired as exc:
        return False, f"test timeout: {exc}"
    finally:
        _git("worktree", "remove", "--force", str(worktree_root))
        shutil.rmtree(worktree_root, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Safely deploy supervision-only GitHub changes.")
    parser.add_argument("--remote", default=DEFAULT_REMOTE)
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--python", type=Path, default=ROOT / ".venv/bin/python")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report: dict[str, object] = {
        "updated": False,
        "remote": args.remote,
        "branch": args.branch,
        "allowed_paths": sorted(ALLOWED_PATHS),
    }
    if not args.python.is_file():
        report["error"] = f"python missing: {args.python}"
        print(json.dumps(report, indent=2))
        return 2
    if _tracked_dirty():
        report["error"] = "tracked working tree is dirty; refusing automatic update"
        print(json.dumps(report, indent=2))
        return 3

    fetch = _git("fetch", args.remote, args.branch)
    if fetch.returncode != 0:
        report["error"] = f"git fetch failed: {fetch.stdout.strip()}"
        print(json.dumps(report, indent=2))
        return 4

    remote_ref = f"{args.remote}/{args.branch}"
    behind = _git("rev-list", "--count", f"HEAD..{remote_ref}")
    ahead = _git("rev-list", "--count", f"{remote_ref}..HEAD")
    if behind.returncode != 0 or ahead.returncode != 0:
        report["error"] = "unable to compare local and remote refs"
        print(json.dumps(report, indent=2))
        return 5
    report["behind_commits"] = int(behind.stdout.strip() or 0)
    report["ahead_commits"] = int(ahead.stdout.strip() or 0)
    if int(ahead.stdout.strip() or 0) != 0:
        report["error"] = "local branch contains commits not on remote; refusing automatic merge"
        print(json.dumps(report, indent=2))
        return 6
    if int(behind.stdout.strip() or 0) == 0:
        report["status"] = "up_to_date"
        print(json.dumps(report, indent=2))
        return 0

    paths = _changed_files(remote_ref)
    report["changed_files"] = paths
    allowed, blocked = _validate_changes(paths)
    if not allowed:
        report["status"] = "manual_review_required"
        report["blocked_files"] = blocked
        print(json.dumps(report, indent=2))
        return 10

    tests_ok, test_output = _run_tests_at_ref(remote_ref, args.python)
    report["tests_ok"] = tests_ok
    report["test_output"] = test_output[-4000:]
    if not tests_ok:
        report["status"] = "tests_failed"
        print(json.dumps(report, indent=2))
        return 11

    if args.dry_run:
        report["status"] = "safe_update_available"
        print(json.dumps(report, indent=2))
        return 0

    merge = _git("merge", "--ff-only", remote_ref)
    if merge.returncode != 0:
        report["status"] = "merge_failed"
        report["error"] = merge.stdout.strip()
        print(json.dumps(report, indent=2))
        return 12

    report["updated"] = True
    report["status"] = "updated"
    report["new_head"] = _git("rev-parse", "HEAD").stdout.strip()
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
