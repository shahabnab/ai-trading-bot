#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATUS = ROOT / "state/ops/system_health.json"
DEFAULT_REPO = "shahabnab/ai-trading-bot"
DEFAULT_BRANCH = "ops-status"
DEFAULT_REMOTE_PATH = "ops/status.json"
DEFAULT_WORKTREE = Path("/var/lib/ai-trading-ops-status")


def _run(args: list[str], *, cwd: Path, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args, cwd=cwd, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, timeout=timeout, check=False,
    )


def _publish_api(status_path: Path, repo: str, branch: str, remote_path: str, token: str) -> str:
    api = f"https://api.github.com/repos/{repo}/contents/{remote_path}"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "ai-trading-ops-supervisor",
    }
    content = status_path.read_bytes()
    sha: str | None = None
    with httpx.Client(timeout=30.0, follow_redirects=True, headers=headers) as client:
        response = client.get(api, params={"ref": branch})
        if response.status_code == 200:
            payload = response.json()
            if isinstance(payload, dict):
                sha = str(payload.get("sha") or "") or None
        elif response.status_code != 404:
            response.raise_for_status()

        body: dict[str, Any] = {
            "message": f"ops: publish server status {datetime.now(UTC).isoformat()}",
            "content": base64.b64encode(content).decode("ascii"),
            "branch": branch,
        }
        if sha:
            body["sha"] = sha
        put = client.put(api, json=body)
        put.raise_for_status()
        result = put.json()
        commit = result.get("commit", {}) if isinstance(result, dict) else {}
        return str(commit.get("sha") or "unknown")


def _ensure_worktree(root: Path, worktree: Path, branch: str, remote: str) -> None:
    if (worktree / ".git").exists():
        return
    if worktree.exists() and any(worktree.iterdir()):
        raise RuntimeError(f"status worktree is not empty: {worktree}")
    worktree.parent.mkdir(parents=True, exist_ok=True)
    fetch = _run(["git", "fetch", remote, branch], cwd=root)
    if fetch.returncode != 0:
        raise RuntimeError(f"git fetch failed: {fetch.stdout.strip()}")
    add = _run(["git", "worktree", "add", "-B", branch, str(worktree), f"{remote}/{branch}"], cwd=root)
    if add.returncode != 0:
        raise RuntimeError(f"git worktree add failed: {add.stdout.strip()}")


def _publish_git(status_path: Path, root: Path, worktree: Path, branch: str, remote: str, remote_path: str) -> str:
    _ensure_worktree(root, worktree, branch, remote)
    target = worktree / remote_path
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(status_path, target)
    _run(["git", "config", "user.name", "ai-trading-ops"], cwd=worktree)
    _run(["git", "config", "user.email", "ai-trading-ops@localhost"], cwd=worktree)
    add = _run(["git", "add", remote_path], cwd=worktree)
    if add.returncode != 0:
        raise RuntimeError(f"git add failed: {add.stdout.strip()}")
    diff = _run(["git", "diff", "--cached", "--quiet"], cwd=worktree)
    if diff.returncode == 0:
        head = _run(["git", "rev-parse", "HEAD"], cwd=worktree)
        return head.stdout.strip()
    commit = _run(
        ["git", "commit", "-m", f"ops: publish server status {datetime.now(UTC).isoformat()}"],
        cwd=worktree,
    )
    if commit.returncode != 0:
        raise RuntimeError(f"git commit failed: {commit.stdout.strip()}")
    push = _run(["git", "push", remote, f"HEAD:{branch}"], cwd=worktree, timeout=120)
    if push.returncode != 0:
        raise RuntimeError(
            "git push failed. Configure server write authentication or set OPS_GITHUB_TOKEN. "
            f"Details: {push.stdout.strip()}"
        )
    head = _run(["git", "rev-parse", "HEAD"], cwd=worktree)
    return head.stdout.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish server ops status to a dedicated GitHub branch.")
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--repo", default=os.getenv("OPS_GITHUB_REPO", DEFAULT_REPO))
    parser.add_argument("--branch", default=os.getenv("OPS_STATUS_BRANCH", DEFAULT_BRANCH))
    parser.add_argument("--remote-path", default=os.getenv("OPS_STATUS_REMOTE_PATH", DEFAULT_REMOTE_PATH))
    parser.add_argument("--remote", default=os.getenv("OPS_GIT_REMOTE", "origin"))
    parser.add_argument("--worktree", type=Path, default=Path(os.getenv("OPS_STATUS_WORKTREE", str(DEFAULT_WORKTREE))))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.status.is_file():
        raise FileNotFoundError(args.status)
    # Validate JSON before publishing so a partial/corrupt file never replaces the last known-good status.
    json.loads(args.status.read_text(encoding="utf-8"))

    token = os.getenv("OPS_GITHUB_TOKEN", "").strip()
    try:
        if token:
            commit = _publish_api(args.status, args.repo, args.branch, args.remote_path, token)
            mode = "github_api"
        else:
            commit = _publish_git(args.status, ROOT, args.worktree, args.branch, args.remote, args.remote_path)
            mode = "git"
    except Exception as exc:
        print(f"OPS STATUS PUBLISH FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3

    print(json.dumps({"published": True, "mode": mode, "branch": args.branch, "path": args.remote_path, "commit": commit}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
