from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

REPO_ROOT = Path(os.getenv("AI_TRADING_REPO_ROOT", "/opt/ai-trading-bot")).resolve()
HOST = os.getenv("AI_TRADING_MCP_HOST", "127.0.0.1")
PORT = int(os.getenv("AI_TRADING_MCP_PORT", "8765"))
ACTION_WRAPPER = REPO_ROOT / "scripts" / "mcp_ops_action.sh"
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"

ALLOWED_SERVICES = {
    "ai-trading-backend.service",
    "ai-trading-frontend.service",
    "ai-trading-shortterm.service",
    "ai-trading-shortterm.timer",
    "ai-trading-shortterm-collector.service",
    "ai-trading-paper-snapshot.service",
    "ai-trading-paper-snapshot.timer",
    "ai-trading-paper-monitor.service",
    "ai-trading-paper-monitor.timer",
    "ai-trading-ops-supervisor.service",
    "ai-trading-ops-supervisor.timer",
}

TEST_SUITES: dict[str, list[str]] = {
    "short_term": [
        "tests/test_short_term.py",
        "tests/test_short_term_diagnostics.py",
        "tests/test_short_term_exploration.py",
    ],
    "monitoring": ["tests/test_six_hour_monitoring.py"],
    "safe_core": [
        "tests/test_short_term.py",
        "tests/test_short_term_diagnostics.py",
        "tests/test_short_term_exploration.py",
        "tests/test_six_hour_monitoring.py",
    ],
}

mcp = FastMCP(
    "AI Trading Ops",
    host=HOST,
    port=PORT,
    instructions=(
        "Restricted operations interface for the AI trading PAPER server. "
        "Never expose secrets, .env files, SSH keys, exchange credentials, or arbitrary shell access. "
        "Trading remains PAPER-only."
    ),
)


def _run(args: list[str], *, cwd: Path | None = None, timeout: int = 120) -> dict[str, Any]:
    proc = subprocess.run(
        args,
        cwd=str(cwd or REPO_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    output = proc.stdout[-120_000:]
    return {"ok": proc.returncode == 0, "returncode": proc.returncode, "output": output}


def _service(name: str) -> str:
    if name not in ALLOWED_SERVICES:
        raise ValueError(f"service is not allowlisted: {name}")
    return name


def _safe_state_path(relative_path: str) -> Path:
    if not relative_path or relative_path.startswith("/"):
        raise ValueError("use a relative path under state/")
    candidate = (REPO_ROOT / "state" / relative_path).resolve()
    state_root = (REPO_ROOT / "state").resolve()
    if candidate != state_root and state_root not in candidate.parents:
        raise ValueError("path escapes state/")
    forbidden = {".env", "id_rsa", "id_ed25519", "authorized_keys", "credentials", "secret", "token"}
    lower = candidate.name.lower()
    if any(word in lower for word in forbidden):
        raise ValueError("secret-like files are blocked")
    if candidate.suffix.lower() not in {".json", ".jsonl", ".log", ".txt"}:
        raise ValueError("only JSON/JSONL/log/text runtime files are readable")
    return candidate


@mcp.tool()
def server_health() -> dict[str, Any]:
    """Read basic server/project health without changing anything."""
    services: dict[str, str] = {}
    for service in sorted(ALLOWED_SERVICES):
        result = _run(["systemctl", "is-active", service], timeout=15)
        services[service] = result["output"].strip() or "unknown"
    branch = _run(["git", "branch", "--show-current"], timeout=15)["output"].strip()
    commit = _run(["git", "rev-parse", "HEAD"], timeout=15)["output"].strip()
    return {"repo_root": str(REPO_ROOT), "branch": branch, "commit": commit, "services": services}


@mcp.tool()
def service_status(service: str) -> dict[str, Any]:
    """Read systemd status for one allowlisted AI-trading service."""
    return _run(["systemctl", "status", _service(service), "--no-pager", "-l"], timeout=30)


@mcp.tool()
def service_logs(service: str, lines: int = 100) -> dict[str, Any]:
    """Read recent journal logs for one allowlisted AI-trading service."""
    lines = max(1, min(int(lines), 500))
    return _run(["journalctl", "-u", _service(service), "-n", str(lines), "--no-pager"], timeout=30)


@mcp.tool()
def git_status() -> dict[str, Any]:
    """Read branch, commit, tracked/untracked status and recent commits."""
    return {
        "branch": _run(["git", "branch", "--show-current"], timeout=15)["output"].strip(),
        "commit": _run(["git", "rev-parse", "HEAD"], timeout=15)["output"].strip(),
        "status": _run(["git", "status", "--short"], timeout=15)["output"],
        "recent": _run(["git", "log", "-5", "--oneline", "--decorate"], timeout=15)["output"],
    }


@mcp.tool()
def read_runtime_file(relative_path: str, tail_lines: int = 200) -> dict[str, Any]:
    """Read an allowlisted runtime file under state/. Secret-like paths are blocked."""
    path = _safe_state_path(relative_path)
    if not path.is_file():
        return {"ok": False, "error": "file not found", "path": str(path)}
    if path.stat().st_size > 10_000_000:
        return {"ok": False, "error": "file is too large; use a summarized monitoring file", "path": str(path)}
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() == ".jsonl":
        text = "\n".join(text.splitlines()[-max(1, min(int(tail_lines), 1000)):])
    else:
        text = text[-250_000:]
    return {"ok": True, "path": str(path.relative_to(REPO_ROOT)), "content": text}


@mcp.tool()
def run_tests(suite: str = "safe_core") -> dict[str, Any]:
    """Run one predefined pytest suite; arbitrary commands and paths are not accepted."""
    tests = TEST_SUITES.get(suite)
    if tests is None:
        raise ValueError(f"unknown suite; choose one of {sorted(TEST_SUITES)}")
    if not PYTHON.is_file():
        return {"ok": False, "returncode": 127, "output": f"missing Python: {PYTHON}"}
    return _run(
        [str(PYTHON), "-m", "pytest", "-q", "-p", "no:cacheprovider", *tests],
        timeout=900,
    )


def _privileged(action: str, *args: str, timeout: int = 900) -> dict[str, Any]:
    if not ACTION_WRAPPER.is_file():
        return {"ok": False, "returncode": 127, "output": f"missing action wrapper: {ACTION_WRAPPER}"}
    return _run(["sudo", str(ACTION_WRAPPER), action, *args], timeout=timeout)


@mcp.tool()
def restart_service(service: str) -> dict[str, Any]:
    """Restart one allowlisted AI-trading systemd service."""
    return _privileged("restart-service", _service(service), timeout=120)


@mcp.tool()
def start_service(service: str) -> dict[str, Any]:
    """Start one allowlisted AI-trading service/oneshot."""
    return _privileged("start-service", _service(service), timeout=300)


@mcp.tool()
def rebuild_frontend() -> dict[str, Any]:
    """Build the checked-out Next.js frontend and restart only the frontend service."""
    return _privileged("rebuild-frontend", timeout=1200)


@mcp.tool()
def publish_monitoring_snapshot() -> dict[str, Any]:
    """Generate and push the current six-hour PAPER monitoring snapshot."""
    return _privileged("publish-monitoring", timeout=1200)


@mcp.tool()
def pull_production_code() -> dict[str, Any]:
    """Fast-forward the production branch only. Refuses dirty tracked files or another checked-out branch."""
    return _privileged("pull-production", timeout=300)


if __name__ == "__main__":
    # Keep this bound to loopback. Connect it to ChatGPT with Secure MCP Tunnel
    # or a separately authenticated HTTPS reverse proxy; do not expose port 8765 publicly.
    mcp.run(transport="streamable-http")
