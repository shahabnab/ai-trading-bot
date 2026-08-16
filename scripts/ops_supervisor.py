#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from backend.config import settings
from backend.paper.model_catalog import ALL_PAPER_MODELS, FROZEN_V3_MODELS
from backend.paper.model_engine import ModelPaperStore

ROOT = Path(__file__).resolve().parents[1]
HOUR_MS = 60 * 60 * 1000
DEFAULT_OUTPUT = ROOT / "state/ops/system_health.json"
DEFAULT_STALE_MINUTES = 100


def _run(*args: str, timeout: int = 15) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            list(args), cwd=ROOT, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, timeout=timeout, check=False,
        )
        return proc.returncode, proc.stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 127, f"{type(exc).__name__}: {exc}"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _parse_iso(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        text = str(value).replace("Z", "+00:00")
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.astimezone(UTC)
    except ValueError:
        return None


def _iso_from_ms(value: Any) -> str | None:
    try:
        return datetime.fromtimestamp(int(value) / 1000.0, UTC).isoformat()
    except (TypeError, ValueError, OSError, OverflowError):
        return None


def _age_minutes(timestamp: datetime | None, now: datetime) -> float | None:
    if timestamp is None:
        return None
    return max(0.0, (now - timestamp).total_seconds() / 60.0)


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _systemctl(unit: str) -> dict[str, Any]:
    props = (
        "Id,LoadState,ActiveState,SubState,Result,ExecMainStatus,"
        "ExecMainStartTimestamp,ExecMainExitTimestamp,NextElapseUSecRealtime"
    )
    code, output = _run("systemctl", "show", unit, f"--property={props}", "--no-pager")
    values: dict[str, Any] = {"unit": unit, "query_rc": code}
    for line in output.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def _meminfo() -> dict[str, Any]:
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if ":" not in line:
                continue
            key, rest = line.split(":", 1)
            token = rest.strip().split()[0]
            values[key] = int(token) * 1024
    except (OSError, ValueError, IndexError):
        return {}
    total = values.get("MemTotal", 0)
    available = values.get("MemAvailable", 0)
    return {
        "total_bytes": total,
        "available_bytes": available,
        "available_fraction": (available / total) if total else None,
    }


def _system_metrics() -> dict[str, Any]:
    disk = shutil.disk_usage(ROOT)
    try:
        load1, load5, load15 = os.getloadavg()
    except OSError:
        load1 = load5 = load15 = 0.0
    return {
        "cpu_count": os.cpu_count(),
        "load_average": {"1m": load1, "5m": load5, "15m": load15},
        "memory": _meminfo(),
        "disk": {
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
            "free_fraction": disk.free / disk.total if disk.total else None,
        },
    }


def _git_status() -> dict[str, Any]:
    _, branch = _run("git", "rev-parse", "--abbrev-ref", "HEAD")
    _, head = _run("git", "rev-parse", "HEAD")
    _, dirty = _run("git", "status", "--porcelain", "--untracked-files=no")
    _, remote = _run("git", "remote", "get-url", "origin")
    return {
        "branch": branch,
        "head": head,
        "tracked_dirty": bool(dirty.strip()),
        "origin": remote,
    }


def _latest_v3_predictions() -> dict[str, dict[str, Any]]:
    path = ROOT / "state/forward_v3/predictions.jsonl"
    latest: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return latest
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                model_id = str(row.get("model_id", ""))
                if not model_id:
                    continue
                previous = latest.get(model_id)
                ts = int(row.get("feature_timestamp", 0) or 0)
                prev_ts = int(previous.get("feature_timestamp", 0) or 0) if previous else -1
                if ts >= prev_ts:
                    latest[model_id] = row
    except OSError:
        return {}
    return latest


def _strategy_runtime(now: datetime) -> dict[str, Any]:
    out: dict[str, Any] = {}
    v3 = _latest_v3_predictions()
    for spec in FROZEN_V3_MODELS:
        row = v3.get(spec.model_id, {})
        feature_iso = _iso_from_ms(row.get("feature_timestamp"))
        feature_dt = _parse_iso(feature_iso)
        out[spec.model_id] = {
            "display_name": spec.display_name,
            "driver": spec.driver,
            "feature_time_utc": feature_iso,
            "age_minutes": _age_minutes(feature_dt, now),
            "signal": row.get("signal"),
            "probability": row.get("calibrated_probability"),
            "decision_ev": row.get("decision_ev"),
            "trade": row.get("trade"),
            "recorded_at": row.get("recorded_at"),
        }

    for spec in ALL_PAPER_MODELS:
        if not spec.driver.startswith("trader_brain"):
            continue
        path = ROOT / f"state/trader_brain/latest_{spec.model_id}.json"
        row = _read_json(path)
        feature_iso = _iso_from_ms(row.get("feature_timestamp"))
        feature_dt = _parse_iso(feature_iso)
        out[spec.model_id] = {
            "display_name": spec.display_name,
            "driver": spec.driver,
            "feature_time_utc": feature_iso,
            "age_minutes": _age_minutes(feature_dt, now),
            "signal": row.get("executed_action") or row.get("requested_action"),
            "probability": (row.get("gate") or {}).get("p_up") if isinstance(row.get("gate"), dict) else None,
            "decision_ev": (row.get("decision") or {}).get("net_edge") if isinstance(row.get("decision"), dict) else None,
            "trade": row.get("trade"),
            "recorded_at": row.get("recorded_at"),
            "policy_source": row.get("policy_source"),
            "learning": row.get("learning"),
        }
    return out


def _latest_market_price(runtime: dict[str, Any]) -> Decimal | None:
    candidates: list[Any] = []
    for model_id in runtime:
        if model_id.startswith("v3-"):
            row = _latest_v3_predictions().get(model_id, {})
            candidates.append(row.get("paper_market_price"))
        else:
            row = _read_json(ROOT / f"state/trader_brain/latest_{model_id}.json")
            candidates.append(row.get("market_price"))
    for value in candidates:
        try:
            number = Decimal(str(value))
            if number > 0:
                return number
        except (InvalidOperation, TypeError, ValueError):
            continue
    return None


def _paper_performance(mark_price: Decimal | None) -> dict[str, Any]:
    db_path = ROOT / settings.paper_db_path
    if not db_path.is_file():
        return {"database": str(db_path), "available": False, "models": {}}
    store = ModelPaperStore(str(db_path), settings.paper_model_initial_balance_eur_equiv)
    models: dict[str, Any] = {}
    for spec in ALL_PAPER_MODELS:
        try:
            account = store.get_account(spec.model_id)
            positions = store.get_positions(spec.model_id)
            perf = store.performance_summary(spec.model_id)
            cash = Decimal(account["cash_usdt"])
            initial = Decimal(account["initial_cash_usdt"])
            position_value = Decimal("0")
            for position in positions:
                qty = Decimal(position["quantity"])
                if str(position["symbol"]).upper() == "BTCUSDT" and mark_price is not None:
                    position_value += qty * mark_price
                else:
                    position_value += qty * Decimal(position["avg_entry_price"])
            equity = cash + position_value
            last_decision = store.list_decisions(spec.model_id, limit=1)
            last_trade = store.list_trades(spec.model_id, limit=1)
            models[spec.model_id] = {
                "display_name": spec.display_name,
                "cash_usdt": str(cash),
                "position_value_usdt": str(position_value),
                "equity_usdt": str(equity),
                "initial_equity_usdt": str(initial),
                "return_fraction": float(equity / initial - Decimal("1")) if initial else None,
                "positions": positions,
                "performance": perf,
                "last_decision": last_decision[0] if last_decision else None,
                "last_trade": last_trade[0] if last_trade else None,
            }
        except Exception as exc:
            models[spec.model_id] = {"display_name": spec.display_name, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "database": str(db_path),
        "available": True,
        "mark_price_btcusdt": str(mark_price) if mark_price is not None else None,
        "models": models,
    }


def _artifacts() -> dict[str, Any]:
    root = ROOT / "artifacts/ml/forward_deployment/v3-paper"
    out: dict[str, Any] = {}
    for spec in FROZEN_V3_MODELS:
        folder = root / spec.model_id
        manifest = _read_json(folder / "manifest.json")
        actual_sha = _sha256(folder / "model.keras")
        expected_sha = manifest.get("model_sha256")
        out[spec.model_id] = {
            "model_exists": (folder / "model.keras").is_file(),
            "manifest_exists": (folder / "manifest.json").is_file(),
            "standardizer_exists": (folder / "standardizer.json").is_file(),
            "model_sha256": actual_sha,
            "expected_model_sha256": expected_sha,
            "sha_matches_manifest": bool(actual_sha and expected_sha and actual_sha == expected_sha),
        }
    return out


def _journal_errors() -> list[str]:
    code, text = _run(
        "journalctl", "-u", "ai-trading-all-forward.service",
        "--since", "-2 hours", "--no-pager", "-n", "120", "-o", "cat", timeout=20,
    )
    if code not in (0, 1):
        return [f"journalctl query failed rc={code}: {text[:300]}"]
    ignored = ("cuda", "cudart_stub", "cuinit", "gpu will not be used")
    needles = (
        "traceback", "forwardv3error", "modulenotfounderror", "filenotfounderror",
        "runtimeerror", "exception", "failed with result", "status=1/failure",
    )
    errors: list[str] = []
    for line in text.splitlines():
        lower = line.lower()
        if any(token in lower for token in ignored):
            continue
        if any(token in lower for token in needles):
            errors.append(line[-600:])
    return errors[-20:]


def _assess(
    services: dict[str, dict[str, Any]], runtime: dict[str, Any], artifacts: dict[str, Any],
    metrics: dict[str, Any], journal_errors: list[str], stale_minutes: int, git: dict[str, Any],
) -> tuple[str, list[dict[str, str]]]:
    issues: list[dict[str, str]] = []

    for unit in ("ai-trading-backend.service", "ai-trading-frontend.service", "ai-trading-all-forward.timer"):
        active = services.get(unit, {}).get("ActiveState")
        if active != "active":
            issues.append({"severity": "critical", "code": "unit_inactive", "message": f"{unit} ActiveState={active!r}"})

    service = services.get("ai-trading-all-forward.service", {})
    result = str(service.get("Result", ""))
    status = str(service.get("ExecMainStatus", ""))
    if result and result not in {"success", ""}:
        issues.append({"severity": "critical", "code": "forward_service_failed", "message": f"forward service result={result}, status={status}"})

    for model_id, row in runtime.items():
        age = row.get("age_minutes")
        if age is None:
            issues.append({"severity": "critical", "code": "missing_strategy_state", "message": f"{model_id} has no forward state"})
        elif float(age) > stale_minutes:
            issues.append({"severity": "critical", "code": "stale_strategy_state", "message": f"{model_id} state is {float(age):.1f} minutes old"})

    for model_id, row in artifacts.items():
        if not row.get("model_exists") or not row.get("manifest_exists") or not row.get("standardizer_exists"):
            issues.append({"severity": "critical", "code": "missing_v3_artifact", "message": f"{model_id} deployment artifact incomplete"})
        elif not row.get("sha_matches_manifest"):
            issues.append({"severity": "critical", "code": "artifact_hash_mismatch", "message": f"{model_id} model SHA no longer matches manifest"})

    disk = metrics.get("disk", {})
    free_fraction = disk.get("free_fraction")
    free_bytes = int(disk.get("free_bytes") or 0)
    if free_fraction is not None and (float(free_fraction) < 0.08 or free_bytes < 3 * 1024**3):
        issues.append({"severity": "critical", "code": "disk_low", "message": f"disk free is {float(free_fraction):.1%} ({free_bytes / 1024**3:.1f} GiB)"})
    elif free_fraction is not None and float(free_fraction) < 0.15:
        issues.append({"severity": "warning", "code": "disk_warning", "message": f"disk free is {float(free_fraction):.1%}"})

    mem_fraction = metrics.get("memory", {}).get("available_fraction")
    if mem_fraction is not None and float(mem_fraction) < 0.08:
        issues.append({"severity": "warning", "code": "memory_low", "message": f"available memory is {float(mem_fraction):.1%}"})

    if journal_errors:
        issues.append({"severity": "critical", "code": "recent_forward_errors", "message": f"{len(journal_errors)} error-like log line(s) in last 2 hours"})

    if git.get("tracked_dirty"):
        issues.append({"severity": "warning", "code": "tracked_code_drift", "message": "server has tracked code changes not committed to Git"})

    if any(issue["severity"] == "critical" for issue in issues):
        return "critical", issues
    if issues:
        return "degraded", issues
    return "healthy", issues


def build_report(stale_minutes: int = DEFAULT_STALE_MINUTES) -> dict[str, Any]:
    now = datetime.now(UTC)
    service_names = (
        "ai-trading-backend.service",
        "ai-trading-frontend.service",
        "ai-trading-all-forward.timer",
        "ai-trading-all-forward.service",
        "ai-trading-ops-supervisor.timer",
    )
    services = {name: _systemctl(name) for name in service_names}
    runtime = _strategy_runtime(now)
    metrics = _system_metrics()
    artifacts = _artifacts()
    git = _git_status()
    errors = _journal_errors()
    mark_price = _latest_market_price(runtime)
    paper = _paper_performance(mark_price)
    overall, issues = _assess(services, runtime, artifacts, metrics, errors, stale_minutes, git)
    return {
        "schema_version": 1,
        "generated_at": now.isoformat(),
        "hostname": os.uname().nodename,
        "overall_status": overall,
        "issues": issues,
        "guardrails": {
            "trading_mode": str(settings.trading_mode.value),
            "live_trading_expected": False,
            "frozen_v3_must_not_retrain": True,
            "automatic_code_fix_scope": "ops/deployment/observability only; no strategy/risk/model/threshold changes",
        },
        "services": services,
        "strategies": runtime,
        "paper": paper,
        "artifacts": artifacts,
        "system": metrics,
        "git": git,
        "recent_forward_errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate compact health/performance status for the AI trading PAPER server.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stale-minutes", type=int, default=DEFAULT_STALE_MINUTES)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(stale_minutes=max(60, args.stale_minutes))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(args.output)
    print(json.dumps({"status": report["overall_status"], "output": str(args.output), "issues": report["issues"]}, indent=2))
    return 0 if report["overall_status"] != "critical" else 2


if __name__ == "__main__":
    raise SystemExit(main())
