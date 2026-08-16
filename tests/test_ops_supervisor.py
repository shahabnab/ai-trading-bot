from __future__ import annotations

from datetime import UTC, datetime

from scripts.ops_supervisor import _age_minutes, _assess, _parse_iso


def _healthy_inputs():
    services = {
        "ai-trading-backend.service": {"ActiveState": "active"},
        "ai-trading-frontend.service": {"ActiveState": "active"},
        "ai-trading-all-forward.timer": {"ActiveState": "active"},
        "ai-trading-all-forward.service": {"Result": "success", "ExecMainStatus": "0"},
        "ai-trading-shortterm-collector.service": {"ActiveState": "active"},
        "ai-trading-shortterm.timer": {"ActiveState": "active"},
        "ai-trading-shortterm.service": {"Result": "success", "ExecMainStatus": "0"},
    }
    runtime = {
        "v3-25bps-fullcontext-12h": {"age_minutes": 20.0, "driver": "frozen_v3"},
        "v3-50bps-technical-3h": {"age_minutes": 20.0, "driver": "frozen_v3"},
        "trader-brain-v1": {"age_minutes": 20.0, "driver": "trader_brain"},
        "trader-brain-bandit-v1": {"age_minutes": 20.0, "driver": "trader_brain_rl"},
        "short-momentum-15m": {"age_minutes": 20.0, "driver": "short_term"},
        "short-mean-reversion-15m": {"age_minutes": 20.0, "driver": "short_term"},
    }
    artifacts = {
        "v3-25bps-fullcontext-12h": {
            "model_exists": True, "manifest_exists": True,
            "standardizer_exists": True, "sha_matches_manifest": True,
        },
        "v3-50bps-technical-3h": {
            "model_exists": True, "manifest_exists": True,
            "standardizer_exists": True, "sha_matches_manifest": True,
        },
    }
    metrics = {
        "disk": {"free_fraction": 0.5, "free_bytes": 20 * 1024**3},
        "memory": {"available_fraction": 0.5},
    }
    git = {"tracked_dirty": False}
    return services, runtime, artifacts, metrics, git


def test_parse_iso_and_age_minutes():
    now = datetime(2026, 8, 16, 12, 0, tzinfo=UTC)
    parsed = _parse_iso("2026-08-16T11:30:00+00:00")
    assert parsed is not None
    assert _age_minutes(parsed, now) == 30.0
    assert _parse_iso("not-a-date") is None


def test_assess_healthy():
    services, runtime, artifacts, metrics, git = _healthy_inputs()
    status, issues = _assess(services, runtime, artifacts, metrics, [], 100, git)
    assert status == "healthy"
    assert issues == []


def test_assess_stale_strategy_is_critical():
    services, runtime, artifacts, metrics, git = _healthy_inputs()
    runtime["trader-brain-v1"]["age_minutes"] = 150.0
    status, issues = _assess(services, runtime, artifacts, metrics, [], 100, git)
    assert status == "critical"
    assert any(issue["code"] == "stale_strategy_state" for issue in issues)


def test_assess_short_term_uses_tighter_staleness_limit():
    services, runtime, artifacts, metrics, git = _healthy_inputs()
    runtime["short-momentum-15m"]["age_minutes"] = 45.0
    status, issues = _assess(services, runtime, artifacts, metrics, [], 100, git)
    assert status == "critical"
    assert any(
        issue["code"] == "stale_strategy_state" and "short-momentum-15m" in issue["message"]
        for issue in issues
    )


def test_assess_missing_shortterm_collector_is_critical():
    services, runtime, artifacts, metrics, git = _healthy_inputs()
    services["ai-trading-shortterm-collector.service"]["ActiveState"] = "inactive"
    status, issues = _assess(services, runtime, artifacts, metrics, [], 100, git)
    assert status == "critical"
    assert any(issue["code"] == "unit_inactive" for issue in issues)


def test_assess_cuda_noise_is_not_needed_for_health():
    services, runtime, artifacts, metrics, git = _healthy_inputs()
    status, issues = _assess(services, runtime, artifacts, metrics, [], 100, git)
    assert status == "healthy"
    assert not issues
