# Autonomous PAPER Ops Supervision

This layer supervises the running PAPER experiment without changing trading semantics.

## What it checks

- backend, frontend, forward timer, and supervisor timer state
- latest V3 and Trader-Brain forward timestamps and decisions
- frozen V3 model/manifest/standardizer presence and model SHA integrity
- paper-account cash, positions, equity, trade counts, decision counts, and realized PnL
- CPU load, RAM availability, and disk space
- recent forward-service errors while ignoring expected CPU-only CUDA noise
- Git branch/head and tracked working-tree drift

The local snapshot is written to `state/ops/system_health.json` and is intentionally ignored on the live code branch.

## GitHub bridge

A dedicated branch named `ops-status` stores the latest published snapshot at `ops/status.json`.

`publish_ops_status.py` uses either:

1. `OPS_GITHUB_TOKEN` (preferred) with a fine-grained token restricted to this repository and `Contents: Read and write`; or
2. existing server-side Git push credentials as a fallback.

The token belongs only in `/etc/ai-trading-ops.env`, never in the repository.

## Install on the VPS

```bash
cd /opt/ai-trading-bot
git pull origin step-5-live-paper-dashboard
source .venv/bin/activate
pytest -q tests/test_ops_supervisor.py tests/test_forward_v3_policy.py tests/test_trader_brain.py
chmod +x scripts/install_ops_supervisor.sh
bash scripts/install_ops_supervisor.sh /opt/ai-trading-bot
```

The health publisher runs at approximately `:15 UTC` every hour, after the forward trading timer at `:05 UTC`.

A second guarded updater runs around `:45 UTC`. It fetches `step-5-live-paper-dashboard`, refuses to deploy if the server has tracked local changes, refuses any commit touching files outside the supervision whitelist, tests the candidate revision in a detached Git worktree, and only then performs a fast-forward merge. A normal trading/model/risk code change therefore stops at `manual_review_required` instead of being auto-deployed.

## Guardrails

Automatic supervision and automatic deployment are limited to the explicit supervision whitelist. They must not silently change:

- frozen V3 models or manifests
- model features or training data
- trading strategy logic
- risk limits
- decision thresholds
- fees/slippage assumptions
- paper balances or forward experiment state

Those changes require explicit human approval because they would alter the experiment being measured.
