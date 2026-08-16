#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/opt/ai-trading-bot}"
PYTHON="$ROOT/.venv/bin/python"
SUPERVISOR="$ROOT/scripts/ops_supervisor.py"
PUBLISHER="$ROOT/scripts/publish_ops_status.py"
ENV_FILE="/etc/ai-trading-ops.env"
SERVICE="/etc/systemd/system/ai-trading-ops-supervisor.service"
TIMER="/etc/systemd/system/ai-trading-ops-supervisor.timer"

for path in "$PYTHON" "$SUPERVISOR" "$PUBLISHER"; do
  [[ -e "$path" ]] || { echo "Missing: $path" >&2; exit 1; }
done

if [[ ! -f "$ENV_FILE" ]]; then
  cat > "$ENV_FILE" <<'EOF'
# Optional: preferred publish method. Use a fine-grained GitHub token limited to
# this repository with Contents: Read and write. Leave blank to use git push
# credentials already configured on the server.
OPS_GITHUB_TOKEN=
OPS_GITHUB_REPO=shahabnab/ai-trading-bot
OPS_STATUS_BRANCH=ops-status
OPS_STATUS_REMOTE_PATH=ops/status.json
EOF
  chmod 600 "$ENV_FILE"
fi

cat > "$SERVICE" <<EOF
[Unit]
Description=AI Trading Bot health/performance supervisor and GitHub status publisher
Wants=network-online.target
After=network-online.target ai-trading-all-forward.service

[Service]
Type=oneshot
User=root
WorkingDirectory=$ROOT
Environment=PYTHONUNBUFFERED=1
Environment=PYTHONPATH=$ROOT
EnvironmentFile=-$ENV_FILE
ExecStart=/bin/bash -lc 'set +e; "$PYTHON" "$SUPERVISOR" --output "$ROOT/state/ops/system_health.json"; SUP=\$?; "$PYTHON" "$PUBLISHER" --status "$ROOT/state/ops/system_health.json"; PUB=\$?; if [ \$PUB -ne 0 ]; then exit \$PUB; fi; exit 0'
TimeoutStartSec=10min
Nice=8
EOF

cat > "$TIMER" <<'EOF'
[Unit]
Description=Check AI trading server after each hourly forward cycle

[Timer]
OnCalendar=*-*-* *:15:00 UTC
Persistent=true
AccuracySec=30s
RandomizedDelaySec=30s
Unit=ai-trading-ops-supervisor.service

[Install]
WantedBy=timers.target
EOF

chmod +x "$SUPERVISOR" "$PUBLISHER"
systemctl daemon-reload
systemctl enable --now ai-trading-ops-supervisor.timer

echo "Running first supervisor snapshot..."
set +e
PYTHONPATH="$ROOT" "$PYTHON" "$SUPERVISOR" --output "$ROOT/state/ops/system_health.json"
SUP_RC=$?
set -e

echo "Publishing first snapshot..."
set +e
PYTHONPATH="$ROOT" "$PYTHON" "$PUBLISHER" --status "$ROOT/state/ops/system_health.json"
PUB_RC=$?
set -e

echo
systemctl list-timers ai-trading-ops-supervisor.timer --no-pager

echo
if [[ $PUB_RC -ne 0 ]]; then
  echo "Supervisor is installed locally, but GitHub publication needs write authentication."
  echo "Preferred: add a fine-grained token to $ENV_FILE as OPS_GITHUB_TOKEN=..."
  echo "Then run: systemctl start ai-trading-ops-supervisor.service"
  exit 3
fi

if [[ $SUP_RC -eq 2 ]]; then
  echo "Status was published, but the supervisor currently reports CRITICAL. Inspect:"
  echo "  cat $ROOT/state/ops/system_health.json"
else
  echo "Ops supervision installed and publishing successfully."
fi
