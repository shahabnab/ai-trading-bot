#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/opt/ai-trading-bot}"
PYTHON="$ROOT/.venv/bin/python"
RUNNER="$ROOT/scripts/run_all_forward_once.py"
SERVICE="/etc/systemd/system/ai-trading-v3-forward.service"
TIMER="/etc/systemd/system/ai-trading-v3-forward.timer"

if [[ ! -x "$PYTHON" ]]; then
  echo "Python environment not found: $PYTHON" >&2
  exit 1
fi
if [[ ! -f "$RUNNER" ]]; then
  echo "Forward runner not found: $RUNNER" >&2
  exit 1
fi

cat > "$SERVICE" <<EOF
[Unit]
Description=AI Trading Bot hourly V3 + Trader-Brain paper inference
Wants=network-online.target
After=network-online.target ai-trading-backend.service

[Service]
Type=oneshot
User=root
WorkingDirectory=$ROOT
ExecStart=$PYTHON $RUNNER
Environment=PYTHONUNBUFFERED=1
TimeoutStartSec=45min
Nice=5

[Install]
WantedBy=multi-user.target
EOF

cat > "$TIMER" <<'EOF'
[Unit]
Description=Run all AI trading paper strategies after each completed UTC hour

[Timer]
OnCalendar=*-*-* *:05:00 UTC
Persistent=true
AccuracySec=20s
RandomizedDelaySec=20s
Unit=ai-trading-v3-forward.service

[Install]
WantedBy=timers.target
EOF

systemctl daemon-reload
systemctl enable --now ai-trading-v3-forward.timer

echo "Installed ai-trading-v3-forward.timer (Frozen V3 + Trader-Brain)"
systemctl list-timers ai-trading-v3-forward.timer --no-pager
