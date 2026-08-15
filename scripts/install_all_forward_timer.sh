#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/opt/ai-trading-bot}"
PYTHON="$ROOT/.venv/bin/python"
V3_RUNNER="$ROOT/scripts/run_v3_forward_once.py"
TB_RUNNER="$ROOT/scripts/run_trader_brain_once.py"
SERVICE="/etc/systemd/system/ai-trading-all-forward.service"
TIMER="/etc/systemd/system/ai-trading-all-forward.timer"

for path in "$PYTHON" "$V3_RUNNER" "$TB_RUNNER"; do
  [[ -e "$path" ]] || { echo "Missing: $path" >&2; exit 1; }
done

cat > "$SERVICE" <<EOF
[Unit]
Description=AI Trading Bot frozen V3 plus Trader-Brain PAPER inference
Wants=network-online.target
After=network-online.target ai-trading-backend.service

[Service]
Type=oneshot
User=root
WorkingDirectory=$ROOT
ExecStart=/bin/bash -lc '$PYTHON $V3_RUNNER; V3_STATUS=\$?; $PYTHON $TB_RUNNER; TB_STATUS=\$?; if [ \$V3_STATUS -ne 0 ] && [ \$TB_STATUS -ne 0 ]; then exit 2; fi'
Environment=PYTHONUNBUFFERED=1
Environment=PYTHONPATH=$ROOT
TimeoutStartSec=45min
Nice=5
EOF

cat > "$TIMER" <<'EOF'
[Unit]
Description=Run all AI trading PAPER strategies after each completed UTC hour

[Timer]
OnCalendar=*-*-* *:05:00 UTC
Persistent=true
AccuracySec=20s
RandomizedDelaySec=20s
Unit=ai-trading-all-forward.service

[Install]
WantedBy=timers.target
EOF

systemctl daemon-reload
systemctl enable --now ai-trading-all-forward.timer
systemctl list-timers ai-trading-all-forward.timer --no-pager
