#!/usr/bin/env bash
set -euo pipefail

# Install a persistent six-hour systemd timer for decision-vs-reality PAPER
# monitoring snapshots. Run once as root on the always-on trading VPS.

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SERVICE=/etc/systemd/system/ai-trading-paper-monitor.service
TIMER=/etc/systemd/system/ai-trading-paper-monitor.timer

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run this installer with sudo/root so it can create systemd units." >&2
  exit 1
fi

chmod +x "$REPO_ROOT/scripts/push_six_hour_monitoring.sh"

cat > "$SERVICE" <<EOF
[Unit]
Description=Export and push six-hour AI trading decision-vs-reality monitor
Wants=network-online.target
After=network-online.target ai-trading-backend.service

[Service]
Type=oneshot
User=root
WorkingDirectory=$REPO_ROOT
Environment=REPO_ROOT=$REPO_ROOT
Environment=PAPER_RESULTS_BRANCH=paper-live-results
Environment=PAPER_API_BASE=http://127.0.0.1:8000
Environment=PAPER_PYTHON_BIN=$REPO_ROOT/.venv/bin/python
ExecStart=/bin/bash $REPO_ROOT/scripts/push_six_hour_monitoring.sh
TimeoutStartSec=10min
Nice=8
EOF

cat > "$TIMER" <<'EOF'
[Unit]
Description=Six-hour AI trading decision-vs-reality monitoring snapshot

[Timer]
# 02:05 / 08:05 / 14:05 / 20:05 UTC. The ChatGPT audit is scheduled about
# ten minutes later, leaving time for export + git push to finish.
OnCalendar=*-*-* 02,08,14,20:05:00 UTC
Persistent=true
AccuracySec=30s
RandomizedDelaySec=20
Unit=ai-trading-paper-monitor.service

[Install]
WantedBy=timers.target
EOF

systemctl daemon-reload
systemctl enable --now ai-trading-paper-monitor.timer

echo
systemctl --no-pager status ai-trading-paper-monitor.timer || true
echo
echo "Installed. Test immediately with:"
echo "  systemctl start ai-trading-paper-monitor.service"
echo "  journalctl -u ai-trading-paper-monitor.service -n 100 --no-pager"
echo
echo "Monitoring output branch: paper-live-results"
echo "Monitoring path: paper_monitoring/latest.json"
