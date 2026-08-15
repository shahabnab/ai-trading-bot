#!/usr/bin/env bash
set -euo pipefail

# Install a persistent daily systemd timer for Git-backed paper snapshots.
# Run once with sudo/root on the always-on paper-trading server.

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SERVICE=/etc/systemd/system/ai-trading-paper-snapshot.service
TIMER=/etc/systemd/system/ai-trading-paper-snapshot.timer

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run this installer with sudo/root so it can create systemd units." >&2
  exit 1
fi

cat > "$SERVICE" <<EOF
[Unit]
Description=Export and push AI trading paper snapshot
Wants=network-online.target
After=network-online.target

[Service]
Type=oneshot
WorkingDirectory=$REPO_ROOT
Environment=REPO_ROOT=$REPO_ROOT
Environment=PAPER_RESULTS_BRANCH=paper-live-results
Environment=PAPER_API_BASE=http://127.0.0.1:8000
ExecStart=/bin/bash $REPO_ROOT/scripts/push_daily_paper_snapshot.sh
EOF

cat > "$TIMER" <<'EOF'
[Unit]
Description=Daily AI trading paper snapshot

[Timer]
# One immutable audit snapshot near the end of every UTC day. Persistent=true
# runs a missed snapshot after a reboot rather than silently losing that day.
OnCalendar=*-*-* 23:55:00 UTC
Persistent=true
RandomizedDelaySec=60
Unit=ai-trading-paper-snapshot.service

[Install]
WantedBy=timers.target
EOF

systemctl daemon-reload
systemctl enable --now ai-trading-paper-snapshot.timer

echo
systemctl --no-pager status ai-trading-paper-snapshot.timer || true
echo
echo "Installed. Test immediately with:"
echo "  systemctl start ai-trading-paper-snapshot.service"
echo "  journalctl -u ai-trading-paper-snapshot.service -n 100 --no-pager"
echo
echo "Daily results branch: paper-live-results"
