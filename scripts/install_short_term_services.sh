#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/opt/ai-trading-bot}"
PYTHON="$ROOT/.venv/bin/python"
COLLECTOR="$ROOT/scripts/run_short_term_collector.py"
RUNNER="$ROOT/scripts/run_short_term_once.py"
COLLECTOR_SERVICE="/etc/systemd/system/ai-trading-shortterm-collector.service"
RUN_SERVICE="/etc/systemd/system/ai-trading-shortterm.service"
RUN_TIMER="/etc/systemd/system/ai-trading-shortterm.timer"

for path in "$PYTHON" "$COLLECTOR" "$RUNNER"; do
  [[ -e "$path" ]] || { echo "Missing: $path" >&2; exit 1; }
done

cat > "$COLLECTOR_SERVICE" <<EOF
[Unit]
Description=AI Trading Bot CoinEx short-term microstructure collector
Wants=network-online.target
After=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=$ROOT
Environment=PYTHONUNBUFFERED=1
Environment=PYTHONPATH=$ROOT
ExecStart=$PYTHON $COLLECTOR --symbol BTCUSDT --state-root $ROOT/state/short_term --poll-seconds 5
Restart=always
RestartSec=10
Nice=5

[Install]
WantedBy=multi-user.target
EOF

cat > "$RUN_SERVICE" <<EOF
[Unit]
Description=AI Trading Bot 15-minute short-term PAPER strategies
Wants=network-online.target
After=network-online.target ai-trading-shortterm-collector.service ai-trading-backend.service

[Service]
Type=oneshot
User=root
WorkingDirectory=$ROOT
Environment=PYTHONUNBUFFERED=1
Environment=PYTHONPATH=$ROOT
ExecStart=$PYTHON $RUNNER
TimeoutStartSec=10min
Nice=5
EOF

cat > "$RUN_TIMER" <<'EOF'
[Unit]
Description=Run short-term PAPER strategies after each completed 15-minute candle

[Timer]
OnCalendar=*-*-* *:02,17,32,47:00 UTC
Persistent=true
AccuracySec=10s
RandomizedDelaySec=10s
Unit=ai-trading-shortterm.service

[Install]
WantedBy=timers.target
EOF

systemctl daemon-reload
systemctl enable --now ai-trading-shortterm-collector.service
systemctl enable --now ai-trading-shortterm.timer

echo "Short-term collector and PAPER timer installed."
systemctl --no-pager --full status ai-trading-shortterm-collector.service || true
systemctl list-timers ai-trading-shortterm.timer --no-pager
