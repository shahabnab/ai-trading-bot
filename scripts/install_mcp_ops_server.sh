#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/opt/ai-trading-bot}"
PYTHON="$ROOT/.venv/bin/python"
PIP="$ROOT/.venv/bin/pip"
SERVER="$ROOT/ops_mcp/server.py"
WRAPPER="$ROOT/scripts/mcp_ops_action.sh"
SERVICE=/etc/systemd/system/ai-trading-mcp.service
SUDOERS=/etc/sudoers.d/ai-trading-mcp
USER_NAME=aiops

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run as root/sudo." >&2
  exit 1
fi

for path in "$PYTHON" "$PIP" "$SERVER" "$WRAPPER" "$ROOT/requirements-mcp.txt"; do
  [[ -e "$path" ]] || { echo "Missing: $path" >&2; exit 2; }
done

if ! id "$USER_NAME" >/dev/null 2>&1; then
  useradd --system --no-create-home --shell /usr/sbin/nologin "$USER_NAME"
fi
usermod -a -G systemd-journal "$USER_NAME" || true

chmod 755 "$WRAPPER"

# Only this audited wrapper can be elevated. The wrapper itself allowlists every
# privileged action and service name; the MCP process never receives a shell.
cat > "$SUDOERS" <<EOF
$USER_NAME ALL=(root) NOPASSWD: $WRAPPER *
EOF
chmod 440 "$SUDOERS"
visudo -cf "$SUDOERS"

"$PIP" install -r "$ROOT/requirements-mcp.txt"

cat > "$SERVICE" <<EOF
[Unit]
Description=AI Trading restricted MCP operations server
Wants=network-online.target
After=network-online.target

[Service]
Type=simple
User=$USER_NAME
Group=$USER_NAME
SupplementaryGroups=systemd-journal
WorkingDirectory=$ROOT
Environment=PYTHONUNBUFFERED=1
Environment=PYTHONPATH=$ROOT
Environment=AI_TRADING_REPO_ROOT=$ROOT
Environment=AI_TRADING_MCP_HOST=127.0.0.1
Environment=AI_TRADING_MCP_PORT=8765
ExecStart=$PYTHON $SERVER
Restart=on-failure
RestartSec=3
PrivateTmp=true
ProtectHome=true
ProtectKernelTunables=true
ProtectKernelModules=true
ProtectControlGroups=true
RestrictSUIDSGID=true
LockPersonality=true

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now ai-trading-mcp.service

echo
systemctl status ai-trading-mcp.service --no-pager -l || true
echo
echo "MCP is bound ONLY to 127.0.0.1:8765."
echo "Do not open port 8765 in the firewall."
echo "Connect it to ChatGPT using Secure MCP Tunnel or an authenticated HTTPS reverse proxy."
