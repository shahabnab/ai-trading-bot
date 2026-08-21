#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/opt/ai-trading-bot}"
MCP_VENV="${AI_TRADING_MCP_VENV:-/opt/ai-trading-mcp-venv}"
PYTHON="$MCP_VENV/bin/python"
SERVER="$ROOT/ops_mcp/server.py"
WRAPPER="$ROOT/scripts/mcp_ops_action.sh"
SERVICE=/etc/systemd/system/ai-trading-mcp.service
SUDOERS=/etc/sudoers.d/ai-trading-mcp
USER_NAME=aiops

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run as root/sudo." >&2
  exit 1
fi

for path in "$SERVER" "$WRAPPER" "$ROOT/requirements-mcp.txt"; do
  [[ -e "$path" ]] || { echo "Missing: $path" >&2; exit 2; }
done

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required." >&2
  exit 2
fi

# Never let a configurable venv path point at a dangerous location. This
# directory is disposable and contains MCP dependencies only.
case "$MCP_VENV" in
  /opt/ai-trading-mcp-venv|/var/lib/ai-trading-mcp/venv) ;;
  *)
    echo "Refusing unexpected MCP venv path: $MCP_VENV" >&2
    echo "Allowed: /opt/ai-trading-mcp-venv or /var/lib/ai-trading-mcp/venv" >&2
    exit 2
    ;;
esac

# Stop an older/restarting unit before repairing its interpreter. The previous
# service may be in a 203/EXEC loop if a partial/broken venv exists.
systemctl stop ai-trading-mcp.service 2>/dev/null || true

if [[ ! -x "$PYTHON" ]]; then
  echo "Creating isolated MCP virtualenv: $MCP_VENV"
  rm -rf -- "$MCP_VENV"
  python3 -m venv --copies "$MCP_VENV"
fi

if [[ ! -x "$PYTHON" ]]; then
  echo "MCP virtualenv creation failed: missing executable $PYTHON" >&2
  exit 2
fi

# Use the interpreter itself to invoke pip. This prevents a stale/broken pip
# launcher from installing packages somewhere other than the venv used by
# systemd.
"$PYTHON" -m pip install -r "$ROOT/requirements-mcp.txt"

# Verify the exact interpreter systemd will execute before writing/enabling the
# unit. This turns a confusing 203/EXEC restart loop into an installer error.
"$PYTHON" -c 'import sys, mcp; print("MCP interpreter:", sys.executable); print("MCP import: OK")'
[[ -x "$PYTHON" ]] || { echo "MCP interpreter disappeared after install: $PYTHON" >&2; exit 2; }

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

# Give systemd a moment to catch immediate EXEC/import failures and fail the
# installer instead of printing a misleading transient "active" state.
sleep 1
if ! systemctl is-active --quiet ai-trading-mcp.service; then
  echo "MCP service failed to stay active." >&2
  systemctl status ai-trading-mcp.service --no-pager -l || true
  journalctl -u ai-trading-mcp.service -n 50 --no-pager || true
  exit 3
fi

echo
systemctl status ai-trading-mcp.service --no-pager -l || true
echo
echo "MCP is bound ONLY to 127.0.0.1:8765."
echo "Do not open port 8765 in the firewall."
echo "Connect it to ChatGPT using Secure MCP Tunnel or an authenticated HTTPS reverse proxy."
