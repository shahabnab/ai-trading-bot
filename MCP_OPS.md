# AI Trading Ops MCP

This MCP server gives ChatGPT a restricted operations interface to the PAPER-trading VPS without exposing an SSH password or an arbitrary root shell.

## Security model

- Runs on the VPS itself; no SSH password is stored in source code.
- Binds to `127.0.0.1:8765` only.
- Intended to be connected with OpenAI Secure MCP Tunnel, or an independently authenticated HTTPS reverse proxy.
- Runs as the dedicated `aiops` system user.
- Secret-like files and paths outside `state/` are not readable through the MCP tool interface.
- No `.env`, SSH key, CoinEx secret, GitHub credential, or generic shell tool is exposed.
- Privileged operations go through `scripts/mcp_ops_action.sh`, which allowlists actions and systemd units.
- Live trading enablement is not exposed.

## Available tools

Read-only:

- `server_health`
- `service_status`
- `service_logs`
- `git_status`
- `read_runtime_file`
- `run_tests`

Controlled writes/operations:

- `restart_service`
- `start_service`
- `rebuild_frontend`
- `publish_monitoring_snapshot`
- `pull_production_code`

`pull_production_code` only fast-forwards `step-6-short-term-trading` and refuses to run if tracked working-tree changes exist.

## Install on the VPS

After the branch containing this code is deployed:

```bash
cd /opt/ai-trading-bot
sudo bash scripts/install_mcp_ops_server.sh /opt/ai-trading-bot
```

Verify locally:

```bash
systemctl status ai-trading-mcp.service --no-pager -l
ss -ltnp | grep 8765
journalctl -u ai-trading-mcp.service -n 100 --no-pager
```

The listener must show `127.0.0.1:8765`, not `0.0.0.0:8765`.

## ChatGPT connection

ChatGPT connects to remote MCP servers. For a private VPS service bound to loopback, use OpenAI Secure MCP Tunnel when available in your workspace. Do not open TCP/8765 publicly just to make the connector work.

Full write/modify MCP actions depend on ChatGPT workspace plan and permissions. Configure the custom app so read actions may run automatically if desired, while write/deploy/restart actions require confirmation.

## Password handling

Never commit the VPS password to this repository and never put it in the MCP source. The MCP server runs locally on the VPS, so it does not need SSH credentials. Use SSH keys for human administration and rotate any password that has previously been pasted into chat or source code.
