#!/usr/bin/env bash
set -euo pipefail

ROOT="${AI_TRADING_REPO_ROOT:-/opt/ai-trading-bot}"
ACTION="${1:-}"
ARG="${2:-}"

allowed_service() {
  case "$1" in
    ai-trading-backend.service|\
    ai-trading-frontend.service|\
    ai-trading-shortterm.service|\
    ai-trading-shortterm.timer|\
    ai-trading-shortterm-collector.service|\
    ai-trading-paper-snapshot.service|\
    ai-trading-paper-snapshot.timer|\
    ai-trading-paper-monitor.service|\
    ai-trading-paper-monitor.timer|\
    ai-trading-ops-supervisor.service|\
    ai-trading-ops-supervisor.timer)
      return 0 ;;
    *) return 1 ;;
  esac
}

case "$ACTION" in
  restart-service)
    allowed_service "$ARG" || { echo "Service not allowlisted: $ARG" >&2; exit 2; }
    systemctl restart "$ARG"
    systemctl status "$ARG" --no-pager -l || true
    ;;

  start-service)
    allowed_service "$ARG" || { echo "Service not allowlisted: $ARG" >&2; exit 2; }
    systemctl start "$ARG"
    systemctl status "$ARG" --no-pager -l || true
    ;;

  rebuild-frontend)
    cd "$ROOT/frontend"
    npm run build
    systemctl restart ai-trading-frontend.service
    systemctl status ai-trading-frontend.service --no-pager -l || true
    ;;

  publish-monitoring)
    cd "$ROOT"
    /bin/bash "$ROOT/scripts/push_six_hour_monitoring.sh"
    ;;

  pull-production)
    cd "$ROOT"
    branch="$(git branch --show-current)"
    if [[ "$branch" != "step-6-short-term-trading" ]]; then
      echo "Refusing deploy: expected step-6-short-term-trading, got '$branch'" >&2
      exit 3
    fi
    if ! git diff --quiet || ! git diff --cached --quiet; then
      echo "Refusing deploy: tracked working-tree changes exist." >&2
      git status --short
      exit 4
    fi
    git fetch origin step-6-short-term-trading
    git pull --ff-only origin step-6-short-term-trading
    ;;

  *)
    echo "Unsupported MCP privileged action: $ACTION" >&2
    exit 64
    ;;
esac
