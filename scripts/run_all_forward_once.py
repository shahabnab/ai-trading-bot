#!/usr/bin/env python3
"""Run legacy frozen V3 and Trader-Brain PAPER cycles independently."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from backend.forward_v3 import DEFAULT_DEPLOYMENT_ROOT, DEFAULT_STATE_ROOT as V3_STATE_ROOT, run_forward_once
from backend.trader_brain.runtime import DEFAULT_STATE_ROOT as TB_STATE_ROOT, run_trader_brain_once


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all forward PAPER strategies once.")
    parser.add_argument("--deployment-root", type=Path, default=DEFAULT_DEPLOYMENT_ROOT)
    parser.add_argument("--v3-state-root", type=Path, default=V3_STATE_ROOT)
    parser.add_argument("--trader-state-root", type=Path, default=TB_STATE_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


async def _main(args: argparse.Namespace) -> int:
    payload: dict[str, object] = {"v3": None, "trader_brain": None, "errors": {}}
    succeeded = 0
    try:
        payload["v3"] = await run_forward_once(
            deployment_root=args.deployment_root,
            state_root=args.v3_state_root,
            dry_run=args.dry_run,
        )
        succeeded += 1
    except Exception as exc:  # isolate strategy families; report and continue
        payload["errors"]["v3"] = f"{type(exc).__name__}: {exc}"
    try:
        payload["trader_brain"] = await run_trader_brain_once(
            state_root=args.trader_state_root,
            v3_state_root=args.v3_state_root,
            dry_run=args.dry_run,
        )
        succeeded += 1
    except Exception as exc:  # isolate strategy families; report and continue
        payload["errors"]["trader_brain"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0 if succeeded else 2


def main() -> int:
    return asyncio.run(_main(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
