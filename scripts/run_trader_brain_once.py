#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from backend.trader_brain.runtime import DEFAULT_STATE_ROOT, DEFAULT_V3_STATE_ROOT, run_trader_brain_once


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Trader-Brain PAPER decision cycle.")
    parser.add_argument("--state-root", type=Path, default=DEFAULT_STATE_ROOT)
    parser.add_argument("--v3-state-root", type=Path, default=DEFAULT_V3_STATE_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


async def _main(args: argparse.Namespace) -> int:
    results = await run_trader_brain_once(state_root=args.state_root, v3_state_root=args.v3_state_root, dry_run=args.dry_run)
    print(json.dumps(results, indent=2, sort_keys=True, default=str))
    return 0


def main() -> int:
    args = parse_args()
    try:
        return asyncio.run(_main(args))
    except (RuntimeError, ValueError) as exc:
        print(f"TRADER_BRAIN_ERROR: {exc}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
