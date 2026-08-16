#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json

from backend.short_term.runtime import run_short_term_once


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one isolated 15-minute short-term PAPER decision cycle.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


async def _main() -> int:
    args = parse_args()
    results = await run_short_term_once(dry_run=args.dry_run)
    print(json.dumps(results, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
