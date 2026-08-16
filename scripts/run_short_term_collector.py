#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from backend.short_term.collector import run_short_term_collector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect public CoinEx short-term order flow/depth into 15m buckets.")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--state-root", type=Path, default=Path("state/short_term"))
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    return parser.parse_args()


async def _main() -> int:
    args = parse_args()
    await run_short_term_collector(symbol=args.symbol, state_root=args.state_root, poll_seconds=args.poll_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
