#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from backend.forward_v3 import (
    DEFAULT_DEPLOYMENT_ROOT,
    DEFAULT_STATE_ROOT,
    ForwardV3Error,
    run_forward_once,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one hourly frozen-V3 inference/paper-decision cycle."
    )
    parser.add_argument(
        "--deployment-root",
        type=Path,
        default=DEFAULT_DEPLOYMENT_ROOT,
    )
    parser.add_argument(
        "--state-root",
        type=Path,
        default=DEFAULT_STATE_ROOT,
    )
    parser.add_argument(
        "--model-id",
        choices=["v3-25bps-fullcontext-12h", "v3-50bps-technical-3h"],
        default=None,
        help="Run only one selected model. Default: both models.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute predictions/decisions but do not write paper trades or runtime state.",
    )
    return parser.parse_args()


async def _main(args: argparse.Namespace) -> int:
    results = await run_forward_once(
        deployment_root=args.deployment_root,
        state_root=args.state_root,
        dry_run=args.dry_run,
        only_model_id=args.model_id,
    )
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


def main() -> int:
    args = parse_args()
    try:
        return asyncio.run(_main(args))
    except ForwardV3Error as exc:
        print(f"FORWARD_V3_ERROR: {exc}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
