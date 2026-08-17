#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.short_term.diagnostics import build_diagnostics_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize short-term decision outcomes and non-executing shadow entry policies."
    )
    parser.add_argument("--state-root", default="state/short_term")
    parser.add_argument("--limit", type=int, default=2000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_diagnostics_summary(Path(args.state_root), limit=max(1, args.limit))
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
