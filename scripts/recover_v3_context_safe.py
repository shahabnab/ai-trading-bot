#!/usr/bin/env python3
"""Resumable, low-memory recovery of the historical V3 Binance inputs.

This is an operational recovery helper for small servers.  It preserves the
existing V3 aggregation contract, but uses *daily* archives for BTC spot
aggTrades so a multi-gigabyte monthly archive is never held in RAM.  Progress
is checkpointed to ``*.partial`` plus ``*.progress.json`` files, so an SSH
session loss, transient failure, or process restart does not force a multi-year
restart.

The final output file names are exactly the same as the original V3 collector.
Once all streams are complete, the script rebuilds ``btc_hourly_v3.jsonl`` from
the recovered BTC spot 1h rows using the original builder.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from backend.data_collection import binance_research_context as brc

CHECKPOINT_EVERY = 14
DAILY_KEYS = {"btc_spot_1h", "btc_spot_aggtrades"}


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _atomic_write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    brc._write_jsonl(tmp, rows)
    tmp.replace(path)


def _read_jsonl_map(path: Path) -> dict[int, dict[str, object]]:
    out: dict[int, dict[str, object]] = {}
    if not path.is_file():
        return out
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[int(row["timestamp"])] = row
    return out


def _read_progress(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    values = payload.get("completed_archives", []) if isinstance(payload, dict) else []
    return {str(value) for value in values}


def _checkpoint(
    *,
    partial_path: Path,
    progress_path: Path,
    output: dict[int, dict[str, object]],
    completed: set[str],
) -> None:
    rows = [output[key] for key in sorted(output)]
    _atomic_write_jsonl(partial_path, rows)
    _atomic_write_json(
        progress_path,
        {
            "updated_at": datetime.now(UTC).isoformat(),
            "hourly_rows": len(rows),
            "completed_archives": sorted(completed),
        },
    )


async def _collect_one(
    client: httpx.AsyncClient,
    stream: brc.ArchiveStream,
    *,
    start: date,
    end: date,
    output_root: Path,
) -> Path:
    output_path = output_root / stream.output_name
    if output_path.is_file():
        print(f"[complete] {stream.key}: {output_path}", flush=True)
        return output_path

    partial_path = output_path.with_suffix(output_path.suffix + ".partial")
    progress_path = output_path.with_suffix(output_path.suffix + ".progress.json")
    output = _read_jsonl_map(partial_path)
    completed = _read_progress(progress_path)
    if output or completed:
        print(
            f"[resume] {stream.key}: {len(output):,} hourly rows, "
            f"{len(completed):,} archives checkpointed",
            flush=True,
        )

    plan = brc._archive_plan(start, end, daily_only=(stream.key in DAILY_KEYS))
    since_checkpoint = 0

    for item, daily in plan:
        archive_name = stream.filename(item, daily=daily)
        if archive_name in completed:
            continue

        url = stream.url(item, daily=daily)
        print(f"[{stream.key}] {archive_name}", flush=True)
        content = await brc._fetch(client, url)

        if content is None and not daily:
            month_end = brc._next_month(item) - timedelta(days=1)
            for day in brc._iter_days(max(start, item), min(end, month_end)):
                daily_name = stream.filename(day, daily=True)
                if daily_name in completed:
                    continue
                daily_content = await brc._fetch(client, stream.url(day, daily=True))
                if daily_content is None:
                    completed.add(daily_name)
                    continue
                for row in brc._aggregate_archive(stream, daily_content, daily_name):
                    output[int(row["timestamp"])] = row
                completed.add(daily_name)
                since_checkpoint += 1
                if since_checkpoint >= CHECKPOINT_EVERY:
                    _checkpoint(
                        partial_path=partial_path,
                        progress_path=progress_path,
                        output=output,
                        completed=completed,
                    )
                    print(f"  checkpoint: {len(output):,} hourly rows", flush=True)
                    since_checkpoint = 0
            completed.add(archive_name)
            continue

        if content is None:
            print(f"  archive unavailable (404): {archive_name}", flush=True)
            completed.add(archive_name)
            since_checkpoint += 1
        else:
            for row in brc._aggregate_archive(stream, content, archive_name):
                output[int(row["timestamp"])] = row
            completed.add(archive_name)
            since_checkpoint += 1
            print(f"  hourly rows so far: {len(output):,}", flush=True)

        if since_checkpoint >= CHECKPOINT_EVERY:
            _checkpoint(
                partial_path=partial_path,
                progress_path=progress_path,
                output=output,
                completed=completed,
            )
            print(f"  checkpoint saved: {len(output):,} hourly rows", flush=True)
            since_checkpoint = 0

    rows = [output[key] for key in sorted(output)]
    _atomic_write_jsonl(output_path, rows)
    partial_path.unlink(missing_ok=True)
    progress_path.unlink(missing_ok=True)
    print(f"[complete] {stream.key}: {len(rows):,} hourly rows -> {output_path}", flush=True)
    return output_path


async def _main(args: argparse.Namespace) -> int:
    if args.end < args.start:
        raise ValueError("--end must be on or after --start")

    print("=" * 78)
    print("SAFE / RESUMABLE V3 BINANCE RECOVERY")
    print(f"Range: {args.start} -> {args.end}")
    print("BTC aggTrades: DAILY archives (lower peak RAM)")
    print(f"Checkpoint cadence: every {CHECKPOINT_EVERY} archives")
    print("=" * 78)

    async with httpx.AsyncClient(timeout=180.0, follow_redirects=True) as client:
        for stream in brc.STREAMS:
            await _collect_one(
                client,
                stream,
                start=args.start,
                end=args.end,
                output_root=args.output_root,
            )

    spot = args.output_root / "btc_spot_hourly.jsonl"
    brc.build_v3_training_dataset(spot_hourly_path=spot, output_path=args.dataset)
    print(f"[complete] training dataset -> {args.dataset}", flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resumable V3 Binance recovery for small servers")
    parser.add_argument("--start", type=_parse_date, default=date(2020, 1, 1))
    parser.add_argument("--end", type=_parse_date, default=date(2026, 8, 10))
    parser.add_argument("--output-root", type=Path, default=brc.OUTPUT_ROOT)
    parser.add_argument("--dataset", type=Path, default=brc.TRAINING_OUTPUT)
    return parser.parse_args()


def main() -> int:
    return asyncio.run(_main(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
