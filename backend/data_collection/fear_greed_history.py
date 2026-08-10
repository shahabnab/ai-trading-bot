from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import httpx

from backend.data_collection.storage import append_unique_jsonl


FNG_URL = "https://api.alternative.me/fng/"
DEFAULT_OUTPUT = Path("data/raw/sentiment/alternative_me/fear_greed.jsonl")


class FearGreedError(RuntimeError):
    """Raised when the Fear & Greed history cannot be downloaded."""


async def backfill_fear_greed(
    *,
    start: date,
    end: date,
    output_file: Path = DEFAULT_OUTPUT,
) -> Path:
    if end < start:
        raise ValueError("end date must be on or after start date")

    start_ms = int(datetime.combine(start, datetime.min.time(), tzinfo=UTC).timestamp() * 1000)
    end_ms_exclusive = int(
        datetime.combine(end + timedelta(days=1), datetime.min.time(), tzinfo=UTC).timestamp() * 1000
    )

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(FNG_URL, params={"limit": 0, "format": "json"})
        try:
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise FearGreedError(f"Fear & Greed request failed: {exc}") from exc

    payload = response.json()
    if not isinstance(payload, dict):
        raise FearGreedError("Fear & Greed API returned an unexpected payload")

    metadata = payload.get("metadata")
    if isinstance(metadata, dict) and metadata.get("error"):
        raise FearGreedError(str(metadata["error"]))

    data = payload.get("data", [])
    if not isinstance(data, list):
        raise FearGreedError("Fear & Greed API returned invalid data")

    records: list[dict[str, object]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        try:
            timestamp_ms = int(item["timestamp"]) * 1000
            value = int(item["value"])
        except (KeyError, TypeError, ValueError):
            continue

        if not start_ms <= timestamp_ms < end_ms_exclusive:
            continue

        records.append(
            {
                "source": "alternative_me_fear_greed",
                "timestamp": timestamp_ms,
                "timestamp_iso": datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC).isoformat(),
                "value": value,
                "classification": str(item.get("value_classification", "")),
            }
        )

    added = append_unique_jsonl(output_file, records, key="timestamp")
    print(f"Fear & Greed records in range: {len(records):,}")
    print(f"New records saved: {added:,}")
    print(f"Saved to: {output_file}")
    return output_file


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must use YYYY-MM-DD") from exc


def parse_args() -> argparse.Namespace:
    today_utc = datetime.now(UTC).date()
    default_end = today_utc - timedelta(days=1)
    default_start = default_end - timedelta(days=180)

    parser = argparse.ArgumentParser(description="Backfill historical crypto Fear & Greed values.")
    parser.add_argument("--start", type=_parse_date, default=default_start)
    parser.add_argument("--end", type=_parse_date, default=default_end)
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    await backfill_fear_greed(start=args.start, end=args.end)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nFear & Greed download stopped.")
