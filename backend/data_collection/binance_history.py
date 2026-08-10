from __future__ import annotations

import argparse
import asyncio
import csv
import io
import zipfile
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Iterable

import httpx

from backend.data_collection.storage import append_unique_jsonl


BINANCE_DATA_BASE = "https://data.binance.vision/data"
DEFAULT_OUTPUT_ROOT = Path("data/raw/market/binance")


class BinanceHistoryError(RuntimeError):
    """Raised when Binance public archive data cannot be downloaded or parsed."""


def normalize_timestamp_ms(value: str | int) -> int:
    """Normalize Binance second/millisecond/microsecond timestamps to milliseconds."""
    timestamp = int(value)
    if timestamp >= 10**14:  # microseconds (spot archives from 2025 onward)
        return timestamp // 1000
    if timestamp >= 10**11:  # milliseconds
        return timestamp
    return timestamp * 1000  # seconds


def _month_start(day: date) -> date:
    return day.replace(day=1)


def _next_month(day: date) -> date:
    if day.month == 12:
        return date(day.year + 1, 1, 1)
    return date(day.year, day.month + 1, 1)


def _iter_months(start: date, end: date) -> Iterable[date]:
    current = _month_start(start)
    last = _month_start(end)
    while current <= last:
        yield current
        current = _next_month(current)


def _iter_days(start: date, end: date) -> Iterable[date]:
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def _monthly_url(symbol: str, interval: str, month: date) -> str:
    filename = f"{symbol}-{interval}-{month:%Y-%m}.zip"
    return f"{BINANCE_DATA_BASE}/spot/monthly/klines/{symbol}/{interval}/{filename}"


def _daily_url(symbol: str, interval: str, day: date) -> str:
    filename = f"{symbol}-{interval}-{day:%Y-%m-%d}.zip"
    return f"{BINANCE_DATA_BASE}/spot/daily/klines/{symbol}/{interval}/{filename}"


async def _download_archive(client: httpx.AsyncClient, url: str) -> bytes | None:
    response = await client.get(url)
    if response.status_code == 404:
        return None
    try:
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise BinanceHistoryError(f"Binance archive request failed: {url}: {exc}") from exc
    return response.content


def _parse_archive(
    content: bytes,
    *,
    symbol: str,
    interval: str,
    archive_name: str,
    start_ms: int,
    end_ms_exclusive: int,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []

    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if not csv_names:
                raise BinanceHistoryError(f"No CSV found in Binance archive {archive_name}")

            with archive.open(csv_names[0]) as raw:
                text_stream = io.TextIOWrapper(raw, encoding="utf-8")
                reader = csv.reader(text_stream)

                for row in reader:
                    if not row or len(row) < 11:
                        continue

                    try:
                        open_time_ms = normalize_timestamp_ms(row[0])
                    except (TypeError, ValueError):
                        # Some archives include a header row.
                        continue

                    if not start_ms <= open_time_ms < end_ms_exclusive:
                        continue

                    close_time_ms = normalize_timestamp_ms(row[6])
                    records.append(
                        {
                            "source": "binance_spot_bulk",
                            "symbol": symbol,
                            "interval": interval,
                            "open_time": open_time_ms,
                            "open_time_iso": datetime.fromtimestamp(open_time_ms / 1000, tz=UTC).isoformat(),
                            "open": row[1],
                            "high": row[2],
                            "low": row[3],
                            "close": row[4],
                            "volume": row[5],
                            "close_time": close_time_ms,
                            "quote_volume": row[7],
                            "number_of_trades": int(row[8]),
                            "taker_buy_base_volume": row[9],
                            "taker_buy_quote_volume": row[10],
                            "archive": archive_name,
                        }
                    )
    except zipfile.BadZipFile as exc:
        raise BinanceHistoryError(f"Invalid Binance ZIP archive: {archive_name}") from exc

    return records


async def backfill_binance_klines(
    *,
    symbol: str = "BTCUSDT",
    interval: str = "1m",
    start: date,
    end: date,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> Path:
    """Download historical spot klines from Binance daily/monthly public archives."""
    if end < start:
        raise ValueError("end date must be on or after start date")

    symbol = symbol.upper().strip()
    start_ms = int(datetime.combine(start, datetime.min.time(), tzinfo=UTC).timestamp() * 1000)
    end_ms_exclusive = int(
        datetime.combine(end + timedelta(days=1), datetime.min.time(), tzinfo=UTC).timestamp() * 1000
    )

    output_file = output_root / symbol / interval / "candles.jsonl"
    current_month = _month_start(datetime.now(UTC).date())

    monthly_plan: list[date] = []
    daily_plan: list[date] = []

    for month in _iter_months(start, end):
        if month < current_month:
            monthly_plan.append(month)
        else:
            month_end = _next_month(month) - timedelta(days=1)
            daily_plan.extend(_iter_days(max(start, month), min(end, month_end)))

    total_added = 0

    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        for month in monthly_plan:
            url = _monthly_url(symbol, interval, month)
            archive_name = url.rsplit("/", 1)[-1]
            print(f"Downloading {archive_name} ...")
            content = await _download_archive(client, url)

            if content is None:
                month_end = _next_month(month) - timedelta(days=1)
                fallback_start = max(start, month)
                fallback_end = min(end, month_end)
                print(f"Monthly archive missing; falling back to daily files for {month:%Y-%m}.")
                for day in _iter_days(fallback_start, fallback_end):
                    daily_plan.append(day)
                continue

            records = _parse_archive(
                content,
                symbol=symbol,
                interval=interval,
                archive_name=archive_name,
                start_ms=start_ms,
                end_ms_exclusive=end_ms_exclusive,
            )
            added = append_unique_jsonl(output_file, records, key="open_time")
            total_added += added
            print(f"  parsed={len(records):,} new={added:,}")

        for day in sorted(set(daily_plan)):
            # Daily files are normally published the following day.
            if day >= datetime.now(UTC).date():
                continue

            url = _daily_url(symbol, interval, day)
            archive_name = url.rsplit("/", 1)[-1]
            print(f"Downloading {archive_name} ...")
            content = await _download_archive(client, url)
            if content is None:
                print("  archive not available yet; skipping")
                continue

            records = _parse_archive(
                content,
                symbol=symbol,
                interval=interval,
                archive_name=archive_name,
                start_ms=start_ms,
                end_ms_exclusive=end_ms_exclusive,
            )
            added = append_unique_jsonl(output_file, records, key="open_time")
            total_added += added
            print(f"  parsed={len(records):,} new={added:,}")

    print()
    print(f"Binance historical backfill complete. New candles: {total_added:,}")
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

    parser = argparse.ArgumentParser(
        description="Backfill Binance BTC historical klines from public ZIP archives."
    )
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="1m")
    parser.add_argument("--start", type=_parse_date, default=default_start)
    parser.add_argument("--end", type=_parse_date, default=default_end)
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    await backfill_binance_klines(
        symbol=args.symbol,
        interval=args.interval,
        start=args.start,
        end=args.end,
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nHistorical download stopped.")
