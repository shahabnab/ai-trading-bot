from __future__ import annotations

import argparse
import asyncio
import csv
import io
import json
import zipfile
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Iterator

import httpx
import numpy as np

BINANCE_DATA_BASE = "https://data.binance.vision/data"
RAW_ROOT = Path("data/raw/research_context/binance")
OUTPUT_ROOT = Path("data/processed/context")
TRAINING_OUTPUT = Path("data/processed/training/btc_hourly_v3.jsonl")
HOUR_MS = 60 * 60 * 1000


class ResearchContextError(RuntimeError):
    pass


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must use YYYY-MM-DD") from exc


def _month_start(day: date) -> date:
    return day.replace(day=1)


def _next_month(day: date) -> date:
    if day.month == 12:
        return date(day.year + 1, 1, 1)
    return date(day.year, day.month + 1, 1)


def _iter_months(start: date, end: date) -> Iterable[date]:
    current = _month_start(start)
    final = _month_start(end)
    while current <= final:
        yield current
        current = _next_month(current)


def _iter_days(start: date, end: date) -> Iterable[date]:
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def _normalize_timestamp_ms(value: str | int) -> int:
    timestamp = int(value)
    if timestamp >= 10**14:
        return timestamp // 1000
    if timestamp >= 10**11:
        return timestamp
    return timestamp * 1000


@dataclass(frozen=True)
class ArchiveStream:
    key: str
    market_path: str
    data_type: str
    symbol: str
    interval: str | None
    output_name: str

    def filename(self, day: date, *, daily: bool) -> str:
        stamp = day.strftime("%Y-%m-%d" if daily else "%Y-%m")
        if self.interval:
            return f"{self.symbol}-{self.interval}-{stamp}.zip"
        return f"{self.symbol}-{self.data_type}-{stamp}.zip"

    def url(self, day: date, *, daily: bool) -> str:
        cadence = "daily" if daily else "monthly"
        if self.interval:
            return (
                f"{BINANCE_DATA_BASE}/{self.market_path}/{cadence}/{self.data_type}/"
                f"{self.symbol}/{self.interval}/{self.filename(day, daily=daily)}"
            )
        return (
            f"{BINANCE_DATA_BASE}/{self.market_path}/{cadence}/{self.data_type}/"
            f"{self.symbol}/{self.filename(day, daily=daily)}"
        )


STREAMS = (
    ArchiveStream("btc_spot_1h", "spot", "klines", "BTCUSDT", "1h", "btc_spot_hourly.jsonl"),
    ArchiveStream("btc_spot_aggtrades", "spot", "aggTrades", "BTCUSDT", None, "btc_spot_aggtrades_hourly.jsonl"),
    ArchiveStream("btc_um_futures_1h", "futures/um", "klines", "BTCUSDT", "1h", "btc_um_futures_hourly.jsonl"),
    ArchiveStream("eth_spot_1h", "spot", "klines", "ETHUSDT", "1h", "eth_spot_hourly.jsonl"),
)


def _csv_rows(content: bytes, archive_name: str) -> Iterator[list[str]]:
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if not names:
                raise ResearchContextError(f"no CSV in {archive_name}")
            with archive.open(names[0]) as raw:
                text = io.TextIOWrapper(raw, encoding="utf-8")
                for row in csv.reader(text):
                    if row:
                        yield row
    except zipfile.BadZipFile as exc:
        raise ResearchContextError(f"invalid ZIP: {archive_name}") from exc


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")


async def _fetch(client: httpx.AsyncClient, url: str) -> bytes | None:
    response = await client.get(url)
    if response.status_code == 404:
        return None
    try:
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise ResearchContextError(f"Binance archive request failed: {url}: {exc}") from exc
    return response.content


def _finalize_agg_hour(hour_start: int, prices: list[float], quote_sizes: list[float], aggressive_buy: list[bool]) -> dict[str, object]:
    q = np.asarray(quote_sizes, dtype=np.float64)
    p = np.asarray(prices, dtype=np.float64)
    buy = np.asarray(aggressive_buy, dtype=bool)
    total = float(np.sum(q))
    buy_quote = float(np.sum(q[buy]))
    sell_quote = total - buy_quote
    p95 = float(np.percentile(q, 95.0)) if len(q) else 0.0
    large = q >= p95 if len(q) else np.zeros(0, dtype=bool)
    large_total = float(np.sum(q[large])) if len(q) else 0.0
    large_buy = float(np.sum(q[large & buy])) if len(q) else 0.0
    quantity = np.divide(q, p, out=np.zeros_like(q), where=p > 0.0)
    quantity_sum = float(np.sum(quantity))
    vwap = float(np.sum(q) / quantity_sum) if quantity_sum > 0.0 else float(p[-1])
    return {
        "timestamp": hour_start + HOUR_MS,
        "agg_trade_count": int(len(q)),
        "aggressive_buy_quote_volume": buy_quote,
        "aggressive_sell_quote_volume": sell_quote,
        "aggressive_quote_volume": total,
        "aggressive_imbalance": (buy_quote - sell_quote) / total if total > 0.0 else 0.0,
        "aggressive_buy_trade_ratio": float(np.mean(buy)) if len(buy) else 0.5,
        "agg_vwap": vwap,
        "avg_trade_quote": float(np.mean(q)) if len(q) else 0.0,
        "median_trade_quote": float(np.median(q)) if len(q) else 0.0,
        "p95_trade_quote": p95,
        "max_trade_quote": float(np.max(q)) if len(q) else 0.0,
        "large_trade_quote_share": large_total / total if total > 0.0 else 0.0,
        "large_buy_quote_share": large_buy / large_total if large_total > 0.0 else 0.5,
        "agg_price_range_pct": (float(np.max(p)) - float(np.min(p))) / float(p[0]) if len(p) and p[0] else 0.0,
        "agg_intrahour_return": float(p[-1] / p[0] - 1.0) if len(p) > 1 and p[0] > 0.0 else 0.0,
    }


def _aggregate_aggtrade_archive(content: bytes, archive_name: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    current_hour: int | None = None
    prices: list[float] = []
    quote_sizes: list[float] = []
    aggressive_buy: list[bool] = []

    def flush() -> None:
        nonlocal prices, quote_sizes, aggressive_buy
        if current_hour is not None and prices:
            rows.append(_finalize_agg_hour(current_hour, prices, quote_sizes, aggressive_buy))
        prices, quote_sizes, aggressive_buy = [], [], []

    for row in _csv_rows(content, archive_name):
        if len(row) < 7:
            continue
        try:
            price = float(row[1])
            quantity = float(row[2])
            timestamp = _normalize_timestamp_ms(row[5])
            buyer_maker = str(row[6]).strip().lower() in {"true", "1"}
        except (TypeError, ValueError):
            continue
        hour = timestamp // HOUR_MS * HOUR_MS
        if current_hour is None:
            current_hour = hour
        if hour != current_hour:
            flush()
            current_hour = hour
        prices.append(price)
        quote_sizes.append(max(price * quantity, 0.0))
        aggressive_buy.append(not buyer_maker)
    flush()
    return rows


def _aggregate_kline_archive(content: bytes, archive_name: str, source: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in _csv_rows(content, archive_name):
        if len(row) < 11:
            continue
        try:
            open_time = _normalize_timestamp_ms(row[0])
            quote_volume = float(row[7])
            taker_quote = float(row[10])
            rows.append({
                "timestamp": open_time + HOUR_MS,
                "source": source,
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "base_volume": float(row[5]),
                "quote_volume": quote_volume,
                "trade_count": int(float(row[8])),
                "taker_buy_quote_ratio": taker_quote / quote_volume if quote_volume > 0.0 else 0.5,
                "minute_count": 60,
            })
        except (TypeError, ValueError):
            continue
    return rows


def _aggregate_archive(stream: ArchiveStream, content: bytes, archive_name: str) -> list[dict[str, object]]:
    if stream.data_type == "aggTrades":
        return _aggregate_aggtrade_archive(content, archive_name)
    return _aggregate_kline_archive(content, archive_name, stream.key)


def _archive_plan(start: date, end: date, *, daily_only: bool = False) -> list[tuple[date, bool]]:
    if daily_only:
        return [(day, True) for day in _iter_days(start, end) if day < datetime.now(UTC).date()]
    current_month = _month_start(datetime.now(UTC).date())
    plan: list[tuple[date, bool]] = []
    for month in _iter_months(start, end):
        if month < current_month:
            plan.append((month, False))
        else:
            month_end = _next_month(month) - timedelta(days=1)
            for day in _iter_days(max(start, month), min(end, month_end)):
                if day < datetime.now(UTC).date():
                    plan.append((day, True))
    return plan


async def collect_stream(
    client: httpx.AsyncClient,
    stream: ArchiveStream,
    *,
    start: date,
    end: date,
    output_root: Path,
    raw_root: Path,
    keep_archives: bool,
) -> Path:
    output_path = output_root / stream.output_name
    output: dict[int, dict[str, object]] = {}
    # Core BTC spot uses daily archives by default. It is slower but avoids
    # depending on a monthly archive when a corrected daily file exists.
    plan = _archive_plan(start, end, daily_only=(stream.key == "btc_spot_1h"))
    for item, daily in plan:
        url = stream.url(item, daily=daily)
        archive_name = stream.filename(item, daily=daily)
        print(f"[{stream.key}] {archive_name}", flush=True)
        content = await _fetch(client, url)
        if content is None and not daily:
            month_end = _next_month(item) - timedelta(days=1)
            for day in _iter_days(max(start, item), min(end, month_end)):
                daily_url = stream.url(day, daily=True)
                daily_name = stream.filename(day, daily=True)
                daily_content = await _fetch(client, daily_url)
                if daily_content is None:
                    continue
                for row in _aggregate_archive(stream, daily_content, daily_name):
                    output[int(row["timestamp"])] = row
            continue
        if content is None:
            continue
        if keep_archives:
            target = raw_root / stream.key / stream.symbol / (stream.interval or "na") / ("daily" if daily else "monthly") / archive_name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        archive_rows = _aggregate_archive(stream, content, archive_name)
        for row in archive_rows:
            output[int(row["timestamp"])] = row
        print(f"  hourly rows so far: {len(output):,}", flush=True)

    rows = [output[key] for key in sorted(output)]
    _write_jsonl(output_path, rows)
    print(f"{stream.key}: {len(rows):,} hourly rows -> {output_path}")
    return output_path


def build_v3_training_dataset(*, spot_hourly_path: Path, output_path: Path = TRAINING_OUTPUT) -> Path:
    rows: list[dict[str, object]] = []
    with spot_hourly_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    by_ts = {int(row["timestamp"]): row for row in rows}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            timestamp = int(row["timestamp"])
            close = float(row["close"])
            future = by_ts.get(timestamp + HOUR_MS)
            target = float(future["close"]) / close - 1.0 if future is not None and close > 0.0 else None
            payload = {
                "timestamp": timestamp,
                "feature_window_start": timestamp - HOUR_MS,
                "feature_window_end": timestamp,
                "symbol": "BTCUSDT",
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": close,
                "volume": float(row["base_volume"]),
                "quote_volume": float(row["quote_volume"]),
                "number_of_trades": int(row["trade_count"]),
                "taker_buy_quote_ratio": float(row["taker_buy_quote_ratio"]),
                "target_return_1h": target,
                "news_count": 0,
                "news_overall_sentiment_mean": 0.0,
                "news_btc_sentiment_mean": 0.0,
                "news_btc_relevance_mean": 0.0,
                "fear_greed_value": None,
            }
            handle.write(json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n")
    print(f"V3 BTC hourly training rows: {len(rows):,} -> {output_path}")
    return output_path


async def collect_all(
    *,
    start: date,
    end: date,
    output_root: Path = OUTPUT_ROOT,
    raw_root: Path = RAW_ROOT,
    keep_archives: bool = False,
    force: bool = False,
) -> list[Path]:
    if end < start:
        raise ValueError("end must be on or after start")
    paths: list[Path] = []
    async with httpx.AsyncClient(timeout=180.0, follow_redirects=True) as client:
        for stream in STREAMS:
            output_path = output_root / stream.output_name
            if output_path.is_file() and not force:
                print(f"Using existing processed context: {output_path}")
                paths.append(output_path)
                continue
            paths.append(await collect_stream(client, stream, start=start, end=end, output_root=output_root, raw_root=raw_root, keep_archives=keep_archives))
    btc_spot = output_root / "btc_spot_hourly.jsonl"
    paths.append(build_v3_training_dataset(spot_hourly_path=btc_spot))
    return paths


def parse_args() -> argparse.Namespace:
    today = datetime.now(UTC).date()
    parser = argparse.ArgumentParser(description="Collect compact Binance market context for V3 research.")
    parser.add_argument("--start", type=_parse_date, default=date(2020, 1, 1))
    parser.add_argument("--end", type=_parse_date, default=today - timedelta(days=1))
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--keep-archives", action="store_true", help="Keep downloaded ZIPs. Default streams and discards to save disk.")
    parser.add_argument("--force", action="store_true", help="Re-download even when processed context already exists.")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    print("=" * 78)
    print("V3 BINANCE RESEARCH CONTEXT")
    print(f"Range: {args.start} -> {args.end}")
    print("Streams: BTC spot 1h daily, BTC spot aggTrades, BTC USD-M futures 1h, ETH spot 1h")
    print(f"Archive retention: {'keep' if args.keep_archives else 'stream + discard (disk efficient)'}")
    print("=" * 78)
    await collect_all(start=args.start, end=args.end, output_root=args.output_root, raw_root=args.raw_root, keep_archives=args.keep_archives, force=args.force)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nResearch-context collection stopped.")
