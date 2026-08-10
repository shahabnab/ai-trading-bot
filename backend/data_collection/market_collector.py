from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from backend.data_collection.storage import append_unique_jsonl
from backend.market.coinex_market import CoinExMarketClient, MarketDataError


DEFAULT_DATA_ROOT = Path("data/raw/market/coinex")


async def collect_once(
    symbol: str = "BTCUSDT",
    period: str = "1min",
    limit: int = 1000,
) -> int:
    """Fetch recent CoinEx candles once and append only unseen timestamps."""
    symbol = symbol.upper().strip()
    limit = max(1, min(limit, 1000))

    print(f"Collecting {symbol} {period} candles...")

    client = CoinExMarketClient()
    candles = await client.get_klines(
        symbol=symbol,
        period=period,
        limit=limit,
    )

    output_file = DEFAULT_DATA_ROOT / symbol / period / "candles.jsonl"
    added = append_unique_jsonl(
        output_file,
        candles,
        key="created_at",
    )

    print(f"Received candles: {len(candles)}")
    print(f"New candles saved: {added}")
    print(f"Output: {output_file}")

    if candles:
        def timestamp_value(candle: dict[str, object]) -> int:
            try:
                return int(candle.get("created_at") or 0)
            except (TypeError, ValueError):
                return 0

        latest = max(candles, key=timestamp_value)

        print()
        print("Latest candle:")
        print(f"Timestamp: {latest.get('created_at')}")
        print(f"Open:      {latest.get('open')}")
        print(f"High:      {latest.get('high')}")
        print(f"Low:       {latest.get('low')}")
        print(f"Close:     {latest.get('close')}")
        print(f"Volume:    {latest.get('volume')}")

    return added


async def watch(
    symbol: str,
    period: str,
    limit: int,
    poll_seconds: int,
) -> None:
    """Continuously poll CoinEx and append new candles."""
    poll_seconds = max(1, poll_seconds)

    print(
        f"Starting continuous collection for {symbol.upper()} "
        f"every {poll_seconds} seconds."
    )
    print("Press Ctrl+C to stop.")
    print()

    while True:
        try:
            await collect_once(
                symbol=symbol,
                period=period,
                limit=limit,
            )
        except MarketDataError as exc:
            print(f"CoinEx error: {exc}")
        except Exception as exc:  # keep long-running collector alive
            print(f"Unexpected error: {exc}")

        print()
        await asyncio.sleep(poll_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect public CoinEx candlestick data for ML/backtesting.",
    )
    parser.add_argument(
        "--symbol",
        default="BTCUSDT",
        help="CoinEx market symbol (default: BTCUSDT).",
    )
    parser.add_argument(
        "--period",
        default="1min",
        help="Candle period such as 1min, 5min, or 1hour (default: 1min).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Recent candles per request, capped at 1000 (default: 1000).",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Keep collecting continuously instead of running once.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=60,
        help="Seconds between requests in watch mode (default: 60).",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    if args.watch:
        await watch(
            symbol=args.symbol,
            period=args.period,
            limit=args.limit,
            poll_seconds=args.poll_seconds,
        )
    else:
        await collect_once(
            symbol=args.symbol,
            period=args.period,
            limit=args.limit,
        )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nCollector stopped.")
