from __future__ import annotations

import argparse
import asyncio
import calendar
from datetime import UTC, date, datetime, timedelta

from backend.config import settings
from backend.data_collection.binance_history import backfill_binance_klines
from backend.data_collection.dataset_builder import build_hourly_training_dataset
from backend.data_collection.fear_greed_history import backfill_fear_greed
from backend.data_collection.news_history import NewsHistoryError, backfill_btc_news


def subtract_months(day: date, months: int) -> date:
    if months < 0:
        raise ValueError("months must be non-negative")

    total_months = day.year * 12 + (day.month - 1) - months
    year, month_index = divmod(total_months, 12)
    month = month_index + 1
    last_day = calendar.monthrange(year, month)[1]
    return date(year, month, min(day.day, last_day))


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must use YYYY-MM-DD") from exc


def parse_args() -> argparse.Namespace:
    today_utc = datetime.now(UTC).date()
    default_end = today_utc - timedelta(days=1)

    parser = argparse.ArgumentParser(
        description=(
            "Create a historical BTC ML dataset from bulk market data, "
            "news/sentiment, and Fear & Greed history."
        )
    )
    parser.add_argument(
        "--months",
        type=int,
        default=6,
        help="Historical calendar months when --start is omitted (default: 6).",
    )
    parser.add_argument("--start", type=_parse_date, default=None)
    parser.add_argument("--end", type=_parse_date, default=default_end)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="1m")
    parser.add_argument(
        "--skip-news",
        action="store_true",
        help="Skip Alpha Vantage news even if an API key is configured.",
    )
    parser.add_argument(
        "--news-window-days",
        type=int,
        default=8,
        help="Days per Alpha Vantage request (default: 8).",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    if args.months < 1:
        raise ValueError("--months must be at least 1")

    start = args.start or subtract_months(args.end, args.months)
    end = args.end

    print("=" * 72)
    print("BTC HISTORICAL DATASET BACKFILL")
    print(f"Range: {start} -> {end} UTC")
    print(f"Market: {args.symbol} {args.interval}")
    print("=" * 72)
    print()

    print("[1/4] Binance historical market data")
    await backfill_binance_klines(
        symbol=args.symbol,
        interval=args.interval,
        start=start,
        end=end,
    )
    print()

    print("[2/4] Fear & Greed history")
    await backfill_fear_greed(start=start, end=end)
    print()

    print("[3/4] BTC historical news/text")
    if args.skip_news:
        print("Skipped by --skip-news.")
    elif settings.alpha_vantage_api_key:
        try:
            await backfill_btc_news(
                start=start,
                end=end,
                window_days=args.news_window_days,
            )
        except NewsHistoryError as exc:
            print(f"News backfill stopped: {exc}")
            print(
                "Market and sentiment data are already saved. "
                "You can rerun the command later; stored records are deduplicated."
            )
    else:
        print(
            "ALPHAVANTAGE_API_KEY is not configured, so news was skipped.\n"
            "Add ALPHAVANTAGE_API_KEY=... to .env and rerun this command "
            "to add historical BTC text/sentiment."
        )
    print()

    print("[4/4] Build hourly training dataset")
    build_hourly_training_dataset()
    print()
    print("Done.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nHistorical dataset creation stopped.")
