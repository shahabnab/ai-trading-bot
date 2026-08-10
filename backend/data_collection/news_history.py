from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import httpx

from backend.config import settings
from backend.data_collection.storage import append_unique_jsonl


ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"
DEFAULT_OUTPUT = Path("data/raw/text/alpha_vantage/btc_news.jsonl")


class NewsHistoryError(RuntimeError):
    """Raised when Alpha Vantage historical news cannot be downloaded."""


def _parse_time_published(value: str) -> datetime:
    for fmt in ("%Y%m%dT%H%M%S", "%Y%m%dT%H%M"):
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=UTC)
        except ValueError:
            continue
    raise ValueError(f"unsupported Alpha Vantage timestamp: {value}")


def _btc_ticker_sentiment(article: dict[str, object]) -> dict[str, object]:
    ticker_sentiment = article.get("ticker_sentiment")
    if not isinstance(ticker_sentiment, list):
        return {}

    for item in ticker_sentiment:
        if isinstance(item, dict) and str(item.get("ticker", "")).upper() == "CRYPTO:BTC":
            return item
    return {}


def _normalize_article(article: dict[str, object]) -> dict[str, object] | None:
    published = str(article.get("time_published", "")).strip()
    title = str(article.get("title", "")).strip()
    url = str(article.get("url", "")).strip()

    if not published or not title:
        return None

    try:
        published_dt = _parse_time_published(published)
    except ValueError:
        return None

    btc = _btc_ticker_sentiment(article)
    topics_raw = article.get("topics")
    topics: list[dict[str, str]] = []
    if isinstance(topics_raw, list):
        for topic in topics_raw:
            if isinstance(topic, dict):
                topics.append(
                    {
                        "topic": str(topic.get("topic", "")),
                        "relevance_score": str(topic.get("relevance_score", "")),
                    }
                )

    record_id = url or f"{published}|{title}"

    return {
        "record_id": record_id,
        "source": "alpha_vantage_news_sentiment",
        "time_published": published,
        "time_published_ms": int(published_dt.timestamp() * 1000),
        "time_published_iso": published_dt.isoformat(),
        "title": title,
        "summary": str(article.get("summary", "")).strip(),
        "source_name": str(article.get("source", "")).strip(),
        "source_domain": str(article.get("source_domain", "")).strip(),
        "url": url,
        "overall_sentiment_score": article.get("overall_sentiment_score"),
        "overall_sentiment_label": article.get("overall_sentiment_label"),
        "btc_relevance_score": btc.get("relevance_score"),
        "btc_sentiment_score": btc.get("ticker_sentiment_score"),
        "btc_sentiment_label": btc.get("ticker_sentiment_label"),
        "topics": topics,
    }


async def backfill_btc_news(
    *,
    start: date,
    end: date,
    window_days: int = 8,
    output_file: Path = DEFAULT_OUTPUT,
    pause_seconds: float = 1.0,
) -> Path:
    """Backfill BTC news/sentiment from Alpha Vantage in bounded historical windows."""
    api_key = settings.alpha_vantage_api_key
    if not api_key:
        raise NewsHistoryError(
            "ALPHAVANTAGE_API_KEY is not configured. Add it to your local .env file."
        )
    if end < start:
        raise ValueError("end date must be on or after start date")
    if window_days < 1:
        raise ValueError("window_days must be positive")

    total_added = 0
    current = start
    windows = 0

    async with httpx.AsyncClient(timeout=60.0) as client:
        while current <= end:
            window_end = min(end, current + timedelta(days=window_days - 1))
            start_dt = datetime.combine(current, datetime.min.time(), tzinfo=UTC)
            end_dt = datetime.combine(window_end, datetime.max.time().replace(microsecond=0), tzinfo=UTC)

            params = {
                "function": "NEWS_SENTIMENT",
                "tickers": "CRYPTO:BTC",
                "time_from": start_dt.strftime("%Y%m%dT%H%M"),
                "time_to": end_dt.strftime("%Y%m%dT%H%M"),
                "sort": "EARLIEST",
                "limit": 1000,
                "apikey": api_key,
            }

            print(f"Downloading BTC news {current} -> {window_end} ...")
            response = await client.get(ALPHA_VANTAGE_URL, params=params)
            try:
                response.raise_for_status()
            except httpx.HTTPError as exc:
                raise NewsHistoryError(f"Alpha Vantage HTTP request failed: {exc}") from exc

            payload = response.json()
            if not isinstance(payload, dict):
                raise NewsHistoryError("Alpha Vantage returned an unexpected payload")

            if "Information" in payload:
                raise NewsHistoryError(str(payload["Information"]))
            if "Note" in payload:
                raise NewsHistoryError(str(payload["Note"]))
            if "Error Message" in payload:
                raise NewsHistoryError(str(payload["Error Message"]))

            feed = payload.get("feed", [])
            if not isinstance(feed, list):
                raise NewsHistoryError("Alpha Vantage returned an invalid news feed")

            records = []
            for item in feed:
                if isinstance(item, dict):
                    normalized = _normalize_article(item)
                    if normalized is not None:
                        records.append(normalized)

            added = append_unique_jsonl(output_file, records, key="record_id")
            total_added += added
            windows += 1

            print(f"  received={len(feed):,} parsed={len(records):,} new={added:,}")
            if len(feed) >= 1000:
                print(
                    "  WARNING: this window hit the 1000-article limit. "
                    "Re-run with a smaller --window-days value for complete coverage."
                )

            current = window_end + timedelta(days=1)
            if current <= end and pause_seconds > 0:
                await asyncio.sleep(pause_seconds)

    print()
    print(f"News backfill complete. Windows: {windows}; new articles: {total_added:,}")
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
        description="Backfill historical BTC news/sentiment from Alpha Vantage."
    )
    parser.add_argument("--start", type=_parse_date, default=default_start)
    parser.add_argument("--end", type=_parse_date, default=default_end)
    parser.add_argument(
        "--window-days",
        type=int,
        default=8,
        help="Days per API request. Eight days keeps ~6 months within 25 requests.",
    )
    parser.add_argument("--pause-seconds", type=float, default=1.0)
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    await backfill_btc_news(
        start=args.start,
        end=args.end,
        window_days=args.window_days,
        pause_seconds=args.pause_seconds,
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nNews download stopped.")
