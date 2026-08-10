from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_MARKET_FILE = Path("data/raw/market/binance/BTCUSDT/1m/candles.jsonl")
DEFAULT_NEWS_FILE = Path("data/raw/text/alpha_vantage/btc_news.jsonl")
DEFAULT_FNG_FILE = Path("data/raw/sentiment/alternative_me/fear_greed.jsonl")
DEFAULT_OUTPUT = Path("data/processed/training/btc_hourly.jsonl")

MINUTE_MS = 60 * 1000
HOUR_MS = 60 * MINUTE_MS
DEFAULT_MIN_CANDLES_PER_HOUR = 60


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                yield item


def _float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _aggregate_market(
    market_file: Path,
    *,
    min_candles_per_hour: int = DEFAULT_MIN_CANDLES_PER_HOUR,
) -> list[dict[str, Any]]:
    if not 1 <= min_candles_per_hour <= 60:
        raise ValueError("min_candles_per_hour must be between 1 and 60")

    grouped: dict[int, dict[int, dict[str, Any]]] = defaultdict(dict)
    for candle in _read_jsonl(market_file):
        timestamp = _int(candle.get("open_time"), -1)
        if timestamp < 0:
            continue
        hour = timestamp // HOUR_MS * HOUR_MS
        grouped[hour][timestamp] = candle

    hourly: list[dict[str, Any]] = []
    for hour in sorted(grouped):
        by_timestamp = grouped[hour]
        candles = [by_timestamp[key] for key in sorted(by_timestamp)]
        valid_minute_slots = {
            timestamp
            for timestamp in by_timestamp
            if hour <= timestamp < hour + HOUR_MS
            and (timestamp - hour) % MINUTE_MS == 0
        }
        candle_count = len(valid_minute_slots)
        if candle_count < min_candles_per_hour:
            continue

        quote_volume = sum(_float(item.get("quote_volume")) for item in candles)
        taker_buy_quote = sum(_float(item.get("taker_buy_quote_volume")) for item in candles)
        hourly.append(
            {
                "timestamp": hour + HOUR_MS,
                "feature_window_start": hour,
                "feature_window_end": hour + HOUR_MS,
                "symbol": str(candles[0].get("symbol", "BTCUSDT")),
                "open": _float(candles[0].get("open")),
                "high": max(_float(item.get("high")) for item in candles),
                "low": min(_float(item.get("low")) for item in candles),
                "close": _float(candles[-1].get("close")),
                "volume": sum(_float(item.get("volume")) for item in candles),
                "quote_volume": quote_volume,
                "number_of_trades": sum(_int(item.get("number_of_trades")) for item in candles),
                "taker_buy_base_volume": sum(
                    _float(item.get("taker_buy_base_volume")) for item in candles
                ),
                "taker_buy_quote_volume": taker_buy_quote,
                "taker_buy_quote_ratio": (
                    taker_buy_quote / quote_volume if quote_volume > 0 else 0.0
                ),
                "candle_count": candle_count,
                "minute_count": candle_count,
            }
        )

    return hourly


def _aggregate_news(news_file: Path) -> dict[int, dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)

    if not news_file.exists():
        return {}

    for article in _read_jsonl(news_file):
        published_ms = _int(article.get("time_published_ms"), -1)
        if published_ms < 0:
            continue
        hour = published_ms // HOUR_MS * HOUR_MS
        grouped[hour].append(article)

    result: dict[int, dict[str, Any]] = {}

    for hour, articles in grouped.items():
        overall_scores = [
            _float(article.get("overall_sentiment_score"))
            for article in articles
            if article.get("overall_sentiment_score") not in (None, "")
        ]
        btc_scores = [
            _float(article.get("btc_sentiment_score"))
            for article in articles
            if article.get("btc_sentiment_score") not in (None, "")
        ]
        relevance_scores = [
            _float(article.get("btc_relevance_score"))
            for article in articles
            if article.get("btc_relevance_score") not in (None, "")
        ]

        text_parts: list[str] = []
        titles: list[str] = []
        for article in articles:
            title = str(article.get("title", "")).strip()
            summary = str(article.get("summary", "")).strip()
            if title:
                titles.append(title)
            if title and summary:
                text_parts.append(f"{title}. {summary}")
            elif title:
                text_parts.append(title)

        result[hour] = {
            "news_count": len(articles),
            "news_overall_sentiment_mean": (
                sum(overall_scores) / len(overall_scores) if overall_scores else 0.0
            ),
            "news_btc_sentiment_mean": (
                sum(btc_scores) / len(btc_scores) if btc_scores else 0.0
            ),
            "news_btc_relevance_mean": (
                sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
            ),
            "news_titles": titles,
            "news_text": "\n".join(text_parts)[:20000],
        }

    return result


def _load_fear_greed(fng_file: Path) -> list[dict[str, Any]]:
    values = []
    if not fng_file.exists():
        return values

    for item in _read_jsonl(fng_file):
        timestamp = _int(item.get("timestamp"), -1)
        if timestamp >= 0:
            values.append(item)

    return sorted(values, key=lambda item: _int(item.get("timestamp")))


def _attach_context(
    hourly: list[dict[str, Any]],
    news_by_hour: dict[int, dict[str, Any]],
    fear_greed: list[dict[str, Any]],
) -> None:
    fng_index = 0
    latest_fng: dict[str, Any] | None = None

    for row in hourly:
        feature_end = _int(row["feature_window_end"])
        news_hour = feature_end - HOUR_MS
        row.update(
            news_by_hour.get(
                news_hour,
                {
                    "news_count": 0,
                    "news_overall_sentiment_mean": 0.0,
                    "news_btc_sentiment_mean": 0.0,
                    "news_btc_relevance_mean": 0.0,
                    "news_titles": [],
                    "news_text": "",
                },
            )
        )

        while fng_index < len(fear_greed):
            candidate = fear_greed[fng_index]
            candidate_ts = _int(candidate.get("timestamp"), -1)
            if candidate_ts > feature_end:
                break
            latest_fng = candidate
            fng_index += 1

        row["fear_greed_value"] = (
            _int(latest_fng.get("value")) if latest_fng is not None else None
        )
        row["fear_greed_classification"] = (
            str(latest_fng.get("classification", "")) if latest_fng is not None else None
        )


def _add_market_features_and_targets(hourly: list[dict[str, Any]]) -> None:
    rows_by_timestamp = {_int(row["timestamp"]): row for row in hourly}

    for row in hourly:
        timestamp = _int(row["timestamp"])
        open_price = _float(row["open"])
        high = _float(row["high"])
        low = _float(row["low"])
        close = _float(row["close"])

        previous = rows_by_timestamp.get(timestamp - HOUR_MS)
        previous_close = _float(previous["close"]) if previous is not None else 0.0
        row["return_1h"] = close / previous_close - 1.0 if previous_close else None
        row["range_pct"] = (high - low) / open_price if open_price else None

        for lookback in (4, 24):
            previous = rows_by_timestamp.get(timestamp - lookback * HOUR_MS)
            previous_close = _float(previous["close"]) if previous is not None else 0.0
            row[f"return_{lookback}h"] = (
                close / previous_close - 1.0 if previous_close else None
            )

        recent_returns: list[float] = []
        complete_window = True
        for offset in range(24):
            current = rows_by_timestamp.get(timestamp - offset * HOUR_MS)
            previous = rows_by_timestamp.get(timestamp - (offset + 1) * HOUR_MS)
            previous_close = _float(previous["close"]) if previous is not None else 0.0
            if current is None or previous is None or previous_close == 0.0:
                complete_window = False
                break
            recent_returns.append(_float(current["close"]) / previous_close - 1.0)
        row["volatility_24h"] = (
            statistics.pstdev(recent_returns)
            if complete_window and len(recent_returns) >= 2
            else None
        )

        for horizon in (1, 4, 24):
            future = rows_by_timestamp.get(timestamp + horizon * HOUR_MS)
            row[f"target_return_{horizon}h"] = (
                _float(future["close"]) / close - 1.0 if future is not None and close else None
            )


def build_hourly_training_dataset(
    *,
    market_file: Path = DEFAULT_MARKET_FILE,
    news_file: Path = DEFAULT_NEWS_FILE,
    fng_file: Path = DEFAULT_FNG_FILE,
    output_file: Path = DEFAULT_OUTPUT,
    min_candles_per_hour: int = DEFAULT_MIN_CANDLES_PER_HOUR,
) -> Path:
    if not market_file.exists():
        raise FileNotFoundError(
            f"Market history not found: {market_file}. Run the historical backfill first."
        )

    hourly = _aggregate_market(market_file, min_candles_per_hour=min_candles_per_hour)
    news_by_hour = _aggregate_news(news_file)
    fear_greed = _load_fear_greed(fng_file)

    _attach_context(hourly, news_by_hour, fear_greed)
    _add_market_features_and_targets(hourly)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        for row in hourly:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")

    print(f"Hourly training rows: {len(hourly):,}")
    print(f"Minimum 1m candles per hourly bar: {min_candles_per_hour}")
    print(f"Market file: {market_file}")
    print(f"News file present: {news_file.exists()}")
    print(f"Fear & Greed file present: {fng_file.exists()}")
    print(f"Training dataset: {output_file}")
    return output_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a leakage-aware hourly BTC training dataset."
    )
    parser.add_argument("--market-file", type=Path, default=DEFAULT_MARKET_FILE)
    parser.add_argument("--news-file", type=Path, default=DEFAULT_NEWS_FILE)
    parser.add_argument("--fng-file", type=Path, default=DEFAULT_FNG_FILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--min-candles-per-hour",
        type=int,
        default=DEFAULT_MIN_CANDLES_PER_HOUR,
        help="Drop hours with fewer unique 1m candles than this threshold (default: 60).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_hourly_training_dataset(
        market_file=args.market_file,
        news_file=args.news_file,
        fng_file=args.fng_file,
        output_file=args.output,
        min_candles_per_hour=args.min_candles_per_hour,
    )


if __name__ == "__main__":
    main()
