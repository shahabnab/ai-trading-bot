# Data collection

This package is the home for datasets used by research, backtesting, feature engineering, and ML training.

## Why this is separate

- `backend/coinex/` handles authenticated read-only account access.
- `backend/market/` handles live/public exchange market API access.
- `backend/data_collection/` coordinates repeated collection and persistence of training data.
- `data/raw/` and `data/processed/` hold downloaded/generated datasets and are ignored by Git.

## Planned data layout

```text
data/
  raw/
    market/
      coinex/
        BTCUSDT/
          1min/
    text/
      news/
  processed/
    features/
    training/
```

## Collection pipeline

```text
CoinEx public market API ----> raw BTC candles -----+
                                                   |
News/text providers ---------> raw text records ----+--> time alignment --> features --> ML dataset
```

The collection process should run on a normal CPU machine or inexpensive VPS. A rented GPU is only needed later for model training.

## Market data

The repository already contains `backend.market.CoinExMarketClient`, including a public `get_klines()` method. The collection layer should call that client and persist the returned candles instead of duplicating CoinEx API logic here.

Recommended initial configuration:

- market: `BTCUSDT`
- raw interval: `1min`
- timestamp standard: UTC
- fields: open, high, low, close, volume, value, created_at
- storage: append-only JSONL initially; Parquet can be added when pandas/pyarrow are introduced

Repeated collection must deduplicate candles by `created_at`.

## Text/news data

Do not store random text without timestamps. Each record should at least contain:

```text
published_at
collected_at
source
headline
summary_or_snippet
url
query
language
```

Later we can derive sentiment, relevance scores, embeddings, and hourly/daily aggregates from these records.

## Important ML rule

For a training row at time `t`, features may use only information that was available at or before `t`. Future price information belongs only in the target/label. This prevents look-ahead leakage.

## Next implementation steps

1. Add a BTC candle collector that reuses `CoinExMarketClient.get_klines()`.
2. Add historical backfilling using CoinEx candle timestamps.
3. Add a news/text source adapter.
4. Add a scheduler for incremental collection.
5. Convert raw JSONL into Parquet training datasets.
6. Add walk-forward dataset splitting and leakage checks.
