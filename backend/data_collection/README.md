# Data collection

This package contains the historical and incremental data pipeline used for research, backtesting, feature engineering, and ML training.

## Design

Do not rent a GPU to collect data. Historical data is downloaded in bulk on a normal CPU machine, processed locally, and only the final training dataset needs to be moved to a rented GPU.

- `backend/coinex/` handles authenticated read-only account access.
- `backend/market/` handles live/public CoinEx market access.
- `backend/data_collection/` builds historical and incremental ML datasets.
- `data/raw/` and `data/processed/` are ignored by Git.

## Six-month historical dataset

The default historical pipeline collects:

1. **BTCUSDT 1-minute spot candles from Binance public bulk archives**
   - OHLCV
   - quote volume
   - number of trades
   - taker-buy base volume
   - taker-buy quote volume
2. **Crypto Fear & Greed history** from Alternative.me.
3. **Historical Bitcoin news/text and sentiment** from Alpha Vantage when `ALPHAVANTAGE_API_KEY` is configured.
4. A leakage-aware **hourly training dataset** with backward-looking market features, aligned news text/sentiment, Fear & Greed, and future 1h/4h/24h return targets.

Binance bulk archives are used for the initial backfill because downloading monthly ZIP files is far faster than polling one candle at a time. CoinEx remains the source for live/recent execution-side market data.

## Run the complete backfill

From the repository root:

```powershell
python -m backend.data_collection.historical_dataset --months 6
```

By default the end date is yesterday in UTC, because daily bulk files are normally available the following day.

You can also specify exact dates:

```powershell
python -m backend.data_collection.historical_dataset --start 2026-02-10 --end 2026-08-09
```

If you do not have an Alpha Vantage key yet, market data and Fear & Greed will still be collected and the dataset will be built without news:

```powershell
python -m backend.data_collection.historical_dataset --months 6 --skip-news
```

## Historical news setup

Create a personal Alpha Vantage API key and put it only in your local `.env` file:

```text
ALPHAVANTAGE_API_KEY=your_key_here
```

Do not commit the real key. The default six-month news download uses 8-day windows so that approximately six months fits within 25 API requests. If a window reaches the provider's 1000-article maximum, the collector prints a warning; rerun with a smaller `--news-window-days` value if your API allowance permits it.

## Output layout

```text
data/
  raw/
    market/
      binance/
        BTCUSDT/
          1m/
            candles.jsonl
      coinex/
        BTCUSDT/
          1min/
            candles.jsonl
    sentiment/
      alternative_me/
        fear_greed.jsonl
    text/
      alpha_vantage/
        btc_news.jsonl
  processed/
    training/
      btc_hourly.jsonl
```

## Training rows

`data/processed/training/btc_hourly.jsonl` contains one row per completed hour. Important fields include:

```text
timestamp
open / high / low / close
volume
quote_volume
number_of_trades
taker_buy_quote_ratio
return_1h
return_4h
return_24h
volatility_24h
news_count
news_overall_sentiment_mean
news_btc_sentiment_mean
news_btc_relevance_mean
news_titles
news_text
fear_greed_value
fear_greed_classification
target_return_1h
target_return_4h
target_return_24h
```

The feature window ends at `timestamp`. News is included only if it was published during or before the completed feature hour. Future prices are used only for `target_return_*` fields.

## Individual commands

Historical Binance market data only:

```powershell
python -m backend.data_collection.binance_history --start 2026-02-10 --end 2026-08-09
```

Fear & Greed only:

```powershell
python -m backend.data_collection.fear_greed_history --start 2026-02-10 --end 2026-08-09
```

Historical BTC news only:

```powershell
python -m backend.data_collection.news_history --start 2026-02-10 --end 2026-08-09
```

Rebuild the processed dataset without downloading anything:

```powershell
python -m backend.data_collection.dataset_builder
```

Continue collecting recent CoinEx candles after the historical backfill:

```powershell
python -m backend.data_collection.market_collector --watch
```

## Important ML rule

For a training row at time `t`, features may use only information available at or before `t`. Future price information belongs only in the target. This prevents look-ahead leakage.

## Next additions

The current historical dataset is a strong first baseline. Later iterations can add futures funding/open-interest features, macro variables, on-chain data, technical indicators, Parquet export, and walk-forward train/validation/test splitting.
