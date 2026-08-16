# Short-Term PAPER Trading Lab

This experiment is separate from the four existing forward strategies. The frozen V3 controls and current Trader-Brain models are not retrained, retuned or given short-term state.

## Why the inputs are different

A 15-minute strategy cannot rely only on the hourly features used by the current Trader-Brain. It needs market-activity and microstructure information that can change materially inside one hour.

The first short-term feature set includes:

- 15m, 30m, 1h and 2h returns
- EMA(8)/EMA(21) gap and short EMA slope
- RSI(14)
- ATR / candle range / candle body
- Bollinger displacement
- 20-bar volume and traded-value z-scores
- rolling VWAP distance
- 1h realized volatility
- public CoinEx trade count
- taker buy and sell notional
- taker-flow imbalance / buy ratio
- top-of-book spread
- 20-level order-book notional imbalance
- data-quality / microstructure-coverage score

CoinEx public market endpoints are read-only. No exchange credentials are required for this collector.

## Live collection

`ai-trading-shortterm-collector.service` polls public BTCUSDT deals and depth and aggregates them into completed 15-minute buckets at:

`state/short_term/microstructure.jsonl`

Each 15-minute decision cycle also stores the point-in-time feature row at:

`state/short_term/features.jsonl`

These rows become forward research data for a later trained Short Trader-Brain model.

## Initial active baselines

1. `short-momentum-15m` — cost-aware trend/momentum benchmark.
2. `short-mean-reversion-15m` — cost-aware oversold/mean-reversion benchmark.

Both are long/flat spot PAPER strategies with independent €1,000-equivalent ledgers. They are evaluated every 15 minutes and target holds up to roughly two hours.

The entry hurdle uses the configured paper fee plus slippage for both entry and exit, plus a 15 bps research buffer. With the current 20 bps fee and 5 bps slippage assumptions, the default hurdle is 65 bps. This is deliberately conservative: the goal is more *candidate opportunities*, not forced turnover.

## Next research stage

After the live collector and historical 15-minute backfill provide enough data, add two learned candidates without changing these benchmarks:

- Short Trader Brain — supervised short-horizon mixture/gate using the same point-in-time features.
- Short Trader Brain + RL — PAPER contextual policy trained only from resolved short-term experiences.

The learned models should be backtested with time-based splits and transaction costs before joining the live short-term scoreboard.
