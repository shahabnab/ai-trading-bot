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

The current benchmark implementation uses simple-window average gain/loss for RSI and a simple 14-bar mean true range for ATR. These are intentionally kept stable for the existing hand-written thresholds; they are not Wilder-smoothed indicators. A future learned model must either use these exact feature definitions or explicitly version a changed definition rather than silently mixing them.

## Live collection

`ai-trading-shortterm-collector.service` polls public BTCUSDT deals and depth and aggregates them into completed 15-minute buckets at:

`state/short_term/microstructure.jsonl`

Each 15-minute decision cycle also stores the point-in-time feature row at:

`state/short_term/features.jsonl`

These rows become forward research data for a later trained Short Trader-Brain model.

At bucket rollover, the previous microstructure accumulator remains alive for one additional collector poll so deals reported a few seconds late are still assigned to the bucket that owns their timestamp.

## Initial active baselines

1. `short-momentum-15m` — cost-aware trend/momentum benchmark.
2. `short-mean-reversion-15m` — cost-aware oversold/mean-reversion benchmark.

Both are long/flat spot PAPER strategies with independent €1,000-equivalent ledgers. They are evaluated every 15 minutes and target holds up to roughly two hours.

The entry hurdle uses the configured paper fee plus slippage for both entry and exit, plus a 15 bps research buffer. With the current 20 bps fee and 5 bps slippage assumptions, the default hurdle is 65 bps. This is deliberately conservative: the goal is more *candidate opportunities*, not forced turnover.

New entries fail closed when the aligned microstructure bucket is incomplete: at least 50% bucket coverage plus both spread and order-book imbalance are required. Missing market microstructure must never increase confirmation strength. Exit paths remain available when microstructure is missing so an existing position cannot be trapped by a collector outage.

Momentum edge is measured only from directional price returns; ATR is treated as dispersion and does not by itself satisfy the cost hurdle. Mean-reversion displacement is signed so only movement below VWAP/Bollinger center contributes to a long oversold edge. Protective stops and maximum-hold exits are always allowed, while a normal mean-reversion target exit must first recover the configured round-trip execution cost.

Each decision result records the completed signal-bar close, live execution quote, signal-to-execution drift in basis points and feature age so later analysis can quantify decay between the 15-minute close and the timer-triggered paper fill.

## Forward comparison maturity

Short-term benchmark results must not be compared with the 3h/12h V3 strategies from raw portfolio return alone because the decision horizons and exposure cadence differ materially. Use net expectancy per completed trade, return on deployed capital, Sharpe/Sortino, drawdown, turnover, invested fraction and cost-adjusted performance alongside total return.

Use the following forward-sample labels per short-term model:

- fewer than 50 completed trades: **observational only**; do not rank against V3
- 50–99 completed trades: **preliminary**
- 100–199 completed trades: **scoreboard eligible**, but conclusions remain tentative
- 200 or more completed trades: **mature comparison sample** for stronger conclusions, subject to regime coverage and the same no-look-ahead/cost rules

Any safety/methodology deployment that changes entry or exit semantics should be timestamped by Git commit so pre-change and post-change forward observations are not silently pooled as one policy version.

## Next research stage

After the live collector and historical 15-minute backfill provide enough data, add two learned candidates without changing these benchmarks:

- Short Trader Brain — supervised short-horizon mixture/gate using the same point-in-time features.
- Short Trader Brain + RL — PAPER contextual policy trained only from resolved short-term experiences.

The learned models should be backtested with time-based splits and transaction costs before joining the live short-term scoreboard.
