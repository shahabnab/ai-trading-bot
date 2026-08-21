# Six-hour AI trading audit snapshot

Generated: `2026-08-21T15:02:13.200803+00:00`  
Runtime commit: `320bb63ef0d500a63f29ac1d3e5ea9db47d2b180`  
Real-order execution: **DISABLED**

## Real market

- BTC close: `$77,341.00`
- 6h return: `-0.69%`
- 24h return: `+7.07%`
- Candle freshness: `0.0 min`

## Algorithms

| Algorithm | Policy | Equity | Net P/L | Return | Closed | Exec | Win rate | Fees | Position | Latest |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| Intraday Momentum 15m | official | €1,003.67 | €3.67 | +0.37% | 13 | 26 | 53.8% | €5.23 | FLAT | HOLD |
| Intraday Mean Reversion 15m | official | €1,000.95 | €0.95 | +0.10% | 3 | 7 | 100.0% | €1.40 | LONG | HOLD |
| V3 12h Economic | official | €1,000.00 | €0.00 | +0.00% | 0 | 0 | — | €0.00 | FLAT | HOLD |
| V3 3h Signal Control | official | €1,000.00 | €0.00 | +0.00% | 0 | 0 | — | €0.00 | FLAT | HOLD |
| Trader Brain V1 | official | €1,000.00 | €0.00 | +0.00% | 0 | 0 | — | €0.00 | FLAT | HOLD |
| Trader Brain + RL | official | €1,000.00 | €0.00 | +0.00% | 0 | 0 | — | €0.00 | FLAT | HOLD |
| Momentum Explore 15m | exploration | €1,000.00 | €0.00 | +0.00% | 0 | 0 | — | €0.00 | FLAT | HOLD |
| Mean Reversion Explore 15m | exploration | €1,000.00 | €0.00 | +0.00% | 0 | 0 | — | €0.00 | FLAT | HOLD |

## Warnings

- No monitoring warning generated in this snapshot.

## Evidence included

- Exact recent CoinEx 15m/1h OHLCV used as real-market truth.
- Short-term decision diagnostics and matured 2h outcomes.
- Frozen V3 policy decisions scored against their real future horizon.
- Trader-Brain experiences, resolved market returns, rewards and shadow rewards.
- Current paper account P/L, fees, positions and latest audit reasons.
