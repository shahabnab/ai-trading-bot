# Trading observability and six-hour audit

This project treats observability as part of the trading experiment. A PAPER result is not useful unless we can reconstruct what the algorithm knew, what it decided, what was executed, and what the real market did afterward.

## Sources of truth

The VPS keeps the detailed local state. The dedicated `paper-live-results` Git branch receives compact immutable snapshots for external review.

- Paper SQLite database: accounts, positions, executions, decisions, risk state and Trader-Brain experiences.
- `state/short_term/decision_diagnostics.jsonl`: point-in-time short-term feature/policy decisions.
- `state/short_term/decision_outcomes.jsonl`: matured 15m/30m/1h/2h real-market outcomes, MFE/MAE and shadow-policy results.
- `state/forward_v3/predictions.jsonl`: frozen V3 probability/EV/policy records.
- `trader_brain_experiences`: Trader-Brain state/action/outcome rows, including resolved real market returns, rewards and shadow rewards.
- CoinEx public 15m/1h OHLCV: real-market truth used to score decisions after their intended horizon.
- `state/ops/system_health.json`: latest server/service/data-health snapshot when the ops supervisor is installed.

Raw tick/order-book history remains on the VPS. It is intentionally not committed every six hours because Git is an audit store, not a high-frequency time-series database.

## Six-hour snapshot

`scripts/export_six_hour_monitoring.py` writes:

- `paper_monitoring/latest.json`
- `paper_monitoring/LATEST.md`
- timestamped JSON/Markdown snapshots under `paper_monitoring/YYYY-MM-DD/`

Each snapshot contains:

1. runtime commit and API/ops health;
2. exact recent real CoinEx candles (24h of 15m OHLCV and 72h of 1h OHLCV);
3. every paper algorithm's equity, P/L, return, fees, closed trades, executions, open positions and latest reason;
4. short-term decisions plus matured real outcomes;
5. frozen V3 policy decisions scored after their own horizon;
6. Trader-Brain experiences/resolved returns/rewards;
7. machine-readable warnings for stale/missing data and stale decision activity.

The exporter never reads `.env` and never exports credentials.

## Scheduling

Run once as root on the VPS:

```bash
bash scripts/install_six_hour_monitoring_timer.sh
```

The timer runs at 02:05, 08:05, 14:05 and 20:05 UTC and pushes to `paper-live-results`. The external six-hour ChatGPT audit is scheduled shortly afterward so it can review the newly pushed snapshot.

The existing daily snapshot may remain enabled. The daily snapshot is a broader archival export; the six-hour monitor is the decision-vs-reality operational/research view.

## Change policy

A six-hour report must distinguish two classes of problems.

**Operational faults** (stopped collectors, stale market data, broken API, missing outcome files, failed PAPER execution, corrupt logging) should be investigated and fixed promptly.

**Strategy underperformance** must not trigger automatic threshold/model changes from one short window. Strategy changes require repeated forward evidence and should be made as a separate tested experiment/PR, preserving the control ledgers.

Real-order execution remains disabled.
