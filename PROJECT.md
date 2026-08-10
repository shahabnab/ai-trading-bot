# AI Trading Bot — Shared Project Specification

## 1. Mission

Build a personal, evidence-driven AI-assisted trading platform. The objective is not to maximize backtest profit; it is to determine whether a strategy has a reproducible out-of-sample edge after realistic costs and risk controls.

## 2. Current phase

**Phase v0.1: professional dashboard + read-only market/account integration + paper execution + offline ML evaluation**

Live trading is out of scope. `TRADING_MODE` must remain `paper` until the project explicitly changes phase. Walk-forward model artifacts may be visualized, but they must not be represented as forward/live deployment predictions.

## 3. Core design principles

1. **No look-ahead bias.** Features at time `t` may only use information available at or before `t`.
2. **Chronological evaluation.** Never randomly shuffle time-series train/test splits for trading evaluation.
3. **Costs matter.** Backtests must support commissions, spread/slippage, and position sizing.
4. **Baselines first.** Compare ML against simple baselines and buy-and-hold where applicable.
5. **Risk layer is mandatory.** No strategy/model/LLM can submit an order directly to a broker/exchange.
6. **Paper before live.** Exchange integration begins read-only; simulated execution comes before live orders.
7. **Reproducibility.** Record dataset versions, time ranges, parameters, random seeds, and model versions.
8. **Separation of concerns.** Market data, feature engineering, prediction, strategy, risk, execution, and UI are separate modules.
9. **Secrets never enter Git.** Use environment variables or a secret manager.
10. **AI is advisory by default.** LLM/model outputs are untrusted inputs and must be validated structurally before use.
11. **Fail closed on automated entries.** Missing required risk inputs, confidence, exposure state, or daily-loss state must reject a non-manual entry rather than bypass a control.

## 4. Current architecture

```text
Browser dashboard
        -> Next.js frontend
        -> FastAPI backend
        -> CoinEx public/read-only clients
        -> paper portfolio + SQLite state

Offline research path:
Historical market data
        -> leakage-aware hourly dataset
        -> causal features
        -> XGBoost / LSTM
        -> chronological walk-forward OOS predictions
        -> cost-aware backtest + deterministic baselines
        -> immutable run artifacts
        -> dashboard OOS visualization

Paper trading path:
Model / indicator / LLM / manual input
        -> signal
        -> strategy engine / request adapter
        -> trade proposal
        -> risk manager
        -> approved simulated order
        -> paper broker
        -> SQLite paper state
```

No shortcuts around the risk manager. CoinEx order endpoints are intentionally not implemented in the current phase.

## 5. Current deliverables

- [x] Repository scaffold and shared project rules.
- [x] Conda environment.
- [x] Minimal FastAPI service and health endpoint.
- [x] Professional Next.js prediction dashboard foundation.
- [x] Environment-based trading mode with paper default.
- [x] Mandatory paper risk-manager interface.
- [x] Read-only CoinEx authentication/client.
- [x] CoinEx spot-balance API endpoint.
- [x] Public CoinEx ticker/candlestick market data.
- [x] Persistent paper-trading and ML-run storage.
- [x] Deterministic EMA and buy-and-hold baselines.
- [x] Cost-aware backtest engine with fees, spread and slippage assumptions.
- [x] Performance metrics and chronological walk-forward evaluation.
- [x] XGBoost ML baseline.
- [x] LSTM sequence baseline.
- [x] Dashboard visualization of completed OOS prediction artifacts.
- [ ] Forward/live model inference recorder.
- [ ] Automated model retraining/promotion loop.
- [ ] Automated strategy loop; do not enable until risk-state controls are exercised in integration tests.
- [ ] Live exchange order execution; explicitly out of scope for v0.1.

## 6. Paper risk policy

All execution remains simulated. New automated/non-manual entries must pass all configured checks:

- Model confidence is required for non-manual proposals and must meet `PAPER_MIN_CONFIDENCE`.
- Manual proposals may omit confidence, but if confidence is supplied it is still validated.
- A single BUY is capped by `PAPER_MAX_ORDER_FRACTION`.
- Projected exposure in one symbol is capped by `PAPER_MAX_SYMBOL_EXPOSURE_FRACTION`.
- Projected total market exposure is capped by `PAPER_MAX_TOTAL_EXPOSURE_FRACTION`.
- A UTC-day drawdown circuit breaker blocks new BUY entries after `PAPER_MAX_DAILY_DRAWDOWN_FRACTION` is reached.
- Exposure and daily-loss limits must not block SELL exits that reduce risk.
- The daily baseline is the first paper-equity value observed and persisted for that UTC day. A future always-on portfolio monitor may replace this with a stricter midnight snapshot without weakening the fail-closed rule.

Safe defaults must satisfy:

```text
max order fraction <= max symbol fraction <= max total exposure fraction
```

## 7. Data-quality policy

- Preserve raw 1-minute history; never rewrite raw data to hide gaps.
- The default hourly training bar requires 60 unique, minute-aligned 1-minute candles.
- Incomplete hours are omitted from the processed training dataset.
- Return and target horizons use exact UTC timestamp alignment, so a gap cannot be mislabeled as a shorter horizon.
- Rebuild processed datasets whenever aggregation or labeling rules change.

## 8. ML evaluation policy

Model selection must be based on chronological out-of-sample evidence after realistic costs, not forecast RMSE alone.

Required reporting/ablations as the research phase expands:

- Cost-adjusted strategy P/L, Sharpe, drawdown, turnover and trade count versus buy-and-hold and deterministic baselines.
- Directional and regression metrics reported separately from trading metrics.
- Magnitude-filtered trading, including a top-decile predicted-magnitude ablation, to test whether sparse high-conviction forecasts survive costs better than frequent trading.
- 1h, 4h and 24h target-horizon comparisons using the same no-look-ahead/walk-forward discipline.
- Per-fold XGBoost feature-importance stability, not only aggregate importance, to detect unstable correlated predictors and possible noise memorization.
- Test windows are never used for tuning, threshold selection, feature selection or model promotion.
- Completed walk-forward artifacts are OOS history, not live predictions.

## 9. Security rules

- Never commit `.env`, Access IDs, Secret Keys, withdrawal credentials, private keys, or account identifiers.
- If a Secret Key is exposed in chat, a screenshot, logs, source code, or Git history, revoke it immediately and create a new key.
- Use the minimum CoinEx permissions required for the current phase.
- Do not enable withdrawal permission for this bot.
- Private CoinEx request signing must include the exact canonical query string used by a read-only request.
- Do not implement live order placement until an explicit later project phase.

## 10. Collaboration protocol: ChatGPT + Claude

- This file is the shared source of truth.
- Claude Code and ChatGPT should implement tasks without silently changing methodology.
- Material architecture/methodology changes must update this file in the same change.
- After implementation, commits/PRs can be reviewed independently by the other assistant.
- Never use account credentials in prompts, test fixtures, screenshots, commits, issues, or PRs.
