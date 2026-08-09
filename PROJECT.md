# AI Trading Bot — Shared Project Specification

## 1. Mission

Build a personal, evidence-driven AI-assisted trading platform. The objective is not to maximize backtest profit; it is to determine whether a strategy has a reproducible out-of-sample edge after realistic costs and risk controls.

## 2. Current phase

**Phase v0.1: research/backtesting foundation**

Live trading is out of scope. `TRADING_MODE` must remain `paper` until the project explicitly changes phase.

## 3. Core design principles

1. **No look-ahead bias.** Features at time `t` may only use information available at or before `t`.
2. **Chronological evaluation.** Never randomly shuffle time-series train/test splits for trading evaluation.
3. **Costs matter.** Backtests must support commissions, spread/slippage, and position sizing.
4. **Baselines first.** Compare ML against simple baselines and buy-and-hold where applicable.
5. **Risk layer is mandatory.** No strategy/model/LLM can submit an order directly to a broker.
6. **Paper before live.** Any broker integration begins in a sandbox/paper account.
7. **Reproducibility.** Record dataset versions, time ranges, parameters, random seeds, and model versions.
8. **Separation of concerns.** Market data, feature engineering, prediction, strategy, risk, execution, and UI are separate modules.
9. **Secrets never enter Git.** Use environment variables or a secret manager.
10. **AI is advisory by default.** LLM outputs are untrusted inputs and must be validated structurally before use.

## 4. Intended architecture

```text
backend/
  api/             HTTP endpoints
  brokers/         broker adapters (paper first)
  data/            ingestion/storage interfaces
  features/        feature engineering
  models/          ML training/inference
  strategies/      signal-to-trade logic
  backtesting/     historical simulation
  risk/            position sizing and hard risk gates
  execution/       approved-order execution

frontend/          personal dashboard (later)
tests/             unit/integration tests
research/          notebooks/experiments only; production logic must migrate to modules
config/            non-secret configuration
```

## 5. Mandatory execution path

```text
Model / indicator / LLM
        -> signal
        -> strategy engine
        -> trade proposal
        -> risk manager
        -> approved order
        -> execution adapter
        -> paper broker
```

No shortcuts around the risk manager.

## 6. v0.1 deliverables

- [x] Repository scaffold and shared project rules.
- [x] Minimal FastAPI service and health endpoint.
- [x] Environment-based trading mode with paper default.
- [x] Initial risk-manager interface.
- [ ] Choose first market/universe and data source.
- [ ] Implement historical OHLCV ingestion.
- [ ] Add persistent storage.
- [ ] Implement a deterministic baseline strategy.
- [ ] Implement backtest engine with fees and slippage.
- [ ] Add performance report: CAGR/return, volatility, Sharpe, max drawdown, win rate, turnover, exposure.
- [ ] Add walk-forward evaluation.
- [ ] Add first ML baseline (likely tree-based) only after deterministic baseline works.

## 7. Proposed later milestones

### v0.2 — ML research
- leakage-safe feature pipeline
- logistic regression / tree-based baseline
- XGBoost or LightGBM candidate
- probability calibration
- walk-forward retraining
- model registry/experiment metadata

### v0.3 — Paper trading
- broker sandbox adapter
- portfolio/account sync
- idempotent order handling
- audit log
- kill switch
- daily loss and exposure limits

### v0.4 — AI research assistant
- news/research ingestion
- LLM-generated structured analysis
- source citations and timestamps
- no direct order permission

### v0.5 — Personal dashboard
- portfolio/P&L
- signals and rationale
- risk state
- backtest comparison
- model health/drift

### v0.6 — Limited live trading
Only after explicit review of paper-trading results, operational safety, and broker/legal constraints.

## 8. Evaluation rules

A strategy is not considered validated because a single backtest is profitable. At minimum evaluate:

- untouched out-of-sample period
- walk-forward/rolling evaluation
- transaction costs and plausible slippage
- performance across market regimes
- drawdowns and tail losses
- sensitivity to parameters
- comparison with simple baseline(s)
- paper-trading behavior

Optimization must not repeatedly tune against the final holdout period.

## 9. Collaboration protocol: ChatGPT + Claude

- This file is the shared source of truth.
- Claude Code should implement tasks without silently changing methodology.
- Material architecture/methodology changes should update this file in the same change.
- After implementation, commits/PRs can be reviewed independently by ChatGPT.
- Prefer small, reviewable commits and tests for every behavioral change.

## 10. Open decisions

Before market-data implementation, decide:

1. First asset class (e.g. US equities, crypto, ETFs).
2. First broker/exchange and whether it provides a paper/sandbox API.
3. Trading horizon (intraday, daily, multi-day).
4. Initial universe (one symbol, a small basket, or broader universe).
5. Market-data provider and history requirements.

These choices should be recorded here once decided.
