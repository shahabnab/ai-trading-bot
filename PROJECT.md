# AI Trading Bot — Shared Project Specification

## 1. Mission

Build a personal, evidence-driven AI-assisted trading platform. The objective is not to maximize backtest profit; it is to determine whether a strategy has a reproducible out-of-sample edge after realistic costs and risk controls.

## 2. Current phase

**Phase v0.1: dashboard + read-only exchange integration**

Live trading is out of scope. `TRADING_MODE` must remain `paper` until the project explicitly changes phase.

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
10. **AI is advisory by default.** LLM outputs are untrusted inputs and must be validated structurally before use.

## 4. Current architecture

```text
Browser dashboard
        -> Next.js frontend
        -> FastAPI backend
        -> CoinEx read-only client
        -> Spot balances

Future trading path:
Model / indicator / LLM
        -> signal
        -> strategy engine
        -> trade proposal
        -> risk manager
        -> approved simulated order
        -> paper execution
```

No shortcuts around the risk manager. CoinEx order endpoints are intentionally not implemented in the current phase.

## 5. Current deliverables

- [x] Repository scaffold and shared project rules.
- [x] Conda environment.
- [x] Minimal FastAPI service and health endpoint.
- [x] Next.js dashboard foundation.
- [x] Environment-based trading mode with paper default.
- [x] Initial risk-manager interface.
- [x] Read-only CoinEx authentication/client.
- [x] CoinEx spot-balance API endpoint.
- [x] Dashboard CoinEx balance display.
- [ ] Implement public CoinEx market data.
- [ ] Add persistent storage.
- [ ] Implement deterministic baseline strategy.
- [ ] Implement backtest engine with fees and slippage.
- [ ] Add performance metrics and walk-forward evaluation.
- [ ] Add first ML baseline only after deterministic baseline works.

## 6. Security rules

- Never commit `.env`, Access IDs, Secret Keys, withdrawal credentials, private keys, or account identifiers.
- If a Secret Key is exposed in chat, a screenshot, logs, source code, or Git history, revoke it immediately and create a new key.
- Use the minimum CoinEx permissions required for the current phase.
- Do not enable withdrawal permission for this bot.
- Do not implement live order placement until an explicit later project phase.

## 7. Collaboration protocol: ChatGPT + Claude

- This file is the shared source of truth.
- Claude Code should implement tasks without silently changing methodology.
- Material architecture/methodology changes should update this file in the same change.
- After implementation, commits/PRs can be reviewed independently by ChatGPT.
- Never use account credentials in prompts, test fixtures, screenshots, commits, issues, or PRs.
