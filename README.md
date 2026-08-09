# AI Trading Bot

Personal AI-assisted trading platform for research, backtesting, paper trading, and eventually tightly controlled live execution.

> **Current status:** v0.1 foundation. Paper trading only. No live-order execution is enabled.

## Goals

1. Collect and store reproducible market data.
2. Engineer transparent features and establish non-ML baselines.
3. Backtest with realistic transaction costs and no look-ahead leakage.
4. Add ML models only after baselines are trustworthy.
5. Add paper-trading broker integration.
6. Add AI-assisted research/news analysis as a separate component.
7. Consider limited live trading only after out-of-sample and paper-trading validation.

## Architecture

```text
Market / Broker APIs
        |
        v
Data ingestion ---> Database
        |
        v
Feature engineering
        |
        +----> Baseline / ML models
        |              |
        v              v
Strategy engine ---> Trade proposal
                         |
                         v
                    Risk manager
                         |
                         v
                  Execution adapter
                         |
                         v
                    PAPER broker
```

The model or LLM must never bypass the risk manager.

## Initial stack

- Python 3.11+
- FastAPI backend
- Pydantic settings
- pytest
- PostgreSQL planned for market/trading data
- React/Next.js frontend planned after the core backtester is stable

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
uvicorn backend.main:app --reload
```

Open `http://127.0.0.1:8000/health`.

Run tests:

```bash
pytest
```

## Shared AI workflow

- `PROJECT.md` is the source of truth for scope, architecture, milestones, and methodology.
- `CLAUDE.md` contains implementation rules for Claude Code.
- GitHub history/PRs are the handoff mechanism between coding agents.

Never commit `.env`, API keys, broker secrets, private keys, account IDs, or raw authentication tokens.
