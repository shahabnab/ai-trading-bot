# AI Trading Bot

Personal AI-assisted trading platform for research, backtesting, paper trading, and eventually tightly controlled live execution.

> **Current status:** v0.1 foundation with dashboard work in progress. Paper trading only. No live-order execution is enabled.

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
Browser dashboard
       |
       v
Next.js frontend ---> FastAPI backend
                          |
                          v
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

- Conda environment (`ai-trading-bot`)
- Python 3.11
- FastAPI backend
- Pydantic settings
- pytest
- Next.js + React + TypeScript dashboard
- PostgreSQL planned for market/trading data

## Create the Conda environment

From the repository root:

```bash
conda env create -f environment.yml
conda activate ai-trading-bot
```

If the environment already exists and `environment.yml` changes later, update it with:

```bash
conda env update -f environment.yml --prune
conda activate ai-trading-bot
```

## Run backend locally

With the Conda environment activated:

### Windows

```bash
copy .env.example .env
uvicorn backend.main:app --reload
```

### Linux/macOS

```bash
cp .env.example .env
uvicorn backend.main:app --reload
```

Open `http://127.0.0.1:8000/health`.

Run backend tests:

```bash
pytest
```

`requirements.txt` is retained as a lightweight pip-compatible dependency list, but Conda users should use `environment.yml` as the primary environment definition.

## Run dashboard locally

The frontend uses Node.js/npm separately from the Python Conda environment.

In another terminal:

```bash
cd frontend
npm install
```

Then create the frontend local environment file:

### Windows

```bash
copy .env.local.example .env.local
npm run dev
```

### Linux/macOS

```bash
cp .env.local.example .env.local
npm run dev
```

Open `http://localhost:3000`.

The dashboard reads the FastAPI `/health` endpoint and clearly displays whether the backend is reachable and which trading mode is active.

## Shared AI workflow

- `PROJECT.md` is the source of truth for scope, architecture, milestones, and methodology.
- `CLAUDE.md` contains implementation rules for Claude Code.
- GitHub history/PRs are the handoff mechanism between coding agents.

Never commit `.env`, `.env.local`, API keys, broker secrets, private keys, account IDs, or raw authentication tokens.
