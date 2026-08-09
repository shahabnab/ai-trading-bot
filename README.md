# AI Trading Bot

Personal AI-assisted trading platform for research, backtesting, paper trading, and eventually tightly controlled live execution.

> **Current status:** v0.3 paper-trading engine. CoinEx is used for read-only market data. All orders, positions, balances, fees, slippage and P/L are simulated locally. No live-order execution exists.

## Goals

1. Collect and store reproducible market data.
2. Engineer transparent features and establish non-ML baselines.
3. Backtest with realistic transaction costs and no look-ahead leakage.
4. Add ML models only after baselines are trustworthy.
5. Evaluate model signals through a persistent paper-trading engine.
6. Add AI-assisted research/news analysis as a separate component.
7. Consider limited live trading only after out-of-sample and paper-trading validation.

## Current architecture

```text
CoinEx public market data
          |
          v
      Market adapter
          |
          v
 Model / strategy signal
          |
          v
      Risk manager
          |
          v
      Paper broker
          |
          +----> SQLite trade + decision log
          |
          v
 Paper portfolio / P&L
          |
          v
    Next.js dashboard
```

The model or LLM must never bypass the risk manager. The paper broker has no CoinEx order-placement capability.

## Stack

- Conda environment (`ai-trading-bot`)
- Python 3.11
- FastAPI backend
- Pydantic settings
- SQLite paper-trading store
- pytest
- Next.js + React + TypeScript dashboard
- PostgreSQL remains a later option for larger historical datasets

## Create the Conda environment

From the repository root:

```bash
conda env create -f environment.yml
conda activate ai-trading-bot
```

If the environment already exists:

```bash
conda env update -f environment.yml --prune
conda activate ai-trading-bot
```

## Configure paper trading

Copy `.env.example` to `.env`. The default paper settings are:

```env
TRADING_MODE=paper
PAPER_INITIAL_BALANCE_USDT=10000
PAPER_FEE_RATE=0.002
PAPER_SLIPPAGE_BPS=5
PAPER_MIN_CONFIDENCE=0.55
PAPER_MAX_ORDER_FRACTION=0.10
PAPER_DB_PATH=data/paper_trading.sqlite3
```

CoinEx API credentials are optional for paper trading because public ticker and candle endpoints do not need them. If supplied, the credentials are used only by the existing read-only balance diagnostics.

Never commit real credentials.

## Run backend locally

From the repository root with the Conda environment activated:

```bash
uvicorn backend.main:app --reload
```

Useful endpoints:

```text
GET  /health
GET  /api/market/BTCUSDT
GET  /api/market/BTCUSDT/klines?period=5min&limit=100
GET  /api/paper/portfolio
GET  /api/paper/positions
GET  /api/paper/trades
GET  /api/paper/decisions
GET  /api/paper/performance
POST /api/paper/signal
```

### Example paper BUY signal

PowerShell:

```powershell
$body = @{
  symbol = "BTCUSDT"
  signal = "BUY"
  confidence = 0.82
  notional_usdt = 500
  model_version = "baseline-v1"
  strategy_version = "signal-test-v1"
} | ConvertTo-Json

Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8000/api/paper/signal" `
  -ContentType "application/json" `
  -Body $body
```

### Example HOLD signal

```powershell
$body = @{
  symbol = "BTCUSDT"
  signal = "HOLD"
  confidence = 0.63
  model_version = "baseline-v1"
  strategy_version = "signal-test-v1"
} | ConvertTo-Json

Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8000/api/paper/signal" `
  -ContentType "application/json" `
  -Body $body
```

### Example full-position SELL signal

Omit `quantity` to close the full paper position:

```powershell
$body = @{
  symbol = "BTCUSDT"
  signal = "SELL"
  confidence = 0.79
  model_version = "baseline-v1"
  strategy_version = "signal-test-v1"
} | ConvertTo-Json

Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8000/api/paper/signal" `
  -ContentType "application/json" `
  -Body $body
```

Every signal is logged, including HOLD and rejected decisions. Filled paper trades also record market price, simulated execution price, fee, quantity, model version, strategy version and realized P/L.

Run tests:

```bash
pytest
```

## Run dashboard locally

The frontend uses Node.js/npm separately from the Python Conda environment.

In another terminal:

```bash
cd frontend
npm install
```

Then create the frontend local environment file and run the dashboard:

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

The dashboard displays the virtual portfolio, total P/L, simulated positions, live CoinEx watchlist prices and recent model decisions.

## Shared AI workflow

- `PROJECT.md` is the source of truth for scope, architecture, milestones, and methodology.
- `CLAUDE.md` contains implementation rules for Claude Code.
- GitHub history/PRs are the handoff mechanism between coding agents.

Never commit `.env`, `.env.local`, API keys, broker secrets, private keys, account IDs, or raw authentication tokens.
