from decimal import Decimal
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from backend.coinex import CoinExAPIError, CoinExClient
from backend.config import settings
from backend.market import CoinExMarketClient, MarketDataError
from backend.ml import ModelRegistry, ModelRegistryError
from backend.paper import PaperBroker, PaperBrokerError, PaperStore
from backend.risk.manager import RiskManager, TradeProposal


app = FastAPI(title=settings.app_name, version="0.4.0")

market_client = CoinExMarketClient()
paper_store = PaperStore(settings.paper_db_path, settings.paper_initial_balance_usdt)
paper_broker = PaperBroker(
    paper_store,
    fee_rate=settings.paper_fee_rate,
    slippage_bps=settings.paper_slippage_bps,
)
risk_manager = RiskManager(
    min_confidence=settings.paper_min_confidence,
    max_order_fraction=settings.paper_max_order_fraction,
)
model_registry = ModelRegistry(settings.model_registry_path)


class PaperSignalRequest(BaseModel):
    symbol: str = Field(min_length=3, max_length=30)
    signal: Literal["BUY", "SELL", "HOLD"]
    confidence: float | None = Field(default=None, ge=0, le=1)
    notional_usdt: Decimal | None = Field(default=None, gt=0)
    quantity: Decimal | None = Field(default=None, gt=0)
    model_version: str = Field(default="manual", min_length=1, max_length=100)
    strategy_version: str = Field(default="baseline", min_length=1, max_length=100)


def _coinex_client() -> CoinExClient:
    if not settings.coinex_configured:
        raise HTTPException(
            status_code=503,
            detail="CoinEx is not configured. Add credentials to the local .env file.",
        )

    return CoinExClient(
        access_id=settings.coinex_access_id or "",
        secret_key=settings.coinex_secret_key or "",
    )


async def _portfolio_snapshot() -> dict[str, object]:
    account = paper_store.get_account()
    cash = Decimal(account["cash_usdt"])
    initial_cash = Decimal(account["initial_cash_usdt"])
    positions = paper_store.get_positions()

    enriched_positions: list[dict[str, object]] = []
    positions_value = Decimal("0")
    unrealized_pnl = Decimal("0")

    for position in positions:
        symbol = position["symbol"]
        quantity = Decimal(position["quantity"])
        avg_entry = Decimal(position["avg_entry_price"])
        price_source = "coinex"
        try:
            quote = await market_client.get_ticker(symbol)
            last = quote.last
        except MarketDataError:
            # Keep the portfolio endpoint usable during a temporary market-data outage.
            last = avg_entry
            price_source = "entry_price_fallback"

        market_value = quantity * last
        pnl = (last - avg_entry) * quantity
        positions_value += market_value
        unrealized_pnl += pnl
        enriched_positions.append(
            {
                "symbol": symbol,
                "quantity": str(quantity),
                "avg_entry_price": str(avg_entry),
                "last_price": str(last),
                "market_value_usdt": str(market_value),
                "unrealized_pnl_usdt": str(pnl),
                "price_source": price_source,
            }
        )

    portfolio_value = cash + positions_value
    total_pnl = portfolio_value - initial_cash
    total_return = (total_pnl / initial_cash) if initial_cash else Decimal("0")

    return {
        "currency": "USDT",
        "initial_cash_usdt": str(initial_cash),
        "cash_usdt": str(cash),
        "positions_value_usdt": str(positions_value),
        "portfolio_value_usdt": str(portfolio_value),
        "unrealized_pnl_usdt": str(unrealized_pnl),
        "total_pnl_usdt": str(total_pnl),
        "total_return": str(total_return),
        "positions": enriched_positions,
    }


@app.get("/health")
def health() -> dict[str, str | bool]:
    return {
        "status": "ok",
        "environment": settings.app_env,
        "trading_mode": settings.trading_mode.value,
        "coinex_configured": settings.coinex_configured,
        "paper_engine": True,
        "ml_registry": True,
    }


@app.get("/api/market/{symbol}")
async def market_ticker(symbol: str) -> dict[str, object]:
    try:
        quote = await market_client.get_ticker(symbol)
    except MarketDataError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"exchange": "coinex", "read_only": True, **quote.as_dict()}


@app.get("/api/market/{symbol}/klines")
async def market_klines(
    symbol: str,
    period: str = Query(default="5min"),
    limit: int = Query(default=100, ge=1, le=1000),
) -> dict[str, object]:
    try:
        candles = await market_client.get_klines(symbol, period=period, limit=limit)
    except MarketDataError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {
        "exchange": "coinex",
        "read_only": True,
        "symbol": symbol.upper(),
        "period": period,
        "candles": candles,
    }


@app.get("/api/ml/registry")
def ml_registry_summary() -> dict[str, object]:
    """Return portable model-registry metadata without loading model binaries."""
    return model_registry.summary()


@app.get("/api/ml/models/{model_id}")
def ml_model_manifest(model_id: str) -> dict[str, object]:
    try:
        return model_registry.get_manifest(model_id)
    except ModelRegistryError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/ml/models/{model_id}/verify")
def ml_model_verify(model_id: str) -> dict[str, object]:
    try:
        return model_registry.verify_model(model_id)
    except ModelRegistryError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/paper/portfolio")
async def paper_portfolio() -> dict[str, object]:
    return await _portfolio_snapshot()


@app.get("/api/paper/positions")
async def paper_positions() -> dict[str, object]:
    snapshot = await _portfolio_snapshot()
    return {"positions": snapshot["positions"]}


@app.get("/api/paper/trades")
def paper_trades(limit: int = Query(default=100, ge=1, le=1000)) -> dict[str, object]:
    return {"trades": paper_store.list_trades(limit)}


@app.get("/api/paper/decisions")
def paper_decisions(limit: int = Query(default=100, ge=1, le=1000)) -> dict[str, object]:
    return {"decisions": paper_store.list_decisions(limit)}


@app.get("/api/paper/performance")
async def paper_performance() -> dict[str, object]:
    portfolio = await _portfolio_snapshot()
    summary = paper_store.performance_summary()
    return {**summary, **portfolio}


@app.post("/api/paper/signal")
async def paper_signal(request: PaperSignalRequest) -> dict[str, object]:
    """Accept a model/strategy signal and simulate it locally after risk checks.

    This endpoint never sends an order to CoinEx. CoinEx is used only to read
    the current public market price used by the paper fill simulation.
    """
    symbol = request.symbol.upper().strip()

    try:
        quote = await market_client.get_ticker(symbol)
    except MarketDataError as exc:
        paper_store.record_decision(
            symbol=symbol,
            signal=request.signal,
            confidence=request.confidence,
            approved=False,
            reason=f"Market data unavailable: {exc}",
            model_version=request.model_version,
            strategy_version=request.strategy_version,
            market_price=None,
        )
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    if request.signal == "HOLD":
        reason = "HOLD signal recorded; no paper order was simulated."
        paper_store.record_decision(
            symbol=symbol,
            signal=request.signal,
            confidence=request.confidence,
            approved=True,
            reason=reason,
            model_version=request.model_version,
            strategy_version=request.strategy_version,
            market_price=quote.last,
        )
        return {
            "status": "hold",
            "symbol": symbol,
            "market_price": str(quote.last),
            "risk": {"approved": True, "reason": reason},
            "trade": None,
        }

    portfolio = await _portfolio_snapshot()
    portfolio_value = Decimal(str(portfolio["portfolio_value_usdt"]))

    if request.signal == "BUY":
        if request.notional_usdt is None:
            raise HTTPException(status_code=422, detail="BUY signals require notional_usdt")
        notional = request.notional_usdt
    else:
        position = paper_store.get_position(symbol)
        if position is None:
            reason = "No paper position exists for this SELL signal."
            paper_store.record_decision(
                symbol=symbol,
                signal=request.signal,
                confidence=request.confidence,
                approved=False,
                reason=reason,
                model_version=request.model_version,
                strategy_version=request.strategy_version,
                market_price=quote.last,
            )
            return {
                "status": "rejected",
                "symbol": symbol,
                "market_price": str(quote.last),
                "risk": {"approved": False, "reason": reason},
                "trade": None,
            }
        held_qty = Decimal(position["quantity"])
        sell_qty = held_qty if request.quantity is None else request.quantity
        notional = sell_qty * quote.last

    decision = risk_manager.evaluate(
        TradeProposal(
            symbol=symbol,
            side=request.signal,
            notional_usdt=notional,
            reference_price=quote.last,
            confidence=request.confidence,
            portfolio_value_usdt=portfolio_value,
        )
    )

    if not decision.approved:
        paper_store.record_decision(
            symbol=symbol,
            signal=request.signal,
            confidence=request.confidence,
            approved=False,
            reason=decision.reason,
            model_version=request.model_version,
            strategy_version=request.strategy_version,
            market_price=quote.last,
        )
        return {
            "status": "rejected",
            "symbol": symbol,
            "market_price": str(quote.last),
            "risk": {"approved": False, "reason": decision.reason},
            "trade": None,
        }

    try:
        if request.signal == "BUY":
            trade = paper_broker.buy(
                symbol=symbol,
                market_price=quote.last,
                notional_usdt=notional,
                model_version=request.model_version,
                strategy_version=request.strategy_version,
                confidence=request.confidence,
            )
        else:
            trade = paper_broker.sell(
                symbol=symbol,
                market_price=quote.last,
                quantity=request.quantity,
                model_version=request.model_version,
                strategy_version=request.strategy_version,
                confidence=request.confidence,
            )
    except PaperBrokerError as exc:
        paper_store.record_decision(
            symbol=symbol,
            signal=request.signal,
            confidence=request.confidence,
            approved=False,
            reason=str(exc),
            model_version=request.model_version,
            strategy_version=request.strategy_version,
            market_price=quote.last,
        )
        return {
            "status": "rejected",
            "symbol": symbol,
            "market_price": str(quote.last),
            "risk": {"approved": False, "reason": str(exc)},
            "trade": None,
        }

    paper_store.record_decision(
        symbol=symbol,
        signal=request.signal,
        confidence=request.confidence,
        approved=True,
        reason=decision.reason,
        model_version=request.model_version,
        strategy_version=request.strategy_version,
        market_price=quote.last,
    )
    return {
        "status": "filled",
        "symbol": symbol,
        "market_price": str(quote.last),
        "risk": {"approved": True, "reason": decision.reason},
        "trade": trade,
    }


@app.get("/api/coinex/debug")
async def coinex_debug() -> dict[str, object]:
    """Return safe diagnostics for the CoinEx Spot balance response."""
    client = _coinex_client()

    try:
        debug = await client.get_spot_balance_debug()
    except CoinExAPIError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return {
        "exchange": "coinex",
        "account": "spot",
        "read_only": True,
        **debug,
    }


@app.get("/api/coinex/balances")
async def coinex_balances() -> dict[str, object]:
    """Return non-zero CoinEx spot balances. This endpoint cannot place orders."""
    client = _coinex_client()

    try:
        balances = await client.get_spot_balances()
    except CoinExAPIError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return {
        "exchange": "coinex",
        "account": "spot",
        "read_only": True,
        "balances": [balance.as_dict() for balance in balances],
    }
