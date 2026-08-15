from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from backend.config import settings
from backend.market import CoinExMarketClient, MarketDataError
from backend.paper.model_catalog import PAPER_MODELS, PaperModelSpec, get_paper_model
from backend.paper.model_engine import ModelPaperError, ModelPaperStore
from backend.risk.manager import RiskManager, TradeProposal


router = APIRouter(prefix="/api/paper/models", tags=["paper-models"])
market_client = CoinExMarketClient()
store = ModelPaperStore(settings.paper_db_path, settings.paper_initial_balance_usdt)
risk_manager = RiskManager(
    min_confidence=settings.paper_min_confidence,
    max_order_fraction=settings.paper_max_order_fraction,
    max_total_exposure_fraction=settings.paper_max_total_exposure_fraction,
    max_symbol_exposure_fraction=settings.paper_max_symbol_exposure_fraction,
    max_daily_drawdown_fraction=settings.paper_max_daily_drawdown_fraction,
)
_signal_lock = asyncio.Lock()

for _model in PAPER_MODELS:
    store.ensure_account(_model.model_id, _model.display_name)


class ModelPaperSignalRequest(BaseModel):
    symbol: str = Field(default="BTCUSDT", min_length=3, max_length=30)
    signal: Literal["BUY", "SELL", "HOLD"]
    confidence: float | None = Field(default=None, ge=0, le=1)
    notional_usdt: Decimal | None = Field(default=None, gt=0)
    quantity: Decimal | None = Field(default=None, gt=0)
    strategy_version: str = Field(default="v3-forward-paper-1", min_length=1, max_length=100)
    reason: str | None = Field(default=None, max_length=500)


def _spec(model_id: str) -> PaperModelSpec:
    try:
        return get_paper_model(model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


async def _snapshot(model_id: str) -> dict[str, object]:
    account = store.get_account(model_id)
    cash = Decimal(account["cash_usdt"])
    initial_cash = Decimal(account["initial_cash_usdt"])
    positions = store.get_positions(model_id)
    enriched: list[dict[str, object]] = []
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
            last = avg_entry
            price_source = "entry_price_fallback"
        market_value = quantity * last
        pnl = (last - avg_entry) * quantity
        positions_value += market_value
        unrealized_pnl += pnl
        enriched.append(
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
    total_return = total_pnl / initial_cash if initial_cash else Decimal("0")
    return {
        "currency": "EUR_EQUIV",
        "ledger_currency": "USDT",
        "initial_cash_usdt": str(initial_cash),
        "cash_usdt": str(cash),
        "positions_value_usdt": str(positions_value),
        "portfolio_value_usdt": str(portfolio_value),
        "unrealized_pnl_usdt": str(unrealized_pnl),
        "total_pnl_usdt": str(total_pnl),
        "total_return": str(total_return),
        "positions": enriched,
        "fx_note": "EUR-equivalent paper notional; EUR/USDT FX variation is not modeled.",
    }


async def _model_payload(spec: PaperModelSpec) -> dict[str, object]:
    portfolio = await _snapshot(spec.model_id)
    performance = store.performance_summary(spec.model_id)
    recent_decisions = store.list_decisions(spec.model_id, 1)
    return {
        **spec.to_dict(),
        "portfolio": portfolio,
        "performance": performance,
        "latest_decision": recent_decisions[0] if recent_decisions else None,
        "live_status": "paper_running" if recent_decisions else "waiting_for_signal",
    }


@router.get("")
async def list_model_accounts() -> dict[str, object]:
    models = [await _model_payload(spec) for spec in PAPER_MODELS]
    return {
        "mode": "paper",
        "starting_capital_eur_equiv_per_model": str(settings.paper_initial_balance_usdt),
        "real_orders_enabled": False,
        "models": models,
    }


@router.get("/{model_id}")
async def model_account(model_id: str) -> dict[str, object]:
    return await _model_payload(_spec(model_id))


@router.get("/{model_id}/trades")
def model_trades(model_id: str, limit: int = Query(default=100, ge=1, le=1000)) -> dict[str, object]:
    _spec(model_id)
    return {"model_id": model_id, "trades": store.list_trades(model_id, limit)}


@router.get("/{model_id}/decisions")
def model_decisions(model_id: str, limit: int = Query(default=100, ge=1, le=1000)) -> dict[str, object]:
    _spec(model_id)
    return {"model_id": model_id, "decisions": store.list_decisions(model_id, limit)}


@router.post("/{model_id}/signal")
async def model_signal(model_id: str, request: ModelPaperSignalRequest) -> dict[str, object]:
    spec = _spec(model_id)
    symbol = request.symbol.upper().strip()
    try:
        quote = await market_client.get_ticker(symbol)
    except MarketDataError as exc:
        store.record_decision(
            model_id,
            symbol=symbol,
            signal=request.signal,
            confidence=request.confidence,
            approved=False,
            reason=f"Market data unavailable: {exc}",
            strategy_version=request.strategy_version,
            market_price=None,
        )
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    if request.signal == "HOLD":
        reason = request.reason or "Model chose HOLD; no simulated order was placed."
        store.record_decision(
            model_id,
            symbol=symbol,
            signal="HOLD",
            confidence=request.confidence,
            approved=True,
            reason=reason,
            strategy_version=request.strategy_version,
            market_price=quote.last,
        )
        return {"status": "hold", "model_id": model_id, "model": spec.to_dict(), "trade": None}

    async with _signal_lock:
        portfolio = await _snapshot(model_id)
        portfolio_value = Decimal(str(portfolio["portfolio_value_usdt"]))
        total_exposure = Decimal(str(portfolio["positions_value_usdt"]))
        symbol_exposure = Decimal("0")
        for item in portfolio["positions"]:
            if isinstance(item, dict) and str(item.get("symbol", "")).upper() == symbol:
                symbol_exposure = Decimal(str(item.get("market_value_usdt", "0")))
                break
        daily_start = store.get_or_create_daily_start_portfolio_value(model_id, portfolio_value)

        if request.signal == "BUY":
            notional = request.notional_usdt or (portfolio_value * settings.paper_max_order_fraction)
        else:
            position = store.get_position(model_id, symbol)
            if position is None:
                reason = "This model has no paper position to sell."
                store.record_decision(
                    model_id,
                    symbol=symbol,
                    signal="SELL",
                    confidence=request.confidence,
                    approved=False,
                    reason=reason,
                    strategy_version=request.strategy_version,
                    market_price=quote.last,
                )
                return {"status": "rejected", "model_id": model_id, "risk": {"approved": False, "reason": reason}, "trade": None}
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
                model_version=model_id,
                total_exposure_usdt=total_exposure,
                symbol_exposure_usdt=symbol_exposure,
                daily_start_portfolio_value_usdt=daily_start,
            )
        )
        reason = request.reason or decision.reason
        if not decision.approved:
            store.record_decision(
                model_id,
                symbol=symbol,
                signal=request.signal,
                confidence=request.confidence,
                approved=False,
                reason=reason,
                strategy_version=request.strategy_version,
                market_price=quote.last,
            )
            return {"status": "rejected", "model_id": model_id, "risk": {"approved": False, "reason": reason}, "trade": None}

        try:
            if request.signal == "BUY":
                trade = store.buy(
                    model_id,
                    symbol=symbol,
                    market_price=quote.last,
                    notional_usdt=notional,
                    fee_rate=settings.paper_fee_rate,
                    slippage_bps=settings.paper_slippage_bps,
                    confidence=request.confidence,
                    strategy_version=request.strategy_version,
                )
            else:
                trade = store.sell(
                    model_id,
                    symbol=symbol,
                    market_price=quote.last,
                    quantity=request.quantity,
                    fee_rate=settings.paper_fee_rate,
                    slippage_bps=settings.paper_slippage_bps,
                    confidence=request.confidence,
                    strategy_version=request.strategy_version,
                )
        except ModelPaperError as exc:
            store.record_decision(
                model_id,
                symbol=symbol,
                signal=request.signal,
                confidence=request.confidence,
                approved=False,
                reason=str(exc),
                strategy_version=request.strategy_version,
                market_price=quote.last,
            )
            return {"status": "rejected", "model_id": model_id, "risk": {"approved": False, "reason": str(exc)}, "trade": None}

        store.record_decision(
            model_id,
            symbol=symbol,
            signal=request.signal,
            confidence=request.confidence,
            approved=True,
            reason=reason,
            strategy_version=request.strategy_version,
            market_price=quote.last,
        )
        return {
            "status": "filled",
            "model_id": model_id,
            "model": spec.to_dict(),
            "market_price": str(quote.last),
            "risk": {"approved": True, "reason": reason},
            "trade": trade,
        }
