from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping

from backend.config import settings
from backend.market import CoinExMarketClient
from backend.paper.model_catalog import SHORT_TERM_MODELS
from backend.paper.model_engine import ModelPaperStore
from backend.risk.manager import RiskManager, TradeProposal

from .features import ShortTermFeatures, build_short_term_features
from .strategies import ShortTermDecision, decide_mean_reversion, decide_momentum

BUCKET_MS = 15 * 60 * 1000
DEFAULT_STATE_ROOT = Path("state/short_term")
MOMENTUM_ID = "short-momentum-15m"
MEAN_REVERSION_ID = "short-mean-reversion-15m"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, default=str), encoding="utf-8")
    tmp.replace(path)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), separators=(",", ":"), sort_keys=True, default=str) + "\n")


def _micro_for_bucket(path: Path, bucket_start: int) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    match: dict[str, Any] | None = None
    try:
        lines = path.read_text(encoding="utf-8").splitlines()[-200:]
    except OSError:
        return None
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and int(row.get("bucket_start", -1)) == bucket_start:
            match = row
    return match


def _last_feature_timestamp(path: Path) -> int | None:
    if not path.is_file():
        return None
    try:
        for line in reversed(path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            row = json.loads(line)
            return int(row.get("timestamp", row.get("feature_timestamp", 0)))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None
    return None


def _position_context(
    store: ModelPaperStore,
    model_id: str,
    price: Decimal,
) -> tuple[Decimal, Decimal, Decimal, bool, float, float]:
    account = store.get_account(model_id)
    cash = Decimal(account["cash_usdt"])
    total_exposure = Decimal("0")
    symbol_exposure = Decimal("0")
    long_open = False
    unrealized_return = 0.0
    hold_minutes = 0.0
    position = store.get_position(model_id, "BTCUSDT")
    if position is not None:
        quantity = Decimal(position["quantity"])
        avg_entry = Decimal(position["avg_entry_price"])
        symbol_exposure = quantity * price
        total_exposure += symbol_exposure
        long_open = quantity > 0
        if avg_entry > 0:
            unrealized_return = float(price / avg_entry - Decimal("1"))
        trades = store.list_trades(model_id, limit=100)
        last_buy = next((row for row in trades if str(row.get("side", "")).upper() == "BUY"), None)
        if last_buy and last_buy.get("created_at"):
            try:
                opened = datetime.fromisoformat(str(last_buy["created_at"]).replace("Z", "+00:00"))
                if opened.tzinfo is None:
                    opened = opened.replace(tzinfo=UTC)
                hold_minutes = max(0.0, (datetime.now(UTC) - opened.astimezone(UTC)).total_seconds() / 60.0)
            except ValueError:
                pass
    return cash + total_exposure, total_exposure, symbol_exposure, long_open, hold_minutes, unrealized_return


def _risk_manager() -> RiskManager:
    return RiskManager(
        min_confidence=settings.paper_min_confidence,
        max_order_fraction=settings.paper_max_order_fraction,
        max_total_exposure_fraction=settings.paper_max_total_exposure_fraction,
        max_symbol_exposure_fraction=settings.paper_max_symbol_exposure_fraction,
        max_daily_drawdown_fraction=settings.paper_max_daily_drawdown_fraction,
    )


def _execute(
    store: ModelPaperStore,
    risk: RiskManager,
    *,
    model_id: str,
    decision: ShortTermDecision,
    price: Decimal,
    dry_run: bool,
) -> tuple[str, dict[str, Any] | None, bool, str]:
    portfolio, total_exposure, symbol_exposure, long_open, _, _ = _position_context(store, model_id, price)
    if decision.action == "HOLD" or (decision.action == "EXIT" and not long_open):
        return "HOLD", None, True, "No simulated order required."

    if decision.action == "ENTER_LONG":
        side = "BUY"
        notional = portfolio * settings.paper_max_order_fraction
        quantity: Decimal | None = None
    else:
        side = "SELL"
        position = store.get_position(model_id, "BTCUSDT")
        if position is None:
            return "HOLD", None, True, "No BTC paper position exists to exit."
        quantity = Decimal(position["quantity"])
        notional = quantity * price

    daily_start = store.get_or_create_daily_start_portfolio_value(model_id, portfolio)
    risk_decision = risk.evaluate(TradeProposal(
        symbol="BTCUSDT",
        side=side,
        notional_usdt=notional,
        reference_price=price,
        confidence=decision.confidence if side == "BUY" else None,
        portfolio_value_usdt=portfolio,
        model_version=model_id,
        total_exposure_usdt=total_exposure,
        symbol_exposure_usdt=symbol_exposure,
        daily_start_portfolio_value_usdt=daily_start,
    ))
    if not risk_decision.approved:
        return "HOLD", None, False, risk_decision.reason
    if dry_run:
        return side, None, True, "Dry-run: approved by RiskManager."

    if side == "BUY":
        trade = store.buy(
            model_id,
            symbol="BTCUSDT",
            market_price=price,
            notional_usdt=notional,
            fee_rate=settings.paper_fee_rate,
            slippage_bps=settings.paper_slippage_bps,
            confidence=decision.confidence,
            strategy_version="short-term-v1",
        )
    else:
        trade = store.sell(
            model_id,
            symbol="BTCUSDT",
            market_price=price,
            quantity=quantity,
            fee_rate=settings.paper_fee_rate,
            slippage_bps=settings.paper_slippage_bps,
            confidence=decision.confidence,
            strategy_version="short-term-v1",
        )
    return side, trade, True, risk_decision.reason


def _decide(
    model_id: str,
    features: ShortTermFeatures,
    *,
    long_open: bool,
    hold_minutes: float,
    unrealized_return: float,
    min_edge_bps: float,
    round_trip_cost_bps: float,
) -> ShortTermDecision:
    kwargs = dict(
        long_open=long_open,
        hold_minutes=hold_minutes,
        unrealized_return=unrealized_return,
        min_edge_bps=min_edge_bps,
        round_trip_cost_bps=round_trip_cost_bps,
    )
    if model_id == MOMENTUM_ID:
        return decide_momentum(features, **kwargs)
    if model_id == MEAN_REVERSION_ID:
        return decide_mean_reversion(features, **kwargs)
    raise KeyError(f"Unknown short-term strategy: {model_id}")


async def run_short_term_once(
    *,
    state_root: Path = DEFAULT_STATE_ROOT,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    settings.assert_safe_mode()
    market = CoinExMarketClient()
    now_ms = int(datetime.now(UTC).timestamp() * 1000)
    current_bucket = now_ms // BUCKET_MS * BUCKET_MS
    candles = await market.get_klines("BTCUSDT", period="15min", limit=300)
    completed = [row for row in candles if int(row.get("created_at", 0)) < current_bucket]
    if len(completed) < 30:
        raise RuntimeError("Not enough completed 15-minute candles for short-term inference")
    feature_bucket = int(completed[-1]["created_at"])
    micro = _micro_for_bucket(state_root / "microstructure.jsonl", feature_bucket)
    features = build_short_term_features(completed, micro)
    quote = await market.get_ticker("BTCUSDT")

    store = ModelPaperStore(settings.paper_db_path, settings.paper_model_initial_balance_eur_equiv)
    for spec in SHORT_TERM_MODELS:
        store.ensure_account(spec.model_id, spec.display_name)
    risk = _risk_manager()

    one_way_cost_bps = float(settings.paper_fee_rate) * 10_000.0 + float(settings.paper_slippage_bps)
    round_trip_cost_bps = 2.0 * one_way_cost_bps
    # Short-term signals must clear the full simulated round-trip cost plus a
    # 15 bps research buffer. This prevents increasing activity by simply
    # trading noise that cannot pay its execution costs.
    min_edge_bps = round_trip_cost_bps + 15.0

    signal_close = Decimal(str(features.close))
    signal_execution_drift_bps = (
        float((quote.last / signal_close - Decimal("1")) * Decimal("10000"))
        if signal_close > 0
        else 0.0
    )
    feature_close_ms = features.timestamp + BUCKET_MS
    feature_age_seconds = max(0.0, (int(datetime.now(UTC).timestamp() * 1000) - feature_close_ms) / 1000.0)

    feature_log = state_root / "features.jsonl"
    if not dry_run and _last_feature_timestamp(feature_log) != features.timestamp:
        _append_jsonl(feature_log, {
            **features.to_dict(),
            "recorded_at": datetime.now(UTC).isoformat(),
            "round_trip_cost_bps": round_trip_cost_bps,
            "microstructure_available": micro is not None,
        })

    results: list[dict[str, Any]] = []
    for spec in SHORT_TERM_MODELS:
        latest_path = state_root / f"latest_{spec.model_id}.json"
        previous = _read_json(latest_path)
        if not dry_run and int(previous.get("feature_timestamp", -1)) == features.timestamp:
            results.append({"model_id": spec.model_id, "status": "already_processed", "feature_timestamp": features.timestamp})
            continue

        try:
            _, _, _, long_open, hold_minutes, unrealized_return = _position_context(store, spec.model_id, quote.last)
            decision = _decide(
                spec.model_id,
                features,
                long_open=long_open,
                hold_minutes=hold_minutes,
                unrealized_return=unrealized_return,
                min_edge_bps=min_edge_bps,
                round_trip_cost_bps=round_trip_cost_bps,
            )
            executed, trade, approved, risk_reason = _execute(
                store,
                risk,
                model_id=spec.model_id,
                decision=decision,
                price=quote.last,
                dry_run=dry_run,
            )
            signal = "BUY" if executed == "BUY" else "SELL" if executed == "SELL" else "HOLD"
            reason = (
                f"15m {spec.display_name}; {decision.reason} "
                f"round_trip_cost={round_trip_cost_bps:.1f}bps; RiskManager={risk_reason}"
            )
            if not dry_run:
                store.record_decision(
                    spec.model_id,
                    symbol="BTCUSDT",
                    signal=signal,
                    confidence=decision.confidence,
                    approved=approved,
                    reason=reason,
                    strategy_version="short-term-v1",
                    market_price=quote.last,
                )

            result = {
                "status": "ok",
                "recorded_at": datetime.now(UTC).isoformat(),
                "model_id": spec.model_id,
                "display_name": spec.display_name,
                "feature_timestamp": features.timestamp,
                "feature_time_utc": datetime.fromtimestamp(features.timestamp / 1000.0, UTC).isoformat(),
                "signal_close": str(signal_close),
                "market_price": str(quote.last),
                "signal_execution_drift_bps": signal_execution_drift_bps,
                "feature_age_seconds": feature_age_seconds,
                "features": features.to_dict(),
                "microstructure": micro,
                "decision": decision.to_dict(),
                "signal": signal,
                "executed_action": executed,
                "approved": approved,
                "trade": trade,
                "round_trip_cost_bps": round_trip_cost_bps,
                "min_edge_bps": min_edge_bps,
                "dry_run": dry_run,
                "reason": reason,
            }
        except Exception as exc:
            # Per-model isolation boundary: one paper ledger/order failure must
            # never prevent the other benchmark from producing its decision.
            result = {
                "status": "error",
                "recorded_at": datetime.now(UTC).isoformat(),
                "model_id": spec.model_id,
                "display_name": spec.display_name,
                "feature_timestamp": features.timestamp,
                "feature_time_utc": datetime.fromtimestamp(features.timestamp / 1000.0, UTC).isoformat(),
                "signal_close": str(signal_close),
                "market_price": str(quote.last),
                "signal_execution_drift_bps": signal_execution_drift_bps,
                "feature_age_seconds": feature_age_seconds,
                "round_trip_cost_bps": round_trip_cost_bps,
                "min_edge_bps": min_edge_bps,
                "dry_run": dry_run,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

        results.append(result)
        if not dry_run:
            _write_json(latest_path, result)
    return results
