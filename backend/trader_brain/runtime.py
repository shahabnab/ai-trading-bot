from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from backend.config import settings
from backend.market import CoinExMarketClient
from backend.paper.model_catalog import TRADER_BRAIN_MODELS
from backend.paper.model_engine import ModelPaperStore
from backend.risk.manager import RiskManager, TradeProposal

from .bandit import LinUCBBandit
from .decision import DecisionConfig, decide
from .experience import TraderExperienceStore
from .experts import ai_probability_expert, derivatives_expert, macro_expert, news_expert, technical_expert
from .features import HOUR_MS, build_trader_feature_history, normalize_candles
from .gate import ReliabilityWeightedGate, XGBoostStackingGate
from .regime import PersistentGaussianRegimeDetector

BASE_MODEL_ID = "trader-brain-v1"
RL_MODEL_ID = "trader-brain-bandit-v1"
DEFAULT_STATE_ROOT = Path("state/trader_brain")
DEFAULT_V3_STATE_ROOT = Path("state/forward_v3")


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


def _external_context(path: Path, timestamp: int) -> dict[str, Any]:
    payload = _read_json(path)
    try:
        ts = int(payload.get("timestamp", 0))
    except (TypeError, ValueError):
        return {}
    # Never accept future or silently stale external context.
    return payload if 0 < ts <= timestamp and timestamp - ts <= 6 * HOUR_MS else {}


def _numeric_section(payload: Mapping[str, Any], name: str, prefix: str) -> dict[str, float]:
    section = payload.get(name)
    if not isinstance(section, Mapping):
        return {}
    out: dict[str, float] = {}
    for key, value in section.items():
        if not str(key).startswith(prefix):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            out[str(key)] = number
    return out


def _relative_eth_btc(eth_rows: list[dict[str, Any]], btc_rows: list[dict[str, Any]]) -> float | None:
    eth = {int(r["timestamp"]): float(r["close"]) for r in normalize_candles(eth_rows)}
    btc = {int(r["timestamp"]): float(r["close"]) for r in normalize_candles(btc_rows)}
    common = sorted(set(eth) & set(btc))
    if len(common) < 25:
        return None
    end = common[-1]
    candidates = [ts for ts in common if ts <= end - 24 * HOUR_MS]
    if not candidates:
        return None
    start = candidates[-1]
    return float(np.log(eth[end] / eth[start]) - np.log(btc[end] / btc[start]))


def _latest_v3_predictions(path: Path, timestamp: int) -> list[tuple[str, float, float]]:
    if not path.is_file():
        return []
    latest: dict[str, dict[str, Any]] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines:
        try:
            row = json.loads(line)
            ts = int(row.get("feature_timestamp", row.get("source_timestamp", 0)))
            model_id = str(row.get("model_id", ""))
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        if model_id and 0 <= timestamp - ts <= 2 * HOUR_MS:
            previous = latest.get(model_id)
            if previous is None or ts > int(previous.get("feature_timestamp", previous.get("source_timestamp", 0))):
                latest[model_id] = row
    out: list[tuple[str, float, float]] = []
    for model_id, row in latest.items():
        try:
            p_up = float(row.get("probability", row.get("prob_up", row.get("p_up", 0.5))))
            expected = float(row.get("expected_log_return", row.get("predicted_log_return", 0.0)))
        except (TypeError, ValueError):
            continue
        if np.isfinite(p_up) and np.isfinite(expected):
            name = "ai_v3_12h" if "12h" in model_id else "ai_v3_3h"
            out.append((name, p_up, expected))
    return out


def _position_state(store: ModelPaperStore, model_id: str, price: Decimal) -> tuple[Decimal, Decimal, Decimal, bool]:
    account = store.get_account(model_id)
    cash = Decimal(account["cash_usdt"])
    total = Decimal("0"); symbol = Decimal("0")
    for pos in store.get_positions(model_id):
        value = Decimal(pos["quantity"]) * price
        total += value
        if str(pos["symbol"]).upper() == "BTCUSDT":
            symbol = value
    return cash + total, total, symbol, symbol > 0


def _bandit_vector(gate, regime, long_open: bool) -> tuple[float, ...]:
    return tuple(gate.feature_vector) + (float(regime.entropy), 1.0 if long_open else 0.0)


def _supervised_action(decision, long_open: bool) -> str:
    if long_open:
        return "EXIT" if decision.action == "SHORT" else "NO_TRADE"
    return "LONG" if decision.action == "LONG" else "NO_TRADE"


def _execute(
    store: ModelPaperStore,
    risk: RiskManager,
    *,
    model_id: str,
    action: str,
    confidence: float,
    target_fraction: float,
    price: Decimal,
    dry_run: bool,
) -> tuple[str, dict[str, Any] | None, bool, str]:
    portfolio, total_exposure, symbol_exposure, long_open = _position_state(store, model_id, price)
    if action == "NO_TRADE" or (action == "EXIT" and not long_open):
        return "NO_TRADE", None, True, "No simulated order required."
    if action == "LONG":
        notional = min(portfolio * settings.paper_max_order_fraction, portfolio * Decimal(str(max(target_fraction, 0.01))))
        side = "BUY"; quantity = None
    else:
        position = store.get_position(model_id, "BTCUSDT")
        if position is None:
            return "NO_TRADE", None, True, "No BTC paper position exists to exit."
        quantity = Decimal(position["quantity"]); notional = quantity * price; side = "SELL"
    daily_start = store.get_or_create_daily_start_portfolio_value(model_id, portfolio)
    risk_decision = risk.evaluate(TradeProposal(
        symbol="BTCUSDT", side=side, notional_usdt=notional, reference_price=price,
        confidence=confidence, portfolio_value_usdt=portfolio, model_version=model_id,
        total_exposure_usdt=total_exposure, symbol_exposure_usdt=symbol_exposure,
        daily_start_portfolio_value_usdt=daily_start,
    ))
    if not risk_decision.approved:
        return "NO_TRADE", None, False, risk_decision.reason
    if dry_run:
        return action, None, True, "Dry-run: approved by RiskManager."
    if side == "BUY":
        trade = store.buy(model_id, symbol="BTCUSDT", market_price=price, notional_usdt=notional,
                          fee_rate=settings.paper_fee_rate, slippage_bps=settings.paper_slippage_bps,
                          confidence=confidence, strategy_version="trader-brain-v1")
    else:
        trade = store.sell(model_id, symbol="BTCUSDT", market_price=price, quantity=quantity,
                           fee_rate=settings.paper_fee_rate, slippage_bps=settings.paper_slippage_bps,
                           confidence=confidence, strategy_version="trader-brain-v1")
    return action, trade, True, risk_decision.reason


async def run_trader_brain_once(
    *,
    state_root: Path = DEFAULT_STATE_ROOT,
    v3_state_root: Path = DEFAULT_V3_STATE_ROOT,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    market = CoinExMarketClient()
    btc_raw = await market.get_klines("BTCUSDT", period="1hour", limit=1000)
    eth_raw = await market.get_klines("ETHUSDT", period="1hour", limit=1000)
    current_hour = int(datetime.now(UTC).timestamp() * 1000) // HOUR_MS * HOUR_MS
    btc_raw = [row for row in btc_raw if int(row.get("created_at", row.get("timestamp", 0))) < current_hour]
    eth_raw = [row for row in eth_raw if int(row.get("created_at", row.get("timestamp", 0))) < current_hour]
    history = build_trader_feature_history(btc_raw)
    latest = history[-1]
    matrix = np.asarray([row.regime_vector for row in history[-700:]], dtype=np.float64)
    regime = PersistentGaussianRegimeDetector(n_regimes=5).fit_predict(matrix, timestamp=latest.timestamp)

    external = _external_context(state_root / "external_context.json", latest.timestamp)
    macro = _numeric_section(external, "macro", "M")
    derivatives = _numeric_section(external, "derivatives", "D")
    news = _numeric_section(external, "news", "N")
    rel = _relative_eth_btc(eth_raw, btc_raw)
    if rel is not None:
        macro.setdefault("M9", rel)

    experts = [
        technical_expert(latest.technical, realized_vol_24h=latest.realized_vol_24h, timestamp=latest.timestamp),
        macro_expert(macro, realized_vol_24h=latest.realized_vol_24h, timestamp=latest.timestamp),
        derivatives_expert(derivatives, realized_vol_24h=latest.realized_vol_24h, timestamp=latest.timestamp),
        news_expert(news, realized_vol_24h=latest.realized_vol_24h, timestamp=latest.timestamp),
    ]
    for name, p_up, expected in _latest_v3_predictions(v3_state_root / "predictions.jsonl", latest.timestamp):
        experts.append(ai_probability_expert(name, p_up=p_up, expected_return=expected, realized_vol_24h=latest.realized_vol_24h, timestamp=latest.timestamp))

    paper = ModelPaperStore(settings.paper_db_path, settings.paper_model_initial_balance_eur_equiv)
    for spec in TRADER_BRAIN_MODELS:
        paper.ensure_account(spec.model_id, spec.display_name)
    experience = TraderExperienceStore(settings.paper_db_path)
    price_map = {int(row["timestamp"]): float(row["close"]) for row in normalize_candles(btc_raw)}
    for spec in TRADER_BRAIN_MODELS:
        experience.resolve_due(model_id=spec.model_id, current_timestamp=latest.timestamp, current_price=latest.close, price_by_timestamp=price_map)

    reliability = experience.expert_reliability(BASE_MODEL_ID)
    fallback = ReliabilityWeightedGate().combine(regime, experts, reliability=reliability)
    gate = fallback
    X, y = experience.gate_training_data(BASE_MODEL_ID)
    if len(X) >= settings.trader_brain_meta_min_samples and len(np.unique(y)) >= 2:
        try:
            gate = XGBoostStackingGate().fit(X, y).predict(fallback.feature_vector, fallback=fallback)
        except Exception:
            gate = fallback

    available = [expert for expert in experts if expert.available and expert.quality > 0]
    data_ok = regime.quality >= 0.5 and len(available) >= 1
    cost = float(settings.paper_fee_rate) + float(settings.paper_slippage_bps) / 10_000.0
    decision = decide(
        gate, regime, one_way_cost_rate=cost, data_quality_ok=data_ok,
        config=DecisionConfig(
            min_edge_bps=settings.trader_brain_min_edge_bps,
            min_direction_probability=settings.trader_brain_min_direction_probability,
            max_regime_entropy=settings.trader_brain_max_regime_entropy,
            uncertainty_penalty=settings.trader_brain_uncertainty_penalty,
            max_position_fraction=float(settings.paper_max_order_fraction),
        ),
    )
    risk = RiskManager(
        min_confidence=settings.paper_min_confidence, max_order_fraction=settings.paper_max_order_fraction,
        max_total_exposure_fraction=settings.paper_max_total_exposure_fraction,
        max_symbol_exposure_fraction=settings.paper_max_symbol_exposure_fraction,
        max_daily_drawdown_fraction=settings.paper_max_daily_drawdown_fraction,
    )
    quote = await market.get_ticker("BTCUSDT")
    results: list[dict[str, Any]] = []
    for spec in TRADER_BRAIN_MODELS:
        if not dry_run and experience.has_experience(spec.model_id, latest.timestamp):
            results.append({"model_id": spec.model_id, "status": "already_processed", "feature_timestamp": latest.timestamp})
            continue
        _, _, _, long_open = _position_state(paper, spec.model_id, quote.last)
        supervised = _supervised_action(decision, long_open)
        action = supervised; policy_source = "supervised_moe"; bandit_diag: dict[str, Any] | None = None
        vector = _bandit_vector(gate, regime, long_open)
        if spec.model_id == RL_MODEL_ID and settings.trader_brain_bandit_paper_enabled:
            samples = experience.bandit_shadow_samples(RL_MODEL_ID)
            bandit = LinUCBBandit(len(vector), alpha=settings.trader_brain_bandit_alpha).fit_shadow_samples(samples)
            valid = ("NO_TRADE", "EXIT") if long_open else ("NO_TRADE", "LONG")
            choice = bandit.choose(vector, valid_actions=valid, min_samples=settings.trader_brain_bandit_min_samples, fallback_action=supervised)
            action = choice.action; policy_source = choice.policy_source
            bandit_diag = {"trained_samples": choice.trained_samples, "score": choice.score, "action_scores": dict(choice.action_scores)}
        executed, trade, approved, risk_reason = _execute(
            paper, risk, model_id=spec.model_id, action=action, confidence=max(gate.confidence, decision.confidence),
            target_fraction=decision.target_position_fraction, price=quote.last, dry_run=dry_run,
        )
        reason = f"policy={policy_source}; gate={gate.gate_type}; regime_entropy={regime.entropy:.3f}; {decision.reason} RiskManager={risk_reason}"
        if not dry_run:
            paper.record_decision(
                spec.model_id, symbol="BTCUSDT", signal="BUY" if executed == "LONG" else "SELL" if executed == "EXIT" else "HOLD",
                confidence=max(gate.confidence, decision.confidence), approved=approved, reason=reason,
                strategy_version="trader-brain-v1", market_price=quote.last,
            )
            experience.record(
                model_id=spec.model_id, feature_timestamp=latest.timestamp,
                target_timestamp=latest.timestamp + spec.horizon_hours * HOUR_MS, reference_price=latest.close,
                position_before="LONG" if long_open else "FLAT", action=executed,
                gate_vector=gate.feature_vector, bandit_vector=vector, gate=gate, experts=experts, regime=regime,
                estimated_one_way_cost=cost,
            )
        result = {
            "recorded_at": datetime.now(UTC).isoformat(), "model_id": spec.model_id, "display_name": spec.display_name,
            "feature_timestamp": latest.timestamp, "market_price": str(quote.last), "regime": regime.to_dict(),
            "experts": [expert.to_dict() for expert in experts], "expert_reliability": reliability,
            "gate": gate.to_dict(), "decision": decision.to_dict(), "policy_source": policy_source,
            "requested_action": action, "executed_action": executed, "approved": approved,
            "reason": reason, "trade": trade, "bandit": bandit_diag, "learning": experience.report(spec.model_id), "dry_run": dry_run,
        }
        results.append(result)
        if not dry_run:
            _write_json(state_root / f"latest_{spec.model_id}.json", result)
    if not dry_run:
        _write_json(state_root / "latest.json", {"generated_at": datetime.now(UTC).isoformat(), "models": results})
    return results
