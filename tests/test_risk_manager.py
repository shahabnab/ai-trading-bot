from decimal import Decimal

from backend.risk.manager import RiskManager, TradeProposal


def _manager() -> RiskManager:
    return RiskManager(
        min_confidence=0.55,
        max_order_fraction=Decimal("0.10"),
        max_symbol_exposure_fraction=Decimal("0.20"),
        max_total_exposure_fraction=Decimal("0.50"),
        max_daily_drawdown_fraction=Decimal("0.05"),
    )


def _proposal(**overrides) -> TradeProposal:
    values = {
        "symbol": "BTCUSDT",
        "side": "BUY",
        "notional_usdt": Decimal("500"),
        "reference_price": Decimal("100000"),
        "confidence": 0.90,
        "portfolio_value_usdt": Decimal("10000"),
        "model_version": "xgboost-v1",
        "total_exposure_usdt": Decimal("1000"),
        "symbol_exposure_usdt": Decimal("500"),
        "daily_start_portfolio_value_usdt": Decimal("10000"),
    }
    values.update(overrides)
    return TradeProposal(**values)


def test_missing_confidence_rejected_for_non_manual_source() -> None:
    decision = _manager().evaluate(_proposal(confidence=None))
    assert decision.approved is False
    assert "require model confidence" in decision.reason.lower()


def test_missing_confidence_allowed_for_manual_source() -> None:
    decision = _manager().evaluate(_proposal(confidence=None, model_version="manual"))
    assert decision.approved is True


def test_risk_manager_rejects_low_confidence_buy() -> None:
    decision = _manager().evaluate(_proposal(confidence=0.40))
    assert decision.approved is False
    assert "confidence" in decision.reason.lower()


def test_risk_manager_limits_new_position_size() -> None:
    decision = _manager().evaluate(_proposal(notional_usdt=Decimal("1500")))
    assert decision.approved is False
    assert "risk limit" in decision.reason.lower()


def test_risk_manager_limits_per_symbol_exposure() -> None:
    decision = _manager().evaluate(
        _proposal(symbol_exposure_usdt=Decimal("1800"), notional_usdt=Decimal("500"))
    )
    assert decision.approved is False
    assert "per-symbol cap" in decision.reason.lower()


def test_risk_manager_limits_total_exposure() -> None:
    decision = _manager().evaluate(
        _proposal(total_exposure_usdt=Decimal("4800"), notional_usdt=Decimal("500"))
    )
    assert decision.approved is False
    assert "total exposure" in decision.reason.lower()


def test_daily_drawdown_blocks_entry_but_not_exit() -> None:
    manager = _manager()
    buy = manager.evaluate(
        _proposal(
            portfolio_value_usdt=Decimal("9400"),
            daily_start_portfolio_value_usdt=Decimal("10000"),
        )
    )
    sell = manager.evaluate(
        _proposal(
            side="SELL",
            portfolio_value_usdt=Decimal("9400"),
            daily_start_portfolio_value_usdt=Decimal("10000"),
        )
    )
    assert buy.approved is False
    assert "circuit breaker" in buy.reason.lower()
    assert sell.approved is True


def test_risk_manager_allows_valid_paper_buy() -> None:
    decision = _manager().evaluate(_proposal())
    assert decision.approved is True
