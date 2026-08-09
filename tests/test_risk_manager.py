from decimal import Decimal

from backend.risk.manager import RiskManager, TradeProposal


def test_risk_manager_rejects_low_confidence_buy() -> None:
    manager = RiskManager(min_confidence=0.55, max_order_fraction=Decimal("0.10"))
    proposal = TradeProposal(
        symbol="BTCUSDT",
        side="BUY",
        notional_usdt=Decimal("500"),
        reference_price=Decimal("100000"),
        confidence=0.40,
        portfolio_value_usdt=Decimal("10000"),
    )

    decision = manager.evaluate(proposal)

    assert decision.approved is False
    assert "confidence" in decision.reason.lower()


def test_risk_manager_limits_new_position_size() -> None:
    manager = RiskManager(min_confidence=0.55, max_order_fraction=Decimal("0.10"))
    proposal = TradeProposal(
        symbol="BTCUSDT",
        side="BUY",
        notional_usdt=Decimal("1500"),
        reference_price=Decimal("100000"),
        confidence=0.90,
        portfolio_value_usdt=Decimal("10000"),
    )

    decision = manager.evaluate(proposal)

    assert decision.approved is False
    assert "risk limit" in decision.reason.lower()


def test_risk_manager_allows_valid_paper_buy() -> None:
    manager = RiskManager(min_confidence=0.55, max_order_fraction=Decimal("0.10"))
    proposal = TradeProposal(
        symbol="BTCUSDT",
        side="BUY",
        notional_usdt=Decimal("500"),
        reference_price=Decimal("100000"),
        confidence=0.90,
        portfolio_value_usdt=Decimal("10000"),
    )

    decision = manager.evaluate(proposal)

    assert decision.approved is True
