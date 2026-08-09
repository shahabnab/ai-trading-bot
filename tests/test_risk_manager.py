from decimal import Decimal

from backend.risk.manager import RiskManager, TradeProposal


def test_unconfigured_risk_manager_fails_closed() -> None:
    proposal = TradeProposal(
        symbol="TEST",
        side="buy",
        quantity=Decimal("1"),
        reference_price=Decimal("100"),
    )

    decision = RiskManager().evaluate(proposal)

    assert decision.approved is False
