from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class TradeProposal:
    symbol: str
    side: str
    notional_usdt: Decimal
    reference_price: Decimal
    confidence: float | None
    portfolio_value_usdt: Decimal


@dataclass(frozen=True)
class RiskDecision:
    approved: bool
    reason: str


class RiskManager:
    """Mandatory pre-execution checks for the simulated paper broker only."""

    def __init__(self, *, min_confidence: float, max_order_fraction: Decimal) -> None:
        self.min_confidence = min_confidence
        self.max_order_fraction = max_order_fraction

    def evaluate(self, proposal: TradeProposal) -> RiskDecision:
        side = proposal.side.upper()
        if side not in {"BUY", "SELL"}:
            return RiskDecision(False, "Only BUY and SELL proposals can reach the paper broker.")
        if proposal.reference_price <= 0:
            return RiskDecision(False, "Reference price must be positive.")
        if proposal.notional_usdt <= 0:
            return RiskDecision(False, "Order notional must be positive.")
        if proposal.portfolio_value_usdt <= 0:
            return RiskDecision(False, "Paper portfolio value must be positive.")
        if proposal.confidence is not None and proposal.confidence < self.min_confidence:
            return RiskDecision(
                False,
                f"Model confidence {proposal.confidence:.3f} is below the configured minimum {self.min_confidence:.3f}.",
            )

        max_notional = proposal.portfolio_value_usdt * self.max_order_fraction
        if proposal.notional_usdt > max_notional:
            return RiskDecision(
                False,
                f"Order notional exceeds the paper risk limit of {self.max_order_fraction:.1%} of portfolio value.",
            )

        return RiskDecision(True, "Approved by paper risk rules.")
