from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class TradeProposal:
    symbol: str
    side: str
    quantity: Decimal
    reference_price: Decimal


@dataclass(frozen=True)
class RiskDecision:
    approved: bool
    reason: str


class RiskManager:
    """Fail-closed placeholder for mandatory pre-execution risk checks.

    v0.1 deliberately approves nothing until concrete portfolio limits and
    account-state checks are implemented.
    """

    def evaluate(self, proposal: TradeProposal) -> RiskDecision:
        del proposal
        return RiskDecision(
            approved=False,
            reason="Risk rules are not configured; execution is blocked by default.",
        )
