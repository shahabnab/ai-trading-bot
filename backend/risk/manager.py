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
    model_version: str = "unknown"
    total_exposure_usdt: Decimal = Decimal("0")
    symbol_exposure_usdt: Decimal = Decimal("0")
    daily_start_portfolio_value_usdt: Decimal | None = None


@dataclass(frozen=True)
class RiskDecision:
    approved: bool
    reason: str


class RiskManager:
    """Mandatory pre-execution checks for the simulated paper broker only."""

    def __init__(
        self,
        *,
        min_confidence: float,
        max_order_fraction: Decimal,
        max_total_exposure_fraction: Decimal = Decimal("0.50"),
        max_symbol_exposure_fraction: Decimal = Decimal("0.20"),
        max_daily_drawdown_fraction: Decimal = Decimal("0.05"),
    ) -> None:
        self.min_confidence = min_confidence
        self.max_order_fraction = max_order_fraction
        self.max_total_exposure_fraction = max_total_exposure_fraction
        self.max_symbol_exposure_fraction = max_symbol_exposure_fraction
        self.max_daily_drawdown_fraction = max_daily_drawdown_fraction

    def evaluate(self, proposal: TradeProposal) -> RiskDecision:
        side = proposal.side.upper()
        if side not in {"BUY", "SELL"}:
            return RiskDecision(False, "Only BUY and SELL proposals can reach the paper broker.")
        if proposal.reference_price <= 0 or proposal.notional_usdt <= 0 or proposal.portfolio_value_usdt <= 0:
            return RiskDecision(False, "Price, notional and portfolio value must be positive.")

        is_manual = proposal.model_version.strip().lower() == "manual"
        if proposal.confidence is None:
            # New entries require model confidence. Exits must stay possible so a data/model
            # problem cannot trap a position that should be reduced.
            if side == "BUY" and not is_manual:
                return RiskDecision(False, "Non-manual paper entries require model confidence.")
        elif not 0.0 <= proposal.confidence <= 1.0:
            return RiskDecision(False, "Model confidence must be between 0 and 1.")
        elif side == "BUY" and proposal.confidence < self.min_confidence:
            return RiskDecision(False, f"Model confidence {proposal.confidence:.3f} is below the configured minimum {self.min_confidence:.3f}.")

        # Entry-only limits. SELL exits remain available after exposure/drawdown limits
        # are breached because they reduce risk rather than create it.
        if side == "BUY":
            if proposal.total_exposure_usdt < 0 or proposal.symbol_exposure_usdt < 0:
                return RiskDecision(False, "Exposure inputs must be non-negative.")
            if proposal.daily_start_portfolio_value_usdt is None or proposal.daily_start_portfolio_value_usdt <= 0:
                return RiskDecision(False, "A positive daily risk baseline is required for new entries.")
            daily_drawdown = max(
                Decimal("0"),
                (proposal.daily_start_portfolio_value_usdt - proposal.portfolio_value_usdt) / proposal.daily_start_portfolio_value_usdt,
            )
            if daily_drawdown >= self.max_daily_drawdown_fraction:
                return RiskDecision(False, f"Daily drawdown {daily_drawdown:.2%} reached the entry circuit breaker {self.max_daily_drawdown_fraction:.2%}.")
            if proposal.notional_usdt > proposal.portfolio_value_usdt * self.max_order_fraction:
                return RiskDecision(False, f"Order notional exceeds {self.max_order_fraction:.1%} of portfolio value.")
            projected_symbol = proposal.symbol_exposure_usdt + proposal.notional_usdt
            if projected_symbol > proposal.portfolio_value_usdt * self.max_symbol_exposure_fraction:
                return RiskDecision(False, f"Projected symbol exposure exceeds {self.max_symbol_exposure_fraction:.1%}.")
            projected_total = proposal.total_exposure_usdt + proposal.notional_usdt
            if projected_total > proposal.portfolio_value_usdt * self.max_total_exposure_fraction:
                return RiskDecision(False, f"Projected total exposure exceeds {self.max_total_exposure_fraction:.1%}.")
        return RiskDecision(True, "Approved by paper risk rules.")
