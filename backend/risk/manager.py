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
        if proposal.reference_price <= 0:
            return RiskDecision(False, "Reference price must be positive.")
        if proposal.notional_usdt <= 0:
            return RiskDecision(False, "Order notional must be positive.")
        if proposal.portfolio_value_usdt <= 0:
            return RiskDecision(False, "Paper portfolio value must be positive.")

        is_manual = proposal.model_version.strip().lower() == "manual"
        if proposal.confidence is None:
            if not is_manual:
                return RiskDecision(False, "Non-manual paper proposals require model confidence.")
        elif not 0.0 <= proposal.confidence <= 1.0:
            return RiskDecision(False, "Model confidence must be between 0 and 1.")
        elif proposal.confidence < self.min_confidence:
            return RiskDecision(
                False,
                f"Model confidence {proposal.confidence:.3f} is below the configured minimum {self.min_confidence:.3f}.",
            )

        # Entry-only limits. Exits must remain available to reduce risk even after
        # exposure or daily-loss limits have been breached.
        if side == "BUY":
            if proposal.total_exposure_usdt < 0 or proposal.symbol_exposure_usdt < 0:
                return RiskDecision(False, "Exposure inputs must be non-negative.")
            if proposal.daily_start_portfolio_value_usdt is None:
                return RiskDecision(False, "Daily risk baseline is required for new entries.")
            if proposal.daily_start_portfolio_value_usdt <= 0:
                return RiskDecision(False, "Daily risk baseline must be positive.")

            daily_drawdown = max(
                Decimal("0"),
                (proposal.daily_start_portfolio_value_usdt - proposal.portfolio_value_usdt)
                / proposal.daily_start_portfolio_value_usdt,
            )
            if daily_drawdown >= self.max_daily_drawdown_fraction:
                return RiskDecision(
                    False,
                    f"Daily drawdown {daily_drawdown:.2%} reached the configured entry circuit breaker of {self.max_daily_drawdown_fraction:.2%}.",
                )

            max_notional = proposal.portfolio_value_usdt * self.max_order_fraction
            if proposal.notional_usdt > max_notional:
                return RiskDecision(
                    False,
                    f"Order notional exceeds the paper risk limit of {self.max_order_fraction:.1%} of portfolio value.",
                )

            projected_symbol = proposal.symbol_exposure_usdt + proposal.notional_usdt
            max_symbol = proposal.portfolio_value_usdt * self.max_symbol_exposure_fraction
            if projected_symbol > max_symbol:
                return RiskDecision(
                    False,
                    f"Projected {proposal.symbol.upper()} exposure exceeds the configured {self.max_symbol_exposure_fraction:.1%} per-symbol cap.",
                )

            projected_total = proposal.total_exposure_usdt + proposal.notional_usdt
            max_total = proposal.portfolio_value_usdt * self.max_total_exposure_fraction
            if projected_total > max_total:
                return RiskDecision(
                    False,
                    f"Projected total exposure exceeds the configured {self.max_total_exposure_fraction:.1%} portfolio cap.",
                )

        return RiskDecision(True, "Approved by paper risk rules.")
