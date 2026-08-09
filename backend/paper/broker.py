from decimal import Decimal

from backend.paper.storage import PaperStore


class PaperBrokerError(RuntimeError):
    """Raised when a simulated order cannot be executed."""


class PaperBroker:
    """Long-only USDT paper broker using live market prices.

    No method in this class can submit an order to CoinEx.
    """

    def __init__(self, store: PaperStore, *, fee_rate: Decimal, slippage_bps: Decimal) -> None:
        self.store = store
        self.fee_rate = fee_rate
        self.slippage_bps = slippage_bps

    def _execution_price(self, market_price: Decimal, side: str) -> Decimal:
        slip = self.slippage_bps / Decimal("10000")
        if side == "BUY":
            return market_price * (Decimal("1") + slip)
        return market_price * (Decimal("1") - slip)

    def buy(
        self,
        *,
        symbol: str,
        market_price: Decimal,
        notional_usdt: Decimal,
        model_version: str,
        strategy_version: str,
        confidence: float | None,
    ) -> dict[str, str | float | None]:
        symbol = symbol.upper()
        if not symbol.endswith("USDT"):
            raise PaperBrokerError("Paper broker currently supports USDT-quoted spot markets only")
        if market_price <= 0 or notional_usdt <= 0:
            raise PaperBrokerError("Market price and notional must be positive")

        execution_price = self._execution_price(market_price, "BUY")
        quantity = notional_usdt / execution_price
        gross = quantity * execution_price
        fee = gross * self.fee_rate
        debit = gross + fee

        with self.store.connection() as conn:
            account = conn.execute("SELECT * FROM paper_account WHERE id = 1").fetchone()
            if account is None:
                raise PaperBrokerError("Paper account is not initialized")
            cash = Decimal(account["cash_usdt"])
            if debit > cash:
                raise PaperBrokerError("Insufficient paper cash for this simulated buy")

            current = conn.execute(
                "SELECT * FROM paper_positions WHERE symbol = ?", (symbol,)
            ).fetchone()
            now = self.store._now()
            if current is None:
                new_qty = quantity
                new_avg = execution_price
                conn.execute(
                    "INSERT INTO paper_positions (symbol, quantity, avg_entry_price, updated_at) VALUES (?, ?, ?, ?)",
                    (symbol, str(new_qty), str(new_avg), now),
                )
            else:
                old_qty = Decimal(current["quantity"])
                old_avg = Decimal(current["avg_entry_price"])
                new_qty = old_qty + quantity
                new_avg = ((old_qty * old_avg) + (quantity * execution_price)) / new_qty
                conn.execute(
                    "UPDATE paper_positions SET quantity = ?, avg_entry_price = ?, updated_at = ? WHERE symbol = ?",
                    (str(new_qty), str(new_avg), now, symbol),
                )

            new_cash = cash - debit
            conn.execute(
                "UPDATE paper_account SET cash_usdt = ?, updated_at = ? WHERE id = 1",
                (str(new_cash), now),
            )
            cur = conn.execute(
                """
                INSERT INTO paper_trades (
                    created_at, symbol, side, quantity, market_price, execution_price,
                    gross_value_usdt, fee_usdt, realized_pnl_usdt,
                    model_version, strategy_version, confidence
                ) VALUES (?, ?, 'BUY', ?, ?, ?, ?, ?, '0', ?, ?, ?)
                """,
                (
                    now,
                    symbol,
                    str(quantity),
                    str(market_price),
                    str(execution_price),
                    str(gross),
                    str(fee),
                    model_version,
                    strategy_version,
                    confidence,
                ),
            )
            trade_id = cur.lastrowid

        return {
            "trade_id": str(trade_id),
            "symbol": symbol,
            "side": "BUY",
            "quantity": str(quantity),
            "market_price": str(market_price),
            "execution_price": str(execution_price),
            "gross_value_usdt": str(gross),
            "fee_usdt": str(fee),
            "cash_after_usdt": str(new_cash),
            "realized_pnl_usdt": "0",
            "confidence": confidence,
        }

    def sell(
        self,
        *,
        symbol: str,
        market_price: Decimal,
        quantity: Decimal | None,
        model_version: str,
        strategy_version: str,
        confidence: float | None,
    ) -> dict[str, str | float | None]:
        symbol = symbol.upper()
        execution_price = self._execution_price(market_price, "SELL")

        with self.store.connection() as conn:
            account = conn.execute("SELECT * FROM paper_account WHERE id = 1").fetchone()
            position = conn.execute(
                "SELECT * FROM paper_positions WHERE symbol = ?", (symbol,)
            ).fetchone()
            if account is None:
                raise PaperBrokerError("Paper account is not initialized")
            if position is None:
                raise PaperBrokerError("No paper position exists for this symbol")

            held_qty = Decimal(position["quantity"])
            avg_entry = Decimal(position["avg_entry_price"])
            sell_qty = held_qty if quantity is None else quantity
            if sell_qty <= 0 or sell_qty > held_qty:
                raise PaperBrokerError("Sell quantity must be positive and cannot exceed the paper position")

            gross = sell_qty * execution_price
            fee = gross * self.fee_rate
            realized_pnl = ((execution_price - avg_entry) * sell_qty) - fee
            cash = Decimal(account["cash_usdt"])
            new_cash = cash + gross - fee
            remaining = held_qty - sell_qty
            now = self.store._now()

            conn.execute(
                "UPDATE paper_account SET cash_usdt = ?, updated_at = ? WHERE id = 1",
                (str(new_cash), now),
            )
            if remaining == 0:
                conn.execute("DELETE FROM paper_positions WHERE symbol = ?", (symbol,))
            else:
                conn.execute(
                    "UPDATE paper_positions SET quantity = ?, updated_at = ? WHERE symbol = ?",
                    (str(remaining), now, symbol),
                )

            cur = conn.execute(
                """
                INSERT INTO paper_trades (
                    created_at, symbol, side, quantity, market_price, execution_price,
                    gross_value_usdt, fee_usdt, realized_pnl_usdt,
                    model_version, strategy_version, confidence
                ) VALUES (?, ?, 'SELL', ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now,
                    symbol,
                    str(sell_qty),
                    str(market_price),
                    str(execution_price),
                    str(gross),
                    str(fee),
                    str(realized_pnl),
                    model_version,
                    strategy_version,
                    confidence,
                ),
            )
            trade_id = cur.lastrowid

        return {
            "trade_id": str(trade_id),
            "symbol": symbol,
            "side": "SELL",
            "quantity": str(sell_qty),
            "market_price": str(market_price),
            "execution_price": str(execution_price),
            "gross_value_usdt": str(gross),
            "fee_usdt": str(fee),
            "cash_after_usdt": str(new_cash),
            "realized_pnl_usdt": str(realized_pnl),
            "confidence": confidence,
        }
