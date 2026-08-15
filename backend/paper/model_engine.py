from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Iterator


class ModelPaperError(RuntimeError):
    """Raised when an isolated model paper account cannot execute a simulated order."""


class ModelPaperStore:
    """SQLite-backed isolated paper accounts used to compare models fairly.

    Every model receives its own starting balance and positions. This avoids one
    model consuming another model's cash and makes forward performance directly
    comparable. The legacy single paper account remains untouched.
    """

    def __init__(self, db_path: str, initial_cash: Decimal) -> None:
        self.db_path = Path(db_path)
        self.initial_cash = initial_cash
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _initialize(self) -> None:
        with self.connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS paper_model_accounts (
                    model_id TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    cash_usdt TEXT NOT NULL,
                    initial_cash_usdt TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS paper_model_positions (
                    model_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    quantity TEXT NOT NULL,
                    avg_entry_price TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (model_id, symbol),
                    FOREIGN KEY (model_id) REFERENCES paper_model_accounts(model_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS paper_model_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity TEXT NOT NULL,
                    market_price TEXT NOT NULL,
                    execution_price TEXT NOT NULL,
                    gross_value_usdt TEXT NOT NULL,
                    fee_usdt TEXT NOT NULL,
                    realized_pnl_usdt TEXT NOT NULL,
                    confidence REAL,
                    strategy_version TEXT NOT NULL,
                    FOREIGN KEY (model_id) REFERENCES paper_model_accounts(model_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS paper_model_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    signal TEXT NOT NULL,
                    confidence REAL,
                    approved INTEGER NOT NULL,
                    reason TEXT NOT NULL,
                    strategy_version TEXT NOT NULL,
                    market_price TEXT,
                    FOREIGN KEY (model_id) REFERENCES paper_model_accounts(model_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS paper_model_daily_risk_state (
                    model_id TEXT NOT NULL,
                    date_utc TEXT NOT NULL,
                    start_portfolio_value_usdt TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (model_id, date_utc),
                    FOREIGN KEY (model_id) REFERENCES paper_model_accounts(model_id) ON DELETE CASCADE
                );
                """
            )

    def ensure_account(self, model_id: str, display_name: str) -> None:
        now = self._now()
        with self.connection() as conn:
            row = conn.execute(
                "SELECT model_id FROM paper_model_accounts WHERE model_id = ?", (model_id,)
            ).fetchone()
            if row is None:
                conn.execute(
                    """
                    INSERT INTO paper_model_accounts (
                        model_id, display_name, cash_usdt, initial_cash_usdt, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (model_id, display_name, str(self.initial_cash), str(self.initial_cash), now, now),
                )
            else:
                conn.execute(
                    "UPDATE paper_model_accounts SET display_name = ?, updated_at = ? WHERE model_id = ?",
                    (display_name, now, model_id),
                )

    def get_account(self, model_id: str) -> dict[str, str]:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM paper_model_accounts WHERE model_id = ?", (model_id,)
            ).fetchone()
            if row is None:
                raise ModelPaperError(f"Paper model account is not initialized: {model_id}")
            return dict(row)

    def get_positions(self, model_id: str) -> list[dict[str, str]]:
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT symbol, quantity, avg_entry_price, updated_at
                FROM paper_model_positions WHERE model_id = ? ORDER BY symbol
                """,
                (model_id,),
            ).fetchall()
            return [dict(row) for row in rows]

    def get_position(self, model_id: str, symbol: str) -> dict[str, str] | None:
        with self.connection() as conn:
            row = conn.execute(
                """
                SELECT symbol, quantity, avg_entry_price, updated_at
                FROM paper_model_positions WHERE model_id = ? AND symbol = ?
                """,
                (model_id, symbol.upper()),
            ).fetchone()
            return dict(row) if row else None

    def list_trades(self, model_id: str, limit: int = 100) -> list[dict[str, object]]:
        limit = max(1, min(limit, 1000))
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM paper_model_trades WHERE model_id = ? ORDER BY id DESC LIMIT ?",
                (model_id, limit),
            ).fetchall()
            return [dict(row) for row in rows]

    def list_decisions(self, model_id: str, limit: int = 100) -> list[dict[str, object]]:
        limit = max(1, min(limit, 1000))
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM paper_model_decisions WHERE model_id = ? ORDER BY id DESC LIMIT ?",
                (model_id, limit),
            ).fetchall()
            return [dict(row) for row in rows]

    def record_decision(
        self,
        model_id: str,
        *,
        symbol: str,
        signal: str,
        confidence: float | None,
        approved: bool,
        reason: str,
        strategy_version: str,
        market_price: Decimal | None,
    ) -> None:
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO paper_model_decisions (
                    model_id, created_at, symbol, signal, confidence, approved,
                    reason, strategy_version, market_price
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    model_id,
                    self._now(),
                    symbol.upper(),
                    signal.upper(),
                    confidence,
                    1 if approved else 0,
                    reason,
                    strategy_version,
                    str(market_price) if market_price is not None else None,
                ),
            )

    def get_or_create_daily_start_portfolio_value(
        self, model_id: str, current_portfolio_value: Decimal
    ) -> Decimal:
        if current_portfolio_value <= 0:
            raise ValueError("current portfolio value must be positive")
        observed_at = datetime.now(timezone.utc)
        date_key = observed_at.date().isoformat()
        with self.connection() as conn:
            row = conn.execute(
                """
                SELECT start_portfolio_value_usdt FROM paper_model_daily_risk_state
                WHERE model_id = ? AND date_utc = ?
                """,
                (model_id, date_key),
            ).fetchone()
            if row is not None:
                return Decimal(row["start_portfolio_value_usdt"])
            conn.execute(
                """
                INSERT INTO paper_model_daily_risk_state (
                    model_id, date_utc, start_portfolio_value_usdt, created_at
                ) VALUES (?, ?, ?, ?)
                """,
                (model_id, date_key, str(current_portfolio_value), observed_at.isoformat()),
            )
            return current_portfolio_value

    def performance_summary(self, model_id: str) -> dict[str, object]:
        with self.connection() as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(*) AS trade_count,
                    COALESCE(SUM(CAST(fee_usdt AS REAL)), 0) AS total_fees,
                    COALESCE(SUM(CAST(realized_pnl_usdt AS REAL)), 0) AS realized_pnl,
                    COALESCE(SUM(CASE WHEN side = 'SELL' THEN 1 ELSE 0 END), 0) AS closed_trades,
                    COALESCE(SUM(CASE WHEN side = 'SELL' AND CAST(realized_pnl_usdt AS REAL) > 0 THEN 1 ELSE 0 END), 0) AS winning_trades
                FROM paper_model_trades WHERE model_id = ?
                """,
                (model_id,),
            ).fetchone()
            decisions = conn.execute(
                "SELECT COUNT(*) AS n FROM paper_model_decisions WHERE model_id = ?", (model_id,)
            ).fetchone()
        closed = int(row["closed_trades"] if row else 0)
        winners = int(row["winning_trades"] if row else 0)
        return {
            "trade_count": int(row["trade_count"] if row else 0),
            "decision_count": int(decisions["n"] if decisions else 0),
            "closed_trades": closed,
            "winning_trades": winners,
            "win_rate": (winners / closed) if closed else None,
            "total_fees_usdt": str(Decimal(str(row["total_fees"] if row else 0))),
            "realized_pnl_usdt": str(Decimal(str(row["realized_pnl"] if row else 0))),
        }

    @staticmethod
    def _execution_price(market_price: Decimal, side: str, slippage_bps: Decimal) -> Decimal:
        slip = slippage_bps / Decimal("10000")
        return market_price * (Decimal("1") + slip if side == "BUY" else Decimal("1") - slip)

    def buy(
        self,
        model_id: str,
        *,
        symbol: str,
        market_price: Decimal,
        notional_usdt: Decimal,
        fee_rate: Decimal,
        slippage_bps: Decimal,
        confidence: float | None,
        strategy_version: str,
    ) -> dict[str, object]:
        symbol = symbol.upper()
        execution_price = self._execution_price(market_price, "BUY", slippage_bps)
        quantity = notional_usdt / execution_price
        gross = quantity * execution_price
        fee = gross * fee_rate
        debit = gross + fee
        unit_cost = debit / quantity
        now = self._now()
        with self.connection() as conn:
            account = conn.execute(
                "SELECT * FROM paper_model_accounts WHERE model_id = ?", (model_id,)
            ).fetchone()
            if account is None:
                raise ModelPaperError(f"Unknown paper model account: {model_id}")
            cash = Decimal(account["cash_usdt"])
            if debit > cash:
                raise ModelPaperError("Insufficient paper cash for this model")
            current = conn.execute(
                "SELECT * FROM paper_model_positions WHERE model_id = ? AND symbol = ?",
                (model_id, symbol),
            ).fetchone()
            if current is None:
                conn.execute(
                    """
                    INSERT INTO paper_model_positions (
                        model_id, symbol, quantity, avg_entry_price, updated_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (model_id, symbol, str(quantity), str(unit_cost), now),
                )
            else:
                old_qty = Decimal(current["quantity"])
                old_avg = Decimal(current["avg_entry_price"])
                new_qty = old_qty + quantity
                new_avg = ((old_qty * old_avg) + (quantity * unit_cost)) / new_qty
                conn.execute(
                    """
                    UPDATE paper_model_positions
                    SET quantity = ?, avg_entry_price = ?, updated_at = ?
                    WHERE model_id = ? AND symbol = ?
                    """,
                    (str(new_qty), str(new_avg), now, model_id, symbol),
                )
            new_cash = cash - debit
            conn.execute(
                "UPDATE paper_model_accounts SET cash_usdt = ?, updated_at = ? WHERE model_id = ?",
                (str(new_cash), now, model_id),
            )
            cur = conn.execute(
                """
                INSERT INTO paper_model_trades (
                    model_id, created_at, symbol, side, quantity, market_price,
                    execution_price, gross_value_usdt, fee_usdt, realized_pnl_usdt,
                    confidence, strategy_version
                ) VALUES (?, ?, ?, 'BUY', ?, ?, ?, ?, ?, '0', ?, ?)
                """,
                (
                    model_id, now, symbol, str(quantity), str(market_price), str(execution_price),
                    str(gross), str(fee), confidence, strategy_version,
                ),
            )
        return {
            "trade_id": str(cur.lastrowid), "model_id": model_id, "symbol": symbol, "side": "BUY",
            "quantity": str(quantity), "market_price": str(market_price),
            "execution_price": str(execution_price), "gross_value_usdt": str(gross),
            "fee_usdt": str(fee), "cash_after_usdt": str(new_cash), "realized_pnl_usdt": "0",
            "confidence": confidence,
        }

    def sell(
        self,
        model_id: str,
        *,
        symbol: str,
        market_price: Decimal,
        quantity: Decimal | None,
        fee_rate: Decimal,
        slippage_bps: Decimal,
        confidence: float | None,
        strategy_version: str,
    ) -> dict[str, object]:
        symbol = symbol.upper()
        execution_price = self._execution_price(market_price, "SELL", slippage_bps)
        now = self._now()
        with self.connection() as conn:
            account = conn.execute(
                "SELECT * FROM paper_model_accounts WHERE model_id = ?", (model_id,)
            ).fetchone()
            position = conn.execute(
                "SELECT * FROM paper_model_positions WHERE model_id = ? AND symbol = ?",
                (model_id, symbol),
            ).fetchone()
            if account is None or position is None:
                raise ModelPaperError("No model paper position exists for this SELL signal")
            held_qty = Decimal(position["quantity"])
            avg_entry = Decimal(position["avg_entry_price"])
            sell_qty = held_qty if quantity is None else quantity
            if sell_qty <= 0 or sell_qty > held_qty:
                raise ModelPaperError("Sell quantity must be positive and cannot exceed the model position")
            gross = sell_qty * execution_price
            fee = gross * fee_rate
            realized_pnl = ((execution_price - avg_entry) * sell_qty) - fee
            cash = Decimal(account["cash_usdt"])
            new_cash = cash + gross - fee
            remaining = held_qty - sell_qty
            conn.execute(
                "UPDATE paper_model_accounts SET cash_usdt = ?, updated_at = ? WHERE model_id = ?",
                (str(new_cash), now, model_id),
            )
            if remaining == 0:
                conn.execute(
                    "DELETE FROM paper_model_positions WHERE model_id = ? AND symbol = ?",
                    (model_id, symbol),
                )
            else:
                conn.execute(
                    """
                    UPDATE paper_model_positions SET quantity = ?, updated_at = ?
                    WHERE model_id = ? AND symbol = ?
                    """,
                    (str(remaining), now, model_id, symbol),
                )
            cur = conn.execute(
                """
                INSERT INTO paper_model_trades (
                    model_id, created_at, symbol, side, quantity, market_price,
                    execution_price, gross_value_usdt, fee_usdt, realized_pnl_usdt,
                    confidence, strategy_version
                ) VALUES (?, ?, ?, 'SELL', ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    model_id, now, symbol, str(sell_qty), str(market_price), str(execution_price),
                    str(gross), str(fee), str(realized_pnl), confidence, strategy_version,
                ),
            )
        return {
            "trade_id": str(cur.lastrowid), "model_id": model_id, "symbol": symbol, "side": "SELL",
            "quantity": str(sell_qty), "market_price": str(market_price),
            "execution_price": str(execution_price), "gross_value_usdt": str(gross),
            "fee_usdt": str(fee), "cash_after_usdt": str(new_cash),
            "realized_pnl_usdt": str(realized_pnl), "confidence": confidence,
        }
