import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Iterator


class PaperStore:
    """SQLite persistence for paper account state, trades, and model decisions."""

    def __init__(self, db_path: str, initial_cash: Decimal) -> None:
        self.db_path = Path(db_path)
        self.initial_cash = initial_cash
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _initialize(self) -> None:
        with self.connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS paper_account (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    cash_usdt TEXT NOT NULL,
                    initial_cash_usdt TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS paper_positions (
                    symbol TEXT PRIMARY KEY,
                    quantity TEXT NOT NULL,
                    avg_entry_price TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS paper_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity TEXT NOT NULL,
                    market_price TEXT NOT NULL,
                    execution_price TEXT NOT NULL,
                    gross_value_usdt TEXT NOT NULL,
                    fee_usdt TEXT NOT NULL,
                    realized_pnl_usdt TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    strategy_version TEXT NOT NULL,
                    confidence REAL
                );

                CREATE TABLE IF NOT EXISTS paper_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    signal TEXT NOT NULL,
                    confidence REAL,
                    approved INTEGER NOT NULL,
                    reason TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    strategy_version TEXT NOT NULL,
                    market_price TEXT
                );

                CREATE TABLE IF NOT EXISTS paper_daily_risk_state (
                    date_utc TEXT PRIMARY KEY,
                    start_portfolio_value_usdt TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            row = conn.execute("SELECT id FROM paper_account WHERE id = 1").fetchone()
            if row is None:
                now = self._now()
                conn.execute(
                    "INSERT INTO paper_account (id, cash_usdt, initial_cash_usdt, updated_at) VALUES (1, ?, ?, ?)",
                    (str(self.initial_cash), str(self.initial_cash), now),
                )

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def get_or_create_daily_start_portfolio_value(
        self,
        current_portfolio_value: Decimal,
        *,
        now: datetime | None = None,
    ) -> Decimal:
        """Return the first observed paper-equity value for the current UTC day."""
        if current_portfolio_value <= 0:
            raise ValueError("current portfolio value must be positive")

        observed_at = now or datetime.now(timezone.utc)
        if observed_at.tzinfo is None:
            observed_at = observed_at.replace(tzinfo=timezone.utc)
        observed_at = observed_at.astimezone(timezone.utc)
        date_key = observed_at.date().isoformat()

        with self.connection() as conn:
            row = conn.execute(
                "SELECT start_portfolio_value_usdt FROM paper_daily_risk_state WHERE date_utc = ?",
                (date_key,),
            ).fetchone()
            if row is not None:
                return Decimal(row["start_portfolio_value_usdt"])

            conn.execute(
                """
                INSERT INTO paper_daily_risk_state (
                    date_utc, start_portfolio_value_usdt, created_at
                ) VALUES (?, ?, ?)
                """,
                (date_key, str(current_portfolio_value), observed_at.isoformat()),
            )
            return current_portfolio_value

    def get_account(self) -> dict[str, str]:
        with self.connection() as conn:
            row = conn.execute("SELECT * FROM paper_account WHERE id = 1").fetchone()
            if row is None:
                raise RuntimeError("Paper account is not initialized")
            return dict(row)

    def get_positions(self) -> list[dict[str, str]]:
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT symbol, quantity, avg_entry_price, updated_at FROM paper_positions ORDER BY symbol"
            ).fetchall()
            return [dict(row) for row in rows]

    def get_position(self, symbol: str) -> dict[str, str] | None:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT symbol, quantity, avg_entry_price, updated_at FROM paper_positions WHERE symbol = ?",
                (symbol.upper(),),
            ).fetchone()
            return dict(row) if row else None

    def list_trades(self, limit: int = 100) -> list[dict[str, object]]:
        limit = max(1, min(limit, 1000))
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM paper_trades ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(row) for row in rows]

    def list_decisions(self, limit: int = 100) -> list[dict[str, object]]:
        limit = max(1, min(limit, 1000))
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM paper_decisions ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(row) for row in rows]

    def record_decision(
        self,
        *,
        symbol: str,
        signal: str,
        confidence: float | None,
        approved: bool,
        reason: str,
        model_version: str,
        strategy_version: str,
        market_price: Decimal | None,
    ) -> None:
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO paper_decisions (
                    created_at, symbol, signal, confidence, approved, reason,
                    model_version, strategy_version, market_price
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self._now(),
                    symbol.upper(),
                    signal.upper(),
                    confidence,
                    1 if approved else 0,
                    reason,
                    model_version,
                    strategy_version,
                    str(market_price) if market_price is not None else None,
                ),
            )

    def performance_summary(self) -> dict[str, object]:
        with self.connection() as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(*) AS trade_count,
                    COALESCE(SUM(CAST(fee_usdt AS REAL)), 0) AS total_fees,
                    COALESCE(SUM(CAST(realized_pnl_usdt AS REAL)), 0) AS realized_pnl,
                    COALESCE(SUM(CASE WHEN side = 'SELL' THEN 1 ELSE 0 END), 0) AS closed_trades,
                    COALESCE(SUM(CASE WHEN side = 'SELL' AND CAST(realized_pnl_usdt AS REAL) > 0 THEN 1 ELSE 0 END), 0) AS winning_trades
                FROM paper_trades
                """
            ).fetchone()
            decisions = conn.execute("SELECT COUNT(*) AS n FROM paper_decisions").fetchone()

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
