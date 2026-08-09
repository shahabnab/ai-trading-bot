from decimal import Decimal
from enum import Enum

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class TradingMode(str, Enum):
    PAPER = "paper"
    LIVE = "live"


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables.

    Live execution is intentionally rejected during the current project phase.
    CoinEx credentials are optional and are used only for read-only account data.
    Paper trading uses public market data and a local SQLite database.
    ML artifacts, checkpoints, and run logs use portable filesystem-backed stores
    so training/inference can resume on another machine after copying/syncing them.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = Field(default="AI Trading Bot", alias="APP_NAME")
    app_env: str = Field(default="development", alias="APP_ENV")
    trading_mode: TradingMode = Field(default=TradingMode.PAPER, alias="TRADING_MODE")

    coinex_access_id: str | None = Field(default=None, alias="COINEX_ACCESS_ID")
    coinex_secret_key: str | None = Field(default=None, alias="COINEX_SECRET_KEY")

    paper_initial_balance_usdt: Decimal = Field(default=Decimal("10000"), alias="PAPER_INITIAL_BALANCE_USDT")
    paper_fee_rate: Decimal = Field(default=Decimal("0.002"), alias="PAPER_FEE_RATE")
    paper_slippage_bps: Decimal = Field(default=Decimal("5"), alias="PAPER_SLIPPAGE_BPS")
    paper_min_confidence: float = Field(default=0.55, alias="PAPER_MIN_CONFIDENCE")
    paper_max_order_fraction: Decimal = Field(default=Decimal("0.10"), alias="PAPER_MAX_ORDER_FRACTION")
    paper_db_path: str = Field(default="data/paper_trading.sqlite3", alias="PAPER_DB_PATH")

    model_registry_path: str = Field(default="state/model_registry", alias="MODEL_REGISTRY_PATH")
    ml_runs_path: str = Field(default="logs/ml", alias="ML_RUNS_PATH")
    ml_checkpoints_path: str = Field(default="models/checkpoints", alias="ML_CHECKPOINTS_PATH")

    @property
    def coinex_configured(self) -> bool:
        return bool(self.coinex_access_id and self.coinex_secret_key)

    def assert_safe_mode(self) -> None:
        if self.trading_mode is not TradingMode.PAPER:
            raise RuntimeError(
                "Live trading is disabled in the current project phase. Set TRADING_MODE=paper."
            )
        if self.paper_initial_balance_usdt <= 0:
            raise RuntimeError("PAPER_INITIAL_BALANCE_USDT must be positive.")
        if not Decimal("0") <= self.paper_fee_rate < Decimal("1"):
            raise RuntimeError("PAPER_FEE_RATE must be between 0 and 1.")
        if self.paper_slippage_bps < 0:
            raise RuntimeError("PAPER_SLIPPAGE_BPS cannot be negative.")
        if not 0 <= self.paper_min_confidence <= 1:
            raise RuntimeError("PAPER_MIN_CONFIDENCE must be between 0 and 1.")
        if not Decimal("0") < self.paper_max_order_fraction <= Decimal("1"):
            raise RuntimeError("PAPER_MAX_ORDER_FRACTION must be in (0, 1].")


settings = Settings()
settings.assert_safe_mode()
