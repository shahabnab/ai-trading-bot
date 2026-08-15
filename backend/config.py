from decimal import Decimal
from enum import Enum

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class TradingMode(str, Enum):
    PAPER = "paper"
    LIVE = "live"


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables.

    Live execution remains intentionally disabled. Trader-Brain and the adaptive
    policy are research/PAPER components and share the existing hard risk layer.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = Field(default="AI Trading Bot", alias="APP_NAME")
    app_env: str = Field(default="development", alias="APP_ENV")
    trading_mode: TradingMode = Field(default=TradingMode.PAPER, alias="TRADING_MODE")
    coinex_access_id: str | None = Field(default=None, alias="COINEX_ACCESS_ID")
    coinex_secret_key: str | None = Field(default=None, alias="COINEX_SECRET_KEY")
    alpha_vantage_api_key: str | None = Field(default=None, alias="ALPHAVANTAGE_API_KEY")

    paper_initial_balance_usdt: Decimal = Field(default=Decimal("1000"), alias="PAPER_INITIAL_BALANCE_USDT")
    paper_model_initial_balance_eur_equiv: Decimal = Field(default=Decimal("1000"), alias="PAPER_MODEL_INITIAL_BALANCE_EUR_EQUIV")
    paper_fee_rate: Decimal = Field(default=Decimal("0.002"), alias="PAPER_FEE_RATE")
    paper_slippage_bps: Decimal = Field(default=Decimal("5"), alias="PAPER_SLIPPAGE_BPS")
    paper_min_confidence: float = Field(default=0.55, alias="PAPER_MIN_CONFIDENCE")
    paper_max_order_fraction: Decimal = Field(default=Decimal("0.10"), alias="PAPER_MAX_ORDER_FRACTION")
    paper_max_symbol_exposure_fraction: Decimal = Field(default=Decimal("0.20"), alias="PAPER_MAX_SYMBOL_EXPOSURE_FRACTION")
    paper_max_total_exposure_fraction: Decimal = Field(default=Decimal("0.50"), alias="PAPER_MAX_TOTAL_EXPOSURE_FRACTION")
    paper_max_daily_drawdown_fraction: Decimal = Field(default=Decimal("0.05"), alias="PAPER_MAX_DAILY_DRAWDOWN_FRACTION")
    paper_db_path: str = Field(default="data/paper_trading.sqlite3", alias="PAPER_DB_PATH")

    trader_brain_min_edge_bps: float = Field(default=5.0, alias="TRADER_BRAIN_MIN_EDGE_BPS")
    trader_brain_min_direction_probability: float = Field(default=0.58, alias="TRADER_BRAIN_MIN_DIRECTION_PROBABILITY")
    trader_brain_max_regime_entropy: float = Field(default=0.92, alias="TRADER_BRAIN_MAX_REGIME_ENTROPY")
    trader_brain_uncertainty_penalty: float = Field(default=0.25, alias="TRADER_BRAIN_UNCERTAINTY_PENALTY")
    trader_brain_meta_min_samples: int = Field(default=200, alias="TRADER_BRAIN_META_MIN_SAMPLES")
    trader_brain_bandit_paper_enabled: bool = Field(default=True, alias="TRADER_BRAIN_BANDIT_PAPER_ENABLED")
    trader_brain_bandit_min_samples: int = Field(default=100, alias="TRADER_BRAIN_BANDIT_MIN_SAMPLES")
    trader_brain_bandit_alpha: float = Field(default=0.35, alias="TRADER_BRAIN_BANDIT_ALPHA")

    model_registry_path: str = Field(default="state/model_registry", alias="MODEL_REGISTRY_PATH")
    ml_runs_path: str = Field(default="logs/ml", alias="ML_RUNS_PATH")
    ml_checkpoints_path: str = Field(default="models/checkpoints", alias="ML_CHECKPOINTS_PATH")

    @property
    def coinex_configured(self) -> bool:
        return bool(self.coinex_access_id and self.coinex_secret_key)

    def assert_safe_mode(self) -> None:
        if self.trading_mode is not TradingMode.PAPER:
            raise RuntimeError("Live trading is disabled in the current project phase. Set TRADING_MODE=paper.")
        if self.paper_initial_balance_usdt <= 0 or self.paper_model_initial_balance_eur_equiv <= 0:
            raise RuntimeError("Paper starting balances must be positive.")
        if not Decimal("0") <= self.paper_fee_rate < Decimal("1") or self.paper_slippage_bps < 0:
            raise RuntimeError("Paper execution costs are invalid.")
        if not 0 <= self.paper_min_confidence <= 1:
            raise RuntimeError("PAPER_MIN_CONFIDENCE must be between 0 and 1.")
        limits = (self.paper_max_order_fraction, self.paper_max_symbol_exposure_fraction, self.paper_max_total_exposure_fraction)
        if any(not Decimal("0") < value <= Decimal("1") for value in limits):
            raise RuntimeError("Paper exposure fractions must be in (0,1].")
        if not self.paper_max_order_fraction <= self.paper_max_symbol_exposure_fraction <= self.paper_max_total_exposure_fraction:
            raise RuntimeError("Paper exposure limits must satisfy order <= symbol <= total.")
        if not Decimal("0") < self.paper_max_daily_drawdown_fraction <= Decimal("1"):
            raise RuntimeError("PAPER_MAX_DAILY_DRAWDOWN_FRACTION must be in (0,1].")
        if self.trader_brain_min_edge_bps < 0 or not 0.5 <= self.trader_brain_min_direction_probability <= 1.0:
            raise RuntimeError("Trader-Brain edge/probability thresholds are invalid.")
        if not 0.0 <= self.trader_brain_max_regime_entropy <= 1.0 or self.trader_brain_uncertainty_penalty < 0:
            raise RuntimeError("Trader-Brain regime/uncertainty thresholds are invalid.")
        if self.trader_brain_meta_min_samples < 30 or self.trader_brain_bandit_min_samples < 1 or self.trader_brain_bandit_alpha < 0:
            raise RuntimeError("Trader-Brain learning thresholds are invalid.")


settings = Settings()
settings.assert_safe_mode()
