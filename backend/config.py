from enum import Enum

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class TradingMode(str, Enum):
    PAPER = "paper"
    LIVE = "live"


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables.

    Live execution is intentionally rejected during the current project phase.
    Secrets are optional at startup so the dashboard can run before an exchange
    account is configured.
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

    @property
    def coinex_configured(self) -> bool:
        return bool(self.coinex_access_id and self.coinex_secret_key)

    def assert_safe_mode(self) -> None:
        if self.trading_mode is not TradingMode.PAPER:
            raise RuntimeError(
                "Live trading is disabled in the current project phase. Set TRADING_MODE=paper."
            )


settings = Settings()
settings.assert_safe_mode()
