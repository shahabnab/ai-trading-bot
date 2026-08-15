from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

import httpx


COINEX_BASE_URL = "https://api.coinex.com/v2"


class MarketDataError(RuntimeError):
    """Raised when public CoinEx market data cannot be loaded or parsed."""


@dataclass(frozen=True)
class MarketQuote:
    symbol: str
    last: Decimal
    open: Decimal
    high: Decimal
    low: Decimal
    volume: Decimal
    value: Decimal

    def as_dict(self) -> dict[str, str]:
        return {
            "symbol": self.symbol,
            "last": str(self.last),
            "open": str(self.open),
            "high": str(self.high),
            "low": str(self.low),
            "volume": str(self.volume),
            "value": str(self.value),
        }


class CoinExMarketClient:
    """Public, read-only CoinEx v2 market-data client.

    This client has no API credentials and contains no order-placement code.
    """

    def __init__(self, *, base_url: str = COINEX_BASE_URL, timeout: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def _get(self, endpoint: str, params: dict[str, object]) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.get(f"{self.base_url}{endpoint}", params=params)
                response.raise_for_status()
            except httpx.HTTPError as exc:
                raise MarketDataError(f"CoinEx market-data request failed: {exc}") from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise MarketDataError("CoinEx returned non-JSON market data") from exc

        if not isinstance(payload, dict):
            raise MarketDataError("CoinEx returned an unexpected market-data payload")
        if payload.get("code") != 0:
            raise MarketDataError(
                f"CoinEx API error {payload.get('code')}: {payload.get('message', 'unknown error')}"
            )
        return payload

    async def get_ticker(self, symbol: str) -> MarketQuote:
        symbol = symbol.upper().strip()
        payload = await self._get("/spot/ticker", {"market": symbol})
        data = payload.get("data")
        if not isinstance(data, list) or not data:
            raise MarketDataError(f"No CoinEx ticker data returned for {symbol}")

        item = data[0]
        if not isinstance(item, dict):
            raise MarketDataError("CoinEx returned an invalid ticker entry")

        try:
            return MarketQuote(
                symbol=str(item.get("market", symbol)),
                last=Decimal(str(item["last"])),
                open=Decimal(str(item["open"])),
                high=Decimal(str(item["high"])),
                low=Decimal(str(item["low"])),
                volume=Decimal(str(item["volume"])),
                value=Decimal(str(item["value"])),
            )
        except (KeyError, InvalidOperation, TypeError, ValueError) as exc:
            raise MarketDataError("CoinEx returned malformed ticker data") from exc

    @staticmethod
    def _normalize_timestamp_ms(value: object) -> int | None:
        """Return an epoch timestamp in milliseconds, or None when invalid.

        CoinEx currently exposes millisecond timestamps, but normalizing here also
        keeps the dashboard safe if an upstream response arrives in epoch seconds.
        """
        try:
            timestamp = int(value)
        except (TypeError, ValueError):
            return None

        if timestamp <= 0:
            return None
        if timestamp < 10_000_000_000:
            timestamp *= 1000
        return timestamp

    async def get_klines(self, symbol: str, period: str = "5min", limit: int = 100) -> list[dict[str, object]]:
        symbol = symbol.upper().strip()
        limit = max(1, min(limit, 1000))
        payload = await self._get(
            "/spot/kline",
            {"market": symbol, "period": period, "limit": limit},
        )
        data = payload.get("data")
        if not isinstance(data, list):
            raise MarketDataError("CoinEx returned an unexpected candlestick payload")

        # Lightweight Charts requires strictly ordered, unique timestamps. Build
        # a timestamp-keyed map so duplicate CoinEx rows cannot crash the client.
        candles_by_timestamp: dict[int, dict[str, object]] = {}
        for item in data:
            if not isinstance(item, dict):
                continue

            created_at = self._normalize_timestamp_ms(item.get("created_at"))
            if created_at is None:
                continue

            try:
                open_price = Decimal(str(item.get("open", "0")))
                close_price = Decimal(str(item.get("close", "0")))
                high_price = Decimal(str(item.get("high", "0")))
                low_price = Decimal(str(item.get("low", "0")))
            except (InvalidOperation, TypeError, ValueError):
                continue

            if any(price <= 0 for price in (open_price, close_price, high_price, low_price)):
                continue

            candles_by_timestamp[created_at] = {
                "symbol": str(item.get("market", symbol)),
                "created_at": created_at,
                "open": str(open_price),
                "close": str(close_price),
                "high": str(high_price),
                "low": str(low_price),
                "volume": str(item.get("volume", "0")),
                "value": str(item.get("value", "0")),
            }

        return [candles_by_timestamp[timestamp] for timestamp in sorted(candles_by_timestamp)]
