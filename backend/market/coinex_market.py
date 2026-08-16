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
        """Return an epoch timestamp in milliseconds, or None when invalid."""
        try:
            timestamp = int(value)
        except (TypeError, ValueError):
            return None
        if timestamp <= 0:
            return None
        if timestamp < 10_000_000_000:
            timestamp *= 1000
        return timestamp

    async def get_klines(
        self,
        symbol: str,
        period: str = "5min",
        limit: int = 100,
        *,
        start_time: int | None = None,
    ) -> list[dict[str, object]]:
        symbol = symbol.upper().strip()
        limit = max(1, min(limit, 1000))
        params: dict[str, object] = {"market": symbol, "period": period, "limit": limit}
        if start_time is not None:
            params["start_time"] = int(start_time)
        payload = await self._get("/spot/kline", params)
        data = payload.get("data")
        if not isinstance(data, list):
            raise MarketDataError("CoinEx returned an unexpected candlestick payload")

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

    async def get_deals(self, symbol: str, *, limit: int = 1000) -> list[dict[str, object]]:
        """Return recent public trades including CoinEx taker side.

        The endpoint is public/read-only. Values are normalized so the short-term
        collector can aggregate buy/sell notional without depending on raw API
        formatting.
        """
        symbol = symbol.upper().strip()
        limit = max(1, min(limit, 1000))
        payload = await self._get("/spot/deals", {"market": symbol, "limit": limit})
        data = payload.get("data")
        if not isinstance(data, list):
            raise MarketDataError("CoinEx returned an unexpected deals payload")
        out: list[dict[str, object]] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            ts = self._normalize_timestamp_ms(item.get("created_at"))
            try:
                deal_id = int(item.get("deal_id", 0))
                side = str(item.get("side", "")).lower()
                price = Decimal(str(item.get("price", "0")))
                amount = Decimal(str(item.get("amount", "0")))
            except (InvalidOperation, TypeError, ValueError):
                continue
            if ts is None or deal_id <= 0 or side not in {"buy", "sell"} or price <= 0 or amount <= 0:
                continue
            out.append({
                "deal_id": deal_id,
                "created_at": ts,
                "side": side,
                "price": str(price),
                "amount": str(amount),
                "notional": str(price * amount),
            })
        out.sort(key=lambda row: (int(row["created_at"]), int(row["deal_id"])))
        return out

    async def get_depth(self, symbol: str, *, limit: int = 20, interval: str = "0.01") -> dict[str, object]:
        """Return a normalized public order-book snapshot."""
        symbol = symbol.upper().strip()
        if limit not in {5, 10, 20, 50}:
            raise ValueError("CoinEx depth limit must be one of 5, 10, 20, 50")
        payload = await self._get("/spot/depth", {"market": symbol, "limit": limit, "interval": interval})
        data = payload.get("data")
        if not isinstance(data, dict) or not isinstance(data.get("depth"), dict):
            raise MarketDataError("CoinEx returned an unexpected depth payload")
        depth = data["depth"]

        def normalize_levels(raw: object) -> list[dict[str, str]]:
            levels: list[dict[str, str]] = []
            if not isinstance(raw, list):
                return levels
            for level in raw:
                if not isinstance(level, list) or len(level) < 2:
                    continue
                try:
                    price = Decimal(str(level[0])); amount = Decimal(str(level[1]))
                except (InvalidOperation, TypeError, ValueError):
                    continue
                if price > 0 and amount > 0:
                    levels.append({"price": str(price), "amount": str(amount)})
            return levels

        return {
            "market": str(data.get("market", symbol)),
            "updated_at": self._normalize_timestamp_ms(depth.get("updated_at")),
            "last": str(depth.get("last", "0")),
            "bids": normalize_levels(depth.get("bids")),
            "asks": normalize_levels(depth.get("asks")),
        }
