import hashlib
import hmac
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import httpx


COINEX_BASE_URL = "https://api.coinex.com/v2"


class CoinExAPIError(RuntimeError):
    """Raised when CoinEx returns an unsuccessful response."""


@dataclass(frozen=True)
class SpotBalance:
    ccy: str
    available: Decimal
    frozen: Decimal

    @property
    def total(self) -> Decimal:
        return self.available + self.frozen

    def as_dict(self) -> dict[str, str]:
        return {
            "ccy": self.ccy,
            "available": str(self.available),
            "frozen": str(self.frozen),
            "total": str(self.total),
        }


class CoinExClient:
    """Minimal CoinEx v2 client for read-only private account data."""

    def __init__(
        self,
        access_id: str,
        secret_key: str,
        *,
        base_url: str = COINEX_BASE_URL,
        timeout: float = 10.0,
    ) -> None:
        self.access_id = access_id
        self.secret_key = secret_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _signature(self, method: str, request_path: str, timestamp_ms: int, body: str = "") -> str:
        prepared = f"{method.upper()}{request_path}{body}{timestamp_ms}"
        return hmac.new(
            self.secret_key.encode("utf-8"),
            prepared.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def _headers(self, method: str, request_path: str, timestamp_ms: int, body: str = "") -> dict[str, str]:
        return {
            "X-COINEX-KEY": self.access_id,
            "X-COINEX-SIGN": self._signature(method, request_path, timestamp_ms, body),
            "X-COINEX-TIMESTAMP": str(timestamp_ms),
            "Content-Type": "application/json",
        }

    async def get_spot_balances(self) -> list[SpotBalance]:
        endpoint = "/assets/spot/balance"
        request_path = f"/v2{endpoint}"
        timestamp_ms = int(time.time() * 1000)
        headers = self._headers("GET", request_path, timestamp_ms)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.get(f"{self.base_url}{endpoint}", headers=headers)
                response.raise_for_status()
            except httpx.HTTPError as exc:
                raise CoinExAPIError(f"CoinEx HTTP request failed: {exc}") from exc

        try:
            payload: dict[str, Any] = response.json()
        except ValueError as exc:
            raise CoinExAPIError("CoinEx returned a non-JSON response") from exc

        if payload.get("code") != 0:
            raise CoinExAPIError(
                f"CoinEx API error {payload.get('code')}: {payload.get('message', 'unknown error')}"
            )

        balances: list[SpotBalance] = []
        for item in payload.get("data", []):
            balance = SpotBalance(
                ccy=str(item["ccy"]),
                available=Decimal(str(item["available"])),
                frozen=Decimal(str(item["frozen"])),
            )
            if balance.total != 0:
                balances.append(balance)

        return sorted(balances, key=lambda item: item.ccy)
