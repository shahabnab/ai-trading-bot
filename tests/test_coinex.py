import hashlib
import hmac

from backend.coinex.client import CoinExClient, SpotBalance
from decimal import Decimal


def test_signature_matches_hmac_sha256() -> None:
    client = CoinExClient("access", "secret")
    timestamp = 1700490703564
    path = "/v2/assets/spot/balance"

    expected = hmac.new(
        b"secret",
        f"GET{path}{timestamp}".encode(),
        hashlib.sha256,
    ).hexdigest()

    assert client._signature("GET", path, timestamp) == expected


def test_spot_balance_total() -> None:
    balance = SpotBalance("BTC", Decimal("1.25"), Decimal("0.05"))
    assert balance.total == Decimal("1.30")
