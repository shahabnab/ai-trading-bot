import hashlib
import hmac
from decimal import Decimal

from backend.coinex.client import CoinExClient, SpotBalance


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


def test_canonical_request_path_includes_query_string() -> None:
    client = CoinExClient("access", "secret")

    path = client._canonical_request_path(
        "/assets/spot/transcation-history",
        [("ccy", "USDT"), ("type", "trade")],
    )

    assert path == "/v2/assets/spot/transcation-history?ccy=USDT&type=trade"


def test_spot_balance_total() -> None:
    balance = SpotBalance("BTC", Decimal("1.25"), Decimal("0.05"))
    assert balance.total == Decimal("1.30")
