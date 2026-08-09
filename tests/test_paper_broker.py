from decimal import Decimal

from backend.paper.broker import PaperBroker
from backend.paper.storage import PaperStore


def test_paper_buy_and_sell_are_persisted(tmp_path) -> None:
    store = PaperStore(str(tmp_path / "paper.sqlite3"), Decimal("10000"))
    broker = PaperBroker(store, fee_rate=Decimal("0.002"), slippage_bps=Decimal("5"))

    buy = broker.buy(
        symbol="BTCUSDT",
        market_price=Decimal("100000"),
        notional_usdt=Decimal("500"),
        model_version="test-model",
        strategy_version="test-strategy",
        confidence=0.9,
    )

    position = store.get_position("BTCUSDT")
    assert buy["side"] == "BUY"
    assert position is not None
    assert Decimal(position["quantity"]) > 0
    assert Decimal(store.get_account()["cash_usdt"]) < Decimal("10000")

    sell = broker.sell(
        symbol="BTCUSDT",
        market_price=Decimal("101000"),
        quantity=None,
        model_version="test-model",
        strategy_version="test-strategy",
        confidence=0.9,
    )

    assert sell["side"] == "SELL"
    assert store.get_position("BTCUSDT") is None
    assert len(store.list_trades()) == 2
