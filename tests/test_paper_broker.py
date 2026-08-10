from datetime import UTC, datetime
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


def test_round_trip_pnl_includes_buy_and_sell_fees(tmp_path) -> None:
    store = PaperStore(str(tmp_path / "paper.sqlite3"), Decimal("10000"))
    broker = PaperBroker(store, fee_rate=Decimal("0.002"), slippage_bps=Decimal("0"))

    buy = broker.buy(
        symbol="BTCUSDT",
        market_price=Decimal("100000"),
        notional_usdt=Decimal("500"),
        model_version="test-model",
        strategy_version="test-strategy",
        confidence=0.9,
    )
    sell = broker.sell(
        symbol="BTCUSDT",
        market_price=Decimal("100000"),
        quantity=None,
        model_version="test-model",
        strategy_version="test-strategy",
        confidence=0.9,
    )

    total_fees = Decimal(str(buy["fee_usdt"])) + Decimal(str(sell["fee_usdt"]))
    assert Decimal(str(sell["realized_pnl_usdt"])) == -total_fees
    assert Decimal(store.get_account()["cash_usdt"]) == Decimal("10000") - total_fees


def test_daily_risk_baseline_is_stable_within_utc_day(tmp_path) -> None:
    store = PaperStore(str(tmp_path / "paper.sqlite3"), Decimal("10000"))
    morning = datetime(2026, 8, 10, 9, tzinfo=UTC)
    afternoon = morning.replace(hour=15)

    first = store.get_or_create_daily_start_portfolio_value(Decimal("10000"), now=morning)
    second = store.get_or_create_daily_start_portfolio_value(Decimal("9500"), now=afternoon)

    assert first == second == Decimal("10000")
