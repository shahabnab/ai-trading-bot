from decimal import Decimal

from backend.paper.model_engine import ModelPaperStore


def test_model_accounts_are_isolated(tmp_path) -> None:
    store = ModelPaperStore(str(tmp_path / "paper.sqlite3"), Decimal("1000"))
    store.ensure_account("model-a", "Model A")
    store.ensure_account("model-b", "Model B")

    store.buy(
        "model-a",
        symbol="BTCUSDT",
        market_price=Decimal("100"),
        notional_usdt=Decimal("100"),
        fee_rate=Decimal("0"),
        slippage_bps=Decimal("0"),
        confidence=0.7,
        strategy_version="test",
    )

    account_a = store.get_account("model-a")
    account_b = store.get_account("model-b")
    assert Decimal(account_a["cash_usdt"]) == Decimal("900")
    assert Decimal(account_b["cash_usdt"]) == Decimal("1000")
    assert store.get_position("model-a", "BTCUSDT") is not None
    assert store.get_position("model-b", "BTCUSDT") is None
    assert len(store.list_trades("model-a")) == 1
    assert store.list_trades("model-b") == []


def test_model_sell_records_realized_pnl(tmp_path) -> None:
    store = ModelPaperStore(str(tmp_path / "paper.sqlite3"), Decimal("1000"))
    store.ensure_account("model-a", "Model A")
    store.buy(
        "model-a",
        symbol="BTCUSDT",
        market_price=Decimal("100"),
        notional_usdt=Decimal("100"),
        fee_rate=Decimal("0"),
        slippage_bps=Decimal("0"),
        confidence=0.7,
        strategy_version="test",
    )
    trade = store.sell(
        "model-a",
        symbol="BTCUSDT",
        market_price=Decimal("110"),
        quantity=None,
        fee_rate=Decimal("0"),
        slippage_bps=Decimal("0"),
        confidence=0.8,
        strategy_version="test",
    )

    assert Decimal(str(trade["realized_pnl_usdt"])) == Decimal("10")
    assert Decimal(store.get_account("model-a")["cash_usdt"]) == Decimal("1010")
    assert store.get_position("model-a", "BTCUSDT") is None
    summary = store.performance_summary("model-a")
    assert summary["closed_trades"] == 1
    assert summary["winning_trades"] == 1
    assert summary["win_rate"] == 1.0
