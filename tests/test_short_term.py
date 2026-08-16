from __future__ import annotations

from backend.paper.model_catalog import ALL_PAPER_MODELS, SHORT_TERM_MODELS
from backend.short_term.features import ShortTermFeatures, build_short_term_features
from backend.short_term.strategies import decide_mean_reversion, decide_momentum


def _candles(n: int = 80):
    rows = []
    price = 100.0
    for i in range(n):
        change = 0.001 if i % 3 != 0 else -0.0004
        op = price
        price *= 1.0 + change
        rows.append({
            "created_at": 1_700_000_000_000 + i * 900_000,
            "open": str(op), "high": str(max(op, price) * 1.001),
            "low": str(min(op, price) * 0.999), "close": str(price),
            "volume": str(100 + i), "value": str((100 + i) * price),
        })
    return rows


def _features(**overrides) -> ShortTermFeatures:
    base = dict(
        timestamp=1_700_000_000_000, close=100.0, return_15m=0.002, return_30m=0.003,
        return_1h=0.009, return_2h=0.012, ema_gap_bps=35.0, ema_fast_slope_bps=8.0,
        rsi_14=63.0, atr_bps=80.0, bollinger_z=0.8, volume_z=1.0, value_z=1.0,
        vwap_distance_bps=55.0, candle_range_bps=45.0, candle_body_bps=20.0,
        realized_vol_1h_bps=50.0, trade_count=500, buy_notional=1000.0, sell_notional=700.0,
        trade_imbalance=0.176, buy_ratio=0.588, spread_bps=2.0, book_imbalance=0.15,
        microstructure_coverage=0.95, quality=0.95,
    )
    base.update(overrides)
    return ShortTermFeatures(**base)


def test_feature_builder_includes_volume_and_microstructure():
    f = build_short_term_features(_candles(), {
        "trade_count": 120, "buy_notional": "800", "sell_notional": "400",
        "trade_imbalance": 1/3, "buy_ratio": 2/3, "spread_bps": 1.5,
        "book_imbalance": 0.2, "coverage_fraction": 0.9,
    })
    assert f.trade_count == 120
    assert f.buy_ratio > 0.6
    assert f.spread_bps == 1.5
    assert f.quality > 0.7
    assert f.volume_z != 0.0


def test_momentum_enters_only_when_cost_hurdle_is_cleared():
    f = _features()
    yes = decide_momentum(f, long_open=False, hold_minutes=0, unrealized_return=0, min_edge_bps=65)
    no = decide_momentum(f, long_open=False, hold_minutes=0, unrealized_return=0, min_edge_bps=150)
    assert yes.action == "ENTER_LONG"
    assert no.action == "HOLD"


def test_mean_reversion_can_enter_oversold_setup():
    f = _features(
        return_15m=-0.004, return_30m=-0.007, return_1h=-0.010, return_2h=-0.008,
        ema_gap_bps=-50.0, ema_fast_slope_bps=-4.0, rsi_14=29.0, atr_bps=70.0,
        bollinger_z=-2.0, vwap_distance_bps=-150.0, trade_imbalance=-0.05,
        book_imbalance=-0.05,
    )
    decision = decide_mean_reversion(f, long_open=False, hold_minutes=0, unrealized_return=0, min_edge_bps=65)
    assert decision.action == "ENTER_LONG"


def test_short_term_models_are_isolated_catalog_entries():
    ids = {model.model_id for model in SHORT_TERM_MODELS}
    all_ids = {model.model_id for model in ALL_PAPER_MODELS}
    assert ids == {"short-momentum-15m", "short-mean-reversion-15m"}
    assert ids <= all_ids
    assert all(model.driver == "short_term" for model in SHORT_TERM_MODELS)
