from __future__ import annotations

from backend.paper.model_catalog import ALL_PAPER_MODELS, SHORT_TERM_MODELS
from backend.short_term.collector import BUCKET_MS, BucketAccumulator, accumulator_for_timestamp
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


def test_missing_microstructure_cannot_turn_momentum_hold_into_entry():
    bearish = _features(trade_imbalance=-0.8, book_imbalance=-0.8, spread_bps=20.0)
    missing = _features(
        trade_count=0,
        buy_notional=0.0,
        sell_notional=0.0,
        trade_imbalance=0.0,
        buy_ratio=0.5,
        spread_bps=None,
        book_imbalance=None,
        microstructure_coverage=0.0,
        quality=0.75,
    )
    with_bearish_micro = decide_momentum(
        bearish, long_open=False, hold_minutes=0, unrealized_return=0, min_edge_bps=65
    )
    without_micro = decide_momentum(
        missing, long_open=False, hold_minutes=0, unrealized_return=0, min_edge_bps=65
    )
    assert with_bearish_micro.action == "HOLD"
    assert without_micro.action == "HOLD"


def test_low_microstructure_coverage_blocks_new_entry():
    f = _features(microstructure_coverage=0.49)
    decision = decide_momentum(f, long_open=False, hold_minutes=0, unrealized_return=0, min_edge_bps=65)
    assert decision.action == "HOLD"
    assert "microstructure incomplete" in decision.reason.lower()


def test_wrong_side_mean_reversion_displacement_does_not_clear_hurdle():
    # Six of eight confirmations are true, but price is above both reference
    # levels. Absolute displacement used to convert this wrong-side setup into
    # a long entry; the signed oversold proxy must remain zero.
    f = _features(
        return_15m=-0.004,
        rsi_14=30.0,
        bollinger_z=2.0,
        vwap_distance_bps=70.0,
        atr_bps=70.0,
        trade_imbalance=0.0,
        book_imbalance=0.0,
        spread_bps=2.0,
    )
    decision = decide_mean_reversion(f, long_open=False, hold_minutes=0, unrealized_return=0, min_edge_bps=65)
    assert decision.action == "HOLD"
    assert decision.edge_proxy_bps == 0.0


def test_atr_alone_cannot_clear_momentum_edge_hurdle():
    f = _features(
        return_15m=0.0001,
        return_30m=0.0002,
        return_1h=0.0010,
        return_2h=0.0010,
        atr_bps=120.0,
    )
    decision = decide_momentum(f, long_open=False, hold_minutes=0, unrealized_return=0, min_edge_bps=65)
    assert decision.action == "HOLD"
    assert decision.edge_proxy_bps < 65.0


def test_protective_and_time_exits_work_without_microstructure():
    f = _features(spread_bps=None, book_imbalance=None, microstructure_coverage=0.0)
    protective = decide_mean_reversion(
        f,
        long_open=True,
        hold_minutes=10,
        unrealized_return=-0.01,
        min_edge_bps=65,
        round_trip_cost_bps=50,
    )
    timed = decide_mean_reversion(
        f,
        long_open=True,
        hold_minutes=121,
        unrealized_return=0.0,
        min_edge_bps=65,
        round_trip_cost_bps=50,
    )
    assert protective.action == "EXIT"
    assert timed.action == "EXIT"


def test_mean_reversion_target_exit_must_clear_round_trip_cost():
    f = _features(bollinger_z=0.0, rsi_14=55.0, vwap_distance_bps=0.0)
    too_small = decide_mean_reversion(
        f,
        long_open=True,
        hold_minutes=30,
        unrealized_return=0.0015,
        min_edge_bps=65,
        round_trip_cost_bps=50,
    )
    enough = decide_mean_reversion(
        f,
        long_open=True,
        hold_minutes=30,
        unrealized_return=0.0060,
        min_edge_bps=65,
        round_trip_cost_bps=50,
    )
    assert too_small.action == "HOLD"
    assert enough.action == "EXIT"


def test_previous_bucket_remains_assignable_during_rollover_grace_poll():
    previous = BucketAccumulator(1_000_000, 1_000_000, 1_000_000)
    current = BucketAccumulator(1_000_000 + BUCKET_MS, 1_000_000 + BUCKET_MS, 1_000_000 + BUCKET_MS)
    late_previous_ts = previous.bucket_start + BUCKET_MS - 1
    current_ts = current.bucket_start + 1
    assert accumulator_for_timestamp(late_previous_ts, current, previous) is previous
    assert accumulator_for_timestamp(current_ts, current, previous) is current


def test_short_term_models_are_isolated_catalog_entries():
    ids = {model.model_id for model in SHORT_TERM_MODELS}
    all_ids = {model.model_id for model in ALL_PAPER_MODELS}
    assert ids == {"short-momentum-15m", "short-mean-reversion-15m"}
    assert ids <= all_ids
    assert all(model.driver == "short_term" for model in SHORT_TERM_MODELS)
