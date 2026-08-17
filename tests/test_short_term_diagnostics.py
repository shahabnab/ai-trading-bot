from __future__ import annotations

from pathlib import Path

from backend.short_term.diagnostics import (
    BUCKET_MS,
    append_decision_diagnostic,
    build_diagnostics_summary,
    build_shadow_policies,
    decision_key,
    resolve_mature_outcomes,
)
from backend.short_term.features import ShortTermFeatures
from backend.short_term.strategies import decide_momentum


def _momentum_features() -> ShortTermFeatures:
    return ShortTermFeatures(
        timestamp=1_700_000_000_000,
        close=100.0,
        return_15m=0.001,
        return_30m=0.002,
        return_1h=0.005,
        return_2h=0.008,
        ema_gap_bps=20.0,
        ema_fast_slope_bps=5.0,
        rsi_14=60.0,
        atr_bps=50.0,
        bollinger_z=0.2,
        volume_z=0.5,
        value_z=0.1,
        vwap_distance_bps=10.0,
        candle_range_bps=10.0,
        candle_body_bps=5.0,
        realized_vol_1h_bps=10.0,
        trade_count=100,
        buy_notional=1000.0,
        sell_notional=500.0,
        trade_imbalance=0.20,
        buy_ratio=2.0 / 3.0,
        spread_bps=2.0,
        book_imbalance=0.20,
        microstructure_coverage=0.95,
        quality=0.95,
    )


def test_shadow_policies_probe_edge_hurdles_without_changing_official_decision():
    features = _momentum_features()
    decision = decide_momentum(
        features,
        long_open=False,
        hold_minutes=0.0,
        unrealized_return=0.0,
        min_edge_bps=65.0,
        round_trip_cost_bps=50.0,
    )
    assert decision.action == "HOLD"
    assert decision.setup_ready is True
    assert decision.confirmation_score == 1.0
    assert decision.edge_proxy_bps == 50.0

    policies = build_shadow_policies(
        decision,
        entry_context=True,
        official_threshold_bps=65.0,
        round_trip_cost_bps=50.0,
    )
    by_name = {row["name"]: row for row in policies}
    assert by_name["official"]["would_enter"] is False
    assert by_name["shadow_55"]["would_enter"] is False
    assert by_name["shadow_45"]["would_enter"] is True
    assert by_name["shadow_35"]["would_enter"] is True
    assert by_name["raw_setup"]["would_enter"] is True


def test_outcome_analyzer_marks_profitable_rejected_setup_as_missed_long(tmp_path: Path):
    features = _momentum_features()
    decision = decide_momentum(
        features,
        long_open=False,
        hold_minutes=0.0,
        unrealized_return=0.0,
        min_edge_bps=65.0,
        round_trip_cost_bps=50.0,
    )
    policies = build_shadow_policies(
        decision,
        entry_context=True,
        official_threshold_bps=65.0,
        round_trip_cost_bps=50.0,
    )
    append_decision_diagnostic(
        tmp_path,
        {
            "decision_key": decision_key("short-momentum-15m", features.timestamp),
            "model_id": "short-momentum-15m",
            "feature_timestamp": features.timestamp,
            "signal_close": features.close,
            "decision_action": decision.action,
            "signal": "HOLD",
            "confidence": decision.confidence,
            "confirmation_score": decision.confirmation_score,
            "setup_ready": decision.setup_ready,
            "edge_proxy_bps": decision.edge_proxy_bps,
            "official_threshold_bps": 65.0,
            "edge_gap_bps": decision.edge_proxy_bps - 65.0,
            "round_trip_cost_bps": 50.0,
            "shadow_policies": policies,
        },
    )

    candles = []
    for i in range(1, 9):
        close = 100.0 * (1.0 + 0.0009 * i)
        candles.append(
            {
                "created_at": features.timestamp + i * BUCKET_MS,
                "close": str(close),
                "high": str(close * 1.0005),
                "low": str(close * 0.9995),
            }
        )

    outcomes = resolve_mature_outcomes(tmp_path, candles)
    assert len(outcomes) == 1
    outcome = outcomes[0]
    assert outcome["classification"] == "MISSED_LONG"
    assert outcome["returns_bps"]["2h"] > 65.0
    shadow_45 = next(row for row in outcome["shadow_results"] if row["name"] == "shadow_45")
    assert shadow_45["would_enter"] is True
    assert shadow_45["win_after_cost"] is True

    # Re-running the resolver must not duplicate the same decision outcome.
    assert resolve_mature_outcomes(tmp_path, candles) == []

    summary = build_diagnostics_summary(tmp_path)
    model = summary["models"][0]
    assert model["classifications"]["MISSED_LONG"] == 1
    policy = next(row for row in model["policies"] if row["name"] == "shadow_45")
    assert policy["signals"] == 1
    assert policy["wins"] == 1
