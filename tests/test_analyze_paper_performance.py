from scripts.analyze_paper_performance import analyze_algorithm, analyze_snapshot


def test_costs_erased_positive_pre_fee_edge():
    row = {
        "model_id": "mean-reversion-v1",
        "display_name": "Mean Reversion",
        "net_pnl": -0.45,
        "fees": 4.05,
        "closed_trades": 10,
        "executions": 20,
        "wins": 6,
        "win_rate": 0.60,
        "experimental": False,
    }

    result = analyze_algorithm(row)

    assert result["pre_fee_pnl"] == 3.5999999999999996
    assert result["costs_erased_positive_pre_fee_edge"] is True
    assert result["high_fee_burden"] is True
    assert result["evidence_status"] == "INSUFFICIENT_SAMPLE"
    assert "COSTS_ERASED_PRE_FEE_EDGE" in result["flags"]
    assert "WAIT_FOR_MORE_CLOSED_TRADES" in result["flags"]


def test_negative_before_and_after_fees_is_not_cost_erasure():
    result = analyze_algorithm(
        {
            "model_id": "explore",
            "net_pnl": -7.0,
            "fees": 2.0,
            "closed_trades": 20,
            "executions": 40,
        }
    )

    assert result["pre_fee_pnl"] == -5.0
    assert result["fee_to_positive_pre_fee_ratio"] is None
    assert result["costs_erased_positive_pre_fee_edge"] is False
    assert result["high_fee_burden"] is False


def test_review_status_only_after_minimum_sample():
    result = analyze_algorithm(
        {
            "model_id": "mature",
            "net_pnl": 8.0,
            "fees": 2.0,
            "closed_trades": 50,
            "executions": 100,
        },
        min_closed_trades=50,
    )

    assert result["evidence_status"] == "PRELIMINARY_POSITIVE"
    assert "WAIT_FOR_MORE_CLOSED_TRADES" not in result["flags"]


def test_snapshot_analysis_is_explicitly_read_only():
    report = analyze_snapshot(
        {
            "generated_at_utc": "2026-08-25T14:05:00+00:00",
            "runtime_git_commit": "abc123",
            "algorithms": [
                {
                    "model_id": "official",
                    "display_name": "Official",
                    "net_pnl": 1.0,
                    "fees": 1.0,
                    "closed_trades": 5,
                    "executions": 10,
                    "experimental": False,
                },
                {
                    "model_id": "experimental",
                    "display_name": "Experimental",
                    "net_pnl": -2.0,
                    "fees": 1.0,
                    "closed_trades": 5,
                    "executions": 10,
                    "experimental": True,
                },
            ],
        }
    )

    assert report["read_only_analysis"] is True
    assert report["strategy_or_risk_parameters_changed"] is False
    assert report["summary"]["official_count"] == 1
    assert report["summary"]["experimental_count"] == 1
