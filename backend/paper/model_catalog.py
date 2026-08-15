from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PaperModelSpec:
    model_id: str
    display_name: str
    role: str
    target_bps: int
    horizon_hours: int
    feature_set: str
    research_auc: float
    research_median_auc: float
    research_sharpe_25bps: float
    research_return_25bps: float
    research_trades: int
    research_gate_passed: bool = False
    driver: str = "frozen_v3"
    algorithm_family: str = "Legacy AI"
    description: str = ""
    adaptive: bool = False
    supports_short: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


FROZEN_V3_MODELS: tuple[PaperModelSpec, ...] = (
    PaperModelSpec(
        model_id="v3-25bps-fullcontext-12h", display_name="V3 12h Economic", role="paper_strategy",
        target_bps=25, horizon_hours=12, feature_set="full_context", research_auc=0.529,
        research_median_auc=0.526, research_sharpe_25bps=0.397, research_return_25bps=0.2907,
        research_trades=20, driver="frozen_v3", algorithm_family="Frozen LSTM V3",
        description="Existing 12h full-context neural forecast with frozen forward policy.",
    ),
    PaperModelSpec(
        model_id="v3-50bps-technical-3h", display_name="V3 3h Signal Control", role="research_control",
        target_bps=50, horizon_hours=3, feature_set="technical", research_auc=0.601,
        research_median_auc=0.593, research_sharpe_25bps=0.225, research_return_25bps=0.1194,
        research_trades=3, driver="frozen_v3", algorithm_family="Frozen LSTM V3",
        description="Existing technical-only V3 control model used as an independent benchmark.",
    ),
)

TRADER_BRAIN_MODELS: tuple[PaperModelSpec, ...] = (
    PaperModelSpec(
        model_id="trader-brain-v1", display_name="Trader Brain V1", role="research_candidate",
        target_bps=25, horizon_hours=4, feature_set="regime_moe", research_auc=0.0,
        research_median_auc=0.0, research_sharpe_25bps=0.0, research_return_25bps=0.0,
        research_trades=0, driver="trader_brain", algorithm_family="Regime-aware Mixture of Experts",
        description=(
            "Probabilistic regimes plus Technical, Macro, Derivatives, News and existing-AI experts. "
            "Uses reliability weighting until enough resolved OOS samples exist for XGBoost stacking. "
            "Costs, uncertainty and regime ambiguity can force NO_TRADE."
        ),
    ),
    PaperModelSpec(
        model_id="trader-brain-bandit-v1", display_name="Trader Brain + RL", role="research_candidate",
        target_bps=25, horizon_hours=4, feature_set="regime_moe_bandit", research_auc=0.0,
        research_median_auc=0.0, research_sharpe_25bps=0.0, research_return_25bps=0.0,
        research_trades=0, driver="trader_brain_rl", algorithm_family="MoE + Contextual Bandit RL",
        description=(
            "Same Trader-Brain forecasts with a PAPER-only LinUCB contextual-bandit policy. "
            "It learns from resolved deal and shadow rewards but can never bypass hard RiskManager limits."
        ), adaptive=True,
    ),
)

# Critical compatibility contract: forward_v3.py iterates PAPER_MODELS and expects frozen Keras artifacts.
PAPER_MODELS = FROZEN_V3_MODELS
ALL_PAPER_MODELS = FROZEN_V3_MODELS + TRADER_BRAIN_MODELS
FROZEN_FORWARD_MODELS = FROZEN_V3_MODELS
MODEL_BY_ID = {model.model_id: model for model in ALL_PAPER_MODELS}


def get_paper_model(model_id: str) -> PaperModelSpec:
    try:
        return MODEL_BY_ID[model_id]
    except KeyError as exc:
        raise KeyError(f"Unknown paper model: {model_id}") from exc
