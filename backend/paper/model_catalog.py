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

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


PAPER_MODELS: tuple[PaperModelSpec, ...] = (
    PaperModelSpec(
        model_id="v3-25bps-fullcontext-12h",
        display_name="V3 12h Economic",
        role="paper_strategy",
        target_bps=25,
        horizon_hours=12,
        feature_set="full_context",
        research_auc=0.529,
        research_median_auc=0.526,
        research_sharpe_25bps=0.397,
        research_return_25bps=0.2907,
        research_trades=20,
    ),
    PaperModelSpec(
        model_id="v3-50bps-technical-3h",
        display_name="V3 3h Signal Control",
        role="research_control",
        target_bps=50,
        horizon_hours=3,
        feature_set="technical",
        research_auc=0.601,
        research_median_auc=0.593,
        research_sharpe_25bps=0.225,
        research_return_25bps=0.1194,
        research_trades=3,
    ),
)

MODEL_BY_ID = {model.model_id: model for model in PAPER_MODELS}


def get_paper_model(model_id: str) -> PaperModelSpec:
    try:
        return MODEL_BY_ID[model_id]
    except KeyError as exc:
        raise KeyError(f"Unknown paper model: {model_id}") from exc
