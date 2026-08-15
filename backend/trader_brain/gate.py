from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .contracts import ExpertForecast, GateResult, RegimePosterior

DEFAULT_EXPERT_ORDER = ("technical", "macro", "derivatives", "news", "ai_v3_12h", "ai_v3_3h")


def build_gate_vector(regime: RegimePosterior, experts: Sequence[ExpertForecast], *, expert_order: Sequence[str] = DEFAULT_EXPERT_ORDER) -> tuple[float, ...]:
    by_name = {expert.name: expert for expert in experts}
    values: list[float] = [*regime.probabilities, regime.entropy, regime.quality]
    for name in expert_order:
        expert = by_name.get(name)
        if expert is None:
            values.extend([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        else:
            values.extend([
                expert.p_down, expert.p_flat, expert.p_up, expert.expected_return,
                expert.sigma, expert.confidence, expert.quality, 1.0 if expert.available else 0.0,
            ])
    vector = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(vector)):
        raise ValueError("gate vector contains non-finite values")
    return tuple(float(value) for value in vector)


class ReliabilityWeightedGate:
    """Auditable fallback until enough true forward outcomes exist for stacking."""

    def combine(
        self,
        regime: RegimePosterior,
        experts: Sequence[ExpertForecast],
        *,
        reliability: Mapping[str, float] | None = None,
    ) -> GateResult:
        reliability = reliability or {}
        usable = [expert for expert in experts if expert.available and expert.quality > 0.0]
        vector = build_gate_vector(regime, experts)
        if not usable:
            return GateResult(0.0, 1.0, 0.0, 0.0, 0.0, {}, "reliability_fallback_no_experts", vector)
        raw: dict[str, float] = {}
        for expert in usable:
            recent = float(np.clip(reliability.get(expert.name, 1.0), 0.05, 1.5))
            raw[expert.name] = max(1e-9, expert.quality * max(expert.confidence, 0.05) * recent)
        total = sum(raw.values())
        weights = {name: value / total for name, value in raw.items()}
        probabilities = np.zeros(3, dtype=np.float64)
        mean = 0.0
        second = 0.0
        by_name = {expert.name: expert for expert in usable}
        for name, weight in weights.items():
            expert = by_name[name]
            probabilities += weight * expert.probabilities
            mean += weight * expert.expected_return
            second += weight * (expert.sigma**2 + expert.expected_return**2)
        probabilities /= probabilities.sum()
        variance = max(second - mean**2, 0.0)
        return GateResult(
            p_down=float(probabilities[0]), p_flat=float(probabilities[1]), p_up=float(probabilities[2]),
            expected_return=float(mean), sigma=float(np.sqrt(variance)), weights=weights,
            gate_type="reliability_weighted", feature_vector=vector,
        )


class XGBoostStackingGate:
    """First learned meta-model; fit only on resolved/OOS expert forecasts."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = int(random_state)
        self.booster = None
        self.feature_count: int | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostStackingGate":
        import xgboost as xgb
        matrix = np.asarray(X, dtype=np.float64)
        target = np.asarray(y, dtype=np.int32)
        if matrix.ndim != 2 or len(matrix) < 30 or len(matrix) != len(target):
            raise ValueError("stacking gate requires >=30 matching OOS rows")
        if len(np.unique(target)) < 2:
            raise ValueError("stacking gate requires at least two target classes")
        params = {
            "objective": "multi:softprob", "num_class": 3, "eval_metric": "mlogloss",
            "tree_method": "hist", "max_depth": 3, "eta": 0.035, "min_child_weight": 12,
            "subsample": 0.8, "colsample_bytree": 0.8, "lambda": 12.0, "alpha": 0.01,
            "seed": self.random_state, "nthread": 0,
        }
        self.booster = xgb.train(params, xgb.DMatrix(matrix, label=target), num_boost_round=180, verbose_eval=False)
        self.feature_count = int(matrix.shape[1])
        return self

    def predict(self, vector: Sequence[float], *, fallback: GateResult) -> GateResult:
        import xgboost as xgb
        if self.booster is None or self.feature_count is None:
            raise RuntimeError("stacking gate is not fitted")
        row = np.asarray(vector, dtype=np.float64).reshape(1, -1)
        if row.shape[1] != self.feature_count:
            raise ValueError("stacking gate feature count mismatch")
        p = np.asarray(self.booster.predict(xgb.DMatrix(row)), dtype=np.float64).reshape(-1)
        if len(p) != 3 or not np.all(np.isfinite(p)):
            raise RuntimeError("invalid stacking probabilities")
        p = np.clip(p, 1e-9, 1.0); p /= p.sum()
        direction = float(p[2] - p[0])
        magnitude = max(abs(fallback.expected_return), fallback.sigma * 0.20 * abs(direction))
        expected = float(np.sign(direction) * magnitude) if abs(direction) > 1e-9 else 0.0
        return GateResult(
            p_down=float(p[0]), p_flat=float(p[1]), p_up=float(p[2]), expected_return=expected,
            sigma=fallback.sigma, weights=fallback.weights, gate_type="xgboost_stacking_oos",
            feature_vector=tuple(float(value) for value in row.reshape(-1)),
        )

    def save(self, path: Path) -> None:
        if self.booster is None:
            raise RuntimeError("stacking gate is not fitted")
        path.parent.mkdir(parents=True, exist_ok=True); self.booster.save_model(path)
