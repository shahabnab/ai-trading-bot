from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ExpertForecast:
    """Common probability/uncertainty contract shared by every specialist."""

    name: str
    p_down: float
    p_flat: float
    p_up: float
    expected_return: float
    sigma: float
    confidence: float
    quality: float
    available: bool
    timestamp: int
    signals: Mapping[str, float] = field(default_factory=dict)
    explanation: Sequence[str] = field(default_factory=tuple)

    @property
    def probabilities(self) -> np.ndarray:
        values = np.asarray([self.p_down, self.p_flat, self.p_up], dtype=np.float64)
        values = np.clip(values, 0.0, 1.0)
        total = float(values.sum())
        return values / total if total > 0.0 else np.asarray([0.0, 1.0, 0.0], dtype=np.float64)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["explanation"] = list(self.explanation)
        payload["signals"] = dict(self.signals)
        return payload


@dataclass(frozen=True)
class RegimePosterior:
    probabilities: tuple[float, ...]
    entropy: float
    labels: tuple[str, ...]
    method: str
    quality: float
    timestamp: int

    @property
    def dominant_index(self) -> int:
        return int(np.argmax(np.asarray(self.probabilities, dtype=np.float64)))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GateResult:
    p_down: float
    p_flat: float
    p_up: float
    expected_return: float
    sigma: float
    weights: Mapping[str, float]
    gate_type: str
    feature_vector: tuple[float, ...]

    @property
    def confidence(self) -> float:
        return float(max(self.p_down, self.p_flat, self.p_up))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["weights"] = dict(self.weights)
        payload["confidence"] = self.confidence
        return payload


@dataclass(frozen=True)
class TraderDecision:
    action: str
    expected_return: float
    expected_cost: float
    uncertainty_penalty: float
    net_edge: float
    confidence: float
    target_position_fraction: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
