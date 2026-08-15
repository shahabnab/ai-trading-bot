from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

ACTIONS = ("NO_TRADE", "LONG", "EXIT")


@dataclass(frozen=True)
class BanditChoice:
    action: str
    policy_source: str
    trained_samples: int
    score: float | None
    action_scores: Mapping[str, float]


class LinUCBBandit:
    """Auditable contextual bandit trained from resolved paper/shadow rewards.

    It is intentionally much simpler than deep RL. The learner never sends orders
    directly; every chosen action still passes through the hard RiskManager.
    """

    def __init__(self, feature_count: int, *, alpha: float = 0.35, ridge: float = 1.0) -> None:
        self.feature_count = int(feature_count)
        self.alpha = float(alpha)
        self.ridge = float(ridge)
        self.A = {action: np.eye(self.feature_count, dtype=np.float64) * self.ridge for action in ACTIONS}
        self.b = {action: np.zeros(self.feature_count, dtype=np.float64) for action in ACTIONS}
        self.sample_count = 0

    def fit_shadow_samples(self, samples: Iterable[tuple[Sequence[float], Mapping[str, float]]]) -> "LinUCBBandit":
        for vector, rewards in samples:
            x = np.asarray(vector, dtype=np.float64).reshape(-1)
            if len(x) != self.feature_count or not np.all(np.isfinite(x)):
                continue
            any_reward = False
            for action, reward in rewards.items():
                if action not in ACTIONS or not np.isfinite(reward):
                    continue
                self.A[action] += np.outer(x, x)
                self.b[action] += x * float(reward)
                any_reward = True
            if any_reward:
                self.sample_count += 1
        return self

    def choose(
        self,
        vector: Sequence[float],
        *,
        valid_actions: Sequence[str],
        min_samples: int,
        fallback_action: str,
    ) -> BanditChoice:
        x = np.asarray(vector, dtype=np.float64).reshape(-1)
        if len(x) != self.feature_count or not np.all(np.isfinite(x)):
            return BanditChoice(fallback_action, "supervised_fallback_invalid_context", self.sample_count, None, {})
        valid = [action for action in valid_actions if action in ACTIONS]
        if not valid:
            return BanditChoice("NO_TRADE", "no_valid_action", self.sample_count, None, {})
        if self.sample_count < int(min_samples):
            action = fallback_action if fallback_action in valid else valid[0]
            return BanditChoice(action, "supervised_warmup", self.sample_count, None, {})
        scores: dict[str, float] = {}
        for action in valid:
            inv = np.linalg.inv(self.A[action])
            theta = inv @ self.b[action]
            mean = float(theta @ x)
            uncertainty = float(np.sqrt(max(x @ inv @ x, 0.0)))
            scores[action] = mean + self.alpha * uncertainty
        action = max(scores, key=scores.get)
        return BanditChoice(action, "linucb_shadow_reward", self.sample_count, scores[action], scores)
