from __future__ import annotations

import numpy as np

from .contracts import RegimePosterior


class PersistentGaussianRegimeDetector:
    """GMM emissions + empirical Markov persistence with forward-only filtering.

    The class intentionally exposes no full-history smoother. Live/posterior features are
    calculated only from observations available through t, avoiding future-state leakage.
    """

    def __init__(self, n_regimes: int = 5, random_state: int = 42, persistence_prior: float = 8.0) -> None:
        if n_regimes < 2:
            raise ValueError("n_regimes must be >= 2")
        self.n_regimes = int(n_regimes)
        self.random_state = int(random_state)
        self.persistence_prior = float(persistence_prior)
        self.scaler = None
        self.model = None
        self.transition: np.ndarray | None = None
        self.order: np.ndarray | None = None

    @staticmethod
    def _entropy(probabilities: np.ndarray) -> float:
        p = np.clip(probabilities, 1e-12, 1.0)
        raw = -float(np.sum(p * np.log(p)))
        return raw / float(np.log(len(p))) if len(p) > 1 else 0.0

    def fit(self, X: np.ndarray) -> "PersistentGaussianRegimeDetector":
        from sklearn.mixture import GaussianMixture
        from sklearn.preprocessing import StandardScaler

        matrix = np.asarray(X, dtype=np.float64)
        if matrix.ndim != 2 or len(matrix) < max(80, self.n_regimes * 12):
            raise ValueError("regime detector needs a 2-D history with enough observations")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("regime matrix contains non-finite values")
        self.scaler = StandardScaler().fit(matrix)
        scaled = self.scaler.transform(matrix)
        model = GaussianMixture(
            n_components=self.n_regimes,
            covariance_type="full",
            reg_covar=1e-5,
            n_init=8,
            max_iter=400,
            random_state=self.random_state,
        ).fit(scaled)
        raw_probs = model.predict_proba(scaled)
        states = np.argmax(raw_probs, axis=1)

        # Unsupervised component IDs can permute on every refit. Canonicalize by the
        # first regime feature (BTC 24h return) so dashboards/experience columns remain stable.
        centers_original = self.scaler.inverse_transform(model.means_)
        order = np.argsort(centers_original[:, 0])
        inverse = np.empty_like(order)
        inverse[order] = np.arange(len(order))
        canonical_states = inverse[states]

        transition = np.ones((self.n_regimes, self.n_regimes), dtype=np.float64)
        transition += np.eye(self.n_regimes, dtype=np.float64) * self.persistence_prior
        for left, right in zip(canonical_states[:-1], canonical_states[1:], strict=True):
            transition[int(left), int(right)] += 1.0
        transition /= transition.sum(axis=1, keepdims=True)

        self.model = model
        self.order = order
        self.transition = transition
        return self

    def _emissions(self, X: np.ndarray) -> np.ndarray:
        if self.model is None or self.scaler is None or self.order is None:
            raise RuntimeError("regime detector is not fitted")
        matrix = np.asarray(X, dtype=np.float64)
        raw = self.model.predict_proba(self.scaler.transform(matrix))
        return raw[:, self.order]

    def filtered_probabilities(self, X: np.ndarray) -> np.ndarray:
        if self.transition is None:
            raise RuntimeError("regime detector is not fitted")
        emissions = self._emissions(X)
        filtered = np.zeros_like(emissions)
        prior = np.full(self.n_regimes, 1.0 / self.n_regimes, dtype=np.float64)
        for idx, likelihood in enumerate(emissions):
            predicted = prior if idx == 0 else filtered[idx - 1] @ self.transition
            posterior = np.clip(predicted * np.clip(likelihood, 1e-12, 1.0), 1e-15, None)
            posterior /= posterior.sum()
            filtered[idx] = posterior
        return filtered

    def predict_filtered(self, X: np.ndarray, *, timestamp: int) -> RegimePosterior:
        probabilities = self.filtered_probabilities(X)[-1]
        return RegimePosterior(
            probabilities=tuple(float(value) for value in probabilities),
            entropy=self._entropy(probabilities),
            labels=tuple(f"state_{idx}" for idx in range(self.n_regimes)),
            method="gmm_markov_forward_filter",
            quality=1.0,
            timestamp=int(timestamp),
        )

    def fit_predict(self, X: np.ndarray, *, timestamp: int) -> RegimePosterior:
        return self.fit(X).predict_filtered(X, timestamp=timestamp)
