from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from backend.ml.evaluation import classification_metrics


@dataclass(frozen=True)
class BootstrapCI:
    estimate: float
    lower: float
    upper: float
    block_length: int
    samples: int
    valid_samples: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "estimate": self.estimate,
            "lower": self.lower,
            "upper": self.upper,
            "block_length": self.block_length,
            "samples": self.samples,
            "valid_samples": self.valid_samples,
        }


def block_bootstrap_auc_ci(
    labels: np.ndarray,
    probability: np.ndarray,
    *,
    block_length: int,
    samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> BootstrapCI:
    """Moving-block bootstrap confidence interval for AUC.

    Blocks preserve local serial dependence instead of pretending hourly market
    observations are independent. V3 reports both 24-hour and 168-hour blocks.
    """
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    p = np.asarray(probability, dtype=np.float64).reshape(-1)
    if y.shape != p.shape or y.size == 0:
        raise ValueError("labels and probability must be identical non-empty arrays")
    if block_length <= 0:
        raise ValueError("block_length must be positive")
    if samples <= 0:
        raise ValueError("samples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")

    estimate = float(classification_metrics(y, p)["auc"])
    n = len(y)
    block = min(block_length, n)
    starts = np.arange(0, n - block + 1, dtype=np.int64)
    rng = np.random.default_rng(seed)
    aucs: list[float] = []
    blocks_needed = int(math.ceil(n / block))

    for _ in range(samples):
        chosen = rng.choice(starts, size=blocks_needed, replace=True)
        indices = np.concatenate(
            [np.arange(start, start + block, dtype=np.int64) for start in chosen]
        )[:n]
        yy = y[indices]
        if np.all(yy == yy[0]):
            continue
        pp = p[indices]
        aucs.append(float(classification_metrics(yy, pp)["auc"]))

    if len(aucs) < max(50, samples // 10):
        raise ValueError("too few valid bootstrap samples to estimate AUC interval")
    alpha = (1.0 - confidence) / 2.0
    array = np.asarray(aucs, dtype=np.float64)
    return BootstrapCI(
        estimate=estimate,
        lower=float(np.quantile(array, alpha)),
        upper=float(np.quantile(array, 1.0 - alpha)),
        block_length=int(block),
        samples=int(samples),
        valid_samples=int(len(array)),
    )
