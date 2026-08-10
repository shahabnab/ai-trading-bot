from __future__ import annotations

from dataclasses import dataclass

import numpy as np


HOUR_MS = 60 * 60 * 1000


@dataclass(frozen=True)
class StandardizationStats:
    mean: np.ndarray
    scale: np.ndarray


def fit_standardizer(values: np.ndarray, *, epsilon: float = 1e-8) -> StandardizationStats:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("cannot fit a standardizer on an empty array")
    if not np.all(np.isfinite(array)):
        raise ValueError("standardizer input must be finite")

    mean = np.mean(array, axis=0)
    scale = np.std(array, axis=0, ddof=0)
    scale = np.where(scale < epsilon, 1.0, scale)
    return StandardizationStats(mean=np.asarray(mean), scale=np.asarray(scale))


def standardize(values: np.ndarray, stats: StandardizationStats) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return ((array - stats.mean) / stats.scale).astype(np.float32, copy=False)


def inverse_standardize(values: np.ndarray, stats: StandardizationStats) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return array * stats.scale + stats.mean


def make_sequence_batch(
    X: np.ndarray,
    y: np.ndarray,
    timestamps_ms: np.ndarray,
    target_indices: np.ndarray,
    *,
    sequence_length: int = 48,
    min_context_index: int = 0,
    expected_step_ms: int = HOUR_MS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build causal sequences ending at each target index.

    A sequence ending at row i uses X[i-sequence_length+1:i+1] to predict y[i].
    Since y[i] is the already-defined next-hour target for bar i, no future
    feature row is included. Timestamp continuity is enforced so missing bars do
    not silently turn into irregular LSTM time steps.
    """

    features = np.asarray(X)
    target = np.asarray(y)
    timestamps = np.asarray(timestamps_ms, dtype=np.int64)
    indices = np.asarray(target_indices, dtype=np.int64)

    if features.ndim != 2:
        raise ValueError("X must be a 2D feature matrix")
    if target.ndim != 1:
        raise ValueError("y must be a 1D target array")
    if len(features) != len(target) or len(features) != len(timestamps):
        raise ValueError("X, y, and timestamps must have identical row counts")
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if expected_step_ms <= 0:
        raise ValueError("expected_step_ms must be positive")
    if min_context_index < 0:
        raise ValueError("min_context_index must be non-negative")
    if indices.size and (np.min(indices) < 0 or np.max(indices) >= len(features)):
        raise IndexError("target index is outside the dataset")

    sequences: list[np.ndarray] = []
    targets: list[float] = []
    kept_indices: list[int] = []

    for target_index in indices:
        end = int(target_index) + 1
        start = end - sequence_length
        if start < min_context_index or start < 0:
            continue

        window_timestamps = timestamps[start:end]
        if len(window_timestamps) != sequence_length:
            continue
        if sequence_length > 1 and np.any(np.diff(window_timestamps) != expected_step_ms):
            continue

        sequences.append(features[start:end])
        targets.append(float(target[target_index]))
        kept_indices.append(int(target_index))

    if not sequences:
        return (
            np.empty((0, sequence_length, features.shape[1]), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    return (
        np.asarray(sequences, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        np.asarray(kept_indices, dtype=np.int64),
    )
