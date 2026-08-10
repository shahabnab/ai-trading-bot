import numpy as np

from backend.ml.sequences import fit_standardizer, make_sequence_batch, standardize


HOUR_MS = 60 * 60 * 1000


def test_standardizer_uses_supplied_training_rows_only() -> None:
    train = np.asarray([[1.0, 10.0], [3.0, 14.0], [5.0, 18.0]], dtype=np.float32)
    future = np.asarray([[1000.0, -500.0]], dtype=np.float32)

    stats = fit_standardizer(train)
    transformed = standardize(np.vstack([train, future]), stats)

    assert np.allclose(stats.mean, [3.0, 14.0])
    assert np.allclose(np.mean(transformed[:3], axis=0), [0.0, 0.0], atol=1e-6)
    assert np.max(np.abs(transformed[3])) > 10.0


def test_sequence_batch_is_causal_and_respects_context_boundary() -> None:
    X = np.arange(20, dtype=np.float32).reshape(10, 2)
    y = np.arange(10, dtype=np.float32)
    timestamps = np.arange(10, dtype=np.int64) * HOUR_MS
    target_indices = np.asarray([2, 3, 4, 5], dtype=np.int64)

    sequences, targets, kept = make_sequence_batch(
        X,
        y,
        timestamps,
        target_indices,
        sequence_length=3,
        min_context_index=2,
    )

    assert kept.tolist() == [4, 5]
    assert np.array_equal(sequences[0], X[2:5])
    assert targets.tolist() == [4.0, 5.0]
    assert np.max(sequences[0]) == X[4].max()


def test_sequence_batch_drops_windows_across_missing_hours() -> None:
    X = np.arange(12, dtype=np.float32).reshape(6, 2)
    y = np.arange(6, dtype=np.float32)
    timestamps = np.asarray([0, 1, 2, 4, 5, 6], dtype=np.int64) * HOUR_MS

    sequences, _, kept = make_sequence_batch(
        X,
        y,
        timestamps,
        np.arange(6),
        sequence_length=3,
    )

    assert kept.tolist() == [2, 5]
    assert len(sequences) == 2
