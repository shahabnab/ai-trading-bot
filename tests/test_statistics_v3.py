import numpy as np

from backend.ml.statistics import block_bootstrap_auc_ci


def test_block_bootstrap_auc_ci_detects_rank_signal():
    labels = np.asarray(([0] * 12 + [1] * 12) * 20, dtype=float)
    probability = np.asarray(([0.2] * 12 + [0.8] * 12) * 20, dtype=float)
    ci = block_bootstrap_auc_ci(labels, probability, block_length=24, samples=200, seed=7)
    assert ci.estimate > 0.9
    assert ci.lower > 0.5
    assert ci.valid_samples > 50
