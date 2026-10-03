"""Sparse interaction identification for 2-additive capacities."""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from .capacity import TwoAdditiveCapacity
from .hesitant import criterion_scores


def association_from_matrix(matrix) -> np.ndarray:
    """Estimate criterion association from alternative-level HFE means.

    Spearman association is used because ordinal structure is often more stable
    than a linear correlation assumption in expert evaluation data.
    """
    scores = criterion_scores(matrix)
    n = scores.shape[1]
    if scores.shape[0] < 3:
        raise ValueError("at least three alternatives are required to estimate association")
    corr = np.eye(n, dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            r = spearmanr(scores[:, i], scores[:, j]).statistic
            if not np.isfinite(r):
                r = 0.0
            corr[i, j] = corr[j, i] = float(np.clip(r, -1.0, 1.0))
    return corr


def _soft_threshold(x, threshold):
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)


def sparse_capacity(importance, association, *, strength=0.15, threshold=0.25,
                    safety_factor=0.98):
    """Create a sparse monotone 2-additive capacity.

    The procedure keeps only associations whose magnitude exceeds ``threshold``.
    Pair interactions are shrunk and globally scaled as needed to satisfy the
    exact compact monotonicity condition of a 2-additive capacity while
    preserving the requested Shapley importance vector.

    Returns
    -------
    capacity, diagnostics
    """
    w = np.asarray(importance, dtype=float)
    a = np.asarray(association, dtype=float)
    if w.ndim != 1 or w.size == 0 or not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError("importance must be non-negative and non-zero")
    w = w / w.sum()
    n = w.size
    if a.shape != (n, n) or not np.isfinite(a).all():
        raise ValueError("association must be an n x n finite matrix")
    if not np.allclose(a, a.T, atol=1e-10):
        raise ValueError("association must be symmetric")
    if not (0 <= threshold < 1):
        raise ValueError("threshold must be in [0, 1)")
    if not np.isfinite(strength) or strength < 0:
        raise ValueError("strength must be non-negative")
    if not 0 < safety_factor <= 1:
        raise ValueError("safety_factor must be in (0, 1]")

    base = _soft_threshold(np.clip(a, -1.0, 1.0), threshold)
    np.fill_diagonal(base, 0.0)
    pair = strength * base

    # Preserve target Shapley importance. For scaled pair matrix gamma*pair,
    # m_i(gamma) = w_i - 0.5*gamma*sum_j pair_ij.
    # The minimum marginal is m_i + gamma*sum_{pair_ij<0} pair_ij.
    coeff = -0.5 * pair.sum(axis=1) + np.minimum(pair, 0.0).sum(axis=1)
    gamma_max = 1.0
    for wi, ci in zip(w, coeff):
        if ci < 0:
            gamma_max = min(gamma_max, wi / (-ci))
    gamma = float(min(1.0, safety_factor * gamma_max))
    scaled = pair * gamma
    m = w - 0.5 * scaled.sum(axis=1)
    capacity = TwoAdditiveCapacity(m, scaled)

    diagnostics = {
        "scale_factor": gamma,
        "retained_pairs": int(np.count_nonzero(np.triu(np.abs(scaled) > 0, 1))),
        "possible_pairs": int(n * (n - 1) // 2),
        "sparsity": float(1.0 - np.count_nonzero(np.triu(np.abs(scaled) > 0, 1)) / max(1, n * (n - 1) // 2)),
        "min_marginal": float((m + np.minimum(scaled, 0.0).sum(axis=1)).min()),
    }
    return capacity, diagnostics
