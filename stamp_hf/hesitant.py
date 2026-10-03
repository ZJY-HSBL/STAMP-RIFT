"""Utilities for hesitant fuzzy elements (HFEs)."""
from __future__ import annotations

import numpy as np


def validate_hfe(cell) -> np.ndarray:
    """Return an HFE as a validated 1-D float array in [0, 1]."""
    a = np.asarray(cell, dtype=float)
    if a.ndim != 1 or a.size == 0:
        raise ValueError("each HFE must be a non-empty one-dimensional vector")
    if not np.isfinite(a).all() or ((a < 0) | (a > 1)).any():
        raise ValueError("HFE membership values must be finite and lie in [0, 1]")
    return a


def hesitation_index(cell) -> float:
    """Normalized variance-based hesitation index in [0, 1].

    Since an HFE lies in [0, 1], its population variance is at most 0.25.
    Multiplication by four therefore maps the variance to [0, 1].
    """
    a = validate_hfe(cell)
    if a.size == 1:
        return 0.0
    return float(np.clip(4.0 * np.var(a), 0.0, 1.0))


def adaptive_theta(cell, risk_preference=0.0, beta=0.5) -> float:
    """Compute a cell-specific completion coefficient.

    risk_preference is in [-1, 1]. Positive values are more conservative.
    beta >= 0 controls how much hesitation increases effective risk aversion.
    The returned theta is in [0, 1], where lower theta pads closer to min(HFE).
    """
    if not np.isfinite(risk_preference) or not -1 <= risk_preference <= 1:
        raise ValueError("risk_preference must be in [-1, 1]")
    if not np.isfinite(beta) or beta < 0:
        raise ValueError("beta must be non-negative")
    effective_risk = np.clip(risk_preference + beta * hesitation_index(cell), -1.0, 1.0)
    return float((1.0 - effective_risk) / 2.0)


def normalize_matrix(matrix, *, benefit=None, completion="adaptive", theta=0.5,
                     risk_preference=0.0, beta=0.5) -> np.ndarray:
    """Pad HFEs to a common length, sort them, and transform cost criteria.

    Parameters
    ----------
    completion:
        ``"fixed"`` uses the same theta for every HFE.
        ``"adaptive"`` uses :func:`adaptive_theta` for each HFE.
    """
    if not matrix or not matrix[0]:
        raise ValueError("matrix must be non-empty")
    n_criteria = len(matrix[0])
    if any(len(row) != n_criteria for row in matrix):
        raise ValueError("matrix must be rectangular")

    arrays = [[validate_hfe(cell) for cell in row] for row in matrix]
    max_len = max(cell.size for row in arrays for cell in row)

    if completion not in {"fixed", "adaptive"}:
        raise ValueError("completion must be 'fixed' or 'adaptive'")
    if completion == "fixed" and (not np.isfinite(theta) or not 0 <= theta <= 1):
        raise ValueError("theta must be in [0, 1]")

    output = np.empty((len(arrays), n_criteria, max_len), dtype=float)
    for i, row in enumerate(arrays):
        for j, cell in enumerate(row):
            local_theta = theta if completion == "fixed" else adaptive_theta(
                cell, risk_preference=risk_preference, beta=beta
            )
            fill = local_theta * cell.max() + (1.0 - local_theta) * cell.min()
            padded = np.r_[cell, np.repeat(fill, max_len - cell.size)]
            output[i, j] = np.sort(padded)

    benefit = [True] * n_criteria if benefit is None else benefit
    if len(benefit) != n_criteria or any(type(v) not in (bool, np.bool_) for v in benefit):
        raise ValueError("benefit must contain one boolean per criterion")
    for j, is_benefit in enumerate(benefit):
        if not is_benefit:
            output[:, j] = np.sort(1.0 - output[:, j], axis=1)
    return output


def criterion_scores(matrix) -> np.ndarray:
    """Return alternatives x criteria mean membership scores without padding."""
    if not matrix or not matrix[0]:
        raise ValueError("matrix must be non-empty")
    n = len(matrix[0])
    if any(len(row) != n for row in matrix):
        raise ValueError("matrix must be rectangular")
    return np.asarray([[validate_hfe(cell).mean() for cell in row] for row in matrix], dtype=float)
