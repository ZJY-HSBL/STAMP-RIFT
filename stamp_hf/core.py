"""Hesitant membership normalization and ideal-point distances."""
import numpy as np


def normalize(matrix, theta=0.0, benefit=None):
    """Pad each HFE using Eq. 11, sort, then complement cost attributes.

    Repeated padding values are retained; theta controls padding only.
    """
    if not np.isfinite(theta) or not 0 <= theta <= 1:
        raise ValueError('theta must be in [0, 1]')
    if not matrix or not matrix[0]:
        raise ValueError('matrix must be nonempty')
    n = len(matrix[0])
    if any(len(row) != n for row in matrix):
        raise ValueError('matrix must be rectangular')
    cells = []
    for row in matrix:
        for cell in row:
            a = np.asarray(cell, dtype=float)
            if a.ndim != 1 or a.size == 0 or not np.isfinite(a).all() or ((a < 0) | (a > 1)).any():
                raise ValueError('each HFE must be a nonempty finite vector in [0, 1]')
            cells.append(a)
    length = max(a.size for a in cells)
    padded = []
    for a in cells:
        fill = theta * a.max() + (1 - theta) * a.min()
        padded.append(np.sort(np.r_[a, np.repeat(fill, length - a.size)]))
    result = np.array(padded).reshape(len(matrix), n, length)
    benefit = [True] * n if benefit is None else benefit
    if len(benefit) != n or any(type(x) not in (bool, np.bool_) for x in benefit):
        raise ValueError('benefit must contain one boolean per attribute')
    for j, flag in enumerate(benefit):
        if not flag:
            result[:, j] = np.sort(1 - result[:, j], axis=1)
    return result


def distances(normalized):
    """Eqs. 12, corrected 13, 32 and 33; coordinate-wise extrema."""
    positive = normalized.max(axis=0)
    negative = normalized.min(axis=0)
    plus = np.sqrt(np.mean((normalized - positive) ** 2, axis=2))
    minus = np.sqrt(np.mean((normalized - negative) ** 2, axis=2))
    return positive, negative, plus, minus


def closeness(plus, minus):
    plus, minus = np.asarray(plus), np.asarray(minus)
    return np.divide(minus, plus + minus, out=np.full_like(minus, 0.5, dtype=float), where=plus + minus > 1e-15)


def evaluate(matrix, capacity, theta=0.0, benefit=None):
    normalized = normalize(matrix, theta, benefit)
    if normalized.shape[1] != capacity.n:
        raise ValueError('capacity dimension differs from matrix')
    positive, negative, dp, dm = distances(normalized)
    zp = np.array([capacity.integral(row) for row in dp])
    zm = np.array([capacity.integral(row) for row in dm])
    scores = closeness(zp, zm)
    return dict(normalized=normalized, positive=positive, negative=negative,
                distance_positive=dp, distance_negative=dm,
                z_positive=zp, z_negative=zm, scores=scores,
                order=np.argsort(-scores, kind='stable'))


def independent_topsis(matrix, weights, theta=0.0, benefit=None):
    """Eqs. 14-15, weighted RMS, distinct from integral of per-attribute RMS."""
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError('invalid weights')
    x = normalize(matrix, theta, benefit)
    if x.shape[1] != w.size:
        raise ValueError('weight dimension mismatch')
    _, _, dp, dm = distances(x)
    w = w / w.sum()
    return closeness(np.sqrt(dp**2 @ w), np.sqrt(dm**2 @ w))


def compromise_baseline(matrix, weights, theta=0.0, benefit=None, v=0.5):
    """Mean-HFE VIKOR-style Q baseline, smaller is better; not Table 10 recreation."""
    if not 0 <= v <= 1:
        raise ValueError('v must be in [0, 1]')
    x = normalize(matrix, theta, benefit).mean(axis=2)
    w = np.asarray(weights, dtype=float)
    if w.shape != (x.shape[1],) or not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError('invalid weights')
    span = x.max(axis=0) - x.min(axis=0)
    loss = np.divide(x.max(axis=0) - x, span, out=np.zeros_like(x), where=span > 0) * (w / w.sum())
    total, worst = loss.sum(axis=1), loss.max(axis=1)
    def scale(a):
        return (a-a.min()) / np.ptp(a) if np.ptp(a) > 0 else np.zeros_like(a)
    return v * scale(total) + (1-v) * scale(worst)
