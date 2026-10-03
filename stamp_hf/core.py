"""TOPSIS-style evaluation with hesitant fuzzy information and capacities."""
from __future__ import annotations

import numpy as np

from .hesitant import normalize_matrix


def ideal_distances(normalized):
    """Return positive/negative ideals and per-criterion RMS distances."""
    x = np.asarray(normalized, dtype=float)
    if x.ndim != 3 or not np.isfinite(x).all():
        raise ValueError("normalized matrix must be a finite 3-D array")
    positive = x.max(axis=0)
    negative = x.min(axis=0)
    d_plus = np.sqrt(np.mean((x - positive) ** 2, axis=2))
    d_minus = np.sqrt(np.mean((x - negative) ** 2, axis=2))
    return positive, negative, d_plus, d_minus


def closeness(d_plus, d_minus):
    a = np.asarray(d_plus, dtype=float)
    b = np.asarray(d_minus, dtype=float)
    if a.shape != b.shape:
        raise ValueError("distance arrays must have identical shapes")
    return np.divide(b, a + b, out=np.full_like(b, 0.5, dtype=float), where=(a + b) > 1e-15)


def evaluate(matrix, capacity, *, benefit=None, completion="adaptive", theta=0.5,
             risk_preference=0.0, beta=0.5):
    """Evaluate alternatives using capacity-aggregated ideal distances."""
    normalized = normalize_matrix(
        matrix,
        benefit=benefit,
        completion=completion,
        theta=theta,
        risk_preference=risk_preference,
        beta=beta,
    )
    if normalized.shape[1] != capacity.n:
        raise ValueError("capacity dimension differs from matrix criteria")
    positive, negative, d_plus, d_minus = ideal_distances(normalized)
    z_plus = np.asarray([capacity.integral(row) for row in d_plus])
    z_minus = np.asarray([capacity.integral(row) for row in d_minus])
    scores = closeness(z_plus, z_minus)
    return {
        "normalized": normalized,
        "positive": positive,
        "negative": negative,
        "distance_positive": d_plus,
        "distance_negative": d_minus,
        "z_positive": z_plus,
        "z_negative": z_minus,
        "scores": scores,
        "order": np.argsort(-scores, kind="stable"),
    }


def independent_topsis(matrix, weights, *, benefit=None, completion="adaptive", theta=0.5,
                        risk_preference=0.0, beta=0.5):
    """Independent weighted RMS comparison model without criterion interactions."""
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError("invalid weights")
    x = normalize_matrix(
        matrix,
        benefit=benefit,
        completion=completion,
        theta=theta,
        risk_preference=risk_preference,
        beta=beta,
    )
    if x.shape[1] != w.size:
        raise ValueError("weight dimension mismatch")
    _, _, d_plus, d_minus = ideal_distances(x)
    w = w / w.sum()
    agg_plus = np.sqrt((d_plus ** 2) @ w)
    agg_minus = np.sqrt((d_minus ** 2) @ w)
    return closeness(agg_plus, agg_minus)
