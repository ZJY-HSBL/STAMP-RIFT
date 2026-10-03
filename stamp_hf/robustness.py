"""Monte Carlo uncertainty propagation and robust ranking."""
from __future__ import annotations

import numpy as np

from .core import evaluate
from .interaction import sparse_capacity


def _perturb_matrix(matrix, rng, sigma):
    out = []
    for row in matrix:
        new_row = []
        for cell in row:
            a = np.asarray(cell, dtype=float)
            noise = rng.normal(0.0, sigma, size=a.shape)
            new_row.append(np.clip(a + noise, 0.0, 1.0).tolist())
        out.append(new_row)
    return out


def robust_evaluate(matrix, importance, association, *, benefit=None,
                    risk_preference=0.0, beta=0.5, interaction_strength=0.15,
                    interaction_threshold=0.25, n_iter=1000, membership_sigma=0.02,
                    weight_sigma=0.05, risk_sigma=0.05, association_sigma=0.03,
                    seed=2026):
    """Estimate score intervals and rank acceptability under uncertainty."""
    if n_iter < 20:
        raise ValueError("n_iter must be at least 20")
    for name, value in {
        "membership_sigma": membership_sigma,
        "weight_sigma": weight_sigma,
        "risk_sigma": risk_sigma,
        "association_sigma": association_sigma,
    }.items():
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be non-negative")

    w0 = np.asarray(importance, dtype=float)
    if w0.ndim != 1 or (w0 < 0).any() or w0.sum() <= 0 or not np.isfinite(w0).all():
        raise ValueError("invalid importance")
    w0 = w0 / w0.sum()
    assoc0 = np.asarray(association, dtype=float)
    n_alt = len(matrix)
    rng = np.random.default_rng(seed)

    scores = np.empty((n_iter, n_alt), dtype=float)
    ranks = np.empty((n_iter, n_alt), dtype=int)

    for b in range(n_iter):
        perturbed = _perturb_matrix(matrix, rng, membership_sigma)
        log_w = np.log(np.clip(w0, 1e-12, None)) + rng.normal(0.0, weight_sigma, size=w0.size)
        w = np.exp(log_w - log_w.max())
        w /= w.sum()
        assoc = assoc0 + rng.normal(0.0, association_sigma, size=assoc0.shape)
        assoc = (assoc + assoc.T) / 2.0
        assoc = np.clip(assoc, -1.0, 1.0)
        np.fill_diagonal(assoc, 1.0)
        cap, _ = sparse_capacity(
            w,
            assoc,
            strength=interaction_strength,
            threshold=interaction_threshold,
        )
        risk = float(np.clip(risk_preference + rng.normal(0.0, risk_sigma), -1.0, 1.0))
        result = evaluate(
            perturbed,
            cap,
            benefit=benefit,
            completion="adaptive",
            risk_preference=risk,
            beta=beta,
        )
        scores[b] = result["scores"]
        order = result["order"]
        inv = np.empty(n_alt, dtype=int)
        inv[order] = np.arange(1, n_alt + 1)
        ranks[b] = inv

    acceptability = np.zeros((n_alt, n_alt), dtype=float)
    for i in range(n_alt):
        for r in range(1, n_alt + 1):
            acceptability[i, r - 1] = np.mean(ranks[:, i] == r)

    superiority = np.zeros((n_alt, n_alt), dtype=float)
    for i in range(n_alt):
        for j in range(n_alt):
            superiority[i, j] = 0.5 if i == j else np.mean(scores[:, i] > scores[:, j])

    return {
        "mean_score": scores.mean(axis=0),
        "score_ci95_low": np.quantile(scores, 0.025, axis=0),
        "score_ci95_high": np.quantile(scores, 0.975, axis=0),
        "mean_rank": ranks.mean(axis=0),
        "rank_acceptability": acceptability,
        "pairwise_superiority": superiority,
        "score_samples": scores,
        "rank_samples": ranks,
    }
