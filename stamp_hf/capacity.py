"""Fuzzy capacities used by the research framework."""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


class LambdaCapacity:
    """Normalized Sugeno lambda capacity derived from singleton measures."""

    def __init__(self, singletons):
        s = np.asarray(singletons, dtype=float)
        if s.ndim != 1 or s.size == 0 or not np.isfinite(s).all() or (s <= 0).any() or (s > 1).any():
            raise ValueError("singletons must lie in (0, 1]")
        self.singletons = s.copy()
        self.n = s.size
        total = float(s.sum())
        if np.isclose(total, 1.0, atol=1e-12, rtol=0):
            self.lam = 0.0
        elif self.n == 1 or (s >= 1).any():
            raise ValueError("no admissible normalized lambda exists")
        else:
            def residual(lam):
                if abs(lam) < 1e-14:
                    return total - 1.0
                return (np.log1p(lam * s).sum() - np.log1p(lam)) / lam

            if total < 1:
                hi = 1.0
                while residual(hi) <= 0:
                    hi *= 2.0
                    if hi > 1e15:
                        raise ValueError("normalization root exceeds supported range")
                self.lam = float(brentq(residual, 0.0, hi, xtol=1e-13))
            else:
                lo = np.nextafter(-1.0, 0.0)
                if residual(lo) >= 0:
                    raise ValueError("lambda root is numerically too close to -1")
                self.lam = float(brentq(residual, lo, 0.0, xtol=1e-14))
        if not np.isclose(self.value(range(self.n)), 1.0, atol=1e-9):
            raise ValueError("capacity normalization failed")

    def value(self, indices) -> float:
        idx = sorted(set(int(i) for i in indices))
        if any(i < 0 or i >= self.n for i in idx):
            raise ValueError("invalid subset index")
        if not idx:
            return 0.0
        s = self.singletons[idx]
        if self.lam == 0:
            return float(s.sum())
        return float(np.expm1(np.log1p(self.lam * s).sum()) / self.lam)

    def integral(self, values) -> float:
        """Choquet integral for a normalized lambda capacity."""
        x = np.asarray(values, dtype=float)
        if x.shape != (self.n,) or not np.isfinite(x).all() or (x < 0).any():
            raise ValueError("integrand must be a finite non-negative vector")
        order = np.argsort(x, kind="stable")
        previous = 0.0
        total = 0.0
        for k, idx in enumerate(order):
            total += (x[idx] - previous) * self.value(order[k:])
            previous = x[idx]
        return float(total)


class TwoAdditiveCapacity:
    """Normalized monotone 2-additive capacity in Möbius representation.

    ``singletons`` are Möbius singleton coefficients m_i and ``interactions``
    are symmetric pair coefficients m_ij. The Choquet integral has the efficient
    form sum(m_i*x_i) + sum(m_ij*min(x_i, x_j)).
    """

    def __init__(self, singletons, interactions):
        m = np.asarray(singletons, dtype=float)
        pair = np.asarray(interactions, dtype=float)
        if m.ndim != 1 or m.size == 0 or not np.isfinite(m).all():
            raise ValueError("singletons must be a finite vector")
        n = m.size
        if pair.shape != (n, n) or not np.isfinite(pair).all():
            raise ValueError("interactions must be an n x n finite matrix")
        if not np.allclose(pair, pair.T, atol=1e-12) or not np.allclose(np.diag(pair), 0.0, atol=1e-12):
            raise ValueError("interactions must be symmetric with a zero diagonal")
        self.singletons = m.copy()
        self.interactions = pair.copy()
        self.n = n
        mass = m.sum() + np.triu(pair, 1).sum()
        if not np.isclose(mass, 1.0, atol=1e-9):
            raise ValueError("2-additive capacity must be normalized to one")
        # For a 2-additive capacity the minimum marginal contribution of i is
        # m_i plus all negative pair terms incident to i.
        min_marginal = m + np.minimum(pair, 0.0).sum(axis=1)
        if (min_marginal < -1e-10).any():
            raise ValueError("interactions violate capacity monotonicity")

    @classmethod
    def from_shapley_and_interactions(cls, importance, interactions):
        """Build a normalized capacity from target Shapley importance values."""
        w = np.asarray(importance, dtype=float)
        pair = np.asarray(interactions, dtype=float)
        if w.ndim != 1 or w.size == 0 or not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
            raise ValueError("importance must be a non-negative finite vector")
        w = w / w.sum()
        if pair.shape != (w.size, w.size):
            raise ValueError("interaction dimension mismatch")
        pair = (pair + pair.T) / 2.0
        np.fill_diagonal(pair, 0.0)
        m = w - 0.5 * pair.sum(axis=1)
        return cls(m, pair)

    def value(self, indices) -> float:
        idx = sorted(set(int(i) for i in indices))
        if any(i < 0 or i >= self.n for i in idx):
            raise ValueError("invalid subset index")
        if not idx:
            return 0.0
        return float(self.singletons[idx].sum() + np.triu(self.interactions[np.ix_(idx, idx)], 1).sum())

    def shapley_importance(self) -> np.ndarray:
        return self.singletons + 0.5 * self.interactions.sum(axis=1)

    def interaction_index(self) -> np.ndarray:
        return self.interactions.copy()

    def integral(self, values) -> float:
        x = np.asarray(values, dtype=float)
        if x.shape != (self.n,) or not np.isfinite(x).all() or (x < 0).any():
            raise ValueError("integrand must be a finite non-negative vector")
        total = float(self.singletons @ x)
        ii, jj = np.triu_indices(self.n, 1)
        total += float(np.sum(self.interactions[ii, jj] * np.minimum(x[ii], x[jj])))
        return total
