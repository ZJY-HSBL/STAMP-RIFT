"""Normalized capacities and the maximum-decomposition integral of Eq. 10."""
import numpy as np
from scipy.optimize import brentq, linprog


def _vector(values, n):
    a = np.asarray(values, dtype=float)
    if a.shape != (n,) or not np.isfinite(a).all() or (a < 0).any():
        raise ValueError('integrand must be a finite nonnegative vector')
    return a


def chain_integral(values, capacity):
    """Exact maximum-decomposition integral for a supermodular capacity."""
    f = _vector(values, capacity.n)
    order = np.argsort(f, kind='stable')
    previous, result = 0.0, 0.0
    for k, index in enumerate(order):
        result += (f[index] - previous) * capacity.value(order[k:])
        previous = f[index]
    return float(result)


def decomposition_lp(values, capacity, max_attributes=14):
    """Exact Eq. 10 reference solver, 2**n-1 variables; refuse huge allocations."""
    f = _vector(values, capacity.n)
    n = capacity.n
    if n > max_attributes:
        raise ValueError(f'exhaustive LP is limited to {max_attributes} attributes')
    masks = np.arange(1, 2**n, dtype=np.int64)
    incidence = ((masks[None, :] >> np.arange(n)[:, None]) & 1).astype(float)
    objective = np.array([capacity.value(np.flatnonzero(incidence[:, k])) for k in range(masks.size)])
    result = linprog(-objective, A_eq=incidence, b_eq=f, bounds=(0, None), method='highs')
    if not result.success:
        raise RuntimeError(result.message)
    return float(-result.fun)


class LambdaCapacity:
    """Eq. 6; solve the nonzero normalization root without selecting spurious zero."""
    def __init__(self, singletons):
        s = np.asarray(singletons, dtype=float)
        if s.ndim != 1 or s.size == 0 or not np.isfinite(s).all() or (s <= 0).any() or (s > 1).any():
            raise ValueError('singletons must be in (0, 1]')
        self.singletons, self.n = s.copy(), s.size
        total = s.sum()
        if abs(total-1) < 1e-12:
            self.lam = 0.0
        elif s.size == 1 or (s >= 1).any():
            raise ValueError('no admissible normalized lambda for these singletons')
        else:
            def residual(lam):
                if abs(lam) < 1e-14:
                    return total-1
                return (np.log1p(lam*s).sum() - np.log1p(lam)) / lam
            if total < 1:
                hi = 1.0
                while residual(hi) <= 0:
                    hi *= 2
                    if hi > 1e15:
                        raise ValueError('normalization root exceeds supported range')
                self.lam = brentq(residual, 0, hi, xtol=1e-13)
            else:
                lo = np.nextafter(-1.0, 0.0)
                if residual(lo) >= 0:
                    raise ValueError('root too close to -1 for floating-point arithmetic')
                self.lam = brentq(residual, lo, 0, xtol=1e-14)
        if not np.isclose(self.value(range(self.n)), 1, atol=1e-9):
            raise ValueError('capacity normalization failed')

    def value(self, indices):
        ix = sorted(set(int(i) for i in indices))
        if any(i < 0 or i >= self.n for i in ix):
            raise ValueError('invalid subset index')
        if not ix:
            return 0.0
        s = self.singletons[ix]
        if self.lam == 0:
            return float(s.sum())
        return float(np.expm1(np.log1p(self.lam*s).sum()) / self.lam)

    def integral(self, values):
        f = _vector(values, self.n)
        if self.lam <= 0:
            # Subadditivity bounds every subset by its singleton sum; attained
            # by the singleton-only decomposition. Do not use a chain here.
            return float(f @ self.singletons)
        return chain_integral(f, self)


class TwoAdditiveCapacity:
    """Separate 2-additive model; importance values plus symmetric interactions.

    The case-study lambda capacity is NOT generally 2-additive.
    """
    def __init__(self, importance, interactions):
        w = np.asarray(importance, dtype=float)
        pair = np.asarray(interactions, dtype=float)
        if w.ndim != 1 or w.size == 0 or not np.isfinite(w).all() or (w < 0).any() or not np.isclose(w.sum(), 1, atol=1e-10, rtol=0):
            raise ValueError('importance must be nonnegative and sum to one')
        n = w.size
        if pair.shape != (n,n) or not np.isfinite(pair).all() or not np.allclose(pair, pair.T) or not np.allclose(pair.diagonal(), 0):
            raise ValueError('interactions must be finite symmetric with zero diagonal')
        pair = (pair + pair.T) / 2
        np.fill_diagonal(pair, 0)
        self.n, self.pair = n, pair
        self.singletons = w - 0.5 * pair.sum(axis=1)
        # Minimum marginal over all subsets is m_i + sum of negative m_ij.
        if (self.singletons + np.minimum(pair, 0).sum(axis=1) < -1e-12).any():
            raise ValueError('interactions violate capacity monotonicity')

    def value(self, indices):
        ix = sorted(set(int(i) for i in indices))
        if any(i < 0 or i >= self.n for i in ix):
            raise ValueError('invalid subset index')
        return float(self.singletons[ix].sum() + self.pair[np.ix_(ix, ix)].sum()/2)

    def integral(self, values):
        f = _vector(values, self.n)
        if (self.pair >= 0).all():
            return chain_integral(f, self)
        if (self.pair <= 0).all():
            return float(f @ self.singletons)
        return decomposition_lp(f, self)
