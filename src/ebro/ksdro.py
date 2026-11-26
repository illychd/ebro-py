"""Kolmogorov-Smirnov (and Levy-KS) distributionally robust baseline in 1-D.

The ambiguity set is {F : F_n(x - a) - eps <= F(x) <= F_n(x + a) + eps} on
X = [lo, hi].  For a loss that is quasiconvex in theta the worst case is
attained by the clipped CDF F_c = clip(c, F_low, F_high) for some level c,
which we maximize over a finite set of candidate levels (validated against
the linear program in the test-suite).
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize_scalar, linprog


def _support(samples, a, lo, hi, extra=()):
    p = np.concatenate([[lo, hi], np.asarray(extra, float), samples + a, samples - a])
    p = np.clip(p, lo, hi)
    return np.unique(np.concatenate([p, np.clip(p - 1e-9, lo, hi)]))


def _band(samples, eps, a, pts, psi=None):
    """Lower/upper CDF bounds at the support points.

    With ``psi`` (array psi[j] = psi(j/n)) the band is F(p) <= psi(#{x_i <= p+a}/n) and
    F(p) >= 1 - psi(#{x_i > p-a}/n); without it, the DKW band of half-width eps.
    """
    s = np.sort(samples); n = s.size
    cnt_le = np.searchsorted(s, pts + a, side="right")
    cnt_gt = n - np.searchsorted(s, pts - a, side="right")
    if psi is None:
        psi = np.minimum(1.0, np.arange(n + 1) / n + eps)
    high = np.minimum(1.0, psi[cnt_le])
    low = np.maximum(0.0, 1.0 - psi[cnt_gt])
    low[-1] = high[-1] = 1.0
    return low, high


def ks_worst_case(f, samples, eps, domain, a=0.0, extra=(), method="clip", psi=None):
    """sup of E_F f over the (Levy-)KS band, or over a general band given by ``psi``.

    ``extra`` adds support points (e.g. kinks of f).
    """
    samples = np.asarray(samples, float)
    lo, hi = domain
    p = _support(samples, a, lo, hi, extra)
    low, high = _band(samples, eps, a, p, psi)
    fv = f(p)
    if method == "lp":
        m = p.size
        c = np.zeros(m); c[:-1] -= fv[:-1] - fv[1:]; c[-1] -= fv[-1]
        A = np.zeros((m - 1, m)); A[np.arange(m - 1), np.arange(m - 1)] = 1
        A[np.arange(m - 1), np.arange(1, m)] = -1
        r = linprog(c, A_ub=A, b_ub=np.zeros(m - 1), bounds=list(zip(low, high)), method="highs")
        return float(-r.fun)
    levels = np.unique(np.concatenate([low, high, [0.0, 1.0]]))

    def value(c):
        F = np.maximum.accumulate(np.clip(c, low, high)); F[-1] = 1.0
        return float(np.diff(np.concatenate([[0.0], F])) @ fv)

    if method == "full":
        return max(value(c) for c in levels)
    # value(c) is concave in the level c for quasiconvex f: ternary search on the sorted levels,
    # then exact evaluation on the final bracket (validated against "full" and "lp" in the tests).
    i, j = 0, levels.size - 1
    while j - i > 6:
        m1 = i + (j - i) // 3; m2 = j - (j - i) // 3
        if value(levels[m1]) < value(levels[m2]): i = m1
        else: j = m2
    return max(value(c) for c in levels[i:j + 1])


def solve_scalar_ks(loss, samples, eps, domain, bounds, a=0.0, grid=201, kinks=None):
    """Minimize the KS worst-case risk over a scalar design (convex in z)."""
    obj = lambda z: ks_worst_case(lambda th: loss(z, th), samples, eps, domain, a,
                                  extra=[z] if kinks is None else kinks(z))
    zs = np.linspace(bounds[0], bounds[1], grid)
    vals = np.array([obj(z) for z in zs])
    i = int(np.argmin(vals))
    lo_, hi_ = zs[max(i - 1, 0)], zs[min(i + 1, grid - 1)]
    r = minimize_scalar(obj, bounds=(lo_, hi_), method="bounded", options={"xatol": 1e-8})
    z = float(r.x) if r.fun <= vals[i] else float(zs[i])
    return z, obj(z)
