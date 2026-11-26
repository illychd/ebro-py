"""Robust (upper) expectations after endographic dilation and classical discountings.

For a possibility distribution u with cut maxima g(t) = max_{(u_t)^a} f
(spatial dilation by a), the three upper expectations at level b are

    add (endograph, D_{a,b}):  b*max_X f + int_0^{1-b} g
    lin (eps-contamination):   b*max_X f + (1-b) int_0^1 g
    max (max-discounting):     b*max_X f + int_b^1 g

The loss f must attain its maximum over an interval (box) at an endpoint
(vertex), e.g. f quasiconvex.  For step possibilities the integrals are exact;
otherwise a bilateral Riemann enclosure with m cells is returned.
"""
from __future__ import annotations
from dataclasses import dataclass
from itertools import product
import numpy as np
from .possibility import IntervalPossibility, BoxPossibility


@dataclass
class RiskBound:
    lower: float
    upper: float

    @property
    def mid(self):
        return 0.5 * (self.lower + self.upper)

    @property
    def width(self):
        return self.upper - self.lower


def _cutmax_interval(f, poss: IntervalPossibility, t, a):
    lo, hi = poss.domain
    L, U = poss.cut(t)
    L = np.clip(L - a, lo, hi); U = np.clip(U + a, lo, hi)
    return np.maximum(f(L), f(U))


def _cutmax_box(f, poss: BoxPossibility, t, a):
    cuts = poss.cut(t)                       # list of (L_j, U_j) arrays
    doms = poss.domain
    ends = []
    for (L, U), (lo, hi) in zip(cuts, doms):
        ends.append((np.clip(L - a, lo, hi), np.clip(U + a, lo, hi)))
    best = None
    for corner in product((0, 1), repeat=poss.dim):
        pt = [ends[j][c] for j, c in enumerate(corner)]
        v = f(*pt)
        best = v if best is None else np.maximum(best, v)
    return best


def _global_max(f, poss):
    if isinstance(poss, BoxPossibility):
        return max(f(*c) for c in product(*[(lo, hi) for lo, hi in poss.domain]))
    lo, hi = poss.domain
    return float(max(f(np.array(lo)), f(np.array(hi))))


def _integral(gfun, t0, t1, breaks, m, global_max):
    """Exact integral of a nonincreasing step function on (t0, t1] when breaks
    are given, otherwise a bilateral Riemann enclosure."""
    if t1 <= t0:
        return 0.0, 0.0
    if breaks is not None:
        b = np.asarray(breaks, float)
        pts = np.unique(np.concatenate([[t0, t1], b[(b > t0) & (b < t1)]]))
        mids = 0.5 * (pts[:-1] + pts[1:])
        val = float(np.sum(gfun(mids) * np.diff(pts)))
        return val, val
    nodes = np.linspace(t0, t1, m + 1)
    if t0 == 0.0:
        g = np.concatenate(([global_max], np.asarray(gfun(nodes[1:]), dtype=float)))
    else:
        g = gfun(nodes)
    h = (t1 - t0) / m
    return float(h * g[1:].sum()), float(h * g[:-1].sum())


def robust_risk(f, poss, a: float = 0.0, b: float = 0.0, kind: str = "add",
                m: int = 256) -> RiskBound:
    """Upper expectation of loss ``f`` under ambiguity around ``poss``.

    Parameters
    ----------
    f : vectorized callable; for BoxPossibility it takes one array per coordinate.
    poss : IntervalPossibility or BoxPossibility.
    a, b : spatial and membership tolerances (a in the max-norm for boxes).
    kind : 'add' (endograph D_{a,b}), 'lin' (eps-contamination) or 'max'.
    """
    if not np.isfinite(a) or a < 0:
        raise ValueError("a must be finite and nonnegative")
    if isinstance(m, bool) or not isinstance(m, (int, np.integer)) or m < 1:
        raise ValueError("m must be a positive integer")
    if not 0 <= b <= 1:
        raise ValueError("b must lie in [0, 1]")
    cutmax = _cutmax_box if isinstance(poss, BoxPossibility) else _cutmax_interval
    g = lambda t: cutmax(f, poss, np.asarray(t, float), a)
    breaks = getattr(poss, "breaks", None)
    if isinstance(poss, BoxPossibility) and all(mg.breaks is not None for mg in poss.marginals):
        breaks = np.unique(np.concatenate([np.minimum(np.asarray(mg.breaks) * poss.scale, 1)
                                           for mg in poss.marginals]))
    M = _global_max(f, poss)
    if kind == "add":
        lo, up = _integral(g, 0.0, 1.0 - b, breaks, m, M)
        return RiskBound(b * M + lo, b * M + up)
    if kind == "lin":
        lo, up = _integral(g, 0.0, 1.0, breaks, m, M)
        return RiskBound(b * M + (1 - b) * lo, b * M + (1 - b) * up)
    if kind == "max":
        lo, up = _integral(g, b, 1.0, breaks, m, M)
        return RiskBound(b * M + lo, b * M + up)
    raise ValueError("kind must be 'add', 'lin' or 'max'")
