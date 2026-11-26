"""Possibility distributions with interval (or box) cuts."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence
import numpy as np


@dataclass
class IntervalPossibility:
    """Normal u.s.c. possibility distribution on X=[lo, hi] with interval cuts.

    The alpha-cut is [lower(alpha), upper(alpha)] for 0 < alpha <= 1, with
    ``lower`` nondecreasing and ``upper`` nonincreasing.  If ``breaks`` is
    given, both functions are constant on each interval (breaks[i], breaks[i+1]]
    and integrals are computed exactly.
    """
    lower: Callable[[np.ndarray], np.ndarray]
    upper: Callable[[np.ndarray], np.ndarray]
    domain: tuple[float, float]
    breaks: Optional[np.ndarray] = None   # increasing, from 0 to 1

    def cut(self, t):
        t = np.asarray(t, float)
        lo, hi = self.domain
        return np.clip(self.lower(t), lo, hi), np.clip(self.upper(t), lo, hi)

    def membership(self, x, grid: int = 2001):
        """Evaluate u(x) = sup{alpha : x in cut(alpha)} on a level grid (for plotting)."""
        x = np.atleast_1d(np.asarray(x, float))
        al = np.linspace(1e-9, 1, grid)
        L, U = self.cut(al)
        inside = (x[:, None] >= L[None, :] - 1e-12) & (x[:, None] <= U[None, :] + 1e-12)
        out = np.where(inside.any(1), np.where(inside, al[None, :], 0).max(1), 0.0)
        return out

    @staticmethod
    def triangular(lo_sup, mode, hi_sup, domain):
        """Triangular fuzzy number with support [lo_sup, hi_sup] and core {mode}."""
        return IntervalPossibility(lambda t: lo_sup + (mode - lo_sup) * t,
                                   lambda t: hi_sup - (hi_sup - mode) * t, domain)


@dataclass
class BoxPossibility:
    """Possibility on a box whose alpha-cuts are products of interval cuts.

    ``marginals`` are IntervalPossibility objects; ``scale`` implements the
    Bonferroni combination pi(x) = min(1, scale * min_j pi_j(x_j)), whose
    alpha-cut is the product of marginal cuts at level alpha/scale.
    """
    marginals: Sequence[IntervalPossibility]
    scale: float = 1.0

    @property
    def dim(self):
        return len(self.marginals)

    @property
    def domain(self):
        return [m.domain for m in self.marginals]

    def cut(self, t):
        t = np.asarray(t, float)
        return [m.cut(np.minimum(t / self.scale, 1.0)) for m in self.marginals]
