"""Certified minimization of the robust risk over a scalar design variable."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize_scalar
from .risk import robust_risk, RiskBound


@dataclass
class DesignResult:
    z: float
    certificate: RiskBound   # bounds on the robust risk of z
    kind: str
    a: float
    b: float


def solve_scalar_design(loss, poss, bounds, a=0.0, b=0.0, kind="add", m=256,
                        grid: int = 401, xtol: float = 1e-8) -> DesignResult:
    """Minimize the certified upper bound of the robust risk over z in ``bounds``.

    ``loss(z, theta)`` must be convex in z and quasiconvex in theta, so the
    objective is convex in z (Theorem on the convex counterpart).  A coarse grid
    locates the basin and a bounded Brent search refines it.
    """
    obj = lambda z: robust_risk(lambda th: loss(z, th), poss, a, b, kind, m).upper
    zs = np.linspace(bounds[0], bounds[1], grid)
    vals = np.array([obj(z) for z in zs])
    i = int(np.argmin(vals))
    lo, hi = zs[max(i - 1, 0)], zs[min(i + 1, grid - 1)]
    if hi > lo:
        r = minimize_scalar(obj, bounds=(lo, hi), method="bounded", options={"xatol": xtol})
        z = float(r.x) if r.fun <= vals[i] else float(zs[i])
    else:
        z = float(zs[i])
    cert = robust_risk(lambda th: loss(z, th), poss, a, b, kind, m)
    return DesignResult(z, cert, kind, a, b)
