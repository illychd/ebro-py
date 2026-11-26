"""EBRO: endograph-based robust optimization under fuzzy (possibilistic) ambiguity.

Core objects
------------
IntervalPossibility : a normal u.s.c. possibility distribution on an interval,
    described by its cut endpoints t -> [lower(t), upper(t)].
robust_risk : exact or bilaterally bounded upper expectation after
    endographic dilation D_{a,b}, epsilon-contamination or max-discounting.
from_samples / calibrate : data-driven possibility distribution and the
    finite-sample tolerance b_n (DKW-Massart) with measurement-error tolerance a.
solve_scalar_design : certified minimization of the robust risk over a scalar design.
ks_worst_case : Kolmogorov-Smirnov (and Levy-KS) DRO baseline in one dimension.
"""
from .possibility import IntervalPossibility, BoxPossibility
from .risk import robust_risk, RiskBound
from .data import (from_samples, from_samples_box, dkw_eps, calibrate,
                   adaptive_psi, dkw_psi, from_samples_graded)
from .design import solve_scalar_design, DesignResult
from .ksdro import ks_worst_case, solve_scalar_ks
from .finite import choquet_finite

__all__ = ["IntervalPossibility", "BoxPossibility", "robust_risk", "RiskBound",
           "from_samples", "from_samples_box", "dkw_eps", "calibrate",
           "adaptive_psi", "dkw_psi", "from_samples_graded",
           "solve_scalar_design", "DesignResult", "ks_worst_case",
           "solve_scalar_ks", "choquet_finite"]
__version__ = "0.2.1"
