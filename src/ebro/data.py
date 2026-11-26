"""Data-driven possibility distributions and finite-sample calibration.

Given observations x_i = theta_i + e_i with |e_i| <= a (bounded measurement
error) and n i.i.d. pairs, the central-interval possibility

    u_n(x) = min{1, 2 min(F_n(x), H_n(x))},   H_n(x) = #{x_i >= x}/n,

satisfies, with probability at least 1 - beta, pi_P in B_{a, b_n}(u_n) with
b_n = min(1, 2*eps_n), eps_n = sqrt(log(2/beta)/(2n))  (DKW-Massart),
where pi_P is the same transform of the true distribution of theta.
Hence E_P f <= robust_risk(f, u_n, a, b_n) for every continuous f simultaneously.
"""
from __future__ import annotations
import math
import numpy as np
from .possibility import IntervalPossibility, BoxPossibility


def dkw_eps(n: int, beta: float, k: int = 1) -> float:
    """DKW-Massart half-width with Bonferroni split over k coordinates."""
    if n < 1 or not 0 < beta < 1:
        raise ValueError("need n >= 1 and 0 < beta < 1")
    return math.sqrt(math.log(2 * k / beta) / (2 * n))


def calibrate(n: int, beta: float, k: int = 1) -> float:
    """Membership tolerance b_n = min(1, 2 k eps_n) of the finite-sample theorem."""
    return min(1.0, 2 * k * dkw_eps(n, beta, k))


def from_samples(x, domain) -> IntervalPossibility:
    """Central-interval possibility of the empirical distribution (exact step cuts).

    The alpha-cut is [x_(k), x_(n+1-k)] with k = ceil(n*alpha/2).
    """
    s = np.sort(np.asarray(x, float))
    n = s.size
    if n == 0:
        raise ValueError("empty sample")
    def idx(t):
        k = np.ceil(np.asarray(t, float) * n / 2 - 1e-12).astype(int)
        return np.clip(k, 1, (n + 1) // 2)
    lower = lambda t: s[idx(t) - 1]
    upper = lambda t: s[n - idx(t)]
    breaks = np.unique(np.clip(2 * np.arange(0, (n + 1) // 2 + 1) / n, 0, 1))
    if breaks[-1] < 1:
        breaks = np.append(breaks, 1.0)
    return IntervalPossibility(lower, upper, tuple(domain), breaks)


def from_samples_box(X, domains) -> BoxPossibility:
    """Bonferroni box possibility min(1, k min_j u_{n,j}) from an (n, k) sample."""
    X = np.asarray(X, float)
    k = X.shape[1]
    return BoxPossibility([from_samples(X[:, j], domains[j]) for j in range(k)], scale=float(k))


# ---------------------------------------------------------------------------
# v0.2: level-dependent (graded) membership tolerance
# ---------------------------------------------------------------------------
def adaptive_psi(n: int, beta: float) -> np.ndarray:
    """Upper confidence map psi(j/n), j = 0..n, valid simultaneously with prob >= 1 - beta.

    psi(j/n) = min( j/n + eps', H_{j+1} ), where eps' = sqrt(ln(4/beta)/(2n)) is the DKW
    half-width at level beta/2 and H_k is the upper (1 - beta/(4n)) quantile of
    Beta(k, n+1-k), the law of the k-th uniform order statistic (Bonferroni over 2n events,
    level beta/2).  On the joint event, for every x:  F(x) <= psi(F_n(x)) and
    H(x) <= psi(H_n(x)), for any (also discrete) distribution.
    """
    from scipy.stats import beta as Beta
    if n < 1 or not 0 < beta < 1:
        raise ValueError("need n >= 1 and 0 < beta < 1")
    j = np.arange(n + 1)
    eps = math.sqrt(math.log(4 / beta) / (2 * n))
    gam = beta / (4 * n)
    Hk = np.ones(n + 1)
    k = j[:-1] + 1
    Hk[:-1] = Beta.ppf(1 - gam, k, n + 1 - k)
    psi = np.minimum(np.minimum(j / n + eps, Hk), 1.0)
    psi = np.maximum(psi, j / n)               # always valid (psi >= identity)
    return np.maximum.accumulate(psi)          # nondecreasing


def dkw_psi(n: int, beta: float) -> np.ndarray:
    """psi for the constant DKW band: psi(j/n) = min(1, j/n + eps_n)."""
    j = np.arange(n + 1)
    return np.minimum(1.0, j / n + dkw_eps(n, beta))


def from_samples_graded(x, domain, beta: float, a: float = 0.0, psi=None) -> IntervalPossibility:
    """Certified envelope Phi = phi(max_{B(.,a)} u_n) of the true central-interval transform.

    phi(t) = min(1, 2 psi(t/2)) on the grid of values of u_n (phi(1) = 1).  With probability
    >= 1 - beta, pi_P <= Phi, hence E_P f <= robust_risk(f, Phi) for all continuous f.
    The returned object already contains the spatial and graded membership tolerance:
    evaluate it with robust_risk(f, Phi) (a = b = 0).
    """
    s = np.sort(np.asarray(x, float)); n = s.size
    lo, hi = domain
    if psi is None:
        psi = adaptive_psi(n, beta)
    J = (n + 1) // 2                           # largest nonempty cut index
    jj = np.arange(J + 1)
    phi = np.where(2 * jj < n, np.minimum(1.0, 2 * psi[np.minimum(jj, n)]), 1.0)
    phi[-1] = 1.0
    phi = np.maximum.accumulate(phi)

    def jstar(t):
        t = np.asarray(t, float)
        return np.clip(np.searchsorted(phi, t - 1e-15, side="left"), 0, J)

    def lower(t):
        j = jstar(t)
        out = np.where(j == 0, lo, s[np.maximum(j, 1) - 1] - a)
        return np.clip(out, lo, hi)

    def upper(t):
        j = jstar(t)
        out = np.where(j == 0, hi, s[n - np.maximum(j, 1)] + a)
        return np.clip(out, lo, hi)

    breaks = np.unique(np.concatenate([[0.0], phi, [1.0]]))
    return IntervalPossibility(lower, upper, (lo, hi), breaks)
