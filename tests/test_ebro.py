from fractions import Fraction as Fr
from itertools import product
import numpy as np
import pytest
import ebro
from ebro import (IntervalPossibility, robust_risk, from_samples, calibrate, dkw_eps,
                  solve_scalar_design, ks_worst_case, choquet_finite)

STEADY = lambda z: (lambda th: (z * th - 1.0) ** 2)
TRI = IntervalPossibility.triangular(0.7, 0.8, 1.2, (0.5, 1.5))


def test_membrane_exact_value():
    # Table 4 of the manuscript: z=1.08, a=b=1/16 -> 3408661291/37324800000
    r = robust_risk(STEADY(1.08), TRI, 1 / 16, 1 / 16, m=40000)
    exact = 3408661291 / 37324800000
    assert r.lower <= exact <= r.upper and r.width < 1e-5


def test_bilateral_enclosure_shrinks():
    w = [robust_risk(STEADY(1.1), TRI, 0.05, 0.05, m=m).width for m in (16, 64, 256)]
    assert w[0] > w[1] > w[2]


def test_design_solver_matches_continuous_optimum():
    res = solve_scalar_design(lambda z, th: (z * th - 1) ** 2, TRI, (0.5, 1.5), 1 / 16, 1 / 16, m=4000)
    assert abs(res.z - 1.08316) < 2e-3


def test_discount_ordering_and_sandwich():
    for z, a, b in product([0.8, 1.0, 1.2], [0, 0.05], [0.1, 0.3, 0.7]):
        f = STEADY(z)
        mx = robust_risk(f, TRI, a, b, "max", m=4000).mid
        ln = robust_risk(f, TRI, a, b, "lin", m=4000).mid
        ad = robust_risk(f, TRI, a, b, "add", m=4000).mid
        Q = robust_risk(f, TRI, a, 0, "add", m=4000).mid
        M = max(f(np.array(0.5)), f(np.array(1.5)))
        assert mx <= ln + 1e-6 <= ad + 2e-6
        assert (1 - b) * Q + b * M <= ad + 1e-6


def test_finite_space_discounts_exact():
    ws = [(1, 0, 0), (1, Fr(1, 4), Fr(3, 4)), (Fr(1, 2), 1, Fr(1, 2))]
    fs = [(0, 1, 2), (-2, 4, 1), (1, 0, 1)]
    for w, f, b in product(ws, fs, [Fr(1, 8), Fr(1, 2)]):
        qm = choquet_finite([max(p, b) for p in w], f)
        ql = choquet_finite([(1 - b) * p + b for p in w], f)
        qa = choquet_finite([min(1, p + b) for p in w], f)
        assert qm <= ql <= qa
        assert ql == (1 - b) * choquet_finite(w, f) + b * max(f)


def test_from_samples_cuts():
    x = np.array([3.0, 1.0, 2.0, 5.0, 4.0])
    u = from_samples(x, (0, 6))
    L, U = u.cut(np.array([0.2, 0.4, 1.0]))
    assert list(L) == [1.0, 1.0, 3.0] and list(U) == [5.0, 5.0, 3.0]


def test_ks_clip_matches_lp():
    rng = np.random.default_rng(1)
    for _ in range(10):
        s = rng.uniform(20, 38, rng.integers(5, 40)); z = rng.uniform(22, 36)
        f = lambda t: np.where(t > z, 9 * (t - z), z - t)
        for a in (0.0, 0.5):
            A = ks_worst_case(f, s, 0.1, (18, 40), a, extra=[z], method="lp")
            B = ks_worst_case(f, s, 0.1, (18, 40), a, extra=[z])
            assert abs(A - B) <= 1e-7 * max(1, abs(A))


def test_ks_set_inside_ebro_credal_set():
    # Proposition: the (Levy-)KS band is contained in M(D_{a,2eps} u_n).
    rng = np.random.default_rng(2)
    for _ in range(10):
        s = rng.normal(30, 3, 60).clip(18, 40); z = rng.uniform(25, 35); eps = dkw_eps(60, 0.1)
        f = lambda t: np.where(t > z, 9 * (t - z), z - t)
        for a in (0.0, 0.5):
            ks = ks_worst_case(f, s, eps, (18, 40), a, extra=[z])
            eb = robust_risk(f, from_samples(s, (18, 40)), a, min(1, 2 * eps))
            assert ks <= eb.upper + 1e-9


def test_finite_sample_coverage():
    # Theorem: with prob >= 1-beta, E_P f <= R_{a,b_n}(f; u_n) for all f.
    rng = np.random.default_rng(3)
    beta, n, a, fails = 0.2, 40, 0.3, 0
    pop = rng.gamma(4, 1.5, 200000).clip(0, 25)
    for _ in range(300):
        th = rng.choice(pop, n)
        obs = np.clip(th + rng.uniform(-a, 0, n), 0, 25)   # biased bounded error
        u = from_samples(obs, (0, 25)); b = calibrate(n, beta)
        bad = False
        for z in (3.0, 6.0, 9.0):
            f = lambda t, z=z: np.where(t > z, 4 * (t - z), z - t)
            if f(pop).mean() > robust_risk(f, u, a, b).upper + 1e-9:
                bad = True
        fails += bad
    assert fails / 300 <= beta


# ----------------------------- v0.2: graded tolerance -----------------------------
from ebro import adaptive_psi, dkw_psi, from_samples_graded


def test_graded_with_dkw_psi_equals_constant_tolerance():
    rng = np.random.default_rng(4)
    s = rng.normal(30, 3, 77).clip(18, 40)
    f = lambda t: np.where(t > 33, 9 * (t - 33), 33 - t)
    b = calibrate(77, 0.1)
    A = robust_risk(f, from_samples(s, (18, 40)), 0.3, b).upper
    B = robust_risk(f, from_samples_graded(s, (18, 40), 0.1, 0.3, psi=dkw_psi(77, 0.1))).upper
    assert abs(A - B) < 1e-9


def test_adaptive_psi_properties():
    for n in (10, 100, 1000):
        p = adaptive_psi(n, 0.1)
        j = np.arange(n + 1) / n
        assert np.all(p >= j - 1e-15) and np.all(np.diff(p) >= -1e-15) and p[-1] == 1
        assert np.all(p <= dkw_psi(n, 0.05) + 1e-12)   # never worse than DKW at beta/2
    assert adaptive_psi(1000, 0.1)[0] < 0.5 * dkw_psi(1000, 0.1)[0]   # much tighter tails


def test_band_clip_matches_lp_and_inclusion():
    rng = np.random.default_rng(5)
    for _ in range(8):
        n = int(rng.integers(10, 80)); s = rng.gamma(3, 2, n).clip(0, 30); z = rng.uniform(3, 15)
        f = lambda t: np.where(t > z, 9 * (t - z), z - t); psi = adaptive_psi(n, 0.1)
        for a in (0.0, 0.4):
            A = ks_worst_case(f, s, None, (0, 30), a, extra=[z], method="lp", psi=psi)
            B = ks_worst_case(f, s, None, (0, 30), a, extra=[z], psi=psi)
            assert abs(A - B) <= 1e-7 * max(1, abs(A))
            E = robust_risk(f, from_samples_graded(s, (0, 30), 0.1, a, psi=psi)).upper
            assert B <= E + 1e-9          # band set inside the EBRO credal set


def test_graded_coverage_discrete_population():
    # coverage of the adaptive certificate, discrete population with many ties
    rng = np.random.default_rng(6)
    pop = np.round(rng.standard_t(3, 100000), 1).clip(-10, 10)
    beta, fails = 0.2, 0
    for _ in range(300):
        n = int(rng.choice([20, 60, 200]))
        obs = rng.choice(pop, n)
        Phi = from_samples_graded(obs, (-10, 10), beta)
        bad = False
        for z in (-1.0, 0.5, 2.0):
            f = lambda t, z=z: np.where(t > z, 9 * (t - z), z - t)
            bad |= f(pop).mean() > robust_risk(f, Phi).upper + 1e-9
        fails += bad
    assert fails / 300 <= beta


def test_band_ternary_matches_full_search():
    rng = np.random.default_rng(7)
    for _ in range(60):
        n = int(rng.integers(5, 150)); s = np.round(rng.standard_t(3, n), 1).clip(-10, 10)
        z = rng.uniform(-3, 4); cu = rng.choice([1.0, 9.0])
        f = lambda t: np.where(t > z, cu * (t - z), z - t)
        psi = adaptive_psi(n, 0.1) if rng.random() < .5 else dkw_psi(n, 0.1)
        a = rng.choice([0.0, 0.3])
        A = ks_worst_case(f, s, None, (-10, 10), a, extra=[z], psi=psi, method="full")
        B = ks_worst_case(f, s, None, (-10, 10), a, extra=[z], psi=psi)
        assert abs(A - B) <= 1e-9 * max(1, abs(A))


def test_tiny_positive_membership_regression():
    u = IntervalPossibility(lambda t: np.zeros_like(t),
                            lambda t: np.where(t <= 1e-16, 1., 0.), (0., 1.))
    r = robust_risk(lambda x: x, u)
    assert r.lower <= 1e-16 <= r.upper


@pytest.mark.parametrize("kwargs", [{"a": -0.1}, {"a": float("inf")}, {"m": 0}, {"m": 1.5}])
def test_invalid_risk_parameters(kwargs):
    with pytest.raises(ValueError):
        robust_risk(STEADY(1.0), TRI, **kwargs)
