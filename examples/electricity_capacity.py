"""Real-data application: contracted electricity capacity under uncertain demand.

Data: half-hourly electricity demand in England and Wales, 5 June - 27 August 2000
(Taylor 2003), 4032 observations, distributed with pmdarima (load_taylor).
The 4032 half-hours form the population P (the season's load-duration distribution).
A planner observes n random half-hours and contracts a capacity z (GW).
Cost per half-hour: c_o (z-theta)^+ for unused capacity, c_u (theta-z)^+ for shortfall.
"""
import json, sys, time
import numpy as np
from pmdarima.datasets import load_taylor
import ebro

CO, CU = 1.0, 9.0
DOM = (18.0, 40.0)                       # a-priori operating range (GW)
POP = load_taylor() / 1000.0            # GW
POP_S = np.sort(POP)
ZGRID = np.linspace(*DOM, 2201)

def set_domain(dom):
    """Change the a-priori operating range (sensitivity analysis)."""
    global DOM, ZGRID
    DOM = tuple(dom); ZGRID = np.linspace(*DOM, int(round((DOM[1] - DOM[0]) * 100)) + 1)

def loss(z, th):
    th = np.asarray(th, float)
    return np.where(th > z, CU * (th - z), CO * (z - th))

def true_cost(z):
    return float(loss(z, POP).mean())

ZSTAR = float(np.quantile(POP, CU / (CO + CU), method="inverted_cdf"))
VSTAR = min(true_cost(z) for z in np.linspace(ZSTAR - 0.05, ZSTAR + 0.05, 201))

# ---------- vectorized possibilistic objectives (checked against the ebro package) -----
def cut_table(obs):
    u = ebro.from_samples(obs, DOM)
    br = u.breaks; mid = 0.5 * (br[:-1] + br[1:]); w = np.diff(br)
    L, U = u.cut(mid)
    return br, L, U, w

def poss_objective(obs, a, b, kind):
    br, L, U, w = cut_table(obs)
    Lz = np.clip(L - a, *DOM); Uz = np.clip(U + a, *DOM)
    Z = ZGRID[:, None]
    g = np.maximum(loss(Z, Lz[None, :]), loss(Z, Uz[None, :]))      # (Z, K)
    M = np.maximum(loss(ZGRID, DOM[0]), loss(ZGRID, DOM[1]))
    if kind == "nominal":
        return g @ w
    if kind == "lin":
        return b * M + (1 - b) * (g @ w)
    # 'add': integrate g over (0, 1-b]
    t1 = 1 - b
    wa = np.clip(np.minimum(br[1:], t1) - br[:-1], 0, None)
    return b * M + g @ wa

def argmin_grid(vals):
    i = int(np.argmin(vals)); return float(ZGRID[i])

# ---------- fast KS / Levy-KS worst case (clip family, concave in the level) -----------
def ks_value(z, obs, eps, a):
    s = np.sort(obs); n = s.size
    p = np.clip(np.concatenate([[DOM[0], DOM[1], z], s + a, s - a]), *DOM)
    p = np.unique(np.concatenate([p, np.clip(p - 1e-9, *DOM)]))
    Fn = lambda x: np.searchsorted(s, x, side="right") / n
    low = np.maximum(0, Fn(p - a) - eps); high = np.minimum(1, Fn(p + a) + eps)
    low[-1] = high[-1] = 1.0
    fv = loss(z, p)
    lev = np.unique(np.concatenate([low, high, [0.0, 1.0]]))
    def E(c):
        F = np.maximum.accumulate(np.clip(c, low, high)); F[-1] = 1.0
        return float(np.diff(np.concatenate([[0.0], F])) @ fv)
    lo_i, hi_i = 0, lev.size - 1
    while hi_i - lo_i > 3:                         # ternary search on a concave sequence
        m1 = lo_i + (hi_i - lo_i) // 3; m2 = hi_i - (hi_i - lo_i) // 3
        if E(lev[m1]) < E(lev[m2]): lo_i = m1
        else: hi_i = m2
    return max(E(c) for c in lev[lo_i:hi_i + 1])

def ks_design(obs, eps, a):
    lo, hi = DOM
    gr = (np.sqrt(5) - 1) / 2
    x1, x2 = hi - gr * (hi - lo), lo + gr * (hi - lo)
    f1, f2 = ks_value(x1, obs, eps, a), ks_value(x2, obs, eps, a)
    for _ in range(60):
        if f1 <= f2: hi, x2, f2 = x2, x1, f1; x1 = hi - gr * (hi - lo); f1 = ks_value(x1, obs, eps, a)
        else: lo, x1, f1 = x1, x2, f2; x2 = lo + gr * (hi - lo); f2 = ks_value(x2, obs, eps, a)
    z = 0.5 * (lo + hi); return z, ks_value(z, obs, eps, a)

# ---------- one replication -----------------------------------------------------------
def replicate(rng, n, beta, a_true, a_model_list):
    th = rng.choice(POP, n)
    obs = np.clip(th - rng.uniform(0, a_true, n), *DOM) if a_true > 0 else th
    b = ebro.calibrate(n, beta); eps = ebro.dkw_eps(n, beta)
    out = {}
    q = np.quantile(obs, CU / (CO + CU), method="inverted_cdf")
    out["SAA"] = (float(q), None)
    out["Nominal possibilistic"] = (argmin_grid(poss_objective(obs, 0.0, 0.0, "nominal")), None)
    for am in a_model_list:
        tag = "" if len(a_model_list) == 1 else f" (a={am})"
        v = poss_objective(obs, am, b, "lin"); out["eps-contamination" + tag] = (argmin_grid(v), None)
        v = poss_objective(obs, am, b, "add"); z = argmin_grid(v)
        cert = ebro.robust_risk(lambda t: loss(z, t), ebro.from_samples(obs, DOM), am, b).upper
        out["EBRO" + tag] = (z, cert)
        zk, ck = ks_design(obs, eps, am); out["KS-DRO" + tag] = (zk, ck)
    return out

def credal_objective(pop):
    """z -> Q_J(pi_P): possibilistic upper expectation of the population transform."""
    uP = ebro.from_samples(pop, DOM)
    return lambda z: ebro.robust_risk(lambda t: loss(z, t), uP, 0.0, 0.0).upper

def run(ns, reps, beta, a_true, a_model_list, seed, pop_train=None, pop_test=None):
    global POP
    full = POP
    pop_train = full if pop_train is None else pop_train
    pop_test = full if pop_test is None else pop_test
    rng = np.random.default_rng(seed)
    cost = lambda z: float(loss(z, pop_test).mean())
    zz = np.linspace(DOM[0], DOM[1], 4401)
    ctest = np.array([cost(z) for z in zz]); vstar = ctest.min()
    Qc = credal_objective(pop_test); qv = np.array([Qc(z) for z in zz[::4]]); qstar = qv.min()
    res = {"vstar": float(vstar), "zstar": float(zz[ctest.argmin()]), "qstar": float(qstar),
           "zq": float(zz[::4][qv.argmin()]), "by_n": {}}
    for n in ns:
        POP = pop_train
        rows = {}
        for r in range(reps):
            for k, (z, cert) in replicate(rng, n, beta, a_true, a_model_list).items():
                c = cost(z); d = rows.setdefault(k, {"z": [], "reg": [], "creg": [], "cert": [], "cover": []})
                d["z"].append(z); d["reg"].append(100 * (c - vstar) / vstar)
                d["creg"].append(100 * (Qc(z) - qstar) / qstar)
                if cert is not None:
                    d["cert"].append(cert / c); d["cover"].append(bool(c <= cert + 1e-12))
        POP = full
        out = {}
        for k, v in rows.items():
            out[k] = {"z_mean": float(np.mean(v["z"])), "regret_mean": float(np.mean(v["reg"])),
                      "regret_p90": float(np.quantile(v["reg"], 0.9)),
                      "credal_regret_mean": float(np.mean(v["creg"])),
                      "cert_ratio": float(np.mean(v["cert"])) if v["cert"] else None,
                      "coverage": float(np.mean(v["cover"])) if v["cover"] else None}
        res["by_n"][n] = out
        print("n =", n, flush=True)
        for k, v in out.items():
            print("   %-28s" % k, {kk: (round(vv, 3) if vv is not None else None) for kk, vv in v.items()}, flush=True)
    return res

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "A"
    if which == "A":      # i.i.d. sampling from the season, exact data
        res = run([25, 50, 100, 200, 500, 1000, 2000], 200, 0.1, 0.0, [0.0], 11)
    elif which == "A45":  # sensitivity to the a-priori upper bound of the domain
        set_domain((18.0, 45.0)); res = run([25, 50, 100, 200, 1000], 200, 0.1, 0.0, [0.0], 11)
    elif which == "B":    # biased bounded metering error theta_obs = theta - U[0, a]
        res = run([50, 200, 1000], 200, 0.1, 0.75, [0.0, 0.75], 12)
    elif which == "S":    # distribution shift: train on weeks 1-4, evaluate on weeks 5-12
        res = run([50, 200, 1000], 200, 0.1, 0.0, [0.0], 13, POP[:1344], POP[1344:])
    json.dump(res, open(f"results_{which}.json", "w"), indent=1)
