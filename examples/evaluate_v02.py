"""Pre-registered evaluation of EBRO v0.2 (see PROTOCOL.md). Usage: python evaluate_v02.py <scenario|all>"""
import json, sys, time
import numpy as np
from pmdarima.datasets import load_taylor, load_msft
import ebro

BETA, REPS, CO = 0.1, 200, 1.0
TAY = load_taylor() / 1000.0
_c = load_msft()["Close"].values
MSFT = -100 * np.diff(np.log(_c)); MSFT = MSFT[np.isfinite(MSFT)]

SCEN = {  # id: (train pop, eval pop, domain, ns, c_u, bias a, seed)
    "T-iid":     (TAY, TAY, (18., 40.), [25, 50, 100, 200, 500, 1000, 2000], 9., 0.0, 101),
    "T-bias":    (TAY, TAY, (18., 40.), [50, 200, 1000], 9., 0.75, 102),
    "T-shift":   (TAY[:1344], TAY[1344:], (18., 40.), [50, 200, 1000], 9., 0.0, 103),
    "T-dom45":   (TAY, TAY, (18., 45.), [25, 100, 1000], 9., 0.0, 104),
    "M-iid":     (MSFT, MSFT, (-21., 39.), [25, 50, 100, 200, 500, 1000, 2000], 9., 0.0, 105),
    "M-shift":   (MSFT[:3991], MSFT[3991:], (-21., 39.), [50, 200, 1000], 9., 0.0, 106),
    "M-iid-med": (MSFT, MSFT, (-21., 39.), [25, 100, 1000], 1., 0.0, 107),
}
METHODS = ["SAA", "Nominal possibilistic", "EBRO-DKW", "EBRO-adaptive", "KS-DRO", "Band-DRO"]


def make_loss(cu):
    return lambda z, th: np.where(np.asarray(th) > z, cu * (np.asarray(th) - z), CO * (z - np.asarray(th)))


def table(poss):
    br = np.asarray(poss.breaks); mid = 0.5 * (br[:-1] + br[1:]); w = np.diff(br)
    L, U = poss.cut(mid); return L, U, w


def poss_values(loss, poss, Z):
    L, U, w = table(poss)
    g = np.maximum(loss(Z[:, None], L[None, :]), loss(Z[:, None], U[None, :]))
    return g @ w


def band_design(loss, obs, dom, psi, a):
    def val(z):
        return ebro.ks_worst_case(lambda t: loss(z, t), obs, None, dom, a, extra=[z], psi=psi)
    lo, hi = dom; gr = (np.sqrt(5) - 1) / 2
    x1, x2 = hi - gr * (hi - lo), lo + gr * (hi - lo); f1, f2 = val(x1), val(x2)
    for _ in range(55):
        if f1 <= f2: hi, x2, f2 = x2, x1, f1; x1 = hi - gr * (hi - lo); f1 = val(x1)
        else: lo, x1, f1 = x1, x2, f2; x2 = lo + gr * (hi - lo); f2 = val(x2)
    z = 0.5 * (lo + hi); return z, val(z)


def run(sid):
    train, evalp, dom, ns, cu, abias, seed = SCEN[sid]
    loss = make_loss(cu); rng = np.random.default_rng(seed)
    Z = np.linspace(dom[0], dom[1], int(round((dom[1] - dom[0]) * 100)) + 1)
    ecost = np.array([loss(z, evalp).mean() for z in Z]); vstar = ecost.min()
    cred = poss_values(loss, ebro.from_samples(evalp, dom), Z); qstar = cred.min()
    cost = lambda z: float(loss(z, evalp).mean())
    credal = lambda z: float(np.interp(z, Z, cred))
    out = {"vstar": float(vstar), "zstar": float(Z[ecost.argmin()]), "qstar": float(qstar),
           "zq": float(Z[cred.argmin()]), "by_n": {}}
    for n in ns:
        t0 = time.time(); rec = {m: {"reg": [], "creg": [], "cov": [], "cr": []} for m in METHODS}
        psiA, psiD = ebro.adaptive_psi(n, BETA), ebro.dkw_psi(n, BETA)
        for _ in range(REPS):
            th = rng.choice(train, n)
            obs = np.clip(th - rng.uniform(0, abias, n), *dom) if abias > 0 else th
            a = abias
            res = {}
            res["SAA"] = (float(np.quantile(obs, cu / (CO + cu), method="inverted_cdf")), None)
            res["Nominal possibilistic"] = (float(Z[np.argmin(poss_values(loss, ebro.from_samples(obs, dom), Z))]), None)
            for name, psi in (("EBRO-DKW", psiD), ("EBRO-adaptive", psiA)):
                Phi = ebro.from_samples_graded(obs, dom, BETA, a, psi=psi)
                z = float(Z[np.argmin(poss_values(loss, Phi, Z))])
                res[name] = (z, ebro.robust_risk(lambda t: loss(z, t), Phi).upper)
            res["KS-DRO"] = band_design(loss, obs, dom, psiD, a)
            res["Band-DRO"] = band_design(loss, obs, dom, psiA, a)
            for m, (z, cert) in res.items():
                c = cost(z); r = rec[m]
                r["reg"].append(100 * (c - vstar) / vstar); r["creg"].append(100 * (credal(z) - qstar) / qstar)
                if cert is not None: r["cov"].append(c <= cert + 1e-12); r["cr"].append(cert / c)
        out["by_n"][n] = {m: {"regret_mean": float(np.mean(r["reg"])), "regret_p90": float(np.quantile(r["reg"], .9)),
                              "credal_regret_mean": float(np.mean(r["creg"])),
                              "coverage": float(np.mean(r["cov"])) if r["cov"] else None,
                              "cert_ratio": float(np.mean(r["cr"])) if r["cr"] else None} for m, r in rec.items()}
        print(sid, "n =", n, f"({time.time()-t0:.0f}s)", {m: round(v["regret_mean"], 2) for m, v in out["by_n"][n].items()}, flush=True)
    json.dump(out, open(f"v02_{sid}.json", "w"), indent=1)


if __name__ == "__main__":
    ids = list(SCEN) if sys.argv[1] == "all" else [sys.argv[1]]
    for sid in ids: run(sid)
