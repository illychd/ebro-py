# ebro — Endograph-Based Robust Optimization

`ebro` implements robust optimization under **possibilistic (fuzzy) ambiguity** measured in the
endograph metric, as developed in *EBRO: Endograph-Based Robust Optimization with
Decision-Independent Worst Cases and Data-Driven Fuzzy Risk Certificates* (Alvarez, 2026).

The key fact is that an endograph ball with spatial tolerance `a` and membership tolerance `b`
has a greatest element, `D_{a,b} u = min(1, b + max_{B(x,a)} u)`. Every order-preserving risk is
therefore maximized at a single, decision-independent membership function.

## What it does

| Feature | Function |
|---|---|
| Interval / box possibility distributions (triangular, from samples, Bonferroni boxes) | `IntervalPossibility`, `from_samples`, `from_samples_box` |
| Robust risk after endographic dilation, eps-contamination or max-discounting (exact for step cuts, bilateral bounds otherwise) | `robust_risk(f, u, a, b, kind)` |
| Finite-sample calibration: `b_n = 2*sqrt(log(2/beta)/(2n))`, measurement-error tolerance `a` | `calibrate`, `dkw_eps` |
| Certified scalar design (convex in the decision) | `solve_scalar_design` |
| Kolmogorov–Smirnov / Lévy–KS DRO baseline (clip formula + LP check) | `ks_worst_case`, `solve_scalar_ks` |
| Exact finite-space Choquet integral | `choquet_finite` |

## Quick start

```python
import numpy as np, ebro
rng = np.random.default_rng(0)
obs = rng.gamma(4, 1.5, 200)                      # observations, |error| <= a
u = ebro.from_samples(obs, domain=(0, 25))
b = ebro.calibrate(n=obs.size, beta=0.1)          # membership tolerance from DKW
loss = lambda z, th: np.where(th > z, 4 * (th - z), z - th)
res = ebro.solve_scalar_design(loss, u, bounds=(0, 25), a=0.2, b=b)
print(res.z, res.certificate.upper)               # decision and out-of-sample certificate
```

With probability at least `1 - beta`, `E_P[loss(res.z, theta)] <= res.certificate.upper`,
uniformly over all decisions (Theorem 6.3 and Corollary 6.4 of the paper).

## Version 0.2.1

Fixes the zero-level quadrature bound and validates spatial tolerance and quadrature size.
Historical experiment outputs are retained from v0.2.0; they have not been regenerated.
This release is a research preview, not an independently verified numerical certification library.
The KS comparator and finite-sample assumptions still require independent review.

## Guarantees and caveats

* Certificates bound the expected loss of the true law **and** of every law in the credal set of
  its central-interval transform. Decisions are consistent for this credal objective, not for the
  expected cost of a single law; for i.i.d. expected cost, sample average approximation can be better.
* In 1-D with exact data, the KS ambiguity set is contained in the EBRO credal set, so EBRO
  certificates are never tighter than KS certificates. EBRO's advantages lie in bounded
  measurement error, box cuts in several dimensions and structure (decision-independent worst cases).
* For small samples the global term `b * max_X loss` dominates, and decisions are driven by the
  a priori domain. Use physically meaningful bounds.
* Floating-point arithmetic is not outward-rounded; bilateral bounds are mathematical bounds in exact arithmetic.

## Reproducing the paper

```bash
pip install -e ".[examples,test]"
pytest -q                                          # run the regression suite
cd examples
python electricity_capacity.py A                   # i.i.d. sampling (Table 10)
python electricity_capacity.py B                   # biased metering error (Table 11)
python electricity_capacity.py S                   # seasonal shift (Table 11)
python electricity_capacity.py A45                 # looser operating range (Table 11)
```

Data: half-hourly electricity demand, England and Wales, 5 June – 27 August 2000
(Taylor, 2003, *JORS* 54:799–805), loaded through `pmdarima.datasets.load_taylor`.

## Citation

See `CITATION.cff`. MIT licence.
