# Pre-registered evaluation protocol — EBRO v0.2 (adaptive tolerance)

Fixed **before** implementing or running the v0.2 experiments. Any later deviation is listed
under "Deviations" at the end of this file, with its reason. All cells are reported, whatever
the outcome.

## 1. Methods (6)
1. **SAA** — empirical c_u/(c_o+c_u) quantile of the observations.
2. **Nominal possibilistic** — argmin_z of the upper expectation under the empirical central-interval transform (no tolerance).
3. **EBRO-DKW** — v0.1: constant membership tolerance b_n = 2 sqrt(ln(2/beta)/(2n)), spatial tolerance a.
4. **EBRO-adaptive** — v0.2: level-dependent (graded) tolerance from the intersection of a DKW band
   (level beta/2) and Bonferroni order-statistic Beta bounds (level beta/2, split over 2n events).
5. **KS-DRO** — Kolmogorov–Smirnov / Lévy–KS ambiguity set with the DKW radius at level beta (v0.1).
6. **Band-DRO** — distributionally robust set defined by **the same** adaptive band as method 4
   (the fair counterpart of EBRO-adaptive).

## 2. Data, decision and loss
Loss J_z(theta) = c_o (z - theta)^+ + c_u (theta - z)^+, with c_o = 1.
* **D1 — Taylor (2003)**: half-hourly electricity demand, England & Wales, 5 Jun–27 Aug 2000, 4032 obs, in GW.
  Domain X = [18, 40] (as already reported in v0.1), plus sensitivity domain [18, 45].
* **D2 — MSFT**: daily loss = -100 * log-return of the Microsoft close (pmdarima `load_msft`), 7982 obs.
  Domain rule fixed now: [min - 0.05 R, max + 0.05 R] rounded outward to integers, R = range -> X = [-21, 39].

## 3. Scenarios (all reported)
| id | data | sampling / evaluation | n | c_u |
|---|---|---|---|---|
| T-iid | D1 | i.i.d. from full season, evaluate on full season | 25, 50, 100, 200, 500, 1000, 2000 | 9 |
| T-bias | D1 | observations theta - U[0, 0.75]; methods 3–6 use a = 0.75 | 50, 200, 1000 | 9 |
| T-shift | D1 | train weeks 1–4, evaluate weeks 5–12 | 50, 200, 1000 | 9 |
| T-dom45 | D1 | as T-iid with X = [18, 45] | 25, 100, 1000 | 9 |
| M-iid | D2 | i.i.d. from full period | 25, 50, 100, 200, 500, 1000, 2000 | 9 |
| M-shift | D2 | train first half (chronological), evaluate second half | 50, 200, 1000 | 9 |
| M-iid-med | D2 | as M-iid with c_u = 1 (median regime) | 25, 100, 1000 | 1 |

beta = 0.1; 200 replications per cell; seeds fixed in the script (one seed per scenario).

## 4. Metrics
* Primary: mean relative regret in expected cost on the evaluation population, 100 (E[J_z] - min) / min.
* Secondary: 90th percentile of that regret; mean relative regret in the credal objective
  (upper expectation under the population central-interval transform); for certified methods
  (3–6): coverage Pr{E[J_zhat] <= certificate} and mean certificate / realized cost.

## 5. Pre-declared comparisons and claims policy
* **H1 (method improvement)**: EBRO-adaptive has lower primary regret than EBRO-DKW for n <= 500.
* **H2 (head-to-head)**: count, over all cells, how often EBRO-adaptive has the lowest primary regret
  among the 6 methods, and how often it beats SAA and Band-DRO separately. No directional prediction.
  Expectation stated in advance: SAA is consistent for expected cost under i.i.d. sampling and should
  win the large-n i.i.d. cells.
* A claim of general superiority is made **only** if EBRO-adaptive has the lowest primary regret in a
  majority of cells of **both** datasets. Otherwise the paper reports the regimes where each method wins.
* Validity: any certified method with coverage below 1 - beta in a cell is reported as a violation.

## Deviations
1. *Computational only.* The band worst case was switched from an exhaustive search over CDF levels
   to an exact ternary search on the (concave) level sequence, because the exhaustive search made the
   n = 1000-2000 cells infeasible. The two agree to 1e-9 in the test-suite (60 random cases, discrete
   data, both bands); seeds, methods, scenarios and metrics are unchanged. Runs were restarted from scratch.
