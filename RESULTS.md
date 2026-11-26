# Results of the pre-registered evaluation (PROTOCOL.md)

Primary metric: mean relative regret in expected cost (%), 200 replications per cell, beta = 0.1. Bold = lowest in the row.

| scenario | n | SAA | Nominal possibilistic | EBRO-DKW | EBRO-adaptive | KS-DRO | Band-DRO |
|---|---|---|---|---|---|---|---|
| T-iid | 25 | **3.03** | 12.18 | 5.63 | 5.63 | 24.08 | 24.08 |
| T-iid | 50 | **1.58** | 10.22 | 5.63 | 5.63 | 26.21 | 26.21 |
| T-iid | 100 | **0.54** | 9.79 | 5.63 | 5.63 | 27.28 | 6.64 |
| T-iid | 200 | **0.25** | 10.11 | 5.63 | 2.06 | 4.64 | 2.13 |
| T-iid | 500 | **0.10** | 9.39 | 5.63 | 5.12 | 1.30 | 0.97 |
| T-iid | 1000 | **0.04** | 9.34 | 4.31 | 6.62 | 0.58 | 0.51 |
| T-iid | 2000 | **0.02** | 9.46 | 6.43 | 7.34 | 0.24 | 0.24 |
| T-bias | 50 | **3.49** | 15.94 | 5.63 | 5.63 | 24.70 | 24.70 |
| T-bias | 200 | 1.61 | 15.83 | 5.63 | **0.78** | 7.45 | 4.39 |
| T-bias | 1000 | **1.22** | 15.19 | 1.87 | 3.59 | 1.91 | 1.77 |
| T-shift | 50 | **1.78** | 3.32 | 7.56 | 7.56 | 29.29 | 29.29 |
| T-shift | 200 | 1.83 | 3.13 | 7.56 | **0.30** | 9.73 | 6.22 |
| T-shift | 1000 | 1.85 | 2.93 | **0.55** | 1.70 | 3.53 | 3.40 |
| T-dom45 | 25 | **4.09** | 13.56 | 62.59 | 62.59 | 81.68 | 81.68 |
| T-dom45 | 100 | **0.69** | 10.18 | 62.59 | 62.59 | 84.82 | 6.20 |
| T-dom45 | 1000 | **0.04** | 9.36 | 4.20 | 6.58 | 0.58 | 0.50 |
| M-iid | 25 | **5.94** | 8.50 | 714.50 | 714.50 | 767.52 | 767.52 |
| M-iid | 50 | **2.71** | 4.74 | 714.50 | 714.50 | 768.09 | 768.09 |
| M-iid | 100 | **1.76** | 2.49 | 714.50 | 714.50 | 768.78 | 88.60 |
| M-iid | 200 | **0.90** | 1.45 | 714.50 | 121.24 | 51.61 | 18.17 |
| M-iid | 500 | **0.39** | 0.67 | 714.50 | 25.44 | 5.30 | 3.83 |
| M-iid | 1000 | **0.17** | 0.34 | 33.21 | 8.14 | 1.98 | 1.77 |
| M-iid | 2000 | **0.08** | 0.32 | 9.41 | 3.92 | 1.11 | 1.09 |
| M-shift | 50 | **10.67** | 19.35 | 977.60 | 977.60 | 1049.40 | 1049.40 |
| M-shift | 200 | **12.67** | 15.84 | 977.60 | 214.87 | 118.28 | 61.40 |
| M-shift | 1000 | **12.83** | 13.68 | 91.58 | 44.36 | 21.66 | 20.67 |
| M-iid-med | 25 | **2.22** | 3.74 | 24.86 | 24.93 | 5.26 | 5.51 |
| M-iid-med | 100 | **0.47** | 1.37 | 1.55 | 1.69 | 1.49 | 1.52 |
| M-iid-med | 1000 | **0.00** | 0.50 | 0.52 | 0.58 | 0.11 | 0.15 |

## Pre-declared checks
* **Validity**: coverage of every certified method (EBRO-DKW, EBRO-adaptive, KS-DRO, Band-DRO) was 1.00 in all 29 cells; no violations.
* **H1** (EBRO-adaptive < EBRO-DKW for n <= 500): strictly better in 7 of 20 cells, equal in 11 (both return the domain-minimax decision), worse in 2.
* **H2** (lowest primary regret among the 6 methods): Taylor - SAA 13/16, EBRO-adaptive 2/16 (T-bias n=200, T-shift n=200), EBRO-DKW 1/16 (T-shift n=1000); MSFT - SAA 13/13.
  EBRO-adaptive beats SAA in 3/16 Taylor and 0/13 MSFT cells; beats its fair counterpart Band-DRO in 10/16 Taylor and 3/13 MSFT cells; beats KS-DRO in 11/16 and 4/13.
* **Claims policy**: EBRO-adaptive does **not** have the lowest regret in a majority of cells of either dataset. No claim of general superiority is made.

## Regimes
* EBRO-adaptive was best with moderate samples (n = 200) under measurement bias or seasonal shift on the bounded electricity problem.
* The graded tolerance made EBRO consistent for the credal objective much faster (e.g. T-iid n = 200: credal regret 4.1% vs 16.0% for EBRO-DKW) but, because the credal and expected-cost minimizers differ, its large-n expected-cost regret is higher than EBRO-DKW's.
* On MSFT daily losses, the a priori domain [-21, 39] is wide relative to typical losses; every certified method is dominated by the global term at small n, and SAA is best in every cell.
