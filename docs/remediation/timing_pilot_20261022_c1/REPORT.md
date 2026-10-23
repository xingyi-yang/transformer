# C1 timing-only pilot

## Status

- **Synthetic diagnostic.**
- **Not design qualification.**
- **No real-data validation.**
- **No operational recommendation.**
- **Timing-only pilot. Not an operating-characteristic evaluation.**

- Seed `20261022`; 20 replicates; complete data; 8 scenarios × 2 sizes × 5 analyses.
- Fits recorded: **1600**. Study datasets: **320**.
- Wall clock: **0.626 s**. Mean **0.360 ms/fit** (median 0.293; p90 0.664).
- Fit failure rate (software stability only): **0.0000**.

## Per-analysis timing

| Analysis | Fits | Mean ms | Median ms | Max ms | Failure rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| `A_endpoint_ancova` | 320 | 0.095 | 0.093 | 0.302 | 0.0000 |
| `B_rm_gls_quadratic` | 640 | 0.395 | 0.417 | 0.870 | 0.0000 |
| `C1_mmrm_visit_contrast` | 640 | 0.458 | 0.453 | 0.745 | 0.0000 |

## Linear projections from this pilot (not guaranteed)

- Exploratory B=600, 48,000 fits: about **17.3 s**.
- Diagnostic locked B=2000, 160,000 fits: about **57.6 s**.

These seconds are `n_fits × mean_ms_per_fit`. They ignore process startup and any later I/O.

## Not computed

FWER, coverage, bias, interval width, and power were not calculated.
