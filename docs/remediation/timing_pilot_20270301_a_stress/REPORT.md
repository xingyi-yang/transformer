# A-default stress timing-only pilot

## Status

- **Synthetic diagnostic.**
- **Not design qualification.**
- **No real-data validation.**
- **No operational recommendation.**
- **Timing-only A-default stress pilot. Not an operating-characteristic evaluation.**

- Seed `20270301`; 20 replicates; complete data; 11 scenarios × 2 sizes × 5 analyses.
- Mechanisms: covariance shift (`sigma_p` ×1.6, `ell` ×2), heavy tails (t(3) measurement ×1.5), delayed alternatives.
- A is the subject. B and C1 are comparators. Production defaults stay `[ancova]`. C1 is not retuned.
- Fits recorded: **2200**. Study datasets: **440**.
- Wall clock: **0.860 s**. Mean **0.360 ms/fit** (median 0.295; p90 0.652).
- Fit failure rate (software stability only): **0.0000**.
- Working covariance: unstressed historical plug-in. Closed-chapter heterogeneity and transient cells were not re-run.

## Per-analysis timing

| Analysis | Fits | Mean ms | Median ms | Max ms | Failure rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| `A_endpoint_ancova` | 440 | 0.095 | 0.093 | 0.334 | 0.0000 |
| `B_rm_gls_quadratic` | 880 | 0.397 | 0.441 | 2.236 | 0.0000 |
| `C1_mmrm_visit_contrast` | 880 | 0.456 | 0.520 | 0.891 | 0.0000 |

## Linear projections from this pilot (not guaranteed)

- Exploratory B=600, 66,000 fits: about **23.8 s**.
- Diagnostic locked B=2000, 220,000 fits: about **79.2 s**.

These seconds are `n_fits × mean_ms_per_fit`. They ignore process startup and any later I/O.

## Not computed

FWER, coverage, bias, interval width, and power were not calculated.

## Constraints on use

- Do not cite this pilot as an operating-characteristic result.
- Do not retune C1 or restore GLS as a default.
- Do not treat a later pass on these mechanisms as design qualification.
