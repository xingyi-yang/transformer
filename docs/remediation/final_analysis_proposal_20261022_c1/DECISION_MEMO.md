# Final-analysis decision memo (C1 specified and implemented as opt-in)

## Status

- **Synthetic diagnostic.**
- **Not design qualification.**
- **No real-data validation.**
- **No operational recommendation.**

This supersedes the 2026-10-18 draft. It does not overwrite that draft. Visit-saturated means are a **candidate**, not a validated primary analysis. Monte Carlo calibration has **not** been run.

## Implementation status (this revision)

Changed in code:

- `src/send_bw/simulation/analysis.py`: added `mmrm_visit_contrast` and `secondary_visit_weights`.
- `src/send_bw/simulation/designs.py`: `mmrm` is an allowed opt-in analysis name.
- `src/send_bw/simulation/optimize.py`: explicit dispatch; unknown names raise.
- `tests/unit/test_mmrm_visit_contrast.py`.
- `docs/methods.md`, `docs/limitations.md` (status notes only).

Not changed:

- `endpoint_ancova` and `rm_gls` formulas, defaults, and df.
- Request configs (`analyses: [ancova, gls]`).
- Production CLI. There is still **no** `calibrate-analysis` command.
- No model refits, no ensemble weights, no new forecasting architecture, no sampling/MCMC, no real-data access.

C1 exists as a library function. Calling it from design search requires adding `"mmrm"` to a request. Default smoke and example requests do not.

## 1. Missingness

| Mechanism | Definition | C1 claim |
| --- | --- | --- |
| Complete | Every scheduled visit observed | Saturated mean recovers \(\Delta_g(T)\) on noiseless fixtures. Sandwich SE is an experimental variance estimator. |
| MCAR | Absence independent of outcomes and of covariates except the realized design | Observed-data GLS functional equals \(\Delta_g(T)\) when the mean is saturated. Not Monte Carlo calibrated. |
| Baseline-covariate-dependent | Absence may depend on \(\mathrm{zb}\), sex, group | Same functional claim, because those covariates are already in the mean. Not outcome-dependent. |
| Prior-outcome MAR dropout | Absence depends on earlier \(Z\) | **Not claimed.** Historical \(V_0\) GLS plus sandwich is not generally consistent when \(V_0\) is misspecified. No justified MAR method is implemented. |

The existing design simulator (`paths.simulate_batch`) draws missed visits and dropout independently of \(Z\) given the design. That is MCAR / design-dependent, not prior-outcome MAR. Do not describe those draws as outcome-dependent MAR.

## 2. Candidate C1 specification

Mean at visit \(j=1,\dots,m\):

\[
E[Z_{ij}\mid x_i]=\alpha_j+\beta_{\mathrm{sex},j}\mathbb{I}(\mathrm{sex}_i=1)+\beta_{\mathrm{zb},j}(\mathrm{zb}_i-\bar{\mathrm{zb}})+\sum_{g=1}^{G}\gamma_{g,j}\mathbb{I}(\mathrm{group}_i=g).
\]

\(p=(3+G)m\). Endpoint contrast \(c_g^\top\theta=\gamma_{g,m}=\Delta_g(T)\).

Observational unit: animal \(i\), contributing \(Z_{i,o_i}\). Within-animal serial dependence is handled only through working \(V_0[o_i,o_i]\). Shared study effects that are common across arms at visit \(j\) are absorbed by \(\alpha_j\) and cancel in concurrent-control contrasts. **Treatment-specific study effects** would leave within-arm dependence that animal-level clustering does not capture. Current generators draw one shared study term per study, not treatment-specific study terms.

GLS point estimate:

\[
\hat\theta=B\sum_i X_i^\top V_0[o_i,o_i]^{-1}Z_{i,o_i},\qquad B=\Bigl(\sum_i X_i^\top V_0[o_i,o_i]^{-1}X_i\Bigr)^{-1}.
\]

Animal scores and sandwich:

\[
u_i=X_i^\top V_0[o_i,o_i]^{-1}\hat r_i,\qquad
\widehat{\mathrm{Cov}}=\frac{N}{N-p_{\mathrm{visit}}}B\Bigl(\sum_i u_i u_i^\top\Bigr)B,
\]

\[
\widehat{\mathrm{Var}}(c_g^\top\hat\theta)=c_g^\top\widehat{\mathrm{Cov}}\,c_g.
\]

**Experimental** finite-sample choices (not claimed conservative; not Kenward–Roger):

- \(p_{\mathrm{visit}}=3+G\), the complete-data per-visit between-animal count (same as ANCOVA). Using \(p=(3+G)m\) as the HC1 divisor overstates complete-data expected variance (~1.24 SE ratio in the earlier algebraic check). \(p_{\mathrm{visit}}\) matches complete-data HC1; it is not a proven small-sample correction under missingness.
- Reference: Student \(t\) with \(\mathrm{df}=N_T-(3+G)\), \(N_T=\) animals observed at \(T\).

Holm across \(G\) comparisons at nominal \(\alpha=0.05\). Fail if any group has fewer than 2 endpoint observations, \(\mathrm{df}<2\), singular \(V_0[o_i,o_i]\), \(\mathrm{cond}>10^{12}\), or \(\mathrm{SE}\le 0\). Failures are failures.

C2 / Satterthwaite is **deferred**. No verified implementation of covariance-scale uncertainty for that route is in this repository.

## 3. Complete-data equivalence (corrected)

| Quantity | A vs C1 under complete data |
| --- | --- |
| Point coefficients \(\hat\Delta_g(T)\) | Exact identity for any PD \(V_0\) (Kronecker design). Fixture max \(\lvert\Delta\mathrm{est}\rvert=1.78\times 10^{-15}\). |
| Population variance of \(\hat\Delta_g(T)\) | Same as ANCOVA when residuals are iid across animals at \(T\) and study effects are shared. |
| Expected estimated variance | Not identical: OLS uses \(s_T^2(X^\top X)^{-1}\); C1 uses HC1/CR1. |
| Realized finite-sample SE | **Not equal.** Deterministic nonzero-residual example: OLS SE \([0.01052, 0.01052]\) vs C1 CR1 \([0.00775, 0.01167]\) (ratios \(0.737\), \(1.109\)). |

See `complete_data_se_check.json`.

## 4. Algebraic estimand check (complete, noiseless)

From `algebraic_estimand_fixtures.csv` (2 schedules \(\times\) 3 shapes \(\times\) 4 scenarios \(\times\) 3 candidates):

| Candidate | Max \(\lvert\mathrm{bias}\rvert\) | Max null-arm spillover | Recovers \(\Delta_g(T)\)? |
| --- | ---: | ---: | --- |
| A `endpoint_ancova` | \(8.3\times 10^{-17}\) | \(5.1\times 10^{-17}\) | Yes |
| B `rm_gls_quadratic` | \(0.01606\) | \(3.8\times 10^{-15}\) | No (weekly transient) |
| C1 `mmrm_visit_contrast` | \(2.7\times 10^{-15}\) | \(2.6\times 10^{-15}\) | Yes |

Weekly transient alternative, historical default \(V_0\): B estimates \((-0.0893,-0.1339)\) versus \((-0.10,-0.15)\). A and C1 recover the endpoint. On biweekly (\(m=k=2\)) B is exact because the quadratic basis spans two visits.

These fixtures are complete data. They do not validate sandwich size or MAR dropout.

## 5. Initial diagnostic matrix (not executed)

Analyses: A, existing B, specified C1. C2 held out.

Missingness: **complete** only.

Scenarios (8): well-specified global and both partial nulls; heterogeneity global and partial-low; transient alternative and both transient partial nulls.

Sizes: \(6/6/6\) and \(8/8/8\) per sex. Equal allocation. Schedules: endpoint-only (A), biweekly and weekly (B and C1). Common random numbers: one latent path per (scenario, size, replicate), then visit subsets.

Deferred (kept for a later locked evaluation, not dropped to claim qualification): covariance shift, heavy tails, delayed alternatives, 5% meaningful effect, control-heavy allocation, twice-weekly, prior-outcome MAR, C2.

## 6. Acceptance criteria and sources

| Role | Value | Source | Meaning |
| --- | ---: | --- | --- |
| Nominal \(\alpha\) | 0.05 | `configs/new_study.example.yaml` `alpha` | Test size, not a Monte Carlo pass rule. |
| FWER point ceiling | \(0.05+0.015=0.065\) | `fwer_max`, `fwer_tolerance` | Design feasibility tolerance, **not** exact 0.05 control. |
| FWER CI ceiling | \(0.05+0.045=0.095\) | `fwer_upper_tolerance` | Applied in `optimize.py::feasibility` to the max over null cells. |
| Coverage floor | 0.92 | `coverage_min` | Per-candidate minimum coverage. |
| Bias fraction | 0.15 | `max_abs_bias_fraction` | Relative to \(\lvert\delta\rvert\) on the alternative. |
| Failure rate | 0.02 | `max_failure_rate` | Failed replicates stay in the denominator. |
| Power target | 0.80 | `power_target` | **Design-specific** on a chosen alternative cell. |
| Power CI floor | \(0.80-0.06=0.74\) | `power_lower_tolerance` | Same selected design, not every small \(n\). |

`Request` dataclass defaults (`fwer_tolerance=0.01`) are **not** the source. The example request YAML is.

Per-cell operating characteristics are reported for every cell. Passing a tolerated FWER gate is not “FWER = 0.05.” \(n=6\) is not required to reach 80% power.

## 7. Cell inventory and resources

| Stream | \(B\) | Unique study datasets | Analysis fits | Seed |
| --- | ---: | ---: | ---: | ---: |
| Timing-only pilot (requested) | 20 | \(8\times 2\times 20=320\) | \(320\times 5=1{,}600\) | after approval |
| Exploratory | 600 | 9{,}600 | 48{,}000 | 20261101 |
| Diagnostic locked | 2000 | 32{,}000 | 160{,}000 | 20261201 |

Fits per unique study, under CRN: A + B weekly + B biweekly + C1 weekly + C1 biweekly \(=5\).

No wall-clock claim is made. A 1{,}600-fit timing-only pilot is requested before any exploratory or locked stream. That pilot is discarded for FWER, coverage, bias, and power.

Dependencies: numpy, scipy, pandas (already required). No new package.

Proposed later commands (not executed; CLI does not exist yet):

```text
python3 -m pytest tests/unit/test_mmrm_visit_contrast.py -q
# after approval: timing-only pilot, then exploratory stream, then locked diagnostic stream
# no send-bw calibrate-analysis command exists
```

## 8. Questions still needing approval

1. Approve the 1{,}600-fit timing-only pilot?
2. After timing, run the exploratory stream (\(B=600\)) on the small matrix only?
3. Keep C2 deferred until a derived Satterthwaite formula exists?
