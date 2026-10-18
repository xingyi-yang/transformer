# Cursor Project Specification v2
## Real-SEND-ready body-weight modeling, Bayesian and transformer benchmarks, and experimental design

This is a self-contained replacement for the original specification. Build a functioning repository, not just a scaffold. Develop offline with synthetic SEND-like data, but implement a shared production-oriented ingestion contract for real local SEND extracts and authorized read-only Snowflake access. Never claim real-data validation from synthetic tests.

## 1. Objective and boundaries

Learn longitudinal body-weight distributions across historical studies and evaluate whether model-informed group allocations and analytical measurement schedules meet prespecified power, false-positive-rate, and precision criteria.

Required benchmarks: hierarchical spline model, Bayesian hierarchical longitudinal model, and a baseline-conditioned transformer. Optional: TFT, Chronos-2, DeepAR.

Primary task: pre-study prediction using biological context, baseline information, and planned regimens. Secondary task: early-history forecasting. Evaluate separately. No synthetic truth or future actual dosing in pre-study features.

Only body weight is targeted. This cannot establish total animal requirements for a toxicology program or justify reducing required welfare observations. No replacement of concurrent controls, unplanned adaptation, or regulatory-grade claims. Initial biological scope: one species, compatible study class, both sexes with explicit estimands. Additional species are exploratory.

## 2. Implementation and structure

Python 3.11+, src layout, pyproject.toml, pytest, ruff, typed interfaces, YAML configs. Core libraries: numpy, pandas, scipy, scikit-learn, pyarrow, pydantic, torch, matplotlib. Bayesian extra: PyMC and ArviZ. Snowflake extra: verified connector or Snowpark. Optional XPT extra: pyreadstat. Offline smoke never downloads weights or connects to services.

```
configs/{smoke,research,bayesian_smoke,bayesian_research}.yaml
configs/{real_local,real_snowflake,new_study}.example.yaml
configs/mappings/example_send.yaml
src/send_bw/
  cli.py
  data/{sources,schema,metadata,generate,harmonize,qc,splits,features}.py
  models/{base,spline,bayesian,transformer,covariance,optional_adapters}.py
  simulation/{truth,paths,analysis,designs,optimize,metrics}.py
  evaluation/{forecast,calibration,report}.py
  utils/{artifacts,seeds,security}.py
tests/{unit,integration,fixtures}/
docs/{methods,data_dictionary,limitations,real_data_qualification}.md
README.md
```

Implement a small end-to-end version before heavy models. Execute code/tests and report actual results. Mark absent dependencies/models as not_run. Do not fabricate metrics, convergence, APIs, or successful execution.

## 3. Single source-independent pipeline

Source -> raw domains and metadata -> version-aware mapping -> canonical tables -> QC/cohort -> frozen study splits -> features -> models -> simulator -> analysis/design report.

Define SendSource.inspect(), read_domain(name, columns, study_ids), and read_metadata(). Implement LocalSendSource and optional SnowflakeSendSource. All fitting code consumes canonical tables, never vendor-specific raw structures.

### Local source

Support Parquet and CSV; optional SAS XPT with explicit dependency handling. Accept per-study folders or combined domains. Preserve variable labels and metadata when available. Detect ambiguous duplicates and naming conflicts. Handle uppercase/lowercase through explicit normalization.

### Snowflake source

Read-only, disabled unless explicitly authorized and configured. Use allowlisted database/schema/tables, safe identifier validation, parameterized value filters, projected columns, batched extraction or server-side preparation. No write statements, DDL, grants or automatic result uploads. Credentials come from approved providers/environment references; never literal secrets. Redact logs and avoid logging raw records/identifiers. Export only to authorized storage. Record extraction timestamps/query IDs and consistency limitations. Mock-test all queries; real credentials are not needed in CI.

### Metadata and SEND variation

Use configured SEND version and mapping profiles. Support study-specific overrides, units, timing conventions, and supplemental qualifiers. Optional define.xml support must specify which features are parsed; disable external entity resolution. If unsupported, require a reviewed metadata mapping rather than guessing.

Do not join trial sets to arms blindly or assume all SEND versions/sponsors use identical assignment conventions. Compound identities require curated mappings; do not infer by splitting EXTRT strings. Planned regimens may require external curated metadata: actual future administrations are not substitutes for planned information.

## 4. Data contract and provenance

Synthetic tables are SEND-like, not certified SEND-conformant. Implement representative fields and document all extensions:
- DM: STUDYID, USUBJID, SUBJID, SEX, SPECIES, STRAIN, ARMCD, ARM, RFSTDTC, RFENDTC.
- BW: STUDYID, USUBJID, BWSEQ, BWTESTCD, BWTEST, BWORRES, BWORRESU, BWSTRESN, BWSTRESU, BWDTC, BWDY.
- EX: STUDYID, USUBJID, EXSEQ, EXTRT, EXDOSE, EXDOSU, EXROUTE, EXSTDTC, EXENDTC.
- Illustrative TX/TS subsets with documented schema, not universal official mappings.
- Separate study_metadata, arm_regimen, compound_reference, animal_status.

Canonical tables: study context, animal context, arm/planned regimen, administration events, body-weight observations, animal status, provenance, and splits. Keep original values, standardized values, units, source file/table/record, mapping version, and QC flags.

Verify positive numeric weights, conversions, keys, duplicates, assignment consistency and no many-to-many row inflation. Distinguish treatment/recovery, main/satellite cohorts, scheduled sacrifice, missed measurement, unscheduled removal and death. Partial dates need an explicit rule and uncertainty flag, not silent completion. Preserve fasting/measurement conditions where available. Never infer sex from weight.

Actual elapsed time is date difference from an explicitly selected reference; preserve source study day. Test the no-day-zero convention. Baseline is last eligible predose measurement by deterministic rules. Do not substitute post-treatment observations silently. Do not score interpolations as observed truth.

Hard mapping errors fail/quarantine. Soft warnings produce limitations. Report cohort exclusions and completeness by study, arm, sex and phase. Distinguish not_supported context from no_feasible_design.

## 5. Synthetic generator and hidden truth

Both synthetic and real inputs traverse the identical harmonizer. Store hidden generator parameters/full latent paths separately; model modules cannot read them. Classify every feature as production-available, optional or truth-only.

For log weight Z=log W, generator A:
Z_sgi(t)=log W_sgi(0)+growth(t,x)+effect(t,r,c)+B_s(t)'u_s+B_a(t)'b_sgi+e_sgi(t).

Anchor change terms at baseline and separate baseline measurement error explicitly. Generate sex-dependent baseline distributions, saturating natural growth, delayed dose effects, recovery, study random effects, animal random effects, and correlated residuals:
K(t,t')=sigma_p^2 exp(-|t-t'|/ell)+sigma_m^2 I(t=t').

Example effect is -A(c)*dose/(ED50(c)+dose)*H_r(t), with a documented administration-history response kernel. Parameters are illustrative, not biologically validated.

Vary study arm counts, compounds, regimens, allocations, schedules and durations. Share study conditions across groups. Use grams plus some original kilogram results for conversion tests. Generate coherent paths before applying schedules. Include MAR missingness; informative termination is a separately labeled extension requiring an estimand.

Independent generator B uses different growth/effect families, covariance mixtures, heavier tails, delayed/transient effects and variance shifts. It is not the fitted model under another name.

Smoke: 12–20 small studies. Research: configurable 120+ studies. Small smoke simulations verify software only.

## 6. Estimand and final analysis

Model log weight. Include baseline log weight as a covariate. Report grams and percent baseline change as secondary summaries.

Primary estimand: baseline-standardized active-control difference in mean log weight at fixed T:
Delta_g(T)=mu_g(T)-mu_0(T).
exp(Delta_g) is a geometric-mean ratio; 100*(exp(Delta_g)-1) is its percent difference, not necessarily an arithmetic-mean difference.

Secondary: fixed weighted average contrast sum_j a_j[mu_g(t_j)-mu_0(t_j)], with fixed query times/weights independent of observation schedule.

Prespecify meaningful effects, sex strategy, alpha, power, contrasts and multiplicity. Use Holm for multiple active-control comparisons; report per-comparison, any and all-required power distinctly.

Implement endpoint ANCOVA at T as a simple validation baseline. It cannot derive extra power from intermediate measurements it does not use. Do not claim schedule benefits for this analysis.

For schedule optimization implement repeated-measures GLS with fixed time basis, baseline, sex, group-by-time effects and training-estimated working within-animal covariance:
beta_hat=(X'V^-1 X)^-1 X'V^-1 y;
Delta_hat=c'beta_hat;
SE=sqrt(c'Var(beta_hat)c).

Specify model-based or cluster-robust covariance and finite-sample test behavior. Historical plug-in covariance is an approximation. Verify null behavior independently; designs failing inference checks are infeasible. Keep estimand fixed across schedules. All candidates include baseline and T.

## 7. Shared model interfaces

fit(train,validation,config), predict_marginals(context,times,history=None), sample_study(context,regimens,design,n_draws,seed), save/load.

Persist preprocessing, vocabularies, features, time/unit conventions, supported contexts and uncertainty with model parameters. IDs are grouping keys, not predictors for future studies. No silent mapping of unseen biological categories to familiar ones.

### Frequentist spline baseline

Mean includes baseline, sex, time splines, compound descriptors and dose/time interactions. Use regularization where needed. Fit study/animal/residual covariance using study-level out-of-fold residuals or a documented joint method. Avoid optimistically small in-sample residuals.

### Required baseline-conditioned transformer

Compact torch model, no download. Context tokens encode biology, baseline, compound descriptors and regimen. Query tokens encode actual future time and planned-dose features. Attention:
Attn(Q,K,V)=softmax(QK'/sqrt(d_k)+M)V.
No future targets enter inputs. Use padding/availability masks and observed-target loss. Normalize study contributions deliberately. Initial hidden size 32–64, 1–2 layers, 2–4 heads, dropout, AdamW and validation early stopping. Verify head/dimension compatibility.

Use the same covariance simulator as the spline for the primary backbone comparison. A separately labeled variance-learning version can be an extension. Avoid claiming a standard TFT when this is a custom architecture.

### Optional models

TFT: verified PyTorch Forecasting adapter for early-history forecasting.
Chronos-2: verified APIs, explicit download permission, zero-shot first, fine-tuning separately.
DeepAR: genuine probabilistic autoregressive recurrent model, not a relabeled point RNN.
Document task/covariate capability differences and never fabricate baseline-only support.

## 8. REQUIRED Bayesian hierarchical benchmark

Implement in PyMC with ArviZ diagnostics. A genuine posterior model is required; dropout/ensembles alone do not count.

Z_sgi(t)=f_beta(t,x,r)+B_s(t)'u_s+B_a(t)'b_sgi+epsilon_sgi(t).
u_s~Normal(0,Omega_s), b_sgi~Normal(0,Omega_a).

Use identifiable baseline-anchored bases. Begin with a compact random-effect structure; do not add unidentifiable intercept components. Add a correlated continuous-time residual process where supported and tested.

After training-only scaling, proposed priors:
beta_k~Normal(0,s_beta^2);
random-effect SDs~HalfNormal(s_scale);
ell~LogNormal(m_ell,s_ell);
correlation matrices~LKJ(eta) only if modeled.

Configure and justify prior scales through prior predictive weights in grams. Use noncentered effects as appropriate. Fit reference research runs using NUTS, multiple chains, documented warmup/draws/seeds. Report R-hat, bulk/tail ESS, divergences and other sampler warnings. Targets such as R-hat<=1.01 and no unresolved divergences are diagnostic goals, not proof of validity. Smoke draws are not adequate inference.

Required prior predictive, posterior predictive and held-out new-study checks; sensitivity to priors, random effects and heavy-tailed likelihood. If diagnostics fail, mark results unreliable and do not rank as a valid successful benchmark.

New-study prediction integrates posterior uncertainty and new random effects:
p(Y_new|D)=integral p(Y_new|theta,u_new,b_new)p(u_new,b_new|theta)p(theta|D)dtheta du db.

Sample one theta per hypothetical study, new study effect once, animal effects per animal. Do not reuse fitted effects for training studies/animals.

In the MAIN benchmark the Bayesian model is a response generator; the FINAL analysis is the same across all generators. Do not automatically introduce historical borrowing or remove controls. Robust MAP priors or Bayesian final decision rules are optional separate projects requiring prior-conflict checks and independent false-positive calibration. Posterior probability thresholds are not automatically frequentist alpha.

## 9. Joint trajectory simulation and uncertainty

Simulator: Z=mu_theta+B_s'u_s+B_a'b_i+epsilon(t). Sample correlated paths, not independent timepoint quantiles. Bound covariance jitter, log failures, use stable Cholesky solves.

Separate biological/process variation, study variation and parameter uncertainty. For non-Bayesian models use study-bootstrap refits or a clearly documented alternative. One parameter/bootstrap member applies to a complete simulated study; it is not independently resampled for every animal. Draw population effects for new studies.

For lognormal predictions, exp(mu) is median; mean is exp(mu+variance/2) under a Gaussian marginal. Prefer sampled summaries when assumptions differ.

Optional joint Gaussian training loss:
L_s=0.5*((y_s-mu_s)'Sigma_s^-1(y_s-mu_s)+log|Sigma_s|+n_s log(2*pi)).
Use observed subvectors for missing outcomes under the stated missingness assumptions. A simpler documented two-stage fit is acceptable for the first implementation.

## 10. Validation

Freeze study-level train/validation/test splits before fitting transforms/windows. Add compound and chronological holdouts where feasible. Do not tune on final tests. Use study-clustered uncertainty estimates.

Evaluate MAE/RMSE, contrast bias, individual/group-mean/contrast coverage and width, CRPS, trajectory energy score, between-animal variance, correlation by gap, increments and weight extrema. Report study-balanced and observation-weighted summaries, sex/arm/horizon results and effective sample sizes. Quantile accuracy alone does not validate dependence.

For missingness distinguish unscheduled removal/death from routine missed visits. Do not impute post-death weights as ordinary outcomes. Exclude informative-termination power claims until a defined estimand and compatible analysis are implemented.

## 11. Simulation design and optimization

Use ADEMP: aims, data-generating mechanisms, estimands, methods and performance measures.

Candidate d=(n_control,n_active...,schedule,T,analysis). Vary group counts and unequal allocations without fixed arm slots. Every study shares context across groups. Generate latent paths before selecting schedules. Use common random numbers for design comparisons where possible.

Conditional power evaluates specified relevant effects, not only optimistic model-predicted effects. Normalize effect shapes to the exact chosen contrast. Independently calculate truth for nonlinear scenarios.

Assurance(d)=integral Power(d;theta)p(theta|historical_data)dtheta is an optional separate quantity, not fixed-effect power.

For B replicates:
power_hat=mean(reject under alternative);
FWER_hat=mean(any true-null rejection);
MCSE=sqrt(p_hat*(1-p_hat)/B);
bias=mean(delta_hat-delta_true);
coverage=mean(CI contains delta_true).

Use binomial confidence intervals, including at zero/all rejections. Run global and partial nulls. Record convergence failures and prespecified handling. Use independent final evaluation batches after search; smoke replication is not scientifically conclusive.

Cost C=c_A*N_animals+c_M*N_analytical_measurements+c_T*T. Prefer Pareto frontiers when costs are arbitrary. Feasibility includes power lower bounds, FWER upper bounds with explicit numerical tolerances, precision, failure rate and operational/welfare constraints. Allow no_feasible_design.

Fit/propose using model simulators, evaluate finalists under generator B and stress scenarios: covariance shift, higher heterogeneity, delayed/transient effects and heavy tails. Add a clearly labeled synthetic oracle reference. Do not claim resource improvement unless both candidate and comparator meet requirements. Resampling historical studies does not directly prove prospective power.

## 12. Tests and acceptance

Unit tests: keys/units, baseline and day conventions, join inflation, no truth/future leakage, study splits, varying arms, shared effects, positive-definite/time-sensitive covariance, lognormal means, known GLS contrasts, Holm, partial null truth, finite MC intervals, no-feasible-design, failure handling. Endpoint analysis must not falsely benefit from intermediate visits.

Independently authored messy SEND fixtures: partial dates, naming variations, unit changes, supplemental mappings, recovery cohorts, missing planned regimens and duplicate extracts. Local/mock-Snowflake paths must yield equivalent canonical outputs. Test read-only SQL, secrets redaction, offline behavior and serialization/new-study inference.

Integration: complete CPU smoke with real training for spline/transformer; Bayesian small smoke if declared dependencies installed, otherwise clearly incomplete. Expensive statistical calibration tests run separately, not as flaky tiny-simulation unit tests.

Real-data qualification stages: (1) implemented/mock-tested, (2) authorized pilot reconciliation, (3) scientist-reviewed cohort mappings, (4) held-out real-study evaluation, (5) prospective shadow validation. State achieved stage, never certify real readiness from stage 1 alone.

## 13. Commands, artifacts and delivery

Implement generate, inspect-source, ingest, prepare, validate-real, train, evaluate, design, report, run-all, validate-request, propose-design. Example:

send-bw run-all --config configs/smoke.yaml
send-bw inspect-source --config configs/real_local.example.yaml
send-bw validate-request --request configs/new_study.example.yaml
send-bw propose-design --model ARTIFACT --request configs/new_study.example.yaml

Future request includes context/baseline distribution, planned groups/regimens, horizon, fixed estimand/effects, analysis, constraints and missing-feature declarations. Run applicability gates before recommendations; refuse unsupported extrapolation.

Artifacts: Parquet cohorts, data dictionary, mapping/QC/exclusion reports, frozen splits, models, training histories, Bayesian diagnostics, predictive metrics, scenario truth, simulation metrics/MC intervals/failures, feasible designs/Pareto set, plots, Markdown/HTML report. Manifest records data/mapping/config/model versions, seeds, software versions, git revision if available, timestamps, source lineage and safe file hashes.

README documents installation, exact executed commands, CPU/research modes, optional dependencies, outputs and limitations. Keep real records out of public examples/version control. Synthetic report must be prominently labeled.

Build in phases: data contracts and generator -> conventional analysis/simulator -> spline/transformer/Bayesian models -> held-out validation -> repeated-measures design engine -> independent evaluation -> real-data qualification and optional adapters. Implement source contracts early, not after coupling models to synthetic truth. Never overwrite user files destructively.

Final handoff must state exactly what ran, tests/results, skipped components, Bayesian convergence status, real-data qualification stage and next steps. Do not call the project complete while required functionality is only a placeholder.

## 14. Background references

Verify installed APIs; references are not validation of this implementation.
- CDISC SEND: https://www.cdisc.org/standards/foundational/send
- TFT documentation: https://pytorch-forecasting.readthedocs.io/en/stable/api/pytorch_forecasting.models.temporal_fusion_transformer.TemporalFusionTransformer.html
- Chronos: https://github.com/amazon-science/chronos-forecasting
- ADEMP: https://pmc.ncbi.nlm.nih.gov/articles/PMC6492164/
- Bayesian animal-study planning: https://arxiv.org/abs/2301.05572
- MAP heterogeneity methods: https://www.jstatsoft.org/article/view/v100i19/4188
- NC3Rs: https://eda.nc3rs.org.uk/

Start by inspecting the workspace, creating a brief execution checklist, and implementing the shared data contract plus offline generator. Ask only when ambiguity blocks safe execution.
