# Focused synthetic objective-validation protocol

> **Synthetic diagnostic. Exploratory. Not design qualification. No real-data
> validation. No operational recommendation.**

This protocol is the execution plan for `send-bw v3-objective`. It does not
open real SEND extracts. Prior checksummed run directories stay untouched.

## Starting assessment

`docs/v3/objective_assessment.md` scores existing artifacts (pilot / eval /
repl / research / closeout / increment / limitations) on functionality versus
demonstrated performance. Those labels are the baseline. This protocol is the
fresh evaluation that can change the **performance** column.

Unresolved limitations carried in:

- O11 failed: matched increment ratio mean-of-dataset-means **1.417** (n=2).
- O12 failed: research-scale Bayesian cells `mean_identifiability`.
- O10 failed: n=2 was not a precision calculation.

## Development versus frozen evaluation

Model changes are allowed only on **development** simulations (generate seed
31001 / train seed 51001; not the frozen namespace).

Allowed development levers:

1. Residual-process SD scale so the validation increment moment approaches 1
   (`increment_moment_split: validation`). Test and `test_compound_holdout`
   are refused.
2. Bayesian `identifiable_mean` (drop strain×time when n_studies < 10; drop
   near-zero columns; QR relative threshold 1e-6). Reliability thresholds stay
   `rhat≤1.01`, `ESS≥400`, `divergences==0`.

Frozen evaluation uses generate seed **41001** and train seed **61001**.
Nothing is retuned after viewing frozen test metrics.

## Known-truth grid

| Axis | Levels |
|---|---|
| Mechanism | `matched` = shared; `growth_shift` = partially shared; `both` = conflicting |
| Target-study history | full; sparse `n_keep=4` |
| Sharing | context-specific (k=0); pooled (k=1); selective v1 (k=1) |
| Models | spline + transformer every cell; Bayesian on one matched/context/full cell |

Target splits are frozen once per (dataset, train) pair and reused across
sharing and banks.

## Monte Carlo precision

Prior matched increment-ratio sd of dataset means = 0.144. Target SE of the
mean ≤ 0.10 requires `n ≥ (0.144/0.10)² = 2.07`. Use **n_datasets = 3**,
**n_train_seeds = 1**. Train-seed split reshuffling is a nested nuisance, not
the outer DGM replicate.

## Distributional pass (then design)

Primary increment slice: spline, matched / shared, context-specific, full
target history. Pass if the mean of dataset means of the matched pred/obs
increment ratio is in **[0.80, 1.20]**.

Bayesian reliability is scored separately against the unchanged gate. Sharing
and the transformer are **not** required to pass or to win.

If the increment check passes, score model-informed designs against
**independent generator B**. `B_final < 2000` is exploratory / nonconclusive.
No operational recommendation is emitted.

## Artifacts

- Config: `configs/v3_objective.frozen.yaml`
- Output: `runs/v3_objective/` (not a preserved directory)
- Verdict: `docs/v3/objective_verdict.md`
