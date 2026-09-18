# Q1-V5.2 one-seed plaintext smoke contract

Status: V5.1 smoke accepted on 2026-08-22. ECG version 3, Heart+ version 5 and
MIMIC version 5 pass their notebook gates and the independent audit. The older
folders named `latest` remain stale old-code downloads; use the `q1_v51_*`
folders listed in `Q1_V51_SMOKE_RESULT_AUDIT_2026-08-22.md`. Smoke numbers are
diagnostics and must not be copied into `Bổ sung.md` as paper evidence.

V5.2 is a new method-validation smoke and does not invalidate the accepted
V5.1 artifacts. It must pass ECG first. There is no one-seed paper promotion;
the final paper experiment is multi-seed.

## Reproducibility scope

- Classifier training/tuning seed: 55 only.
- CounterGAN and DP-CounterGAN training seed: 11 only.
- Main search seed: 11 only.
- MIA attack seeds are repeated calibration/evaluation replicates and are not
  classifier, generator or CFE-search retraining seeds.
- The one-seed design is a runtime-bounded smoke experiment. It does not support
  a cross-training-seed stability or paper-result claim.

## Shared CFE search contract

- Requested set size: `K=10`; missing slots are failures.
- Main population: `P=32` for every dataset and method.
- Smoke and paper budgets: `B={64,128,256,512}`, corresponding to at most
  `{2,4,8,16}` rounds. Smoke reduces factuals, not the budget frontier.
- Population/round sensitivity is matched by candidate budget. ECG uses
  `B=128` with `P=16,32,64`; Heart+ and MIMIC use `B=512` with
  `P=32,64,128`. This table does not silently change the primary operating
  population.
- Disease-to-no-disease and no-disease-to-disease are evaluated separately,
  then combined with an equal-direction macro-average.
- Every oracle-scored proposal, including prune/fill proposals, consumes the
  candidate budget. Search stops at the budget cap; no uncounted rescue or
  post-benchmark pruning is allowed.

## Leakage-free inner selection

The predeclared profiles (`balanced`, `sparse`, `diverse`, `sparse_diverse`,
plus `utility_balanced` for Heart+/MIMIC) are compared only on an inner
calibration cohort with the DP-epsilon-4 checkpoint. ECG uses inner `B=128`
because it is already saturated there; Heart+ and MIMIC use inner `B=512`
because frozen-checkpoint diagnostics show that `B=128` is below their useful
search region. ECG retains macro and worst-direction noninferiority. Heart+ and
MIMIC deliberately do not optimize equality between directions: eligibility
uses macro Valid-CFE yield, Full-K success, robust completeness and constraint
validity, while the direction gap remains a reported outcome. Their frozen
score uses macro utility, feature/group Sparsity, Diversity and Robustness.
Proximity is reported and may trade off; no outer-test row is used for
selection. The selected profile is reused unchanged for DP epsilon 16/8/4/2.

## Methods and fairness

The fixed-budget primary panel contains uniform random, Genetic CFE,
DiCE-style gradient, Wachter-style optimization, CounterGAN one-shot,
iterative CounterGAN, and the proposed DP16/DP8/DP4/DP2 sparse-diverse search.
Methods receive the same requested K and declared budget semantics; missing K
positions remain failures. Official/native DiCE and Wachter runs are emitted in
a separate native-baseline panel because native optimization steps are not
identical to counted population-based oracle calls.

## Per-factual endpoints

Every endpoint is first computed for each factual point, never from one chosen
example. Outputs include Valid-CFE yield@10, Coverage@1, Full-K success,
Proximity, feature and group Sparsity, value and changed-set Diversity,
plausibility distance, plausibility inlier rate, target-data support, local
surrogate Fidelity, Constraint Validity, target margin, perturbation Robustness,
round/candidate/time milestones to K, time-to-K-or-cap and throughput.
Direction tables report mean, standard deviation, median and quartiles; the
equal-direction summary includes factual-clustered bootstrap 95% intervals.
Paired per-factual comparisons use bootstrap intervals, Wilcoxon tests and Holm
correction.

Conditional validity of an archive containing only accepted rows is an
integrity diagnostic and may equal one. The comparative validity endpoint is
Valid-CFE yield@10, whose denominator always remains ten.

## Security and qualitative scope

Every CounterGAN/DP checkpoint receives the calibrated MIA-V2 panel,
memorization-distance and attribute-inference diagnostics. Formal DP accounting
is reported separately from empirical attacks. Explanation-Linkage uses the
exact released CFE set and is reported by method and direction. The qualitative
grid contains three representatives per direction in smoke and five per
direction in paper, comparing all baselines and proposed epsilon checkpoints.
No outer utility or security outcome is used to tune the search profile or
epsilon.

V5.2 saves the jointly accounted discriminator with each GAN checkpoint. DP
search uses its bounded realism score and never queries raw train kNN for CFE
selection. kNN distance, LOF inlier rate and target-data support remain post-hoc
evaluation only. The MIA panel adds a released-discriminator attack and pairs
latent draws across epsilon checkpoints using common random numbers.

Every V5.2 DP checkpoint is trained with Opacus `secure_mode=true`. Kaggle T4
compatibility was verified with Python 3.12, PyTorch 2.10/CUDA 12.8 and a source
build of `torchcsprng` commit `13e04cd`. The smoke acceptance gate requires
`secure_rng=true` and backend `torchcsprng:/dev/urandom` for DP16/8/4/2. This
hardens randomness but does not change epsilon, delta or the requested noise
scale. Exact bitwise reruns are not expected, so checkpoints and checksums are
mandatory artifacts.

## Plaintext/HE boundary

These notebooks retrain and evaluate the complete plaintext pipeline and write
no CKKS benchmark table. A later HE notebook must load the checksum-locked
handoff, disable dropout at inference, evaluate only the add/multiply graph and
compare ciphertext logits/labels with the exact plaintext HE-compatible graph.

## Submitted smoke kernels

- ECG: `REDACTED_KAGGLE_OWNER/leipzig-ecg-cfe-q1-v5-one-seed-plaintext-smoke-t4`
- Heart+: `REDACTED_KAGGLE_OWNER/heart-cfe-q1-v5-one-seed-plaintext-smoke-t4`
- MIMIC-IV: `buiquocviet/mimic-iv-cfe-q1-v5-one-seed-plaintext-smoke-t4`

Version 1 of all three reached the late manuscript-export stage but referenced
two stale V4 variable names. Version 2 fixes the export names, adds a static
rejection for those names, and was submitted for all three datasets. All three
version-2 kernels were `RUNNING` immediately after resubmission. Do not submit
paper until outputs are downloaded and independently audited.

ECG version 2 subsequently completed and passed the independent audit. Heart+
and MIMIC version 2 exposed a visualization-only failure when a Pareto scatter
received NaN coordinates from methods with zero returned CFEs. Version 3
filters non-finite plot rows without altering any table, metric, search result
or selection rule; Heart+ and MIMIC were resubmitted. The ECG version-2 result
remains valid because its plot and final acceptance gate both completed.

## Latest downloaded-result review and V5.1 decision

The three kernels report `COMPLETE`, but the newly downloaded utility and
linkage files have the same SHA-1 hashes as ECG-v2/Heart+-v4/MIMIC-v4. Their
acceptance JSON also lacks the new all-method/all-direction linkage gate.
Therefore these are old-code reruns, not V5.1 validation.

At `B=128`, old-code smoke macro Valid-CFE yield@10 was approximately 1.00 for
ECG, 0.10 for Heart+ and 0.37--0.47 for MIMIC DP variants. This does not justify
paper submission from the old smoke operating point. A frozen-checkpoint
diagnostic (no retraining, hence design evidence only) is stored under:

- `experiments/cloud/results/heartplus/q1_v5_frozen_search_diagnostic/`
- `experiments/cloud/results/mimic/q1_v5_frozen_search_diagnostic/`

For DP epsilon 4, Heart+ `gentle_prune` at `B=512` achieved smoke
Valid-CFE yield@10 about 0.70 with feature/group Sparsity about 0.62; the two
directions were about 0.73 and 0.67. MIMIC's frozen selected profile at `B=512`
achieved about 0.83 yield and 0.55 feature Sparsity; directions were about 0.67
and 1.00. A more validity-first MIMIC profile reached about 0.92 yield but cut
Sparsity to about 0.25, so it is not automatically preferable. These values
must not be presented as paper results because the diagnostic inspected the
smoke outer cohort. They only motivate the predeclared V5.1 inner-selection
grid and dataset-specific inner operating budgets.

V5.1 also represents zero-release Explanation-Linkage cells explicitly with
`released_cfes=0`, `release_available=false` and undefined attack rates instead
of silently dropping those method-direction pairs. All six regenerated
notebooks pass the 22-check validator. The next valid action is to upload and
run the V5.1 smoke notebooks; do not run or cite paper notebooks until those
new smoke artifacts independently pass. V5.1 was submitted to the existing
slugs as ECG version 3, Heart+ version 5 and MIMIC version 5; all three reported
`RUNNING` immediately after upload.
