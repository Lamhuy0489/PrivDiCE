# Q1-V5.2 method, utility and security contract

## Research claim and scope

The proposed contribution is one conditional CounterGAN CFE pipeline with a
warm-up/ramp training schedule, record-level DP-SGD operating points, counted
validity-preserving semantic-group pruning, Pareto/MMR set selection, and an
HE-compatible one-logit MLP oracle. Counterfactuals are model contrasts, not
causal treatment recommendations.

Formal differential privacy, empirical generator attacks, release-linkage
risk and HE input confidentiality answer different questions and must be
reported in separate tables. An empirical attack near random guessing is not a
proof of privacy, and HE inference does not strengthen the GAN training epsilon.

## Leakage-free selection order

1. MLP architecture, preprocessing and threshold are selected on development
   data only. The one-logit threshold is absorbed into the final bias.
2. GAN loss schedule and DP mechanism are selected without observing the outer
   CFE cohort. DP checkpoints use the final epoch; no private-validation epoch
   selection is performed.
3. Search profile and operating budget are selected on inner validation only.
   ECG retains a worst-direction noninferiority gate. Heart+ and MIMIC use
   macro utility without a direction-equality gate; their direction gap is an
   outcome, not a tuning penalty.
4. The outer cohort is used once for utility, release-linkage and attack
   evaluation. It never changes a weight, threshold, orientation or method.
5. The exact frozen plaintext graph, preprocessing, generator, query manifest
   and search contract are handed to the separate CKKS notebook.

For every DP checkpoint, the discriminator is released together with the
generator under the already declared basic composition of their two DP-SGD
mechanisms. During DP-CFE search, its bounded realism cost
`1-sigmoid(D(x'))` replaces the former 5-NN cost against raw training rows.
The train-fitted kNN/LOF quantities remain post-hoc evaluation metrics only;
they cannot rank or select a released DP CFE. This makes the search a
post-processing operation of the jointly accounted DP GAN outputs, conditional
on the explicitly public oracle, preprocessing and constraint contract.

Because the accepted ECG outer result has already been inspected during prior
development, a newly promoted ECG adaptive profile requires nested
subject-disjoint confirmation or must be labelled post-hoc sensitivity
analysis. It cannot silently replace the accepted result.

## Shared benchmark contract

- K=10 requested CFE per factual.
- Population=32 base proposals per round.
- Paper candidate budgets are 64/128/256/512 for ECG, Heart+ and MIMIC-IV.
- The ECG-only 32/64/96/128 frontier is a development saturation diagnostic,
  not a replacement for the shared paper frontier.
- Search seeds are 11/22/33 and missing CFE positions remain failures.
- Disease-to-no-disease and no-disease-to-disease are reported separately,
  followed by an equal-direction macro-average.
- Directional yield is not forced to be equal. A gap is retained and discussed
  when the target classes have different prevalence or boundary difficulty.
- Every proposal scored by the oracle, including pruning/refinement proposals,
  consumes candidate budget and time.
- Donor rows, safe banks, rescue points, padding and uncounted post-processing
  are forbidden.

Random search and Genetic use the same oracle-call budget. Official DiCE and
native Wachter retain their native optimization budgets and are reported in a
separate native-baseline panel rather than being described as equal oracle-call
algorithms.

## Utility and quality endpoints

Primary endpoints:

- Valid-CFE yield@10;
- Coverage@1;
- Full-K success;
- Robust completeness@10 under the frozen perturbation protocol;
- constraint validity.

Secondary endpoints:

- Proximity (lower is better);
- feature and semantic-group Sparsity (higher is better);
- value Diversity and changed-set Diversity (higher is better);
- plausibility distance (lower), plausibility inlier rate and target-data
  support (higher);
- target-logit margin;
- generation time to K or cap, rounds and candidate evaluations.

All endpoints are computed per factual before direction-specific and
equal-direction aggregation. Conditional validity of an archive containing
only valid rows is a diagnostic and cannot replace Yield@K.

## Predeclared Pareto rule allowing a Proximity trade-off

An adaptive candidate is compared with the accepted DP4 sparse-diverse search
using identical checkpoint, factuals, search seeds and scored-candidate budget.
It is eligible only when all non-inferiority gates pass on inner validation:

- Yield@10 difference at least -0.01;
- Full-K difference at least -0.02;
- Robust completeness difference at least -0.02;
- constraint validity remains 1.00;
- Sparsity difference at least -0.05;
- Proximity increase no greater than both +0.02 absolute and +25% relative.

After these gates, the candidate must improve at least two of Robust
completeness, Sparsity, Diversity, target-data support or time-to-cap, including
at least one improvement in Robust completeness or Diversity. Only then is it
retained on the Pareto frontier. A scalar score may break a tie among eligible
profiles but may never compensate for a failed gate.

This rule explicitly permits a bounded Proximity cost for a reproducible gain
in robust validity or set diversity. It does not permit lowering Validity until
a table looks less saturated.

## Formal DP table

Every CounterGAN/DP16/DP8/DP4/DP2 checkpoint is a separate row containing:

- requested epsilon ceiling, achieved epsilon and delta;
- adjacency unit and privacy scope;
- accountant and library version;
- clipping norm, noise multiplier and sample rate for generator and
  discriminator;
- requested and actual optimizer steps;
- warm-up/ramp schedule and checkpoint-selection policy;
- secure RNG state;
- per-checkpoint and joint-release composition.

The generator and discriminator both directly process sensitive factual rows
in this conditional architecture, so their privacy losses are conservatively
composed. Releasing several epsilon checkpoints or several independently
trained seeds requires an additional joint-release composition row.

Historical V4/V5.1 checkpoints use `secure_rng=false`; they are research
controls, not cryptographically hardened releases. Active V5.2 DP checkpoints
must use Opacus `secure_mode=true` with `torchcsprng` backed by `/dev/urandom`.
The source dependency is pinned to the Kaggle-tested commit `13e04cd`; every DP
row must pass an acceptance gate for the backend. Secure RNG changes the
randomness implementation and training cost, not the declared noise multiplier
or privacy accountant formula. The frozen oracle and development-fitted preprocessing
are public auxiliary inputs to the GAN accounting. The claim is record-level
GAN-training DP, not whole-pipeline or patient-level DP. In particular, ECG
beat-level adjacency does not imply patient-level protection.

## Empirical generator-security panel

Protocol `GENERATOR_PRIVACY_ATTACK_V2` evaluates every frozen checkpoint with:

- ten attack seeds;
- 1,000 calibration members and 1,000 calibration nonmembers;
- 1,000 disjoint evaluation members and 1,000 outer-test nonmembers;
- matched class counts inside each member/nonmember comparison;
- 16 latent draws per factual for reconstruction attacks.

Calibration train/validation rows choose score orientation and threshold.
Outer evaluation attack labels never select either. Common latent random
numbers are used across epsilon checkpoints within an attack seed. Five attacks
are reported:

1. single-draw loss-threshold baseline;
2. multi-draw minimum-reconstruction attack;
3. multi-draw mean-objective attack;
4. calibrated Gaussian reconstruction-likelihood attack.
5. released-discriminator score attack, following the LOGAN threat surface.

Each attack reports AUC, balanced accuracy, calibrated-threshold TPR/FPR,
membership advantage `TPR-FPR`, TPR at FPR 0.1%/1%/5%, attack-seed standard
deviation and bootstrap 95% interval. The strongest measured attack for a
checkpoint is reported without cherry-picking a favourable weak attack.

Memorization reporting contains exact match, near-duplicate rates, DCR mean/
median/1st/5th percentile, NNDR mean/median and train-to-holdout DCR ratio.
Attribute inference uses a predeclared sensitive feature and reports the attack
beside a trivial median/majority baseline. It measures sensitive-query-attribute
leakage through a conditional CFE output; it is not a membership attack and is
not a quantity protected by GAN-training DP when the user supplies the factual.

DP epsilon is selected by the formal privacy-utility frontier, not by minimizing
outer-test MIA. Non-monotonic empirical attacks do not invalidate the formal
accountant and do not support a claim that DP universally defeats MIA.

## Released-CFE linkage panel

Explanation-Linkage is run on the exact CFE release used by the quantitative
and qualitative evaluation. Every method and direction is a separate row. The
table contains singleton release groups, single-sensitive-value groups, direct
one-map linkage, one-to-four linkage, no auxiliary match and matched-group
size. Quasi-identifiers, sensitive attribute, development-fit discretization
and auxiliary-table assumptions are persisted. This is a simulated linkage
risk, not real-patient re-identification.

## HE separation

The CKKS notebook performs only add/multiply inference with dropout disabled.
It compares the exact plaintext HE-compatible graph against ciphertext logits
and reports label/margin agreement, logit error, encryption/server/decryption/
selection latency, time to K, ciphertext/context size, peak RAM and throughput
for populations 64/128/256/512. HE results never substitute for plaintext CFE
utility or GAN privacy attacks.

## Current evidence boundary — 2026-08-22

The retained V5.1 smoke artifacts contain the nearest audited generator attack,
memorization, attribute-inference and method-by-direction linkage diagnostics;
their interpretation is summarized in
`Q1_V51_SMOKE_RESULT_AUDIT_2026-08-22.md`. They use one generator-training seed
and `secure_rng=false`, so they are historical controls rather than current
paper security evidence.

V5.2 must regenerate the complete five-attack panel and exact-release linkage
tables with secure RNG enabled. The ten attack seeds quantify attack-protocol
variation, not GAN retraining. Generator-training seeds 22 and 33 are still
required if the final paper claims training-seed stability.
