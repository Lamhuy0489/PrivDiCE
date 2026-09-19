# What the public release establishes

Audit date: 2026-09-19.

## The manuscript numbers come from the one-seed experimental record

`Bổ sung.md` is the source document for the experiments reported in the manuscript. Its public rendering is [`EXPERIMENTAL_SUPPLEMENT_ONE_SEED.md`](EXPERIMENTAL_SUPPLEMENT_ONE_SEED.md); account identifiers and machine-local paths were redacted without changing the numerical tables. The pre-redaction source had SHA-256:

```text
c8a945ba9e9885744404cdd8949f19611c5ae3a4de5cfcdb2407139547580d18
```

`Bổ sung full3.md` contains later confirmation runs with training seeds 22 and 33. Those runs are useful supporting evidence, but they are not pooled with the one-seed values reported in the current manuscript. All manuscript tables remain anchored to training seed 11.

## All nine reported runs pass their acceptance checks

| Dataset | Plaintext CFE | Official DiCE | CKKS/HE | Primary budget and population |
| --- | --- | --- | --- | --- |
| Leipzig ECG | Accepted; paper numbers | Accepted; 200 queries and 400 method rows | Accepted; paper numbers | B=512, P=64 |
| Heart+ | Accepted; paper numbers | Accepted; 200 queries and 400 method rows | Accepted; paper numbers | B=1,024, P=128 |
| MIMIC-IV | Accepted; paper numbers | Accepted; 200 queries and 400 method rows | Accepted; paper numbers | B=512, P=64 |

Each plaintext run records classifier seed 55, generator-training seed 11, search seed 11, 100 factuals per direction, `K=10`, both target directions, and 8,800 factual–method–budget rows. Each CKKS run records candidate-cohort seeds 101, 202, and 303, with five timing repetitions per cohort.

## The audit recomputes the values most likely to affect the conclusions

The automated audit reads the canonical CSV and JSON files and checks:

- classifier Accuracy, F1, and ROC-AUC for all three datasets;
- directional Yield@10, Coverage, Full-10, Robust Yield, Proximity, Sparsity, Diversity, Plausibility, Fidelity, and runtime for the primary ε=4 method;
- primary generator-only MIA AUC and its confidence interval;
- selected-population HE latency, communication, maximum logit error, and label agreement;
- acceptance flags, random-seed contracts, factual counts, and input checksums.

The recomputed values match the supplement and manuscript at the reported precision. Running `python3 tools/audit_release.py` prints the individual checks.

## The release has explicit limitations rather than hidden substitutions

1. All nine checked-in notebooks are manually saved executed copies: every code cell retains its execution count and outputs, and the accepted copies contain no error output. The aggregate numerical record is also preserved in `results/` and, for ECG and Heart+, in `artifacts/`.
2. The saved metadata marks the Heart+ and MIMIC-IV Official DiCE notebooks as private. Their executed local copies, aggregate tables, and acceptance records remain available for review.
3. MIMIC-IV input records, row-level outputs, checkpoints, and the complete run archive are excluded because the source is governed by credentialed access and a data-use agreement.
4. Ten privacy-attack seeds and 15 CKKS timing repetitions do not replace independent generator retraining. The current manuscript is correctly described as a one-training-seed study.
5. The reported encrypted search time is a composed estimate based on measured round latency and frozen search rounds; it is not presented as a directly timed full-cohort encrypted run.
6. Third-party datasets retain their original licenses. A code license cannot override the ECG, BRFSS-derived, or MIMIC-IV data conditions.

## Baseline comparisons are matched where the implementations permit it

Uniform Random, Genetic CFE, and CounterGAN-SD use the same counted candidate budgets as the proposed forward-only search. DiCE-style and Wachter-style methods require backward access to the predictor and are therefore reported in a separate access stratum. Official DiCE shares the factual cohort, targets, `K`, margin, and feasibility constraints, but uses native stopping controls and a 20-second per-factual timeout. Timeout and no-CF cases remain failures in the denominator. This design supports a matched-outcome comparison without claiming identical computation across incompatible APIs.

## Public availability stops at the boundary imposed by the source data

The release includes the manuscript source, implementation, notebook snapshots, aggregate tables, acceptance records, redistributable ECG and Heart+ inputs, and extracted ECG and Heart+ run artifacts.

It excludes credentials, API tokens, caches, the MIMIC-IV input, MIMIC-IV-trained checkpoints, row-level MIMIC-IV outputs, and the complete MIMIC-IV run archive.
