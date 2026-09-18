# Reproducibility release audit

Audit date: 2026-09-19.

## Source-of-truth decision

`Bổ sung.md` is the correct experimental source for this manuscript. `EXPERIMENTAL_SUPPLEMENT_ONE_SEED.md` is its public rendering: account identifiers and machine-local paths were redacted without altering the numeric tables. The pre-redaction source had SHA-256:

```text
c8a945ba9e9885744404cdd8949f19611c5ae3a4de5cfcdb2407139547580d18
```

`Bổ sung full3.md` contains later seeds 22/33 and must not be mixed into the current manuscript's one-seed claims. The supplement links to those confirmatory runs for context, but the canonical tables and manuscript numbers remain anchored to training seed 11.

## Acceptance status

| Dataset | Plaintext | Official DiCE | HE | Primary budget/population |
| --- | --- | --- | --- | --- |
| Leipzig ECG | accepted; paper numbers | accepted; 200 queries/400 rows | accepted; paper numbers | B=512, P=64 |
| Heart+ | accepted; paper numbers | accepted; 200 queries/400 rows | accepted; paper numbers | B=1,024, P=128 |
| MIMIC-IV | accepted; paper numbers | accepted; 200 queries/400 rows | accepted; paper numbers | B=512, P=64 |

Every plaintext run records classifier seed 55, generator-training seed 11, search seed 11, 100 factuals per direction, K=10, both directions, and 8,800 per-factual method/budget rows. Every HE run records candidate cohort seeds 101/202/303 and five timing repetitions per cohort.

## Headline-number verification

The automated audit recomputes the following directly from canonical CSVs:

- classifier Accuracy/F1/ROC-AUC for all three datasets;
- primary ε=4 directional Yield@10, Coverage, Full-10, Robust Yield, Proximity, Sparsity, Diversity, Plausibility, Fidelity, and runtime;
- primary generator-only MIA AUC and confidence intervals;
- selected-population HE latency, communication, maximum logit error, and label agreement;
- acceptance flags, seed contracts, and exact input checksums.

The values match the one-seed supplement and manuscript to the shown precision. Run `python3 tools/audit_release.py` for the detailed pass list.

## Publication issues found

1. **Notebook outputs:** eight Kaggle API downloads contain source cells only and must not be described as executed-output notebooks. The manually saved Heart+ Official DiCE 20-second notebook retains outputs. Canonical output evidence for all runs is supplied in `results/` and `artifacts/`.
2. **Kaggle visibility:** Heart+ Official DiCE and MIMIC-IV Official DiCE are marked private in their downloaded metadata. Their owners must publish them before peer review.
3. **MIMIC-IV redistribution:** the protected input, row-level outputs, checkpoints, and complete ZIP are excluded. Publishing them in a public GitHub repository would be inappropriate under the source access contract.
4. **Single training seed:** 10 privacy-attack seeds and 15 HE timings do not turn the CFE experiment into a multi-training-seed study. The limitation remains explicit.
5. **HE end-to-end wording:** the manuscript implementation section was clarified so that the reported encrypted system cost is described as a composed estimate, consistent with the Results and Limitations sections.
6. **Repository license:** no blanket license should be applied to third-party datasets. A separate code license may be chosen by the authors, but it must not override dataset licenses or the MIMIC-IV agreement.

## Fairness interpretation

Uniform Random, Genetic CFE, and CounterGAN-SD share counted candidate budgets with the proposed forward-only search. DiCE-style and Wachter-style methods have white-box backward access and are reported in a separate access stratum. Official DiCE uses native controls and a 20-second factual timeout; it shares factuals, targets, K, margin, and constraints but not an equal candidate-compute budget. Timeout and no-CF rows remain failures in the denominator.

## Public-release boundary

Included: paper source, pipeline code, source notebooks, aggregate tables, acceptance records, public ECG/Heart+ data, and extracted redistributable ECG/Heart+ artifacts.

Excluded: credentials, API tokens, caches, MIMIC-IV input, MIMIC-IV-trained checkpoints, MIMIC-IV row-level outputs, and the complete MIMIC-IV artifact archive.
