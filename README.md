# PrivDiCE: reproducibility materials for peer review

This repository provides the code, experimental records, and manuscript source for **“PrivDiCE: Privacy-Preserving Actionable Counterfactual Explanations with Differential Privacy and Homomorphic Encryption.”** The release is organized so that reported values can be traced to saved tables and acceptance records without rerunning model training.

For a first review, consult the following files in order:

1. [`ManuScript.tex`](ManuScript.tex) for the article;
2. [`docs/REPRODUCIBILITY_AUDIT.md`](docs/REPRODUCIBILITY_AUDIT.md) for the scope and known limitations of the release;
3. [`docs/RUN_MANIFEST.md`](docs/RUN_MANIFEST.md) for the nine accepted experimental runs;
4. [`results/`](results/) for the tables used to verify the manuscript's reported values.

## The release supports the manuscript's one-training-seed claims

The manuscript reports one fixed training run for each dataset. The classifier seed is `55`, the CounterGAN/DP-CounterGAN training seed is `11`, and the outer counterfactual search seed is `11`. Each dataset contributes 200 factual instances, with 100 evaluated in each prediction direction.

Privacy attacks use 10 attacker or resampling seeds. CKKS timing uses three fixed candidate cohorts and five timing repetitions per cohort. These repetitions estimate attack and timing variability; they are not additional generator-training runs. Likewise, factual-level bootstrap intervals and paired tests quantify variation within the frozen run, not variation across independent model retraining.

## Each manuscript claim is linked to a saved record

| Path | Role in the evidence chain |
| --- | --- |
| `ManuScript.tex`, `ref.bib`, `figures/`, `images/` | Article source, references, and figures |
| `src/` | Frozen plaintext, privacy-attack, and CKKS benchmark implementations |
| `notebooks/` | Nine executed Kaggle notebook snapshots and their saved kernel metadata |
| `results/<dataset>/plaintext/` | Aggregate counterfactual results and plaintext acceptance record |
| `results/<dataset>/official_dice/` | Full-cohort Official DiCE results and failure accounting |
| `results/<dataset>/he/` | CKKS latency, communication, error, and agreement tables |
| `artifacts/` | Extracted run outputs for ECG and Heart+, where redistribution is permitted |
| `data/` | Redistributable ECG and Heart+ inputs and the MIMIC-IV access contract |
| `docs/EXPERIMENTAL_SUPPLEMENT_ONE_SEED.md` | Audited public rendering of the one-seed experimental supplement |
| `tools/audit_release.py` | Automated checks of acceptance records and headline values |

## Nine accepted runs form the experimental record

| Dataset | Plaintext experiment | Official DiCE extension | CKKS benchmark |
| --- | --- | --- | --- |
| Leipzig ECG | [Kaggle](https://www.kaggle.com/code/lamhuy8904/leipzig-ecg-q1-v5-5-gen-only-one-seed-paper-t4) · [executed snapshot](notebooks/ecg/plaintext/leipzig-ecg-q1-v5-5-gen-only-one-seed-paper-t4.ipynb) | [Kaggle](https://www.kaggle.com/code/lamhuy8904/leipzig-ecg-v5-5-official-dice-extension-paper-sav) · [executed snapshot](notebooks/ecg/official_dice/leipzig-ecg-v5-5-official-dice-extension-paper-sav.ipynb) | [Kaggle](https://www.kaggle.com/code/lamhuy8904/leipzig-ecg-v5-5-he-population-paper) · [executed snapshot](notebooks/ecg/he/leipzig-ecg-v5-5-he-population-paper.ipynb) |
| Heart+ | [Kaggle](https://www.kaggle.com/code/huylmhuhu/heart-q1-v5-5-gen-only-one-seed-paper-t4) · [executed snapshot](notebooks/heartplus/plaintext/heart-q1-v5-5-gen-only-one-seed-paper-t4.ipynb) | [Kaggle](https://www.kaggle.com/code/huylmhuhu/heart-v5-5-official-dice-extension-paper)¹ · [executed snapshot](notebooks/heartplus/official_dice/heart-v5-5-official-dice-extension-paper.ipynb) | [Kaggle](https://www.kaggle.com/code/huylmhuhu/heart-v5-5-he-population-paper) · [executed snapshot](notebooks/heartplus/he/heart-v5-5-he-population-paper.ipynb) |
| MIMIC-IV | [Kaggle](https://www.kaggle.com/code/buiquocviet/mimic-iv-q1-v5-5-gen-only-one-seed-paper-t4) · [executed snapshot](notebooks/mimic/plaintext/mimic-iv-q1-v5-5-gen-only-one-seed-paper-t4.ipynb) | [Kaggle](https://www.kaggle.com/code/buiquocviet/mimic-iv-v5-5-official-dice-extension-paper)¹ · [executed snapshot](notebooks/mimic/official_dice/mimic-iv-v5-5-official-dice-extension-paper.ipynb) | [Kaggle](https://www.kaggle.com/code/buiquocviet/mimic-iv-v5-5-he-population-paper) · [executed snapshot](notebooks/mimic/he/mimic-iv-v5-5-he-population-paper.ipynb) |

¹ The saved metadata marks the Heart+ and MIMIC-IV Official DiCE pages as private at audit time. Their executed snapshots, aggregate tables, and acceptance records are included locally, so the numerical record remains reviewable even when an external page is unavailable.

All nine checked-in notebooks are manually saved executed versions: every code cell retains its execution count and the saved cell outputs, and none contains an error output. The machine-readable CSV/JSON material under `results/` and the extracted ECG/Heart+ outputs under `artifacts/` remain the canonical numerical evidence.

## Data availability follows the original access conditions

| Dataset | Included | Source and release condition |
| --- | --- | --- |
| Leipzig ECG | Yes | [PhysioNet v1.0.0](https://physionet.org/content/leipzig-heart-center-ecg/1.0.0/), DOI [10.13026/7a4j-vn37](https://doi.org/10.13026/7a4j-vn37), ODC Attribution 1.0 |
| Heart+ | Yes | Derived from processed CDC BRFSS 2020/2022 tables; provenance and checksum are recorded in [`data/heartplus/README.md`](data/heartplus/README.md) |
| MIMIC-IV | No | [MIMIC-IV v2.2](https://physionet.org/content/mimiciv/2.2/), DOI [10.13026/6mm1-ek67](https://doi.org/10.13026/6mm1-ek67); credentialed access and a data-use agreement apply |

The repository does not redistribute the MIMIC-IV extract, row-level counterfactuals, trained checkpoints, or complete run archive. It retains only aggregate tables that do not disclose patient rows. Authorized users can reconstruct the expected local input by following [`data/mimic/README.md`](data/mimic/README.md).

## Reported values can be checked without retraining the models

The plaintext runs used Kaggle Tesla T4 sessions. CKKS measurements used CPU-only Kaggle sessions with four logical and two physical CPU cores. Core software versions were Python `3.12.13`, PyTorch `2.10.0+cu128`, Opacus `1.6.0`, and TenSEAL `0.3.16`.

From the repository root, run:

```bash
python3 tools/audit_release.py
shasum -a 256 -c SHA256SUMS.txt
```

The first command verifies the nine acceptance records, seed contracts, factual counts, headline metrics, CKKS correctness, and notebook status. The second verifies every tracked file against the release checksum manifest.

## Comparisons must be read within their access and compute conditions

- Counterfactual validity is reported as `Valid-CFE Yield@10`: the number of valid, feasible CFEs divided by `K=10`. Missing slots remain failures.
- Official DiCE uses the same factual cohort, target labels, margin, and constraints as the proposed method, but follows its native `sample_size`, `maxiterations`, and timeout controls. It is a matched-outcome comparison, not an equal-candidate-budget comparison.
- DiCE-style and Wachter-style baselines use backward access to the predictor, whereas the forward-only search methods use counted candidate evaluations. Their access conditions are therefore reported separately.
- HE label agreement measures numerical consistency between CKKS and the frozen plaintext predictor. It does not measure counterfactual utility.
- Composed HE search cost is calculated from measured latency per encrypted population round and the frozen number of search rounds. It is an estimate of end-to-end search cost, not a directly timed full-cohort encrypted run.
- The supplied MIMIC-IV extract has no patient identifier; its split is stratified by row rather than patient-disjoint.

## Citation and reuse remain subject to dataset licenses

Citation metadata is provided in [`CITATION.cff`](CITATION.cff). Dataset licenses and attribution requirements continue to apply. In particular, no repository-wide permission overrides the MIMIC-IV credentialed-data agreement or the licenses of the ECG and BRFSS-derived inputs.
