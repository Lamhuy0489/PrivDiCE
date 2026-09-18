# PrivDiCE reproducibility package

This repository accompanies **“PrivDiCE: Privacy-Preserving Actionable Counterfactual Explanations with Differential Privacy and Homomorphic Encryption.”** It contains the manuscript source, exact one-training-seed experiment code, canonical aggregate results, source notebooks, public-data inputs, and machine-readable acceptance records used for the reported tables.

## Scope of the released evidence

The manuscript is based on one fixed training run per dataset:

- classifier training seed: `55`;
- CounterGAN/DP-CounterGAN training seed: `11`;
- outer counterfactual search seed: `11`;
- 200 factuals per dataset: 100 in each prediction direction;
- MIA, memorization, and attribute attacks: 10 attacker/resampling seeds, **not** 10 generator-training runs;
- CKKS timing: 3 fixed candidate-cohort seeds × 5 repetitions = 15 timing runs per population.

Accordingly, the plaintext CFE tables are one-generator-training-seed results. Factual-level bootstrap intervals and paired tests quantify uncertainty within this frozen run; they do not establish between-training-run variability.

## Repository map

| Path | Contents |
| --- | --- |
| `ManuScript.tex`, `ref.bib`, `figures/`, `images/` | Paper source and figures |
| `src/` | Frozen pipeline, privacy-attack, HE benchmark, and notebook-builder code |
| `notebooks/` | Kaggle notebook source snapshots and kernel metadata |
| `results/<dataset>/plaintext/` | Canonical aggregate plaintext tables and acceptance JSON |
| `results/<dataset>/official_dice/` | Full-cohort Official DiCE aggregate results and audit |
| `results/<dataset>/he/` | CKKS timing, communication, correctness, and acceptance tables |
| `artifacts/` | Fully extracted output directories where redistribution is permitted |
| `data/` | Publicly redistributable ECG and Heart+ inputs; MIMIC-IV is intentionally excluded |
| `docs/EXPERIMENTAL_SUPPLEMENT_ONE_SEED.md` | Audited public rendering of the one-seed supplement derived from `Bổ sung.md` |
| `docs/REPRODUCIBILITY_AUDIT.md` | Release audit, limitations, and manuscript-to-artifact mapping |
| `docs/RUN_MANIFEST.md` | All nine canonical runs, local files, output records, and available public links |
| `tools/audit_release.py` | Integrity and headline-number audit |

## Canonical Kaggle runs

| Dataset | Plaintext paper | Official DiCE extension | HE population benchmark |
| --- | --- | --- | --- |
| Leipzig ECG | [local notebook](notebooks/ecg/plaintext/leipzig-ecg-q1-v5-5-gen-only-one-seed-paper-t4.ipynb) | [local notebook](notebooks/ecg/official_dice/leipzig-ecg-v5-5-official-dice-extension-paper-sav.ipynb) | [local notebook](notebooks/ecg/he/leipzig-ecg-v5-5-he-population-paper.ipynb) |
| Heart+ | [local notebook](notebooks/heartplus/plaintext/heart-q1-v5-5-gen-only-one-seed-paper-t4.ipynb) | [local notebook](notebooks/heartplus/official_dice/heart-v5-5-official-dice-extension-paper.ipynb) | [local notebook](notebooks/heartplus/he/heart-v5-5-he-population-paper.ipynb) |
| MIMIC-IV | [one-seed plaintext](https://www.kaggle.com/code/buiquocviet/mimic-iv-q1-v5-5-gen-only-one-seed-paper-t4) | [Official DiCE](https://www.kaggle.com/code/buiquocviet/mimic-iv-v5-5-official-dice-extension-paper)¹ | [CKKS/HE](https://www.kaggle.com/code/buiquocviet/mimic-iv-v5-5-he-population-paper) |

¹ The downloaded Kaggle metadata reports the MIMIC-IV Official DiCE notebook as private. Its code snapshot and aggregate results are present here, but the Kaggle owner must make the notebook public before a reviewer can open the external link. The Heart+ external account link is intentionally omitted; its complete local record is checked in.

Kaggle's kernel pull API returned source-only `.ipynb` files with cleared execution counts and outputs. Eight checked-in notebooks are therefore labeled source snapshots. The original Heart+ Official DiCE 20-second notebook was also available as a manually saved executed notebook and is checked in with its cell outputs. For every run, the canonical output record remains the audited CSV/JSON tables in `results/` and, where licensing permits, the fully extracted output directories in `artifacts/`.

## Data provenance and redistribution

| Dataset | Released here? | Source and status |
| --- | --- | --- |
| Leipzig ECG | Yes | [PhysioNet v1.0.0](https://physionet.org/content/leipzig-heart-center-ecg/1.0.0/), DOI [10.13026/7a4j-vn37](https://doi.org/10.13026/7a4j-vn37), ODC Attribution 1.0 |
| Heart+ | Yes | Derived from CDC BRFSS 2020/2022 processed tables; detailed provenance and checksum are in `data/heartplus/README.md` |
| MIMIC-IV | No | [MIMIC-IV v2.2](https://physionet.org/content/mimiciv/2.2/), DOI [10.13026/6mm1-ek67](https://doi.org/10.13026/6mm1-ek67); credentialed access and DUA apply |

The MIMIC-IV CSV, trained checkpoints, row-level counterfactuals, and complete output archive are deliberately not committed. Public aggregate tables that do not disclose patient rows are retained. An authorized user can place the exact extract locally as described in `data/mimic/README.md`.

## Reproduce and audit

The original runs used Kaggle Tesla T4 notebooks for plaintext training and CPU-only Kaggle sessions for CKKS. Core versions were PyTorch `2.10.0+cu128`, Opacus `1.6.0`, Python `3.12.13`, and TenSEAL `0.3.16`. See `requirements.txt` and the notebook install cells for the remaining dependencies.

Run the release audit from the repository root:

```bash
python3 tools/audit_release.py
shasum -a 256 -c SHA256SUMS.txt
```

The audit checks all nine acceptance records, seed contracts, factual/result counts, input checksums, primary manuscript metrics, HE timing/correctness, notebook-output status, and Kaggle visibility metadata.

## Important interpretation limits

- CFE validity is `Valid-CFE Yield@10`: valid and feasible returned CFEs divided by `K=10`; missing slots remain failures.
- Official DiCE uses the same cohort, targets, margin, and constraints, but native `sample_size`/`maxiterations` and timeouts. It is a matched-outcome comparison, not an equal-compute comparison.
- HE label agreement evaluates numerical equivalence of CKKS and the frozen plaintext graph. It is not CFE utility.
- Reported composed HE search cost multiplies measured latency per encrypted population round by frozen search-round counts; it is not a directly observed full-cohort encrypted end-to-end run.
- MIMIC-IV lacks a patient identifier in the supplied experimental extract, so its split is stratified by row rather than patient-disjoint.

## Citation and licensing

Citation metadata is provided in `CITATION.cff`. Dataset-specific licenses and attribution requirements remain in force. No single repository-wide license is asserted over third-party data; users must follow each source license and the MIMIC-IV credentialed-data agreement.
