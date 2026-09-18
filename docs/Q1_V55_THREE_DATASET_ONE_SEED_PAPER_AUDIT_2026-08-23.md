# Q1-V5.5 three-dataset one-seed paper audit — 2026-08-23

## Decision

ECG, Heart+ and MIMIC-IV version-1 one-seed paper kernels completed and pass
both their notebook acceptance gates and an independent local audit. All three
report `accepted=true` and `paper_numbers=true`. They are accepted one-seed
paper artifacts, not multi-seed estimates.

| Dataset | Kernel | Rows | Factuals | Methods | Budgets | Actual kernel time |
|---|---|---:|---:|---:|---:|---:|
| ECG | `REDACTED_KAGGLE_OWNER/leipzig-ecg-q1-v5-5-gen-only-one-seed-paper-t4` | 8,800 | 100/direction | 11 | 4 | 4.63 h |
| Heart+ | `REDACTED_KAGGLE_OWNER/heart-q1-v5-5-gen-only-one-seed-paper-t4` | 8,800 | 100/direction | 11 | 4 | 2.29 h |
| MIMIC-IV | `buiquocviet/mimic-iv-q1-v5-5-gen-only-one-seed-paper-t4` | 8,800 | 100/direction | 11 | 4 | 6.88 h |

Actual kernel time is taken from the last Kaggle log timestamp. Do not sum
`phase_wall_times.csv`, because nested phases overlap and that sum double
counts wall time.

Independent checks reproduce both label directions, unique factual/method/
budget keys, common `run_seed`, paired `robustness_seed`, and full candidate
cap consumption for Uniform Random, Genetic, the matched non-private
sparse/diverse method and DP16/8/4/2. All security contracts declare
`deployment_release=generator_only`; discriminator MIA is a supplementary
hypothetical-release ablation only.

## Classifier results

| Dataset | Accuracy | F1 | ROC AUC | PR AUC | Balanced accuracy | MCC |
|---|---:|---:|---:|---:|---:|---:|
| ECG | 0.9046 | 0.9249 | 0.9833 | 0.9923 | 0.9254 | 0.8095 |
| Heart+ | 0.8771 | 0.3490 | 0.8392 | 0.2943 | 0.6925 | 0.3001 |
| MIMIC-IV | 0.8840 | 0.8574 | 0.9496 | 0.9414 | 0.8767 | 0.7609 |

Heart+ is highly imbalanced. Accuracy must therefore be reported with F1,
PR AUC, balanced accuracy and MCC rather than used alone.

## Selected-budget equal-direction macro results

| Dataset/method | Yield@10 | Full-K | Sparsity | Diversity | Robust completeness | Proximity |
|---|---:|---:|---:|---:|---:|---:|
| ECG Genetic | 1.0000 | 1.000 | 0.4148 | 0.0806 | 0.9010 | 0.0745 |
| ECG iterative CounterGAN | 1.0000 | 1.000 | 0.1713 | 0.0446 | 0.9610 | 0.1110 |
| ECG matched non-DP sparse/diverse | 1.0000 | 1.000 | 0.7735 | 0.0689 | 0.8160 | 0.0704 |
| ECG DP16 | 1.0000 | 1.000 | 0.7677 | 0.0730 | 0.7725 | 0.0678 |
| ECG DP8 | 1.0000 | 1.000 | 0.7784 | 0.0666 | 0.7585 | 0.0626 |
| ECG DP4 | 1.0000 | 1.000 | 0.7684 | 0.0709 | 0.7845 | 0.0698 |
| ECG DP2 | 1.0000 | 1.000 | 0.7645 | 0.0743 | 0.8155 | 0.0694 |
| Heart+ Genetic | 0.3975 | 0.210 | 0.6539 | 0.0287 | 0.3650 | 0.0524 |
| Heart+ iterative CounterGAN | 0.7605 | 0.675 | 0.3322 | 0.0442 | 0.7540 | 0.1600 |
| Heart+ matched non-DP sparse/diverse | 0.7875 | 0.750 | 0.6240 | 0.0660 | 0.7585 | 0.1048 |
| Heart+ DP16 | 0.7295 | 0.685 | 0.6422 | 0.0615 | 0.7030 | 0.0971 |
| Heart+ DP8 | 0.7490 | 0.690 | 0.6461 | 0.0618 | 0.7155 | 0.0957 |
| Heart+ DP4 | 0.7265 | 0.675 | 0.6431 | 0.0604 | 0.6945 | 0.0958 |
| Heart+ DP2 | 0.7550 | 0.705 | 0.6457 | 0.0602 | 0.7275 | 0.0955 |
| MIMIC Genetic | 0.9430 | 0.910 | 0.2726 | 0.0869 | 0.8445 | 0.0870 |
| MIMIC iterative CounterGAN | 0.9950 | 0.995 | 0.1238 | 0.0376 | 0.9765 | 0.1367 |
| MIMIC matched non-DP sparse/diverse | 0.9950 | 0.995 | 0.5418 | 0.0893 | 0.9185 | 0.1040 |
| MIMIC DP16 | 0.9950 | 0.995 | 0.5687 | 0.0876 | 0.9000 | 0.0971 |
| MIMIC DP8 | 0.9950 | 0.995 | 0.5872 | 0.0811 | 0.8970 | 0.0909 |
| MIMIC DP4 | 0.9950 | 0.995 | 0.5632 | 0.0829 | 0.9230 | 0.1030 |
| MIMIC DP2 | 0.9950 | 0.995 | 0.5175 | 0.0922 | 0.9095 | 0.1183 |

ECG validity is saturated over 200 factuals, but the endpoints are not all
perfect. The proposed variants gain about 0.35 absolute Sparsity over Genetic
and about 0.60 over iterative CounterGAN, with an explicit robustness
trade-off. Heart+ remains non-saturated and directionally asymmetric. MIMIC
retains 0.995 macro Yield@10 while materially improving Sparsity and Diversity
over iterative CounterGAN.

## Directional proposed-method Yield@10

| Dataset/method | Disease → no disease | No disease → disease |
|---|---:|---:|
| ECG DP16/8/4/2 | 1.000 | 1.000 |
| Heart+ DP16 | 0.908 | 0.551 |
| Heart+ DP8 | 0.912 | 0.586 |
| Heart+ DP4 | 0.907 | 0.546 |
| Heart+ DP2 | 0.904 | 0.606 |
| MIMIC DP16/8/4/2 | 1.000 | 0.990 |

Natural direction gaps are retained and must be shown alongside macro values.

## Privacy accounting and MIA

Every DP checkpoint uses secure RNG and its achieved composed epsilon is below
and close to the requested 16/8/4/2 ceiling. Formal accounting is the primary
privacy claim; empirical attacks are diagnostics.

| Dataset | DP16 MIA AUC | DP8 | DP4 | DP2 |
|---|---:|---:|---:|---:|
| ECG | 0.4475 | 0.4538 | 0.5016 | 0.6054 |
| Heart+ | 0.4983 | 0.4997 | 0.4988 | 0.5031 |
| MIMIC-IV | 0.5042 | 0.5098 | 0.5052 | 0.5026 |

Heart+ and MIMIC attacks are near chance. ECG DP2 is an empirical warning:
its strongest attack AUC is 0.6054 with a 95% interval about 0.5942–0.6155.
It must not be hidden or used to retune the method on outer attack labels.

## Explanation-Linkage

Explanation-Linkage is nonzero and dataset/direction dependent. For proposed
DP variants:

- ECG exact one-MAP linkage is about 0.086–0.097 for disease-to-no-disease
  and 0.044–0.052 in the reverse direction.
- Heart+ exact one-MAP linkage is 0 for disease-to-no-disease and about
  0.022–0.030 in the reverse direction.
- MIMIC exact one-MAP linkage is about 0.083–0.097 for disease-to-no-disease
  and 0.226–0.245 in the reverse direction; small-map linkage reaches roughly
  0.50–0.56 in that reverse direction.

The MIMIC reverse-direction linkage is the largest outstanding release-risk
finding. The supplied MIMIC derived table has no patient identifier, so the
notebook cannot manufacture a patient-disjoint split. This limitation and the
linkage results must remain explicit. A separately predeclared minimal-release
ablation may be justified later, but these outer results must not be used to
silently tune the accepted primary experiment.

## Promotion and next stage

The three artifacts are accepted one-seed paper evidence. They may populate a
clearly labelled one-seed section of the supplement after the table mapping is
checked. They are not multi-seed estimates. The next confirmatory stage is a
multi-seed run or paper sharding plan; CKKS/HE should consume only a
checksum-locked plaintext handoff selected prospectively from this accepted
contract.
