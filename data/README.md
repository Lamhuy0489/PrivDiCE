# Data manifest

This directory intentionally distinguishes redistributable inputs from access-controlled clinical data.

| Dataset | File | Size | SHA-256 | Publicly included |
| --- | --- | ---: | --- | --- |
| Leipzig ECG | `ecg/ecg_beats_features_selected_20.csv` | 26,070,248 bytes | `ffa8482f426ca0fdb6055f745610e88befe7e0cb36889a0bd2678dfe7bca5a9b` | Yes |
| Heart+ | `heartplus/merged_data.csv` | 75,465,590 bytes | `8c3611f95469b3a06fe33628c2d82283dc3571b57bd0ff9b8c5c63ec44dac27e` | Yes |
| MIMIC-IV derived extract | `mimic_icu_mortality_cfe_12671.csv` | 3,666,081 bytes | `d79aa6ecdc9dd04ad540923b30323c4d693a5e84fcad28444c366d2fd703eadb` | **No** |

Dataset-specific provenance and access instructions are in the subdirectory README files. Checksums identify the exact experimental inputs and must be verified before reusing frozen checkpoints or comparing regenerated outputs.
