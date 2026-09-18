# MIMIC-IV input is not redistributed

The experiment used a 12,671-row derived ICU mortality extract from [MIMIC-IV v2.2](https://physionet.org/content/mimiciv/2.2/) (DOI [10.13026/6mm1-ek67](https://doi.org/10.13026/6mm1-ek67)). MIMIC-IV is credentialed health data governed by the PhysioNet Credentialed Health Data License and its data-use agreement.

For that reason, this public repository does **not** include the CSV, row-level counterfactuals, complete artifact archive, or trained checkpoints derived from it.

Authorized users may reconstruct the expected local input as:

```text
data/mimic/mimic_icu_mortality_cfe_12671.csv
```

Verification contract:

- rows: 12,671;
- size: 3,666,081 bytes;
- SHA-256: `d79aa6ecdc9dd04ad540923b30323c4d693a5e84fcad28444c366d2fd703eadb`.

The public aggregate CSV/JSON results in `results/mimic/` are retained for evaluation without redistributing protected records.
