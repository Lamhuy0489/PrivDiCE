# The result tables separate utility, external baselines, and encrypted execution

Each dataset has three independently accepted result groups:

1. `plaintext/`: 200 factuals, both directions, four candidate budgets, one generator-training seed and one search seed;
2. `official_dice/`: Official DiCE Random and Genetic on the same 200 factuals, retaining timeout and no-CF failures;
3. `he/`: real TenSEAL/CKKS CPU timing, communication, memory, numerical error, and label agreement over the declared population grid.

The canonical acceptance files are:

- `plaintext/v55_paper_acceptance.json`;
- `official_dice/official_dice_extension_acceptance.json`;
- `he/12_he_acceptance.json`.

All nine report `accepted=true`; plaintext and HE also report `paper_numbers=true`. Detailed interpretation and table mapping are in `docs/EXPERIMENTAL_SUPPLEMENT_ONE_SEED.md` and `docs/REPRODUCIBILITY_AUDIT.md`.
