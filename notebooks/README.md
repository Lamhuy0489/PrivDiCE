# Notebook snapshots

This directory contains the nine canonical Kaggle notebooks listed in the root README. The Kaggle pull API cleared execution counts and cell outputs for eight notebooks, so those files are reproducible source snapshots rather than rendered run records. The Heart+ Official DiCE 20-second notebook came from the manually saved Kaggle download and retains its executed cells and outputs.

The authoritative saved outputs are:

- machine-readable aggregate tables and acceptance JSON in `results/`;
- complete artifact ZIPs in `artifacts/` for Leipzig ECG and Heart+;
- the live Kaggle notebook pages where public;
- no protected MIMIC-IV row-level artifact or checkpoint is redistributed.

The adjacent `kernel-metadata.json` files preserve owner, slug, accelerator, and visibility settings at download time. Heart+ and MIMIC-IV Official DiCE are currently marked private and should be made public by their Kaggle owners. The checked-in executed Heart+ notebook is the accepted 20-second paper run; it must not be confused with any later extended-time revision under the same Kaggle slug.
