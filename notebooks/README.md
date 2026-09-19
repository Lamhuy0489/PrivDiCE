# Notebook snapshots preserve the experimental protocols

This directory contains the nine canonical Kaggle notebooks listed in the root README. Each file is a manually saved executed version: all code cells retain execution counts and saved outputs, and the accepted copies contain no error outputs.

The authoritative saved outputs are:

- machine-readable aggregate tables and acceptance JSON in `results/`;
- extracted artifact directories in `artifacts/` for Leipzig ECG and Heart+;
- the live Kaggle notebook pages linked from the root README and run manifest;
- no protected MIMIC-IV row-level artifact or checkpoint is redistributed.

The adjacent `kernel-metadata.json` files preserve the accelerator, input sources, and visibility settings recorded at download time. The saved metadata marks the Heart+ and MIMIC-IV Official DiCE notebooks as private; their executed local copies and result records remain available. The checked-in Heart+ notebook is the accepted 20-second paper run and should not be confused with a later extended-time revision under the same kernel slug.
