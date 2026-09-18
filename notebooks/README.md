# Notebook snapshots preserve the experimental protocols

This directory contains the nine canonical Kaggle notebooks listed in the root README. The Kaggle pull API cleared execution counts and cell outputs for eight notebooks, so those files are reproducible source snapshots rather than rendered run records. The Heart+ Official DiCE 20-second notebook came from the manually saved Kaggle download and retains its executed cells and outputs.

The authoritative saved outputs are:

- machine-readable aggregate tables and acceptance JSON in `results/`;
- extracted artifact directories in `artifacts/` for Leipzig ECG and Heart+;
- the live Kaggle notebook pages where public;
- no protected MIMIC-IV row-level artifact or checkpoint is redistributed.

The adjacent `kernel-metadata.json` files preserve the slug, accelerator, input sources, and visibility settings recorded at download time. The saved metadata marks the Heart+ and MIMIC-IV Official DiCE notebooks as private; their local source and result records remain available. The checked-in executed Heart+ notebook is the accepted 20-second paper run and should not be confused with a later extended-time revision under the same kernel slug.
