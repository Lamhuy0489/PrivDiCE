# Q1 V5.5 official DiCE paper extension audit — 2026-08-24

## Acceptance and scope

All three artifacts have `accepted=true`, 200 frozen outer-test factuals (100
per direction), and 400 factual-method rows. They reuse the accepted V5.5
classifier, fitted preprocessing, split, query IDs and DomainProjector. No MLP
or GAN is retrained, and no DP accountant/release is changed.

Official `dice-ml==0.12` Random uses `sample_size=10,000`; Genetic uses
`maxiterations=300`. Every factual-method call has a 20-second cap. Timeout and
no-CF rows remain in the unconditional Yield/Coverage/Full-10 denominator.

This is a matched **outcome, cohort, K, margin and constraint** comparison. It
is not an equal-compute comparison: official DiCE native samples/iterations do
not map to the proposed method's counted candidate budget B, and oracle access
and stopping behavior differ.

## Direction-specific official results

| Dataset | Method | Direction | Yield@10 | Coverage@1 | Full-10 | Robust Yield@10 | Proximity | Sparsity | Diversity | Mean runtime/factual s | Timeout | No-CF |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ECG | Genetic | disease→no disease | 0.843 | 1.00 | 0.30 | 0.826 | 0.3472 | 0.0586 | 0.2041 | 5.4926 | 0 | 0 |
| ECG | Genetic | no disease→disease | 0.562 | 0.99 | 0.03 | 0.539 | 0.3524 | 0.0385 | 0.1690 | 3.8908 | 0 | 0 |
| ECG | Random | disease→no disease | 0.496 | 0.90 | 0.00 | 0.390 | 0.0472 | 0.8354 | 0.0615 | 4.0897 | 10 | 0 |
| ECG | Random | no disease→disease | 0.366 | 0.89 | 0.00 | 0.275 | 0.0464 | 0.8291 | 0.0620 | 4.2692 | 11 | 0 |
| Heart+ | Genetic | disease→no disease | 0.000 | 0.00 | 0.00 | 0.000 | — | — | — | 20.0012 | 100 | 0 |
| Heart+ | Genetic | no disease→disease | 0.000 | 0.00 | 0.00 | 0.000 | — | — | — | 20.0012 | 100 | 0 |
| Heart+ | Random | disease→no disease | 0.244 | 0.75 | 0.00 | 0.230 | 0.0592 | 0.7431 | 0.0520 | 1.3825 | 0 | 11 |
| Heart+ | Random | no disease→disease | 0.171 | 0.63 | 0.00 | 0.160 | 0.0742 | 0.7023 | 0.0368 | 1.6593 | 0 | 33 |
| MIMIC-IV | Genetic | disease→no disease | 0.025 | 0.08 | 0.00 | 0.018 | 0.2271 | 0.0567 | 0.1575 | 19.7477 | 91 | 0 |
| MIMIC-IV | Genetic | no disease→disease | 0.975 | 0.99 | 0.86 | 0.972 | 0.2768 | 0.0395 | 0.2324 | 3.3546 | 0 | 0 |
| MIMIC-IV | Random | disease→no disease | 0.281 | 0.96 | 0.00 | 0.181 | 0.0406 | 0.8687 | 0.0337 | 3.0174 | 0 | 0 |
| MIMIC-IV | Random | no disease→disease | 0.213 | 0.82 | 0.00 | 0.125 | 0.0263 | 0.8905 | 0.0259 | 2.7704 | 0 | 0 |

Quality metrics are success-conditioned. The effective quality denominators
are printed in `Bổ sung.md` and must be read with Yield/Coverage. In
particular, sparse/proximal Random outputs do not compensate for zero Full-10.

## Main scientific interpretation

- ECG Genetic reaches high coverage but is much less sparse and farther from
  factuals than Random/proposed. Proposed DP ε=4 reaches Full-10=1.0 in both
  directions; Genetic reaches 0.30 and 0.03.
- Heart+ Genetic timing out on 200/200 calls is an implementation-and-cap
  result, not proof that every possible Genetic configuration must fail.
- MIMIC Genetic is strongly directional: Yield 0.025 versus 0.975. This must
  not be hidden by its equal-direction macro of 0.500.
- On MIMIC no disease→disease, Genetic Robust Yield 0.972 is slightly above
  proposed DP ε=4 at 0.954, while proposed has higher Yield (0.990), Full-10
  (0.990), much higher Sparsity (0.620 versus 0.040), and lower Proximity
  (0.089 versus 0.277). This is a genuine multi-objective trade-off.
- Runtime is descriptive only. Do not claim an algorithmic speedup from the
  linked table because equal compute/query budget is explicitly false.

## Independent consistency checks

- Each query-method key appears exactly once.
- Each direction has 100 predeclared query IDs.
- Raw-to-direction summaries and direction-to-macro recomputations match to
  floating-point precision.
- There are no implementation errors; timeouts and no-CF are retained.
- All source checkpoint/preprocessor checksums match the accepted V5.5 roots.

## Canonical artifacts

- ECG saved kernel: `REDACTED_KAGGLE_OWNER/leipzig-ecg-v5-5-official-dice-extension-paper-sav`;
  local root `experiments/cloud/results/ecg/q1_v55_official_dice_extension_paper_v1/`;
  ZIP SHA-256 `63d4f543f07624ab512432844eb0259284f35ba9502b38264371cc6fceeee872`.
- Heart+ manual saved artifact: local root
  `experiments/cloud/results/heartplus/q1_v55_official_dice_extension_paper_v1/`;
  ZIP SHA-256 `f7f7e58eadf26298059a7bd1cc1bb0f8c2870976fd0996025840eec40b276fee`.
- MIMIC-IV kernel: `buiquocviet/mimic-iv-v5-5-official-dice-extension-paper`;
  local root `experiments/cloud/results/mimic/q1_v55_official_dice_extension_paper_v1/`;
  ZIP SHA-256 `fa2a8114d8b17d28e7208afaa0e57a793db8b330ee483331070a02771a7c0fea`.

The old four-factual smoke ZIPs remain in `benchmark_official/` only for
implementation diagnosis. They are not paper numbers and must not be merged
with these full-cohort outputs.
