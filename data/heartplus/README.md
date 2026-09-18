# Heart+ input

`Heart+` is the study name for a merged heart-risk cohort, not a standalone clinical registry. The file combines processed 2020 and 2022 tables derived from the CDC Behavioral Risk Factor Surveillance System (BRFSS).

- Experimental file: `merged_data.csv`
- Raw rows: 764,927
- Usable rows after the frozen cleaning contract: 761,862
- SHA-256: `8c3611f95469b3a06fe33628c2d82283dc3571b57bd0ff9b8c5c63ec44dac27e`
- Split contract: exact-feature duplicate groups remain disjoint
- The experimental Kaggle mirror identifier is intentionally omitted from the public release; the upstream sources below are canonical.

Upstream sources:

- [CDC BRFSS annual data](https://www.cdc.gov/brfss/annual_data/annual_data.htm)
- [2020 survey data](https://www.cdc.gov/brfss/annual_data/annual_2020.html)
- [2022 survey data](https://www.cdc.gov/brfss/annual_data/annual_2022.html)
- [Processed Kaggle tables by Kamil Pytlak](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease)

Users remain responsible for observing the attribution terms of each upstream source.
