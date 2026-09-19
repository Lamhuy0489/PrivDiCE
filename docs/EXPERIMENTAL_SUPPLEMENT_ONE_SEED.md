# Kết quả thực nghiệm bổ sung — protocol Q1 V5.5

Cập nhật từ artifact ngày 2026-08-24. **Toàn bộ kết quả plaintext trong tài liệu này là kết quả paper một seed**; không được diễn giải như mean ± std qua nhiều generator-training seed. Ba official DiCE full-cohort extensions và ba HE paper artifacts đã hoàn tất, qua checksum/acceptance audit. HE dùng 3 candidate-cohort seeds × 5 timing repetitions cho mỗi population. Bảng chỉ chứa chỉ số đã đo. `bo_sung_first.md` được giữ riêng làm đối chiếu lịch sử, không trộn số vào đây.

## 0. Nguồn dữ liệu, chuỗi artifact và khác biệt với bản đầu tiên

### 0.1 Nguồn khoa học và đúng input đã dùng

| Dataset | Nguồn khoa học phải trích dẫn | Bản dữ liệu thực sự gắn vào Kaggle | File đầu vào canonical | Ghi chú provenance |
| --- | --- | --- | --- | --- |
| Leipzig ECG | [PhysioNet v1.0.0](https://physionet.org/content/leipzig-heart-center-ecg/1.0.0/); [DOI 10.13026/7a4j-vn37](https://doi.org/10.13026/7a4j-vn37) | Kaggle mirror identifier omitted | `ecg_beats_features_selected_20.csv` | Mirror Kaggle chứa feature table dựng từ 39 subject của nguồn PhysioNet; paper dùng 113.846 beats và split subject-disjoint. |
| Heart+ | [CDC BRFSS annual data](https://www.cdc.gov/brfss/annual_data/annual_data.htm), [BRFSS 2020](https://www.cdc.gov/brfss/annual_data/annual_2020.html), [BRFSS 2022](https://www.cdc.gov/brfss/annual_data/annual_2022.html); [Kaggle upstream của Kamil Pytlak](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease) | Kaggle mirror identifier omitted | `merged_data.csv` | `Heart+` là tên rút gọn trong nghiên cứu, không phải tên một registry lâm sàng. File raw 764.927 dòng ghép bảng đã xử lý 2020 (319.795) và 2022 (445.132) từ BRFSS; còn 761.862 dòng sau contract làm sạch. |
| MIMIC-IV | [PhysioNet MIMIC-IV v2.2](https://physionet.org/content/mimiciv/2.2/); [DOI 10.13026/6mm1-ek67](https://doi.org/10.13026/6mm1-ek67) | [meanalways/mimiciv-full](https://www.kaggle.com/datasets/meanalways/mimiciv-full/data) | `mimic_icu_mortality_cfe_12671.csv` | CSV 12.671 dòng là extract dẫn xuất dùng cho thí nghiệm, không phải toàn bộ MIMIC-IV. Việc truy cập/chia sẻ phải tuân theo PhysioNet Credentialed Health Data License và DUA; link Kaggle chỉ mô tả input thực thi, không thay citation/giấy phép gốc. |

`Raw rows` trong các bảng kết quả phải khớp đúng các file canonical trên. Không được thay bằng bản ECG legacy, một file MIMIC 12.000 dòng hoặc một biến thể Heart+ khác rồi vẫn dùng các checkpoint hiện tại.

### 0.2 BibTeX của ba nguồn dữ liệu

#### MIMIC-IV v2.2

```bibtex
@article{PhysioNet-mimiciv-2.2,
  author = {Johnson, Alistair and Bulgarelli, Lucas and Pollard, Tom and Horng, Steven and Celi, Leo Anthony and Mark, Roger},
  title = {{MIMIC-IV}},
  journal = {{PhysioNet}},
  year = {2023},
  month = jan,
  note = {Version 2.2},
  doi = {10.13026/6mm1-ek67},
  url = {https://doi.org/10.13026/6mm1-ek67}
}
```

#### Leipzig Heart Center ECG v1.0.0

```bibtex
@article{PhysioNet-leipzig-heart-center-ecg-1.0.0,
  author = {Klehs, Sophia and Franke, Daniel and Alhamad, Bayhas and Gebauer, Roman and Teich, Linus and Teich, Tobias and Paech, Christian},
  title = {{Leipzig Heart Center ECG-Database: Arrhythmias in Children and Patients with Congenital Heart Disease}},
  journal = {{PhysioNet}},
  year = {2025},
  month = mar,
  note = {Version 1.0.0},
  doi = {10.13026/7a4j-vn37},
  url = {https://doi.org/10.13026/7a4j-vn37}
}
```

#### Heart+ / Personal Key Indicators of Heart Disease / CDC BRFSS

Heart+ không có DOI riêng. Nên trích dẫn nguồn survey CDC và ghi thêm provenance của bản Kaggle đã xử lý:

```bibtex
@misc{CDC-BRFSS-2020,
  author = {{Centers for Disease Control and Prevention}},
  title = {{2020 BRFSS Survey Data and Documentation}},
  year = {2022},
  url = {https://www.cdc.gov/brfss/annual_data/annual_2020.html},
  note = {Behavioral Risk Factor Surveillance System}
}

@misc{CDC-BRFSS-2022,
  author = {{Centers for Disease Control and Prevention}},
  title = {{2022 BRFSS Survey Data and Documentation}},
  year = {2023},
  url = {https://www.cdc.gov/brfss/annual_data/annual_2022.html},
  note = {Behavioral Risk Factor Surveillance System}
}

@misc{Pytlak-heart-disease-indicators,
  author = {Pytlak, Kamil},
  title = {{Indicators of Heart Disease (2022 UPDATE)}},
  howpublished = {Kaggle},
  url = {https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease},
  note = {Processed tables derived from CDC BRFSS annual survey data}
}
```

### 0.3 Chuỗi notebook nguồn và ba notebook merge

| Dataset | Plaintext seed 11 | Plaintext seed 22 | Plaintext seed 33 | HE paper | Merge full3 + HE |
| --- | --- | --- | --- | --- | --- |
| Leipzig ECG | [one-seed paper](https://www.kaggle.com/code/lamhuy8904/leipzig-ecg-q1-v5-5-gen-only-one-seed-paper-t4) | `leipzig-v55-ms-s22-paper-t4` | `leipzig-v55-ms-s33-paper-t4` | [HE population paper](https://www.kaggle.com/code/lamhuy8904/leipzig-ecg-v5-5-he-population-paper) | `leipzig-ecg-v5-5-full3-he-merge-tables` |
| Heart+ | [one-seed paper](https://www.kaggle.com/code/huylmhuhu/heart-q1-v5-5-gen-only-one-seed-paper-t4) | `heart-v55-ms-s22-paper-t4` | `heart-v55-ms-s33-paper-t4` | [HE population paper](https://www.kaggle.com/code/huylmhuhu/heart-v5-5-he-population-paper) | `heart-v5-5-full3-he-merge-tables` |
| MIMIC-IV | [one-seed paper](https://www.kaggle.com/code/buiquocviet/mimic-iv-q1-v5-5-gen-only-one-seed-paper-t4) | [seed 22 paper](https://www.kaggle.com/code/buiquocviet/mimic-v55-ms-s22-paper-t4) | [seed 33 paper — public cross-account](https://www.kaggle.com/code/meanalways/mimic-v55-ms-s33-paper-t4) | [HE population paper](https://www.kaggle.com/code/buiquocviet/mimic-iv-v5-5-he-population-paper) | [full3 + HE merge](https://www.kaggle.com/code/buiquocviet/mimic-iv-v5-5-full3-he-merge-tables) |

Ba notebook merge đều đã `COMPLETE`. Mỗi notebook nhận đúng bốn output, kiểm checksum/acceptance, kiểm query manifest và baseline invariance, rồi chỉ ghép và in bảng; không train lại MLP/GAN, không sinh lại CFE, không chạy lại attack và không chạy lại CKKS.

Official DiCE full-cohort là nhánh baseline độc lập, không phải một trong bốn input của notebook merge. Các bảng Official DiCE ở Mục 3.5 truy về đúng ba notebook sau:

| Dataset | Official DiCE full-cohort extension |
| --- | --- |
| Leipzig ECG | [leipzig-ecg-v5-5-official-dice-extension-paper-sav](https://www.kaggle.com/code/lamhuy8904/leipzig-ecg-v5-5-official-dice-extension-paper-sav) |
| Heart+ | [heart-v5-5-official-dice-extension-paper](https://www.kaggle.com/code/huylmhuhu/heart-v5-5-official-dice-extension-paper) |
| MIMIC-IV | [mimic-iv-v5-5-official-dice-extension-paper](https://www.kaggle.com/code/buiquocviet/mimic-iv-v5-5-official-dice-extension-paper) |

### 0.4 Khác biệt chính với bản đầu tiên

| Thành phần | Bản đầu tiên trong `Heart-plus/` | Protocol V5.5 hiện tại | Hệ quả khi đọc kết quả |
| --- | --- | --- | --- |
| Phạm vi | Chỉ Heart+ | Leipzig ECG, Heart+ và MIMIC-IV | Có kiểm tra tính nhất quán trên ba hình học dữ liệu khác nhau. |
| Hướng đổi nhãn | Chỉ bệnh → không bệnh | 100 factual bệnh → không bệnh và 100 factual chiều ngược lại trên mỗi dataset | Không dùng macro để che chiều khó. |
| Cohort | 50 factual bệnh được phân loại đúng | 200 factual/dataset, stratified theo khoảng cách tới biên | Mẫu đánh giá lớn hơn và bao gồm near/mid/far boundary. |
| MLP | Heart+ `23→160→80→1`, activation `x+0.125x²` | Kiến trúc HE-friendly riêng từng dataset; output một logit, threshold hấp thụ vào bias | Raw metric cũ–mới không phải paired vì oracle đã đổi. |
| Training GAN | Loss ramp toàn lịch, không có warm-up tách riêng | Warm-up rồi ramp classification/proximity/feature-group sparsity/diversity; direction-balanced | Sparse/diverse là một phần của training và search, không phải sửa CFE sau benchmark. |
| Search | Generator + anchor + mutation + crossover; chọn dần | Counted full archive, project/prune semantic group trước oracle call, MMR chọn K=10; không donor/rescue/train-kNN | Slot thiếu vẫn là failure và candidate budget có thể nối với HE. |
| Population/budget | `P=200`, tối đa 20 iterations | ECG `P=64/B=512/R=8`; Heart+ `P=128/B=1024/R=8`; MIMIC `P=64/B=512/R=8` | P là candidates/round; B là tổng candidate cap, không phải batch bệnh nhân. |
| Validity | Dễ đọc nhầm conditional validity trên returned CFE | `Valid-CFE Yield@10`, `Coverage@1`, `Full-10`; missing slots nằm trong denominator | `Constraint validity=1` không được dùng thay Yield. |
| Quality | Geometry cũ cho Proximity/Diversity/Plausibility | Normalized Proximity, input/group Sparsity, set Diversity, kNN distance, LOF inlier và target support | Không trừ trực tiếp số quality cũ–mới khi công thức khác. |
| Robustness | Chưa có stress protocol đầy đủ | Robust Yield/Full-10 và Changed-set Dice với perturbation cố định | Validity hoàn hảo vẫn có thể đi kèm Robust Yield thấp hơn. |
| DP | Generator + critic DP-SGD; ε split 80/20; secure mode tắt | Panel ε=16/8/4/2, exact composed accounting, secure RNG; generator-only release | ε là bảo đảm cho từng checkpoint; phát hành nhiều checkpoint phải composition. |
| Privacy evaluation | MIA/exact/DCR/NNDR/BMI inversion giới hạn | Generator-only multi-attack MIA, memorization, normalized attribute inference và Explanation-Linkage | Không dùng AUC gần 0.5 như formal proof; vẫn giữ cảnh báo ECG/MIMIC. |
| Baseline | Baseline/cohort hẹp hơn | Matched-budget forward-only baselines, gradient access stratum và official DiCE full-cohort extension | Official DiCE là matched outcome/constraint nhưng không equal-compute. |
| HE | Notebook tách, population sensitivity hẹp | Real TenSEAL CKKS, nhiều P, phase/communication/RAM/error/agreement và boundary stress | HE agreement chỉ đo numerical correctness, không đo CFE utility. |
| Trạng thái thống kê | Ba seed trên 50 factual một hướng trong bản cũ | Tài liệu `Bổ sung.md` này giữ anchor generator seed 11; `Bổ sung full3.md` xác nhận bằng seeds 11/22/33 | Không gọi bảng một-seed là mean ± SD qua training runs. |

Đây là đối chiếu protocol, không phải thí nghiệm paired old-versus-new. Do classifier, cohort, margin, search và metric geometry đã đổi, không lấy hiệu hai số cũ–mới để tuyên bố cải thiện định lượng. Bảng chi tiết và SHA-256 của notebook legacy vẫn được giữ ở Mục 10.

### 0.5 Audit thông số khóa và phép tổng hợp

| Dataset | K | Target logit margin | Operating P | Candidate budget B | Max rounds | Processed HE inputs | HE N | CKKS scale bits | HE timing repetitions/P |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Leipzig ECG | 10 | 0.10 | 64 | 512 | 8 | 20 | 16,384 | 45 | 3 cohort seeds × 5 = 15 |
| Heart+ | 10 | 0.10 | 128 | 1,024 | 8 | 23 | 8,192 | 29 | 3 cohort seeds × 5 = 15 |
| MIMIC-IV | 10 | 0.10 | 64 | 512 | 8 | 40 | 16,384 | 40 | 3 cohort seeds × 5 = 15 |

| Dataset | Search profile | Group-off threshold | Feature-off threshold | Base fraction | Search proximity weight | Search sparsity weight | MMR diversity weight | Selection scope |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Leipzig ECG | sparse | 0.120 | 0.025 | 0.50 | 0.12 | 0.72 | 1.00 | inner validation only |
| Heart+ | balanced | 0.060 | 0.030 | 0.50 | 0.15 | 0.45 | 4.00 | inner validation only |
| MIMIC-IV | sparse | 0.080 | 0.040 | 0.50 | 0.10 | 0.72 | 3.00 | inner validation only |

Classifier seed là 55, generator-training seeds của full3 là 11/22/33 và outer search seed là 11. Phép tổng hợp full3 lấy mean trên 100 factual của từng hướng **trong từng training seed**, sau đó mới tính mean ± sample SD qua ba seed-level means. Baseline không phụ thuộc generator chỉ giữ một giá trị, không tạo `±0` giả. HE giữ median `[Q1,Q3]` và bổ sung mean ± sample SD qua 15 timing runs; các lần timing không được gọi là GAN seeds.

Audit lại 66 bảng của ba notebook merge cho thấy 22/22 bảng mỗi dataset khớp chính xác với phép tính lại từ 12 ZIP canonical sau CSV round-trip; query manifest trùng, duplicate key bằng 0 và baseline scientific max absolute delta bằng 0. Các thông số trên vì vậy nhất quán với artifact. Các giới hạn vẫn phải nêu: Heart+ mất cân bằng mạnh, MIMIC không có patient ID để split patient-disjoint, Heart+ HE dùng 217/218 modulus bits nên rất ít dư địa depth, và HE timing là biến thiên trong cùng môi trường Kaggle chứ không phải benchmark đa phần cứng.

## 1. Quy ước đọc kết quả

- `Valid-CFE Yield@10 ↑`: số CFE vừa đổi nhãn vừa thỏa constraint chia cho K=10; slot thiếu vẫn tính thất bại. Đây là validity chính, không dùng conditional validity vốn dễ bằng 1.
- `Coverage@1 ↑`: tỷ lệ factual có ít nhất một CFE hợp lệ. `Full-10 ↑`: tỷ lệ factual đủ cả 10 CFE.
- `Robust Yield@10 ↑`: số CFE bền vững chia K=10; một CFE chỉ được tính robust khi ít nhất 95% trong 16 perturbations khả thi vẫn đúng target. Nhiễu Gaussian có độ lệch chuẩn bằng 1% feature scale và không tác động immutable features.
- `Proximity ↓`: mean absolute distance sau chuẩn hóa theo feature scale. `Sparsity ↑`: tỷ lệ **actionable processed inputs** không đổi; không tính immutable inputs. `Group sparsity ↑`: tỷ lệ actionable semantic feature groups không đổi.
- `Diversity ↑`: mean pairwise normalized L1 distance giữa các CFE hợp lệ của cùng factual.
- `Plausibility distance ↓` là mean 5-NN distance tới development-data support; vì đây là khoảng cách nên **càng thấp càng tốt**. `Inlier rate ↑` là LOF inlier rate và `Target support ↑` là tỷ lệ 5 láng giềng development thuộc target class.
- `Fidelity ↑` là accuracy của local logistic surrogate khi bắt chước frozen MLP trong lân cận factual–CFE. `Constraint validity ↑` là tỷ lệ CFE đã trả về thỏa constraint.
- Proximity, Sparsity, Diversity, Plausibility và Fidelity là quality metrics có điều kiện trên CFE hợp lệ/đã trả về; phải đọc cùng Yield/Coverage để tránh phương pháp trả rất ít CFE nhưng có quality đẹp.
- Mọi bảng đổi nhãn giữ riêng `Bệnh → không bệnh` và `Không bệnh → bệnh`; bảng macro chỉ dùng làm tổng quan, không thay bảng tách hướng.

### 1.1 Từ điển chỉ số và cách nhận diện

| Chỉ số | Cách tính trong V5.5 | Chiều tốt | Cách nhận diện/giới hạn |
| --- | --- | --- | --- |
| Valid-CFE Yield@10 | mean(valid + feasible CFE count / 10) | ↑ | Validity chính; slot thiếu là failure |
| Coverage@1 | factuals có ≥1 valid CFE / tất cả factuals | ↑ | Đo có giải thích hay không, chưa phản ánh đủ 10 điểm |
| Full-10 | factuals có đủ 10 valid CFE / tất cả factuals | ↑ | Khắt khe hơn Coverage |
| Robust Yield@10 | robust CFE count / 10; robust nếu ≥95% của 16 perturbations vẫn feasible + target-valid | ↑ | Với 16 lần lặp, ngưỡng 95% thực tế yêu cầu 16/16 lần thành công |
| Robust Full-10 | factuals có đủ 10 robust CFE / tất cả factuals | ↑ | Stress test khắt khe nhất của tập CFE |
| Proximity | mean absolute factual–CFE distance / train-fitted feature scale | ↓ | Chỉ tính trên valid CFE; không so trị tuyệt đối giữa dataset |
| Sparsity | 1 − tỷ lệ actionable processed inputs thay đổi quá 0.02 scale | ↑ | Processed-input level; one-hot components có thể là nhiều cột |
| Group sparsity | 1 − tỷ lệ actionable semantic groups thay đổi quá 0.02 scale | ↑ | Phù hợp hơn để diễn giải raw clinical features |
| Diversity | mean pairwise normalized L1 distance trong valid CFE set | ↑ | Đọc cùng Proximity/Sparsity; quá cao có thể là thay đổi xa factual |
| Plausibility distance | mean 5-NN distance tới development reference / √input dimension | ↓ | Evaluation-only; không được query trong generator-only CFE search |
| Inlier rate | tỷ lệ valid CFE được development-fitted LOF nhận là inlier | ↑ | Bổ sung cho distance; phụ thuộc detector/reference |
| Target support | mean fraction của 5 development neighbours thuộc desired class | ↑ | Kiểm tra CFE nằm gần support của lớp đích |
| Fidelity | test accuracy của local logistic surrogate so với frozen MLP | ↑ | Fidelity của local explanation, không phải HE/plaintext agreement |
| Changed-set Dice | Dice overlap của changed-feature mask trước/sau perturbation | ↑ | Đo ổn định tập feature bị thay đổi |
| Constraint validity | feasible returned CFEs / returned CFEs | ↑ | Có thể bằng 1 do archive chỉ lưu feasible CFE; không thay Yield@10 |
| Time to K-or-cap | wall time đến đủ K hoặc hết declared cap | ↓ | Failure vẫn giữ thời gian đến cap |
| Raw MIA AUC | membership attack AUC trên release surface đã khai báo | →0.5 | Gần 0.5 là random; dưới 0.5 có thể chỉ là score đảo chiều |
| MIA distance from random | \|AUC−0.5\| | ↓ | 0 là random; không cho AUC dưới 0.5 che tín hiệu đảo chiều |
| Orientation-normalized MIA AUC | 0.5 + \|AUC−0.5\| = max(AUC, 1−AUC) | ↓ | Cùng thông tin với distance nhưng nằm trên thang 0.5–1.0 |
| Absolute MIA advantage | \|TPR − FPR\| tại threshold chọn trên calibration split | ↓ | Metric threshold-based; khác AUC và không phải formal DP proof |
| Attribute inversion normalized MAE | ECG/MIMIC: MAE của attacker / robust range cho thuộc tính liên tục | ↑ | Attack MAE thấp hơn trivial MAE nghĩa là attacker có lợi thế |
| Attribute inversion balanced accuracy | Heart+: balanced accuracy khi suy đoán thuộc tính Diabetic dạng categorical/ordinal | ↓ | Attack accuracy cao hơn trivial accuracy nghĩa là attacker có lợi thế |
| Attribute inversion attacker advantage | ECG/MIMIC: trivial MAE − attack MAE; Heart+: attack balanced accuracy − trivial accuracy | ↓ | Đưa hai metric về cùng chiều; dương là cảnh báo, gần 0 là tốt hơn |
| Exact/Near-match rate | tỷ lệ generated records trùng/gần train reference | ↓ | Leakage diagnostic |
| DCR | distance tới training record gần nhất | ↑ | Phải đọc cùng holdout-normalized ratio |
| NNDR | nearest distance / second-nearest distance | ↑ | Gần 1: không có một training neighbour cô lập nổi bật |
| Explanation-Linkage | khả năng nối released CFE về factual trong auxiliary pool | ↓ | Release-view attack; không phải membership attack |
| HE label agreement | tỷ lệ nhãn CKKS trùng nhãn của cùng dropout-free plaintext graph | ↑ | Đo correctness của HE, không đo utility CFE |
| HE max logit error | max \|logit CKKS − logit plaintext\| trên cohort | ↓ | Đọc cùng khoảng cách logit tới decision boundary |
| Median [Q1,Q3] | trung vị và phân vị 25%–75% qua timing repetitions | IQR hẹp hơn ổn định hơn | Không phải confidence interval và không phải GAN-seed variability |

Các họ metric Validity/Coverage, Proximity, Sparsity, Diversity và Plausibility là chuẩn trong đánh giá counterfactual. `Robust Yield@10`, `Target support`, `Changed-set Dice` và attack panels là các metric mở rộng theo protocol; chúng hợp lệ về học thuật khi công thức, reference set, perturbation và denominator được khóa như trên, nhưng không nên gọi là một chuẩn duy nhất cho mọi paper.

### 1.2 Phạm vi seed và đơn vị lặp

| Thành phần | Giá trị | Cách diễn giải |
| --- | --- | --- |
| Classifier training | seed 55 | Một lần huấn luyện MLP cho mỗi dataset |
| GAN training | seed 11 | Một lần huấn luyện cho từng variant CounterGAN/DP trên mỗi dataset |
| Outer CFE search | seed 11 | Cùng factual manifest và paired search randomness giữa phương pháp |
| Outer factual cohort | 200/dataset | 100 Bệnh → không bệnh và 100 Không bệnh → bệnh |
| Official DiCE extension | baseline seed 11; 200 factuals/dataset | Không retrain; timeout/no-CF giữ trong denominator |
| MIA/memorization/attribute attack | 10 attack seeds | Lặp attacker/resampling; không phải 10 GAN-training seeds |
| HE smoke | cohort seed 101; 1 timing repeat | Chẩn đoán correctness/khả thi; không phải paper estimate |
| HE paper | 3 cohort seeds × 5 timing repeats/population | Final CPU timing estimate; vẫn dùng cùng checkpoint plaintext một training seed |

Do đó, bootstrap CI và kiểm định paired ở dưới định lượng bất định **giữa factuals trong đúng một lần train**. Chúng không thay thế bất định giữa các lần train GAN; phần xác nhận multi-seed phải chạy riêng sau khi protocol một-seed này được khóa.

## 2. Dữ liệu, biểu diễn và mạng MLP

### 2.1 Kiểm kê dữ liệu

| Dataset | Raw rows | Usable rows | Dropped | Class 0 | Class 1 | Raw predictors | Processed HE inputs | Split |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Leipzig ECG | 113,846 | 113,846 | 0 | 37,153 | 76,693 | 20 | 20 | subject-disjoint |
| Heart+ | 764,927 | 761,862 | 3,065 | 709,381 | 52,481 | 17 | 23 | exact-feature-group-disjoint |
| MIMIC-IV | 12,671 | 12,671 | 0 | 7,356 | 5,315 | 37 | 40 | stratified-row-no-patient-id |

`Raw rows` là số dòng trong CSV; `Usable rows` là số còn lại sau chuẩn hóa nhãn và loại missing theo contract; `Raw predictors` đếm biến có ý nghĩa trước encoding, còn `Processed HE inputs` đếm scalar thực sự đi vào MLP/CKKS. Heart+ có 17 raw predictors nhưng 23 inputs vì Race tạo 7 one-hot columns; MIMIC-IV có 37 predictors và 40 inputs sau encoding gender/race. MIMIC có đúng 12,671 usable rows và class count 7,356 + 5,315 = 12,671; số 12,000 trong bản cũ đã bị loại. ECG chia subject-disjoint; Heart+ giữ exact-duplicate groups disjoint; MIMIC không có patient ID trong file nguồn nên dùng stratified row split và ghi rõ hạn chế.

### 2.2 Chỉ số phân loại outer test

| Dataset | Accuracy ↑ | Precision ↑ | Recall ↑ | F1 ↑ | F2 ↑ | ROC-AUC ↑ | PR-AUC ↑ | Specificity ↑ | Balanced accuracy ↑ | MCC ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Leipzig ECG | 0.9046 | 0.9913 | 0.8667 | 0.9249 | 0.8891 | 0.9833 | 0.9923 | 0.9841 | 0.9254 | 0.8095 |
| Heart+ | 0.8771 | 0.2748 | 0.4783 | 0.3490 | 0.4166 | 0.8392 | 0.2943 | 0.9066 | 0.6925 | 0.3001 |
| MIMIC-IV | 0.8840 | 0.8849 | 0.8316 | 0.8574 | 0.8417 | 0.9496 | 0.9414 | 0.9219 | 0.8767 | 0.7609 |

Accuracy là tỷ lệ nhãn đúng; Precision là độ tin cậy của dự đoán bệnh; Recall là tỷ lệ bệnh được phát hiện; F1 cân bằng Precision–Recall, còn F2 ưu tiên Recall. ROC-AUC đo khả năng xếp hạng hai lớp qua mọi threshold; PR-AUC nhạy hơn khi lớp bệnh hiếm. Specificity đo nhận diện đúng lớp không bệnh. Balanced accuracy lấy trung bình Recall và Specificity. MCC dùng cả bốn ô confusion matrix và thường đáng tin hơn Accuracy khi mất cân bằng. ECG và MIMIC có F1 lần lượt 0.9249 và 0.8574. Heart+ có Accuracy 0.8771 nhưng F1 chỉ 0.3490 và PR-AUC 0.2943 do class bệnh chiếm thiểu số; vì vậy không dùng Accuracy một mình để tuyên bố classifier tốt.

### 2.3 Kiến trúc deploy được bằng HE

| Dataset | Input | Hidden 1 | Hidden 2 | Output | Activation | α | Training dropout | Selected epoch | Probability threshold trước hấp thụ | Absorbed logit threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Leipzig ECG | 20 | 160 | 80 | 1 | poly | 0.1250 | 0.0500 | 7 | 0.9996 | 7.8281 |
| Heart+ | 23 | 128 | 64 | 1 | square | 0.0000 | 0.0500 | 48 | 0.4284 | -0.2885 |
| MIMIC-IV | 40 | 128 | 64 | 1 | square | 0.0000 | 0.0500 | 44 | 0.5684 | 0.2753 |

Input là số processed scalars; hai hidden widths và output=1 xác định graph MLP. `poly` của ECG là x+αx² với α=0.125; Heart+ và MIMIC dùng x². `Selected epoch` được chọn trên inner validation, không trên outer test. Dropout=0.05 chỉ hoạt động khi training. Plaintext reference và HE inference đều dùng graph evaluation dropout-free. Threshold xác suất được chọn trên validation rồi hấp thụ vào final bias; `Absorbed logit threshold` là logit cũ đã trừ khỏi bias, không phải threshold mới khi inference. Server chỉ tính cộng, nhân plaintext weight và activation bậc hai; client giải mã rồi so logit mới với 0.

### 2.4 Chuẩn hóa và encoding của ba bộ dữ liệu

Preprocessor chỉ được fit trên training split. Validation và outer test chỉ gọi `transform`; vì vậy median, quantile, category order và one-hot vocabulary không nhìn thấy outer test. Mã nhãn dùng thống nhất `0 = negative class` và `1 = positive/adverse class`; CFE đặt `desired = 1 − factual label`, nên cả hai hướng đều được sinh và đánh giá. Ý nghĩa lâm sàng cụ thể được giữ theo từng dataset trong bảng dưới.

| Dataset | Target gốc | Nhãn 0 | Nhãn 1 | Tên hướng rút gọn trong bảng |
| --- | --- | --- | --- | --- |
| Leipzig ECG | Target_Abnormal | normal beat | abnormal beat | bất thường → bình thường / bình thường → bất thường |
| Heart+ | HeartDisease | no heart disease | heart disease | bệnh → không bệnh / không bệnh → bệnh |
| MIMIC-IV | Target_Mortality | survival | mortality | tử vong dự đoán → sống / sống → tử vong dự đoán |

Để bảng liên-dataset gọn, cột Direction tiếp tục dùng `Bệnh → không bệnh` và chiều ngược lại như một nhãn quy ước. Với ECG phải đọc là abnormal/normal; với MIMIC phải đọc là predicted mortality/survival. CFE chỉ đổi dự đoán của model dưới các thay đổi đặc trưng khả thi; nó không chứng minh can thiệp đó sẽ chữa bệnh, biến một beat bất thường thành bình thường hay thay đổi kết cục tử vong ngoài đời thực.

| Dataset | Biến liên tục | Biến thứ bậc/nhị phân | Biến danh mục | Raw → processed | Ghi chú diễn giải |
| --- | --- | --- | --- | --- | --- |
| Leipzig ECG | 19 biến số: median imputation, QuantileTransformer sang phân phối normal, clip [-5,5] | Không có | gender: one-hot drop-first, F là reference | 20 → 20 | age và gender immutable; các ràng buộc ECG được chiếu lại sau mỗi proposal |
| Heart+ | PhysicalHealth, MentalHealth, SleepTime, BMI: median imputation + quantile-normal + clip [-5,5] | GenHealth, AgeCategory, Diabetic, Smoking ánh xạ có thứ tự vào [-1,1]; 8 biến nhị phân vào {0,1} | Race dùng đủ 7 one-hot, không drop level | 17 → 23 | Race, tuổi, giới và tiền sử bệnh đã khai báo immutable; numeric ngày/giờ được snap về miền hợp lệ |
| MIMIC-IV | 35 biến số: median imputation, quantile-normal, clip [-5,5] | Không có ordinal riêng | gender và race_group one-hot drop-first; 1 + 4 cột | 37 → 40 | Min/Mean/Max của cùng dấu hiệu được xem là một semantic group; age, gender, race immutable |

Quantile encoding được chọn vì ba lý do: giảm ảnh hưởng của outlier, đưa các biến có đơn vị rất khác nhau về cùng thang và giúp cả MLP đa thức lẫn GAN tối ưu ổn định hơn. Nó không phải bằng chứng rằng khoảng cách trong encoded space có ý nghĩa lâm sàng. Vì vậy kết quả còn được kiểm bằng raw-domain constraints, semantic-group sparsity, inverse transform trong qualitative grid và các thước đo support/plausibility chỉ dùng ở bước đánh giá.


### 2.6 Luồng phương pháp từ dữ liệu tới CFE và HE

1. Dữ liệu được chia train/validation/test trước khi fit preprocessing. MLP được huấn luyện trên train, chọn epoch và threshold trên validation, rồi khóa hoàn toàn trước outer test.
2. Threshold xác suất được hấp thụ vào bias cuối. Vì thế một CFE hợp lệ phải có logit `≥ +0.10` khi đích là lớp 1 hoặc `≤ −0.10` khi đích là lớp 0, ngoài điều kiện domain/immutable.
3. CounterGAN nhận factual đã encode, desired label và latent noise. Generator học đồng thời realism, đổi nhãn, proximity, sparsity theo feature/group và diversity. Warm-up giữ loss dễ học ở đầu; các trọng số được ramp dần, nên sparsity/diversity là một phần của training chứ không phải sửa số sau benchmark.
4. Khi search, mỗi round sinh proposal mới, mutation từ elite và proposal phục hồi từng semantic group. Tất cả proposal đều được project và sparsify trước model call; oracle call của pruning vẫn được tính vào budget. Valid CFE đi vào archive; sau khi dùng hết budget, MMR chọn tối đa K=10 điểm theo quality và độ khác nhau.
5. V5.5 chỉ triển khai generator. Discriminator dùng trong training nhưng không được phát hành hay query khi inference; raw training kNN cũng không được dùng để xếp hạng trong search. Plausibility kNN/LOF chỉ là đánh giá hậu nghiệm.
6. Ở HE, client encode bằng đúng preprocessor, mã hóa từng processed feature, server chỉ chạy MLP đa thức bằng cộng/nhân, client giải mã logit. GAN và bước chọn CFE vẫn ở phía sinh/search; HE benchmark đo phần oracle inference trên population đã đóng gói.

| Dataset | Epoch cap | Steps/epoch cap | Warm-up | Ramp | Classification weight | Proximity weight | Feature sparsity weight | Group sparsity weight | Diversity weight | Non-DP checkpoint | DP checkpoint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Leipzig ECG | 300 | 20 | 30 | 180 | 8→12 | 0.04→0.2 | 0.01→0.1 | 0→0.15 | 0→0.08 | inner-validation best epoch | final epoch; no private-validation selection |
| Heart+ | 300 | 20 | 30 | 180 | 8→12 | 0.04→0.2 | 0.01→0.1 | 0→0.15 | 0→0.08 | inner-validation best epoch | final epoch; no private-validation selection |
| MIMIC-IV | 210 | 20 | 21 | 126 | 8→12 | 0.04→0.2 | 0.01→0.1 | 0→0.15 | 0→0.08 | inner-validation best epoch | final epoch; no private-validation selection |

Mũi tên trong bảng là trọng số đầu → cuối của ramp. ECG/Heart+ dùng 300 epochs với warm-up 30 và ramp 180; MIMIC dùng 210/21/126 để giữ tỷ lệ lịch trình tương ứng trên bộ nhỏ hơn. Non-DP được chọn epoch tốt nhất bằng inner validation. DP dùng final epoch để tránh dùng private-validation utility cho checkpoint selection mà không có accounting bổ sung.

## 3. So sánh CFE plaintext tại operating budget — paper một seed

Mỗi dataset dùng 200 factuals, 100 mỗi hướng, generator seed 11 và search seed 11. Baseline và proposed dùng cùng factual/query manifest và candidate cap. Uniform và Genetic fixed-cap tiêu thụ đủ budget; gradient methods được báo riêng vì có backward oracle access. Proposed DP-CounterGAN-SD dùng full counted archive rồi prune/MMR; không rescue bằng donor và không query raw training kNN khi search.

### 3.0 Các phương pháp được so sánh

| Tên trong bảng | Cách sinh CFE | Quyền truy cập oracle | Budget/stopping | Vai trò trong ablation |
| --- | --- | --- | --- | --- |
| Uniform Random | Lấy mẫu nhiễu Gaussian độc lập quanh factual, tăng dần bán kính; project về miền hợp lệ; không giữ elite và không học từ score vòng trước | Chỉ forward logit | Cùng P, B và K; slot thiếu là failure | Mốc không học, kiểm tra bài toán có quá dễ hay không |
| Genetic CFE | Tournament selection từ population trước, uniform crossover và sparse mutation; archive chỉ nhận điểm distinct, feasible và đạt margin | Forward logit để tính fitness | Cùng P, B và K với proposed | Baseline tiến hóa gần nhất với search lặp; không dùng generator |
| DiCE-style gradient | Nhiều restart tối ưu đồng thời classification + proximity − diversity rồi project | White-box backward qua MLP | Dùng fixed optimization-step proxy; không đồng nhất chi phí với forward-only methods | Baseline gradient có diversity; đây là implementation style, không phải official dice-ml |
| Wachter-style | Nhiều restart tối ưu target loss + proximity, không có diversity reward | White-box backward qua MLP | Cùng proxy như DiCE-style | Tách tác dụng của diversity trong gradient search |
| CounterGAN one-shot | Sinh một population trực tiếp từ generator; không mutation, không elite, không lặp | Một lượt forward scoring sau generation | Một round; vẫn báo slot thiếu | Đo riêng chất lượng generator trước iterative search |
| CounterGAN iterative | Generator proposal + elite mutation qua nhiều round; dừng khi đủ K | Forward logit, black-box | Native early stop nên thường dùng ít hơn B | Cho biết lợi ích của search lặp so với one-shot |
| CounterGAN-SD non-DP | Generator không DP + semantic-group stabilization, counted group restoration, full archive và MMR | Forward logit, black-box | Chạy đủ fixed cap B | Ablation trực tiếp của DP: cùng search với proposed nhưng không DP-SGD |
| DP-CounterGAN-SD ε=16/8/4/2 | Generator huấn luyện DP-SGD; inference dùng cùng sparse-diverse search như CG-SD | Forward logit, black-box | Cùng P, B, K, seed coupling và missing-slot rule | Đo privacy–utility trade-off; ε=4 là primary được khóa từ inner validation |
| Official DiCE Random | Native `dice-ml==0.12` random sampling trong raw feature interface | Official library/model wrapper | sample_size=10,000; timeout 20 s/factual | External implementation check; không phải equal-compute với B |
| Official DiCE Genetic | Native `dice-ml==0.12` genetic search | Official library/model wrapper | maxiterations=300; timeout 20 s/factual | External implementation check; timeout/no-CF vẫn là failure |

So sánh chính trong Mục 3.1–3.4 là matched-cohort và, với nhóm forward-only, matched counted budget. Hai baseline gradient được giữ để đối chiếu white-box nhưng phải báo riêng backward access. Official DiCE ở Mục 3.5 dùng cùng factual, desired label, K, margin và constraints, song không thể gọi là equal-compute vì control native không quy đổi được sang B.

#### Cấu hình sparse-diverse được khóa cho từng dataset

| Dataset | Profile | P | B | Max rounds | Group-off threshold | Feature-off threshold | Base fraction | Search proximity weight | Search sparsity weight | MMR diversity weight | Selection scope |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Leipzig ECG | sparse | 64 | 512 | 8 | 0.1200 | 0.0250 | 0.5000 | 0.1200 | 0.7200 | 1.0000 | inner validation only |
| Heart+ | balanced | 128 | 1,024 | 8 | 0.0600 | 0.0300 | 0.5000 | 0.1500 | 0.4500 | 4.0000 | inner validation only |
| MIMIC-IV | sparse | 64 | 512 | 8 | 0.0800 | 0.0400 | 0.5000 | 0.1000 | 0.7200 | 3.0000 | inner validation only |

Group/feature-off threshold quyết định thay đổi nhỏ nào được phục hồi về factual trước khi chấm; base fraction chia mỗi round giữa proposal gốc và prune/fill proposal; MMR weight điều khiển đánh đổi quality–set diversity khi chọn K từ full archive. Đây là trọng số search, khác với loss weights trong bảng GAN training. Các giá trị được chọn trên inner validation và khóa trước outer test; ba dataset không bị ép dùng cùng threshold vì số semantic groups và hình học feature space khác nhau.

### 3.1 Leipzig ECG — B=512, K=10

#### Đổi nhãn, coverage và robustness

| Method | Direction | Factuals | Valid-CFE Yield@10 ↑ | Coverage@1 ↑ | Full-10 ↑ | Robust Yield@10 ↑ | Robust Full-10 ↑ | Constraint validity ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Uniform Random | Bệnh → không bệnh | 100 | 1.0000 | 1.0000 | 1.0000 | 0.8520 | 0.1900 | 1.0000 |
| Uniform Random | Không bệnh → bệnh | 100 | 1.0000 | 1.0000 | 1.0000 | 0.8200 | 0.1200 | 1.0000 |
| Genetic CFE | Bệnh → không bệnh | 100 | 1.0000 | 1.0000 | 1.0000 | 0.9060 | 0.4700 | 1.0000 |
| Genetic CFE | Không bệnh → bệnh | 100 | 1.0000 | 1.0000 | 1.0000 | 0.8960 | 0.3400 | 1.0000 |
| DiCE-style gradient | Bệnh → không bệnh | 100 | 0.1040 | 1.0000 | 0.0000 | 0.1030 | 0.0000 | 1.0000 |
| DiCE-style gradient | Không bệnh → bệnh | 100 | 0.1000 | 1.0000 | 0.0000 | 0.1000 | 0.0000 | 1.0000 |
| Wachter-style | Bệnh → không bệnh | 100 | 0.1030 | 1.0000 | 0.0000 | 0.1020 | 0.0000 | 1.0000 |
| Wachter-style | Không bệnh → bệnh | 100 | 0.1000 | 1.0000 | 0.0000 | 0.1000 | 0.0000 | 1.0000 |
| CounterGAN one-shot | Bệnh → không bệnh | 100 | 0.9440 | 1.0000 | 0.8400 | 0.8530 | 0.5800 | 1.0000 |
| CounterGAN one-shot | Không bệnh → bệnh | 100 | 1.0000 | 1.0000 | 1.0000 | 0.9860 | 0.9200 | 1.0000 |
| CounterGAN iterative | Bệnh → không bệnh | 100 | 1.0000 | 1.0000 | 1.0000 | 0.9290 | 0.6200 | 1.0000 |
| CounterGAN iterative | Không bệnh → bệnh | 100 | 1.0000 | 1.0000 | 1.0000 | 0.9930 | 0.9400 | 1.0000 |
| CounterGAN-SD non-DP | Bệnh → không bệnh | 100 | 1.0000 | 1.0000 | 1.0000 | 0.8240 | 0.2000 | 1.0000 |
| CounterGAN-SD non-DP | Không bệnh → bệnh | 100 | 1.0000 | 1.0000 | 1.0000 | 0.8080 | 0.1800 | 1.0000 |
| DP-CounterGAN-SD ε=16 | Bệnh → không bệnh | 100 | 1.0000 | 1.0000 | 1.0000 | 0.8150 | 0.2100 | 1.0000 |
| DP-CounterGAN-SD ε=16 | Không bệnh → bệnh | 100 | 1.0000 | 1.0000 | 1.0000 | 0.7300 | 0.1500 | 1.0000 |
| DP-CounterGAN-SD ε=8 | Bệnh → không bệnh | 100 | 1.0000 | 1.0000 | 1.0000 | 0.8160 | 0.2100 | 1.0000 |
| DP-CounterGAN-SD ε=8 | Không bệnh → bệnh | 100 | 1.0000 | 1.0000 | 1.0000 | 0.7010 | 0.0700 | 1.0000 |
| DP-CounterGAN-SD ε=4 (primary) | Bệnh → không bệnh | 100 | 1.0000 | 1.0000 | 1.0000 | 0.8330 | 0.1700 | 1.0000 |
| DP-CounterGAN-SD ε=4 (primary) | Không bệnh → bệnh | 100 | 1.0000 | 1.0000 | 1.0000 | 0.7360 | 0.1400 | 1.0000 |
| DP-CounterGAN-SD ε=2 | Bệnh → không bệnh | 100 | 1.0000 | 1.0000 | 1.0000 | 0.8640 | 0.2800 | 1.0000 |
| DP-CounterGAN-SD ε=2 | Không bệnh → bệnh | 100 | 1.0000 | 1.0000 | 1.0000 | 0.7670 | 0.1700 | 1.0000 |

#### Chất lượng và search effort

| Method | Direction | Proximity ↓ | Sparsity ↑ | Group sparsity ↑ | Diversity ↑ | Plausibility distance ↓ | Inlier rate ↑ | Target support ↑ | Fidelity ↑ | Changed-set Dice ↑ | Time to K-or-cap (s) ↓ | Candidates | Rounds ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Uniform Random | Bệnh → không bệnh | 0.0918 | 0.2155 | 0.2155 | 0.1187 | 0.6810 | 0.0280 | 0.0286 | 0.8424 | 0.9621 | 0.3686 | 512.0000 | 8.0000 |
| Uniform Random | Không bệnh → bệnh | 0.0946 | 0.2061 | 0.2061 | 0.1224 | 0.6528 | 0.0060 | 0.7628 | 0.8248 | 0.9617 | 0.3620 | 512.0000 | 8.0000 |
| Genetic CFE | Bệnh → không bệnh | 0.0725 | 0.4258 | 0.4258 | 0.0790 | 0.6344 | 0.0320 | 0.0280 | 0.8733 | 0.9225 | 0.4118 | 512.0000 | 8.0000 |
| Genetic CFE | Không bệnh → bệnh | 0.0765 | 0.4038 | 0.4038 | 0.0822 | 0.6267 | 0.0100 | 0.7666 | 0.8742 | 0.9251 | 0.4058 | 512.0000 | 8.0000 |
| DiCE-style gradient | Bệnh → không bệnh | 0.0761 | 0.0417 | 0.0417 | 0.0014 | 0.5371 | 0.1000 | 0.0150 | 0.9258 | 0.9956 | 0.1104 | 512.0000 | 1.0000 |
| DiCE-style gradient | Không bệnh → bệnh | 0.0689 | 0.0539 | 0.0539 | 0.0000 | 0.5245 | 0.0100 | 0.6360 | 0.9250 | 0.9930 | 0.1094 | 512.0000 | 1.0000 |
| Wachter-style | Bệnh → không bệnh | 0.0762 | 0.0367 | 0.0367 | 0.0009 | 0.5364 | 0.1000 | 0.0150 | 0.9238 | 0.9972 | 0.1011 | 512.0000 | 1.0000 |
| Wachter-style | Không bệnh → bệnh | 0.0694 | 0.0489 | 0.0489 | 0.0000 | 0.5255 | 0.0100 | 0.6320 | 0.9269 | 0.9931 | 0.1008 | 512.0000 | 1.0000 |
| CounterGAN one-shot | Bệnh → không bệnh | 0.0947 | 0.2250 | 0.2250 | 0.0315 | 0.4383 | 0.2619 | 0.1642 | 0.8639 | 0.9609 | 0.0544 | 512.0000 | 1.0000 |
| CounterGAN one-shot | Không bệnh → bệnh | 0.1106 | 0.2289 | 0.2289 | 0.0370 | 0.4385 | 0.0410 | 0.9966 | 0.8655 | 0.9639 | 0.0515 | 512.0000 | 1.0000 |
| CounterGAN iterative | Bệnh → không bệnh | 0.1040 | 0.1762 | 0.1762 | 0.0486 | 0.4675 | 0.2460 | 0.1654 | 0.8615 | 0.9664 | 0.0648 | 92.1600 | 1.4400 |
| CounterGAN iterative | Không bệnh → bệnh | 0.1181 | 0.1663 | 0.1663 | 0.0407 | 0.4492 | 0.0330 | 0.9968 | 0.8673 | 0.9713 | 0.0509 | 72.3200 | 1.1300 |
| CounterGAN-SD non-DP | Bệnh → không bệnh | 0.0645 | 0.7980 | 0.7980 | 0.0624 | 0.6188 | 0.0630 | 0.0818 | 0.8603 | 0.9154 | 1.9929 | 512.0000 | 8.0000 |
| CounterGAN-SD non-DP | Không bệnh → bệnh | 0.0762 | 0.7491 | 0.7491 | 0.0755 | 0.6336 | 0.0140 | 0.9456 | 0.8339 | 0.9246 | 2.0233 | 512.0000 | 8.0000 |
| DP-CounterGAN-SD ε=16 | Bệnh → không bệnh | 0.0707 | 0.7689 | 0.7689 | 0.0734 | 0.6611 | 0.0660 | 0.1206 | 0.8585 | 0.9245 | 2.0084 | 512.0000 | 8.0000 |
| DP-CounterGAN-SD ε=16 | Không bệnh → bệnh | 0.0649 | 0.7664 | 0.7664 | 0.0725 | 0.6047 | 0.0140 | 0.8710 | 0.8361 | 0.9133 | 2.0190 | 512.0000 | 8.0000 |
| DP-CounterGAN-SD ε=8 | Bệnh → không bệnh | 0.0682 | 0.7663 | 0.7663 | 0.0709 | 0.6616 | 0.0560 | 0.1068 | 0.8597 | 0.9242 | 2.0081 | 512.0000 | 8.0000 |
| DP-CounterGAN-SD ε=8 | Không bệnh → bệnh | 0.0570 | 0.7906 | 0.7906 | 0.0624 | 0.5541 | 0.0320 | 0.8676 | 0.8282 | 0.9058 | 2.0153 | 512.0000 | 8.0000 |
| DP-CounterGAN-SD ε=4 (primary) | Bệnh → không bệnh | 0.0724 | 0.7761 | 0.7761 | 0.0677 | 0.6704 | 0.0810 | 0.1548 | 0.8673 | 0.9222 | 2.0121 | 512.0000 | 8.0000 |
| DP-CounterGAN-SD ε=4 (primary) | Không bệnh → bệnh | 0.0673 | 0.7608 | 0.7608 | 0.0740 | 0.5994 | 0.0210 | 0.8648 | 0.8352 | 0.9177 | 2.0293 | 512.0000 | 8.0000 |
| DP-CounterGAN-SD ε=2 | Bệnh → không bệnh | 0.0789 | 0.7644 | 0.7644 | 0.0761 | 0.6907 | 0.0640 | 0.1988 | 0.8618 | 0.9258 | 2.0250 | 512.0000 | 8.0000 |
| DP-CounterGAN-SD ε=2 | Không bệnh → bệnh | 0.0600 | 0.7646 | 0.7646 | 0.0726 | 0.5674 | 0.0220 | 0.8424 | 0.8245 | 0.9170 | 2.0505 | 512.0000 | 8.0000 |

**Nhận xét Leipzig ECG.** Ở budget 512, Uniform, Genetic, CounterGAN iterative và toàn bộ CG-SD/DP-CG-SD đều đạt Yield@10 và Full-10 bằng 1 ở cả hai hướng. Vì vậy bảng validity tại operating point chỉ cho biết bài toán đã bão hòa; đóng góp phải đọc ở quality và robustness. DP ε=4 giữ Sparsity 0.7761/0.7608, cao hơn rõ rệt CounterGAN iterative 0.1762/0.1663, trong khi Proximity giảm từ 0.1040/0.1181 xuống 0.0724/0.0673. Đánh đổi là Robust Yield giảm từ 0.9290/0.9930 xuống 0.8330/0.7360. Genetic có Robust Yield cao hơn DP ε=4 nhưng Sparsity chỉ 0.4258/0.4038. Kết luận phù hợp là sparse search làm lời giải ngắn gọn hơn, không phải mọi mặt đều tốt hơn.

### 3.2 Heart+ — B=1024, K=10

#### Đổi nhãn, coverage và robustness

| Method | Direction | Factuals | Valid-CFE Yield@10 ↑ | Coverage@1 ↑ | Full-10 ↑ | Robust Yield@10 ↑ | Robust Full-10 ↑ | Constraint validity ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Uniform Random | Bệnh → không bệnh | 100 | 0.6760 | 0.7900 | 0.5700 | 0.6390 | 0.3400 | 1.0000 |
| Uniform Random | Không bệnh → bệnh | 100 | 0.2540 | 0.4200 | 0.1700 | 0.2480 | 0.1500 | 1.0000 |
| Genetic CFE | Bệnh → không bệnh | 100 | 0.5340 | 0.7300 | 0.3000 | 0.4900 | 0.1500 | 1.0000 |
| Genetic CFE | Không bệnh → bệnh | 100 | 0.2610 | 0.4800 | 0.1200 | 0.2400 | 0.0600 | 1.0000 |
| DiCE-style gradient | Bệnh → không bệnh | 100 | 0.3280 | 0.8300 | 0.0300 | 0.3280 | 0.0300 | 1.0000 |
| DiCE-style gradient | Không bệnh → bệnh | 100 | 0.1870 | 0.6100 | 0.0000 | 0.1860 | 0.0000 | 1.0000 |
| Wachter-style | Bệnh → không bệnh | 100 | 0.3180 | 0.8200 | 0.0300 | 0.3180 | 0.0300 | 1.0000 |
| Wachter-style | Không bệnh → bệnh | 100 | 0.1780 | 0.6100 | 0.0000 | 0.1760 | 0.0000 | 1.0000 |
| CounterGAN one-shot | Bệnh → không bệnh | 100 | 0.1370 | 0.8400 | 0.0000 | 0.1370 | 0.0000 | 1.0000 |
| CounterGAN one-shot | Không bệnh → bệnh | 100 | 0.2740 | 0.7000 | 0.0100 | 0.2670 | 0.0000 | 1.0000 |
| CounterGAN iterative | Bệnh → không bệnh | 100 | 0.8940 | 0.9600 | 0.8400 | 0.8870 | 0.8200 | 1.0000 |
| CounterGAN iterative | Không bệnh → bệnh | 100 | 0.6270 | 0.7700 | 0.5100 | 0.6210 | 0.5100 | 1.0000 |
| CounterGAN-SD non-DP | Bệnh → không bệnh | 100 | 0.9140 | 0.9400 | 0.8900 | 0.8860 | 0.6900 | 1.0000 |
| CounterGAN-SD non-DP | Không bệnh → bệnh | 100 | 0.6610 | 0.7600 | 0.6100 | 0.6310 | 0.4100 | 1.0000 |
| DP-CounterGAN-SD ε=16 | Bệnh → không bệnh | 100 | 0.9080 | 0.9300 | 0.9000 | 0.8760 | 0.6600 | 1.0000 |
| DP-CounterGAN-SD ε=16 | Không bệnh → bệnh | 100 | 0.5510 | 0.6800 | 0.4700 | 0.5300 | 0.3500 | 1.0000 |
| DP-CounterGAN-SD ε=8 | Bệnh → không bệnh | 100 | 0.9120 | 0.9300 | 0.8900 | 0.8780 | 0.6300 | 1.0000 |
| DP-CounterGAN-SD ε=8 | Không bệnh → bệnh | 100 | 0.5860 | 0.6800 | 0.4900 | 0.5530 | 0.3600 | 1.0000 |
| DP-CounterGAN-SD ε=4 (primary) | Bệnh → không bệnh | 100 | 0.9070 | 0.9400 | 0.8800 | 0.8720 | 0.6100 | 1.0000 |
| DP-CounterGAN-SD ε=4 (primary) | Không bệnh → bệnh | 100 | 0.5460 | 0.7000 | 0.4700 | 0.5170 | 0.3100 | 1.0000 |
| DP-CounterGAN-SD ε=2 | Bệnh → không bệnh | 100 | 0.9040 | 0.9400 | 0.8800 | 0.8760 | 0.6800 | 1.0000 |
| DP-CounterGAN-SD ε=2 | Không bệnh → bệnh | 100 | 0.6060 | 0.7000 | 0.5300 | 0.5790 | 0.3900 | 1.0000 |

#### Chất lượng và search effort

| Method | Direction | Proximity ↓ | Sparsity ↑ | Group sparsity ↑ | Diversity ↑ | Plausibility distance ↓ | Inlier rate ↑ | Target support ↑ | Fidelity ↑ | Changed-set Dice ↑ | Time to K-or-cap (s) ↓ | Candidates | Rounds ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Uniform Random | Bệnh → không bệnh | 0.0615 | 0.5945 | 0.5945 | 0.0565 | 0.3573 | 0.5856 | 0.8679 | 0.9117 | 0.9895 | 0.0562 | 1024.0000 | 8.0000 |
| Uniform Random | Không bệnh → bệnh | 0.0729 | 0.5505 | 0.5505 | 0.0461 | 0.3663 | 0.5732 | 0.1347 | 0.8796 | 0.9884 | 0.0501 | 1024.0000 | 8.0000 |
| Genetic CFE | Bệnh → không bệnh | 0.0455 | 0.6807 | 0.6807 | 0.0313 | 0.3250 | 0.7200 | 0.8685 | 0.9354 | 0.9790 | 0.0590 | 1024.0000 | 8.0000 |
| Genetic CFE | Không bệnh → bệnh | 0.0592 | 0.6272 | 0.6272 | 0.0261 | 0.3446 | 0.6371 | 0.1471 | 0.9264 | 0.9786 | 0.0533 | 1024.0000 | 8.0000 |
| DiCE-style gradient | Bệnh → không bệnh | 0.0924 | 0.4938 | 0.4938 | 0.0546 | 0.3607 | 0.5552 | 0.8717 | 0.9425 | 0.9960 | 0.1308 | 1024.0000 | 1.0000 |
| DiCE-style gradient | Không bệnh → bệnh | 0.1085 | 0.4521 | 0.4521 | 0.0492 | 0.3793 | 0.3634 | 0.1701 | 0.9465 | 0.9927 | 0.1308 | 1024.0000 | 1.0000 |
| Wachter-style | Bệnh → không bệnh | 0.0916 | 0.4953 | 0.4953 | 0.0541 | 0.3612 | 0.5461 | 0.8770 | 0.9450 | 0.9935 | 0.1120 | 1024.0000 | 1.0000 |
| Wachter-style | Không bệnh → bệnh | 0.1073 | 0.4526 | 0.4526 | 0.0466 | 0.3787 | 0.3583 | 0.1748 | 0.9416 | 0.9930 | 0.1129 | 1024.0000 | 1.0000 |
| CounterGAN one-shot | Bệnh → không bệnh | 0.1691 | 0.3486 | 0.3486 | 0.0114 | 0.3885 | 0.2070 | 0.9169 | 0.9507 | 0.9967 | 0.0297 | 1024.0000 | 1.0000 |
| CounterGAN one-shot | Không bệnh → bệnh | 0.1389 | 0.3919 | 0.3919 | 0.0457 | 0.3846 | 0.4008 | 0.1894 | 0.9357 | 0.9954 | 0.0256 | 1024.0000 | 1.0000 |
| CounterGAN iterative | Bệnh → không bệnh | 0.1758 | 0.2960 | 0.2960 | 0.0406 | 0.3817 | 0.3130 | 0.9239 | 0.9331 | 0.9966 | 0.0469 | 608.0000 | 4.7500 |
| CounterGAN iterative | Không bệnh → bệnh | 0.1442 | 0.3684 | 0.3684 | 0.0477 | 0.3930 | 0.4422 | 0.1962 | 0.9172 | 0.9946 | 0.0490 | 742.4000 | 5.8000 |
| CounterGAN-SD non-DP | Bệnh → không bệnh | 0.1041 | 0.6360 | 0.6360 | 0.0755 | 0.3487 | 0.5589 | 0.8927 | 0.9197 | 0.9904 | 0.2735 | 1024.0000 | 8.0000 |
| CounterGAN-SD non-DP | Không bệnh → bệnh | 0.1055 | 0.6120 | 0.6120 | 0.0564 | 0.3468 | 0.5579 | 0.1761 | 0.9059 | 0.9910 | 0.2384 | 1024.0000 | 8.0000 |
| DP-CounterGAN-SD ε=16 | Bệnh → không bệnh | 0.1012 | 0.6462 | 0.6462 | 0.0749 | 0.3387 | 0.6090 | 0.8988 | 0.9180 | 0.9899 | 0.2747 | 1024.0000 | 8.0000 |
| DP-CounterGAN-SD ε=16 | Không bệnh → bệnh | 0.0930 | 0.6382 | 0.6382 | 0.0480 | 0.3360 | 0.5575 | 0.1752 | 0.9091 | 0.9915 | 0.1836 | 1024.0000 | 8.0000 |
| DP-CounterGAN-SD ε=8 | Bệnh → không bệnh | 0.1019 | 0.6420 | 0.6420 | 0.0735 | 0.3409 | 0.6079 | 0.8873 | 0.9230 | 0.9902 | 0.2777 | 1024.0000 | 8.0000 |
| DP-CounterGAN-SD ε=8 | Không bệnh → bệnh | 0.0895 | 0.6503 | 0.6503 | 0.0501 | 0.3310 | 0.6443 | 0.1584 | 0.9027 | 0.9903 | 0.1883 | 1024.0000 | 8.0000 |
| DP-CounterGAN-SD ε=4 (primary) | Bệnh → không bệnh | 0.0988 | 0.6472 | 0.6472 | 0.0730 | 0.3399 | 0.6121 | 0.8917 | 0.9206 | 0.9899 | 0.2766 | 1024.0000 | 8.0000 |
| DP-CounterGAN-SD ε=4 (primary) | Không bệnh → bệnh | 0.0928 | 0.6390 | 0.6390 | 0.0478 | 0.3324 | 0.5903 | 0.1700 | 0.8922 | 0.9905 | 0.1871 | 1024.0000 | 8.0000 |
| DP-CounterGAN-SD ε=2 | Bệnh → không bệnh | 0.0970 | 0.6514 | 0.6514 | 0.0702 | 0.3430 | 0.6248 | 0.8906 | 0.9319 | 0.9895 | 0.2785 | 1024.0000 | 8.0000 |
| DP-CounterGAN-SD ε=2 | Không bệnh → bệnh | 0.0940 | 0.6399 | 0.6399 | 0.0503 | 0.3322 | 0.6153 | 0.1707 | 0.9130 | 0.9905 | 0.2027 | 1024.0000 | 8.0000 |

**Nhận xét Heart+.** Đây là bộ khó và bất đối xứng nhất. DP ε=4 đạt Yield@10 0.9070 khi đổi bệnh → không bệnh nhưng chỉ 0.5460 ở chiều ngược lại; Coverage tương ứng 0.9400 và 0.7000. Chênh lệch này phù hợp với mất cân bằng lớp và hình học decision boundary, không được làm phẳng bằng cách cân bằng hậu nghiệm. So với CounterGAN iterative, DP ε=4 tăng Sparsity từ 0.2960/0.3684 lên 0.6472/0.6390 và giảm Proximity từ 0.1758/0.1442 xuống 0.0988/0.0928. Ở chiều khó, non-DP CG-SD có Yield 0.6610 cao hơn DP ε=4 0.5460; đây là chi phí utility quan sát được của checkpoint DP/search profile, không nên che bằng macro.

### 3.3 MIMIC-IV — B=512, K=10

#### Đổi nhãn, coverage và robustness

| Method | Direction | Factuals | Valid-CFE Yield@10 ↑ | Coverage@1 ↑ | Full-10 ↑ | Robust Yield@10 ↑ | Robust Full-10 ↑ | Constraint validity ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Uniform Random | Bệnh → không bệnh | 100 | 0.6920 | 0.9000 | 0.5900 | 0.5420 | 0.0400 | 1.0000 |
| Uniform Random | Không bệnh → bệnh | 100 | 0.9900 | 0.9900 | 0.9900 | 0.8540 | 0.2900 | 1.0000 |
| Genetic CFE | Bệnh → không bệnh | 100 | 0.8960 | 0.9800 | 0.8300 | 0.7700 | 0.2000 | 1.0000 |
| Genetic CFE | Không bệnh → bệnh | 100 | 0.9900 | 0.9900 | 0.9900 | 0.9190 | 0.5400 | 1.0000 |
| DiCE-style gradient | Bệnh → không bệnh | 100 | 0.0790 | 0.7900 | 0.0000 | 0.0780 | 0.0000 | 1.0000 |
| DiCE-style gradient | Không bệnh → bệnh | 100 | 0.0900 | 0.8700 | 0.0000 | 0.0860 | 0.0000 | 1.0000 |
| Wachter-style | Bệnh → không bệnh | 100 | 0.0790 | 0.7900 | 0.0000 | 0.0780 | 0.0000 | 1.0000 |
| Wachter-style | Không bệnh → bệnh | 100 | 0.0890 | 0.8700 | 0.0000 | 0.0850 | 0.0000 | 1.0000 |
| CounterGAN one-shot | Bệnh → không bệnh | 100 | 0.9650 | 0.9900 | 0.9500 | 0.9420 | 0.9000 | 1.0000 |
| CounterGAN one-shot | Không bệnh → bệnh | 100 | 0.9900 | 0.9900 | 0.9900 | 0.9870 | 0.9700 | 1.0000 |
| CounterGAN iterative | Bệnh → không bệnh | 100 | 1.0000 | 1.0000 | 1.0000 | 0.9640 | 0.8500 | 1.0000 |
| CounterGAN iterative | Không bệnh → bệnh | 100 | 0.9900 | 0.9900 | 0.9900 | 0.9890 | 0.9800 | 1.0000 |
| CounterGAN-SD non-DP | Bệnh → không bệnh | 100 | 1.0000 | 1.0000 | 1.0000 | 0.8970 | 0.4900 | 1.0000 |
| CounterGAN-SD non-DP | Không bệnh → bệnh | 100 | 0.9900 | 0.9900 | 0.9900 | 0.9400 | 0.7300 | 1.0000 |
| DP-CounterGAN-SD ε=16 | Bệnh → không bệnh | 100 | 1.0000 | 1.0000 | 1.0000 | 0.8810 | 0.3900 | 1.0000 |
| DP-CounterGAN-SD ε=16 | Không bệnh → bệnh | 100 | 0.9900 | 0.9900 | 0.9900 | 0.9190 | 0.5700 | 1.0000 |
| DP-CounterGAN-SD ε=8 | Bệnh → không bệnh | 100 | 1.0000 | 1.0000 | 1.0000 | 0.8790 | 0.3800 | 1.0000 |
| DP-CounterGAN-SD ε=8 | Không bệnh → bệnh | 100 | 0.9900 | 0.9900 | 0.9900 | 0.9150 | 0.5000 | 1.0000 |
| DP-CounterGAN-SD ε=4 (primary) | Bệnh → không bệnh | 100 | 1.0000 | 1.0000 | 1.0000 | 0.8920 | 0.4000 | 1.0000 |
| DP-CounterGAN-SD ε=4 (primary) | Không bệnh → bệnh | 100 | 0.9900 | 0.9900 | 0.9900 | 0.9540 | 0.7000 | 1.0000 |
| DP-CounterGAN-SD ε=2 | Bệnh → không bệnh | 100 | 1.0000 | 1.0000 | 1.0000 | 0.8780 | 0.3800 | 1.0000 |
| DP-CounterGAN-SD ε=2 | Không bệnh → bệnh | 100 | 0.9900 | 0.9900 | 0.9900 | 0.9410 | 0.6200 | 1.0000 |

#### Chất lượng và search effort

| Method | Direction | Proximity ↓ | Sparsity ↑ | Group sparsity ↑ | Diversity ↑ | Plausibility distance ↓ | Inlier rate ↑ | Target support ↑ | Fidelity ↑ | Changed-set Dice ↑ | Time to K-or-cap (s) ↓ | Candidates | Rounds ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Uniform Random | Bệnh → không bệnh | 0.1238 | 0.1274 | 0.0045 | 0.1346 | 0.9433 | 0.2096 | 0.5171 | 0.8532 | 0.9788 | 0.6851 | 512.0000 | 8.0000 |
| Uniform Random | Không bệnh → bệnh | 0.1003 | 0.1672 | 0.0088 | 0.1309 | 0.8882 | 0.2828 | 0.3949 | 0.8672 | 0.9694 | 0.7208 | 512.0000 | 8.0000 |
| Genetic CFE | Bệnh → không bệnh | 0.0892 | 0.2448 | 0.0217 | 0.0851 | 0.8607 | 0.4848 | 0.4493 | 0.8840 | 0.9591 | 0.7073 | 512.0000 | 8.0000 |
| Genetic CFE | Không bệnh → bệnh | 0.0849 | 0.3003 | 0.0382 | 0.0888 | 0.8860 | 0.3727 | 0.5422 | 0.9076 | 0.9485 | 0.7417 | 512.0000 | 8.0000 |
| DiCE-style gradient | Bệnh → không bệnh | 0.0781 | 0.0879 | 0.0000 | 0.0000 | 0.7644 | 0.7975 | 0.5418 | 0.9321 | 0.9898 | 0.1483 | 512.0000 | 1.0000 |
| DiCE-style gradient | Không bệnh → bệnh | 0.0818 | 0.0918 | 0.0010 | 0.0010 | 0.7565 | 0.6379 | 0.3494 | 0.9523 | 0.9849 | 0.1486 | 512.0000 | 1.0000 |
| Wachter-style | Bệnh → không bệnh | 0.0783 | 0.0853 | 0.0000 | 0.0000 | 0.7645 | 0.7975 | 0.5418 | 0.9311 | 0.9898 | 0.1394 | 512.0000 | 1.0000 |
| Wachter-style | Không bệnh → bệnh | 0.0817 | 0.0899 | 0.0010 | 0.0006 | 0.7563 | 0.6379 | 0.3517 | 0.9509 | 0.9848 | 0.1402 | 512.0000 | 1.0000 |
| CounterGAN one-shot | Bệnh → không bệnh | 0.1353 | 0.1566 | 0.0120 | 0.0279 | 0.7744 | 0.7205 | 0.6961 | 0.9335 | 0.9804 | 0.0963 | 512.0000 | 1.0000 |
| CounterGAN one-shot | Không bệnh → bệnh | 0.1301 | 0.1634 | 0.0073 | 0.0392 | 0.8253 | 0.5899 | 0.6182 | 0.9415 | 0.9770 | 0.0935 | 512.0000 | 1.0000 |
| CounterGAN iterative | Bệnh → không bệnh | 0.1387 | 0.1248 | 0.0057 | 0.0350 | 0.7832 | 0.6910 | 0.6946 | 0.9339 | 0.9840 | 0.1048 | 78.7200 | 1.2300 |
| CounterGAN iterative | Không bệnh → bệnh | 0.1346 | 0.1227 | 0.0034 | 0.0403 | 0.8324 | 0.5889 | 0.6388 | 0.9452 | 0.9815 | 0.0879 | 68.4800 | 1.0700 |
| CounterGAN-SD non-DP | Bệnh → không bệnh | 0.1039 | 0.5497 | 0.4595 | 0.0913 | 0.8819 | 0.3450 | 0.6022 | 0.8982 | 0.9522 | 3.4054 | 512.0000 | 8.0000 |
| CounterGAN-SD non-DP | Không bệnh → bệnh | 0.1042 | 0.5340 | 0.4353 | 0.0873 | 0.8764 | 0.5111 | 0.6297 | 0.9317 | 0.9510 | 3.4496 | 512.0000 | 8.0000 |
| DP-CounterGAN-SD ε=16 | Bệnh → không bệnh | 0.0997 | 0.5528 | 0.4565 | 0.0854 | 0.8620 | 0.4220 | 0.6288 | 0.9027 | 0.9481 | 3.4058 | 512.0000 | 8.0000 |
| DP-CounterGAN-SD ε=16 | Không bệnh → bệnh | 0.0945 | 0.5846 | 0.5048 | 0.0898 | 0.7965 | 0.7475 | 0.6188 | 0.9226 | 0.9378 | 3.5336 | 512.0000 | 8.0000 |
| DP-CounterGAN-SD ε=8 | Bệnh → không bệnh | 0.0977 | 0.5670 | 0.4673 | 0.0816 | 0.8822 | 0.3300 | 0.6452 | 0.8988 | 0.9431 | 3.3835 | 512.0000 | 8.0000 |
| DP-CounterGAN-SD ε=8 | Không bệnh → bệnh | 0.0840 | 0.6074 | 0.5246 | 0.0807 | 0.8226 | 0.6889 | 0.6032 | 0.9190 | 0.9337 | 3.5034 | 512.0000 | 8.0000 |
| DP-CounterGAN-SD ε=4 (primary) | Bệnh → không bệnh | 0.1175 | 0.5062 | 0.4017 | 0.0918 | 0.9465 | 0.1390 | 0.6656 | 0.8994 | 0.9560 | 3.4133 | 512.0000 | 8.0000 |
| DP-CounterGAN-SD ε=4 (primary) | Không bệnh → bệnh | 0.0885 | 0.6203 | 0.5438 | 0.0740 | 0.8503 | 0.7051 | 0.7046 | 0.9259 | 0.9352 | 3.4471 | 512.0000 | 8.0000 |
| DP-CounterGAN-SD ε=2 | Bệnh → không bệnh | 0.1232 | 0.5134 | 0.4098 | 0.0963 | 0.9391 | 0.1530 | 0.6986 | 0.9076 | 0.9511 | 3.3595 | 512.0000 | 8.0000 |
| DP-CounterGAN-SD ε=2 | Không bệnh → bệnh | 0.1133 | 0.5215 | 0.4355 | 0.0881 | 0.8494 | 0.7232 | 0.7655 | 0.9388 | 0.9463 | 3.5367 | 512.0000 | 8.0000 |

**Nhận xét MIMIC-IV.** DP ε=4 gần như giữ trọn utility tại B=512: Yield@10 là 1.0000 và 0.9900, Full-10 cũng 1.0000 và 0.9900. So với CounterGAN iterative, Sparsity tăng từ 0.1248/0.1227 lên 0.5062/0.6203; Diversity cũng tăng từ 0.0350/0.0403 lên 0.0918/0.0740. Robust Yield giảm từ 0.9640/0.9890 xuống 0.8920/0.9540, và Plausibility distance ở chiều bệnh → không bệnh tăng từ 0.7832 lên 0.9465. Vì vậy kết quả ủng hộ sparse/diverse selection nhưng đồng thời cho thấy trade-off về robustness và data support.

### 3.4 Bảng tổng quan gộp hai hướng — equal-direction macro

Bảng này không phân nhánh hướng. Mỗi số là trung bình của `Bệnh → không bệnh` và `Không bệnh → bệnh` với trọng số bằng nhau. Vì V5.5 có đúng 100 factual mỗi hướng, các metric đầy đủ không missing cũng bằng pooled mean trên 200 factuals. Đây là bảng đọc nhanh; kết luận về bất đối xứng vẫn phải dựa vào Mục 3.1–3.3.

| Dataset | B | Method | Yield@10 ↑ | Coverage@1 ↑ | Full-10 ↑ | Robust Yield@10 ↑ | Proximity ↓ | Sparsity ↑ | Group sparsity ↑ | Diversity ↑ | Plausibility distance ↓ | Inlier rate ↑ | Target support ↑ | Fidelity ↑ | Changed-set Dice ↑ | Time to K-or-cap (s) ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Leipzig ECG | 512 | Uniform Random | 1.0000 | 1.0000 | 1.0000 | 0.8360 | 0.0932 | 0.2108 | 0.2108 | 0.1205 | 0.6669 | 0.0170 | 0.3957 | 0.8336 | 0.9619 | 0.3653 |
| Leipzig ECG | 512 | Genetic CFE | 1.0000 | 1.0000 | 1.0000 | 0.9010 | 0.0745 | 0.4148 | 0.4148 | 0.0806 | 0.6306 | 0.0210 | 0.3973 | 0.8738 | 0.9238 | 0.4088 |
| Leipzig ECG | 512 | DiCE-style gradient | 0.1020 | 1.0000 | 0.0000 | 0.1015 | 0.0725 | 0.0478 | 0.0478 | 0.0007 | 0.5308 | 0.0550 | 0.3255 | 0.9254 | 0.9943 | 0.1099 |
| Leipzig ECG | 512 | Wachter-style | 0.1015 | 1.0000 | 0.0000 | 0.1010 | 0.0728 | 0.0428 | 0.0428 | 0.0004 | 0.5310 | 0.0550 | 0.3235 | 0.9254 | 0.9952 | 0.1010 |
| Leipzig ECG | 512 | CounterGAN one-shot | 0.9720 | 1.0000 | 0.9200 | 0.9195 | 0.1026 | 0.2270 | 0.2270 | 0.0342 | 0.4384 | 0.1514 | 0.5804 | 0.8647 | 0.9624 | 0.0529 |
| Leipzig ECG | 512 | CounterGAN iterative | 1.0000 | 1.0000 | 1.0000 | 0.9610 | 0.1110 | 0.1713 | 0.1713 | 0.0446 | 0.4583 | 0.1395 | 0.5811 | 0.8644 | 0.9689 | 0.0579 |
| Leipzig ECG | 512 | CounterGAN-SD non-DP | 1.0000 | 1.0000 | 1.0000 | 0.8160 | 0.0704 | 0.7735 | 0.7735 | 0.0689 | 0.6262 | 0.0385 | 0.5137 | 0.8471 | 0.9200 | 2.0081 |
| Leipzig ECG | 512 | DP-CounterGAN-SD ε=16 | 1.0000 | 1.0000 | 1.0000 | 0.7725 | 0.0678 | 0.7677 | 0.7677 | 0.0730 | 0.6329 | 0.0400 | 0.4958 | 0.8473 | 0.9189 | 2.0137 |
| Leipzig ECG | 512 | DP-CounterGAN-SD ε=8 | 1.0000 | 1.0000 | 1.0000 | 0.7585 | 0.0626 | 0.7784 | 0.7784 | 0.0666 | 0.6079 | 0.0440 | 0.4872 | 0.8439 | 0.9150 | 2.0117 |
| Leipzig ECG | 512 | DP-CounterGAN-SD ε=4 (primary) | 1.0000 | 1.0000 | 1.0000 | 0.7845 | 0.0698 | 0.7684 | 0.7684 | 0.0709 | 0.6349 | 0.0510 | 0.5098 | 0.8512 | 0.9200 | 2.0207 |
| Leipzig ECG | 512 | DP-CounterGAN-SD ε=2 | 1.0000 | 1.0000 | 1.0000 | 0.8155 | 0.0694 | 0.7645 | 0.7645 | 0.0743 | 0.6290 | 0.0430 | 0.5206 | 0.8432 | 0.9214 | 2.0378 |
| Heart+ | 1,024 | Uniform Random | 0.4650 | 0.6050 | 0.3700 | 0.4435 | 0.0672 | 0.5725 | 0.5725 | 0.0513 | 0.3618 | 0.5794 | 0.5013 | 0.8956 | 0.9889 | 0.0531 |
| Heart+ | 1,024 | Genetic CFE | 0.3975 | 0.6050 | 0.2100 | 0.3650 | 0.0524 | 0.6539 | 0.6539 | 0.0287 | 0.3348 | 0.6786 | 0.5078 | 0.9309 | 0.9788 | 0.0561 |
| Heart+ | 1,024 | DiCE-style gradient | 0.2575 | 0.7200 | 0.0150 | 0.2570 | 0.1004 | 0.4730 | 0.4730 | 0.0519 | 0.3700 | 0.4593 | 0.5209 | 0.9445 | 0.9944 | 0.1308 |
| Heart+ | 1,024 | Wachter-style | 0.2480 | 0.7150 | 0.0150 | 0.2470 | 0.0994 | 0.4739 | 0.4739 | 0.0503 | 0.3699 | 0.4522 | 0.5259 | 0.9433 | 0.9933 | 0.1124 |
| Heart+ | 1,024 | CounterGAN one-shot | 0.2055 | 0.7700 | 0.0050 | 0.2020 | 0.1540 | 0.3703 | 0.3703 | 0.0286 | 0.3865 | 0.3039 | 0.5531 | 0.9432 | 0.9961 | 0.0277 |
| Heart+ | 1,024 | CounterGAN iterative | 0.7605 | 0.8650 | 0.6750 | 0.7540 | 0.1600 | 0.3322 | 0.3322 | 0.0442 | 0.3873 | 0.3776 | 0.5601 | 0.9251 | 0.9956 | 0.0480 |
| Heart+ | 1,024 | CounterGAN-SD non-DP | 0.7875 | 0.8500 | 0.7500 | 0.7585 | 0.1048 | 0.6240 | 0.6240 | 0.0660 | 0.3478 | 0.5584 | 0.5344 | 0.9128 | 0.9907 | 0.2559 |
| Heart+ | 1,024 | DP-CounterGAN-SD ε=16 | 0.7295 | 0.8050 | 0.6850 | 0.7030 | 0.0971 | 0.6422 | 0.6422 | 0.0615 | 0.3374 | 0.5832 | 0.5370 | 0.9136 | 0.9907 | 0.2291 |
| Heart+ | 1,024 | DP-CounterGAN-SD ε=8 | 0.7490 | 0.8050 | 0.6900 | 0.7155 | 0.0957 | 0.6461 | 0.6461 | 0.0618 | 0.3360 | 0.6261 | 0.5228 | 0.9128 | 0.9902 | 0.2330 |
| Heart+ | 1,024 | DP-CounterGAN-SD ε=4 (primary) | 0.7265 | 0.8200 | 0.6750 | 0.6945 | 0.0958 | 0.6431 | 0.6431 | 0.0604 | 0.3362 | 0.6012 | 0.5309 | 0.9064 | 0.9902 | 0.2318 |
| Heart+ | 1,024 | DP-CounterGAN-SD ε=2 | 0.7550 | 0.8200 | 0.7050 | 0.7275 | 0.0955 | 0.6457 | 0.6457 | 0.0602 | 0.3376 | 0.6200 | 0.5307 | 0.9225 | 0.9900 | 0.2406 |
| MIMIC-IV | 512 | Uniform Random | 0.8410 | 0.9450 | 0.7900 | 0.6980 | 0.1121 | 0.1473 | 0.0067 | 0.1327 | 0.9158 | 0.2462 | 0.4560 | 0.8602 | 0.9741 | 0.7029 |
| MIMIC-IV | 512 | Genetic CFE | 0.9430 | 0.9850 | 0.9100 | 0.8445 | 0.0870 | 0.2726 | 0.0299 | 0.0869 | 0.8733 | 0.4288 | 0.4958 | 0.8958 | 0.9538 | 0.7245 |
| MIMIC-IV | 512 | DiCE-style gradient | 0.0845 | 0.8300 | 0.0000 | 0.0820 | 0.0800 | 0.0898 | 0.0005 | 0.0005 | 0.7605 | 0.7177 | 0.4456 | 0.9422 | 0.9873 | 0.1484 |
| MIMIC-IV | 512 | Wachter-style | 0.0840 | 0.8300 | 0.0000 | 0.0815 | 0.0800 | 0.0876 | 0.0005 | 0.0003 | 0.7604 | 0.7177 | 0.4467 | 0.9410 | 0.9873 | 0.1398 |
| MIMIC-IV | 512 | CounterGAN one-shot | 0.9775 | 0.9900 | 0.9700 | 0.9645 | 0.1327 | 0.1600 | 0.0097 | 0.0335 | 0.7998 | 0.6552 | 0.6572 | 0.9375 | 0.9787 | 0.0949 |
| MIMIC-IV | 512 | CounterGAN iterative | 0.9950 | 0.9950 | 0.9950 | 0.9765 | 0.1367 | 0.1238 | 0.0045 | 0.0376 | 0.8078 | 0.6399 | 0.6667 | 0.9396 | 0.9827 | 0.0963 |
| MIMIC-IV | 512 | CounterGAN-SD non-DP | 0.9950 | 0.9950 | 0.9950 | 0.9185 | 0.1040 | 0.5418 | 0.4474 | 0.0893 | 0.8791 | 0.4281 | 0.6159 | 0.9150 | 0.9516 | 3.4275 |
| MIMIC-IV | 512 | DP-CounterGAN-SD ε=16 | 0.9950 | 0.9950 | 0.9950 | 0.9000 | 0.0971 | 0.5687 | 0.4806 | 0.0876 | 0.8292 | 0.5847 | 0.6238 | 0.9126 | 0.9430 | 3.4697 |
| MIMIC-IV | 512 | DP-CounterGAN-SD ε=8 | 0.9950 | 0.9950 | 0.9950 | 0.8970 | 0.0909 | 0.5872 | 0.4959 | 0.0811 | 0.8524 | 0.5094 | 0.6242 | 0.9089 | 0.9384 | 3.4435 |
| MIMIC-IV | 512 | DP-CounterGAN-SD ε=4 (primary) | 0.9950 | 0.9950 | 0.9950 | 0.9230 | 0.1030 | 0.5632 | 0.4728 | 0.0829 | 0.8984 | 0.4220 | 0.6851 | 0.9127 | 0.9456 | 3.4302 |
| MIMIC-IV | 512 | DP-CounterGAN-SD ε=2 | 0.9950 | 0.9950 | 0.9950 | 0.9095 | 0.1183 | 0.5175 | 0.4227 | 0.0922 | 0.8942 | 0.4381 | 0.7320 | 0.9232 | 0.9487 | 3.4481 |

**Cách đọc bảng macro.** Macro cho trọng số 0.5 cho mỗi hướng nên không bị hướng có nhiều factual hoặc lớp phổ biến hơn chi phối. ECG cho thấy nhiều method cùng Yield=1 nhưng khác mạnh về Sparsity/Robustness; Heart+ macro thấp chủ yếu do chiều không bệnh → bệnh; MIMIC macro cao nhưng vẫn phải giữ cảnh báo Plausibility và linkage theo hướng. Bảng này thích hợp để tóm tắt, không dùng để tuyên bố hai hướng tương đương.

### 3.5 Official DiCE paper extension trên toàn bộ outer cohort

Ba extension artifact dùng đúng accepted V5.5 checkpoint, fitted preprocessor, 200 query IDs (100 mỗi hướng), K=10, logit margin 0.10 và DomainProjector của bảng chính; MLP/GAN không được train lại. Timeout và no-CF vẫn nằm trong mẫu số với Yield/Coverage bằng 0. Official DiCE Random dùng `sample_size=10,000`; Official DiCE Genetic dùng tối đa 300 native iterations; mỗi factual-method call có timeout 20 giây. Vì native controls không quy đổi thành counted candidate budget B của proposed, đây là **matched outcome/constraint comparison nhưng không phải equal-compute comparison**.

#### Official DiCE theo hướng; quality chỉ tính khi có valid CFE

| Dataset | Official method | Direction | Factuals | OK | Timeout | No-CF | Quality-defined n | Yield@10 ↑ | Coverage@1 ↑ | Full-10 ↑ | Robust Yield@10 ↑ | Robust Full-10 ↑ | Proximity ↓ | Sparsity ↑ | Diversity ↑ | Plausibility distance ↓ | Mean runtime/factual (s) ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Leipzig ECG | Official DiCE genetic | Bệnh → không bệnh | 100 | 100 | 0 | 0 | 100 | 0.8430 | 1.0000 | 0.3000 | 0.8260 | 0.2800 | 0.3472 | 0.0586 | 0.2041 | 1.5653 | 5.4926 |
| Leipzig ECG | Official DiCE genetic | Không bệnh → bệnh | 100 | 100 | 0 | 0 | 99 | 0.5620 | 0.9900 | 0.0300 | 0.5390 | 0.0300 | 0.3524 | 0.0385 | 0.1690 | 1.5086 | 3.8908 |
| Leipzig ECG | Official DiCE random | Bệnh → không bệnh | 100 | 90 | 10 | 0 | 90 | 0.4960 | 0.9000 | 0.0000 | 0.3900 | 0.0000 | 0.0472 | 0.8354 | 0.0615 | 0.6597 | 4.0897 |
| Leipzig ECG | Official DiCE random | Không bệnh → bệnh | 100 | 89 | 11 | 0 | 89 | 0.3660 | 0.8900 | 0.0000 | 0.2750 | 0.0000 | 0.0464 | 0.8291 | 0.0620 | 0.6461 | 4.2692 |
| Heart+ | Official DiCE genetic | Bệnh → không bệnh | 100 | 0 | 100 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | — | — | — | — | 20.0012 |
| Heart+ | Official DiCE genetic | Không bệnh → bệnh | 100 | 0 | 100 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | — | — | — | — | 20.0012 |
| Heart+ | Official DiCE random | Bệnh → không bệnh | 100 | 89 | 0 | 11 | 75 | 0.2440 | 0.7500 | 0.0000 | 0.2300 | 0.0000 | 0.0592 | 0.7431 | 0.0520 | 0.3439 | 1.3825 |
| Heart+ | Official DiCE random | Không bệnh → bệnh | 100 | 67 | 0 | 33 | 63 | 0.1710 | 0.6300 | 0.0000 | 0.1600 | 0.0000 | 0.0742 | 0.7023 | 0.0368 | 0.3293 | 1.6593 |
| MIMIC-IV | Official DiCE genetic | Bệnh → không bệnh | 100 | 9 | 91 | 0 | 8 | 0.0250 | 0.0800 | 0.0000 | 0.0180 | 0.0000 | 0.2271 | 0.0567 | 0.1575 | 1.0881 | 19.7477 |
| MIMIC-IV | Official DiCE genetic | Không bệnh → bệnh | 100 | 100 | 0 | 0 | 99 | 0.9750 | 0.9900 | 0.8600 | 0.9720 | 0.8400 | 0.2768 | 0.0395 | 0.2324 | 1.3511 | 3.3546 |
| MIMIC-IV | Official DiCE random | Bệnh → không bệnh | 100 | 100 | 0 | 0 | 96 | 0.2810 | 0.9600 | 0.0000 | 0.1810 | 0.0000 | 0.0406 | 0.8687 | 0.0337 | 0.8385 | 3.0174 |
| MIMIC-IV | Official DiCE random | Không bệnh → bệnh | 100 | 100 | 0 | 0 | 82 | 0.2130 | 0.8200 | 0.0000 | 0.1250 | 0.0000 | 0.0263 | 0.8905 | 0.0259 | 0.7162 | 2.7704 |

`Quality-defined n` là số factual có ít nhất một valid CFE để định nghĩa Proximity/Sparsity/Diversity/Plausibility. Do đó không được kết luận một method tốt hơn về quality chỉ từ các cột này khi Yield/Coverage rất thấp. `Constraint validity=1` không in lại vì archive chỉ lưu feasible CFE và chỉ số đó không thay Yield@10.

#### Equal-direction macro của official DiCE

| Dataset | Official method | Yield@10 ↑ | Coverage@1 ↑ | Full-10 ↑ | Robust Yield@10 ↑ | Robust Full-10 ↑ | Proximity ↓ | Sparsity ↑ | Diversity ↑ | Plausibility distance ↓ | Mean runtime/factual (s) ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Leipzig ECG | Official DiCE genetic | 0.7025 | 0.9950 | 0.1650 | 0.6825 | 0.1550 | 0.3498 | 0.0485 | 0.1865 | 1.5369 | 4.6917 |
| Leipzig ECG | Official DiCE random | 0.4310 | 0.8950 | 0.0000 | 0.3325 | 0.0000 | 0.0468 | 0.8323 | 0.0617 | 0.6529 | 4.1795 |
| Heart+ | Official DiCE genetic | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | — | — | — | — | 20.0012 |
| Heart+ | Official DiCE random | 0.2075 | 0.6900 | 0.0000 | 0.1950 | 0.0000 | 0.0667 | 0.7227 | 0.0444 | 0.3366 | 1.5209 |
| MIMIC-IV | Official DiCE genetic | 0.5000 | 0.5350 | 0.4300 | 0.4950 | 0.4200 | 0.2519 | 0.0481 | 0.1949 | 1.2196 | 11.5511 |
| MIMIC-IV | Official DiCE random | 0.2470 | 0.8900 | 0.0000 | 0.1530 | 0.0000 | 0.0335 | 0.8796 | 0.0298 | 0.7774 | 2.8939 |

**Nhận xét official baseline.** ECG Genetic đạt Coverage gần 1 nhưng Full-10 chỉ 0.30 và 0.03; Random tạo lời giải gần và sparse hơn nhưng Full-10 bằng 0. Heart+ Genetic chạm timeout ở toàn bộ 200 calls, vì vậy các số 0 là kết quả dưới implementation và time cap đã khai báo, không phải chứng minh Genetic nói chung không hoạt động. MIMIC Genetic cực kỳ bất đối xứng: Yield 0.025 ở bệnh → không bệnh và 0.975 ở chiều ngược lại. Macro 0.500 che mất hiện tượng này. Ở MIMIC không bệnh → bệnh, Genetic có Robust Yield 0.972 cao hơn proposed 0.954, nhưng proposed có Full-10 0.990 so với 0.860, Sparsity 0.620 so với 0.040 và Proximity 0.089 so với 0.277. Đây là trade-off nhiều mục tiêu, không có một method thắng mọi cột.

#### Linked comparison: proposed primary và official DiCE

| Dataset | Method | Direction | Factuals | Yield@10 ↑ | Coverage@1 ↑ | Full-10 ↑ | Robust Yield@10 ↑ | Proximity ↓ | Sparsity ↑ | Diversity ↑ | Mean runtime/factual (s) ↓ | Counted B | Native control | Access/compute stratum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Leipzig ECG | DP-CounterGAN-SD ε=4 (primary) | Bệnh → không bệnh | 100 | 1.0000 | 1.0000 | 1.0000 | 0.8330 | 0.0724 | 0.7761 | 0.0677 | 2.0121 | 512.0000 | counted candidate evaluations | custom counted-candidate implementation |
| Leipzig ECG | Official DiCE genetic | Bệnh → không bệnh | 100 | 0.8430 | 1.0000 | 0.3000 | 0.8260 | 0.3472 | 0.0586 | 0.2041 | 5.4926 | — | maxiterations=300 | official dice-ml native implementation |
| Leipzig ECG | Official DiCE random | Bệnh → không bệnh | 100 | 0.4960 | 0.9000 | 0.0000 | 0.3900 | 0.0472 | 0.8354 | 0.0615 | 4.0897 | — | sample_size=10000 | official dice-ml native implementation |
| Leipzig ECG | DP-CounterGAN-SD ε=4 (primary) | Không bệnh → bệnh | 100 | 1.0000 | 1.0000 | 1.0000 | 0.7360 | 0.0673 | 0.7608 | 0.0740 | 2.0293 | 512.0000 | counted candidate evaluations | custom counted-candidate implementation |
| Leipzig ECG | Official DiCE genetic | Không bệnh → bệnh | 100 | 0.5620 | 0.9900 | 0.0300 | 0.5390 | 0.3524 | 0.0385 | 0.1690 | 3.8908 | — | maxiterations=300 | official dice-ml native implementation |
| Leipzig ECG | Official DiCE random | Không bệnh → bệnh | 100 | 0.3660 | 0.8900 | 0.0000 | 0.2750 | 0.0464 | 0.8291 | 0.0620 | 4.2692 | — | sample_size=10000 | official dice-ml native implementation |
| Heart+ | DP-CounterGAN-SD ε=4 (primary) | Bệnh → không bệnh | 100 | 0.9070 | 0.9400 | 0.8800 | 0.8720 | 0.0988 | 0.6472 | 0.0730 | 0.2766 | 1024.0000 | counted candidate evaluations | custom counted-candidate implementation |
| Heart+ | Official DiCE genetic | Bệnh → không bệnh | 100 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | — | — | — | 20.0012 | — | maxiterations=300 | official dice-ml native implementation |
| Heart+ | Official DiCE random | Bệnh → không bệnh | 100 | 0.2440 | 0.7500 | 0.0000 | 0.2300 | 0.0592 | 0.7431 | 0.0520 | 1.3825 | — | sample_size=10000 | official dice-ml native implementation |
| Heart+ | DP-CounterGAN-SD ε=4 (primary) | Không bệnh → bệnh | 100 | 0.5460 | 0.7000 | 0.4700 | 0.5170 | 0.0928 | 0.6390 | 0.0478 | 0.1871 | 1024.0000 | counted candidate evaluations | custom counted-candidate implementation |
| Heart+ | Official DiCE genetic | Không bệnh → bệnh | 100 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | — | — | — | 20.0012 | — | maxiterations=300 | official dice-ml native implementation |
| Heart+ | Official DiCE random | Không bệnh → bệnh | 100 | 0.1710 | 0.6300 | 0.0000 | 0.1600 | 0.0742 | 0.7023 | 0.0368 | 1.6593 | — | sample_size=10000 | official dice-ml native implementation |
| MIMIC-IV | DP-CounterGAN-SD ε=4 (primary) | Bệnh → không bệnh | 100 | 1.0000 | 1.0000 | 1.0000 | 0.8920 | 0.1175 | 0.5062 | 0.0918 | 3.4133 | 512.0000 | counted candidate evaluations | custom counted-candidate implementation |
| MIMIC-IV | Official DiCE genetic | Bệnh → không bệnh | 100 | 0.0250 | 0.0800 | 0.0000 | 0.0180 | 0.2271 | 0.0567 | 0.1575 | 19.7477 | — | maxiterations=300 | official dice-ml native implementation |
| MIMIC-IV | Official DiCE random | Bệnh → không bệnh | 100 | 0.2810 | 0.9600 | 0.0000 | 0.1810 | 0.0406 | 0.8687 | 0.0337 | 3.0174 | — | sample_size=10000 | official dice-ml native implementation |
| MIMIC-IV | DP-CounterGAN-SD ε=4 (primary) | Không bệnh → bệnh | 100 | 0.9900 | 0.9900 | 0.9900 | 0.9540 | 0.0885 | 0.6203 | 0.0740 | 3.4471 | 512.0000 | counted candidate evaluations | custom counted-candidate implementation |
| MIMIC-IV | Official DiCE genetic | Không bệnh → bệnh | 100 | 0.9750 | 0.9900 | 0.8600 | 0.9720 | 0.2768 | 0.0395 | 0.2324 | 3.3546 | — | maxiterations=300 | official dice-ml native implementation |
| MIMIC-IV | Official DiCE random | Không bệnh → bệnh | 100 | 0.2130 | 0.8200 | 0.0000 | 0.1250 | 0.0263 | 0.8905 | 0.0259 | 2.7704 | — | sample_size=10000 | official dice-ml native implementation |

Bảng linked dùng cùng cohort/constraints/metrics nhưng cố ý để `Counted B=—` cho official DiCE. Runtime được báo để mô tả implementation đã chạy, không đủ để tuyên bố speedup thuật toán vì oracle access, native stopping rule và số candidate evaluations không đồng nhất.

### 3.6 Paired factual-level comparisons với Holm correction

Delta luôn là `DP ε=4 primary − comparator` trên cùng factual. Dấu tốt phụ thuộc cột `Higher is better`; p-value đã Holm-correct trên family được xuất bởi notebook. Đây là paired factual evidence trong một generator-training seed, không thay multi-seed confirmation.

| Dataset | Comparator | Metric | Paired factuals | Mean Δ primary−comparator | Bootstrap 95% low | Bootstrap 95% high | Higher is better | Holm p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Leipzig ECG | CounterGAN one-shot | diversity | 200 | 0.0366 | 0.0323 | 0.0407 | Có | 3.3786e-31 |
| Leipzig ECG | CounterGAN one-shot | proximity | 200 | -0.0328 | -0.0375 | -0.0289 | Không | 5.1289e-24 |
| Leipzig ECG | CounterGAN one-shot | robust_completeness_at_k | 200 | -0.1350 | -0.1655 | -0.1050 | Có | 4.0714e-10 |
| Leipzig ECG | CounterGAN one-shot | sparsity | 200 | 0.5415 | 0.5294 | 0.5529 | Có | 6.5291e-32 |
| Leipzig ECG | CounterGAN one-shot | valid_cfe_yield_at_k | 200 | 0.0280 | 0.0150 | 0.0445 | Có | 0.0983 |
| Leipzig ECG | CounterGAN-SD non-DP | diversity | 200 | 0.0019 | -0.0015 | 0.0055 | Có | 1.0000 |
| Leipzig ECG | CounterGAN-SD non-DP | proximity | 200 | -0.0005 | -0.0034 | 0.0025 | Không | 1.0000 |
| Leipzig ECG | CounterGAN-SD non-DP | robust_completeness_at_k | 200 | -0.0315 | -0.0551 | -0.0084 | Có | 1.0000 |
| Leipzig ECG | CounterGAN-SD non-DP | sparsity | 200 | -0.0051 | -0.0133 | 0.0021 | Có | 1.0000 |
| Leipzig ECG | CounterGAN-SD non-DP | valid_cfe_yield_at_k | 200 | 0.0000 | 0.0000 | 0.0000 | Có | 1.0000 |
| Leipzig ECG | DiCE-style gradient | diversity | 200 | 0.0702 | 0.0666 | 0.0743 | Có | 6.5291e-32 |
| Leipzig ECG | DiCE-style gradient | proximity | 200 | -0.0027 | -0.0066 | 0.0006 | Không | 1.0000 |
| Leipzig ECG | DiCE-style gradient | robust_completeness_at_k | 200 | 0.6830 | 0.6600 | 0.7050 | Có | 3.4211e-32 |
| Leipzig ECG | DiCE-style gradient | sparsity | 200 | 0.7206 | 0.7037 | 0.7346 | Có | 6.5291e-32 |
| Leipzig ECG | DiCE-style gradient | valid_cfe_yield_at_k | 200 | 0.8980 | 0.8965 | 0.8995 | Có | 6.6919e-42 |
| Leipzig ECG | Genetic CFE | diversity | 200 | -0.0098 | -0.0142 | -0.0054 | Có | 0.0006 |
| Leipzig ECG | Genetic CFE | proximity | 200 | -0.0047 | -0.0084 | -0.0012 | Không | 0.1402 |
| Leipzig ECG | Genetic CFE | robust_completeness_at_k | 200 | -0.1165 | -0.1435 | -0.0940 | Có | 6.9860e-11 |
| Leipzig ECG | Genetic CFE | sparsity | 200 | 0.3536 | 0.3393 | 0.3655 | Có | 6.5291e-32 |
| Leipzig ECG | Genetic CFE | valid_cfe_yield_at_k | 200 | 0.0000 | 0.0000 | 0.0000 | Có | 1.0000 |
| Leipzig ECG | Uniform Random | diversity | 200 | -0.0497 | -0.0546 | -0.0450 | Có | 7.3220e-28 |
| Leipzig ECG | Uniform Random | proximity | 200 | -0.0234 | -0.0265 | -0.0194 | Không | 1.7545e-18 |
| Leipzig ECG | Uniform Random | robust_completeness_at_k | 200 | -0.0515 | -0.0835 | -0.0195 | Có | 0.1812 |
| Leipzig ECG | Uniform Random | sparsity | 200 | 0.5576 | 0.5468 | 0.5697 | Có | 6.5291e-32 |
| Leipzig ECG | Uniform Random | valid_cfe_yield_at_k | 200 | 0.0000 | 0.0000 | 0.0000 | Có | 1.0000 |
| Leipzig ECG | Wachter-style | diversity | 200 | 0.0704 | 0.0656 | 0.0747 | Có | 6.5291e-32 |
| Leipzig ECG | Wachter-style | proximity | 200 | -0.0030 | -0.0065 | 0.0007 | Không | 0.7776 |
| Leipzig ECG | Wachter-style | robust_completeness_at_k | 200 | 0.6835 | 0.6615 | 0.7055 | Có | 3.3590e-32 |
| Leipzig ECG | Wachter-style | sparsity | 200 | 0.7256 | 0.7103 | 0.7400 | Có | 6.5291e-32 |
| Leipzig ECG | Wachter-style | valid_cfe_yield_at_k | 200 | 0.8985 | 0.8965 | 0.9000 | Có | 4.2157e-42 |
| Heart+ | CounterGAN one-shot | diversity | 152 | 0.0371 | 0.0307 | 0.0427 | Có | 2.1728e-15 |
| Heart+ | CounterGAN one-shot | proximity | 152 | -0.0646 | -0.0701 | -0.0586 | Không | 8.8710e-24 |
| Heart+ | CounterGAN one-shot | robust_completeness_at_k | 200 | 0.4925 | 0.4460 | 0.5495 | Có | 2.4888e-24 |
| Heart+ | CounterGAN one-shot | sparsity | 152 | 0.2944 | 0.2769 | 0.3167 | Có | 8.0679e-24 |
| Heart+ | CounterGAN one-shot | valid_cfe_yield_at_k | 200 | 0.5210 | 0.4744 | 0.5690 | Có | 1.2480e-24 |
| Heart+ | CounterGAN-SD non-DP | diversity | 163 | -0.0059 | -0.0081 | -0.0035 | Có | 0.0001 |
| Heart+ | CounterGAN-SD non-DP | proximity | 163 | -0.0061 | -0.0076 | -0.0042 | Không | 7.5340e-08 |
| Heart+ | CounterGAN-SD non-DP | robust_completeness_at_k | 200 | -0.0640 | -0.0911 | -0.0365 | Có | 0.0188 |
| Heart+ | CounterGAN-SD non-DP | sparsity | 163 | 0.0095 | 0.0021 | 0.0158 | Có | 1.0000 |
| Heart+ | CounterGAN-SD non-DP | valid_cfe_yield_at_k | 200 | -0.0610 | -0.0900 | -0.0310 | Có | 0.0410 |
| Heart+ | DiCE-style gradient | diversity | 142 | 0.0129 | 0.0081 | 0.0175 | Có | 0.0003 |
| Heart+ | DiCE-style gradient | proximity | 142 | -0.0119 | -0.0155 | -0.0085 | Không | 2.9074e-06 |
| Heart+ | DiCE-style gradient | robust_completeness_at_k | 200 | 0.4375 | 0.3990 | 0.4825 | Có | 4.1144e-24 |
| Heart+ | DiCE-style gradient | sparsity | 142 | 0.1994 | 0.1801 | 0.2143 | Có | 2.4225e-22 |
| Heart+ | DiCE-style gradient | valid_cfe_yield_at_k | 200 | 0.4690 | 0.4259 | 0.5150 | Có | 1.3638e-24 |
| Heart+ | Genetic CFE | diversity | 119 | 0.0384 | 0.0342 | 0.0421 | Có | 2.2098e-17 |
| Heart+ | Genetic CFE | proximity | 119 | 0.0266 | 0.0235 | 0.0302 | Không | 3.1841e-17 |
| Heart+ | Genetic CFE | robust_completeness_at_k | 200 | 0.3295 | 0.2810 | 0.3785 | Có | 5.8723e-20 |
| Heart+ | Genetic CFE | sparsity | 119 | 0.0437 | 0.0331 | 0.0542 | Có | 2.9509e-08 |
| Heart+ | Genetic CFE | valid_cfe_yield_at_k | 200 | 0.3290 | 0.2753 | 0.3755 | Có | 5.6490e-18 |
| Heart+ | Uniform Random | diversity | 119 | 0.0164 | 0.0110 | 0.0214 | Có | 3.1985e-05 |
| Heart+ | Uniform Random | proximity | 119 | 0.0141 | 0.0094 | 0.0189 | Không | 4.9288e-06 |
| Heart+ | Uniform Random | robust_completeness_at_k | 200 | 0.2510 | 0.2084 | 0.3020 | Có | 2.5020e-14 |
| Heart+ | Uniform Random | sparsity | 119 | 0.1185 | 0.1035 | 0.1355 | Có | 3.6624e-17 |
| Heart+ | Uniform Random | valid_cfe_yield_at_k | 200 | 0.2615 | 0.2130 | 0.3165 | Có | 1.6887e-13 |
| Heart+ | Wachter-style | diversity | 141 | 0.0143 | 0.0098 | 0.0185 | Có | 9.0340e-06 |
| Heart+ | Wachter-style | proximity | 141 | -0.0114 | -0.0149 | -0.0084 | Không | 2.7903e-06 |
| Heart+ | Wachter-style | robust_completeness_at_k | 200 | 0.4475 | 0.3965 | 0.4921 | Có | 3.6132e-24 |
| Heart+ | Wachter-style | sparsity | 141 | 0.1998 | 0.1831 | 0.2151 | Có | 3.5117e-22 |
| Heart+ | Wachter-style | valid_cfe_yield_at_k | 200 | 0.4785 | 0.4385 | 0.5190 | Có | 1.2694e-24 |
| MIMIC-IV | CounterGAN one-shot | diversity | 198 | 0.0493 | 0.0454 | 0.0530 | Có | 1.4263e-31 |
| MIMIC-IV | CounterGAN one-shot | proximity | 198 | -0.0297 | -0.0348 | -0.0242 | Không | 5.1874e-16 |
| MIMIC-IV | CounterGAN one-shot | robust_completeness_at_k | 200 | -0.0415 | -0.0660 | -0.0125 | Có | 1.2351e-06 |
| MIMIC-IV | CounterGAN one-shot | sparsity | 198 | 0.4038 | 0.3837 | 0.4229 | Có | 1.3708e-31 |
| MIMIC-IV | CounterGAN one-shot | valid_cfe_yield_at_k | 200 | 0.0175 | 0.0030 | 0.0355 | Có | 1.0000 |
| MIMIC-IV | CounterGAN-SD non-DP | diversity | 199 | -0.0064 | -0.0114 | -0.0022 | Có | 1.0000 |
| MIMIC-IV | CounterGAN-SD non-DP | proximity | 199 | -0.0009 | -0.0053 | 0.0038 | Không | 1.0000 |
| MIMIC-IV | CounterGAN-SD non-DP | robust_completeness_at_k | 200 | 0.0045 | -0.0130 | 0.0225 | Có | 1.0000 |
| MIMIC-IV | CounterGAN-SD non-DP | sparsity | 199 | 0.0210 | 0.0025 | 0.0371 | Có | 1.0000 |
| MIMIC-IV | CounterGAN-SD non-DP | valid_cfe_yield_at_k | 200 | 0.0000 | 0.0000 | 0.0000 | Có | 1.0000 |
| MIMIC-IV | DiCE-style gradient | diversity | 166 | 0.0796 | 0.0764 | 0.0837 | Có | 2.2544e-26 |
| MIMIC-IV | DiCE-style gradient | proximity | 166 | 0.0149 | 0.0114 | 0.0191 | Không | 6.5975e-05 |
| MIMIC-IV | DiCE-style gradient | robust_completeness_at_k | 200 | 0.8410 | 0.8200 | 0.8575 | Có | 4.1304e-33 |
| MIMIC-IV | DiCE-style gradient | sparsity | 166 | 0.4925 | 0.4726 | 0.5088 | Có | 2.2544e-26 |
| MIMIC-IV | DiCE-style gradient | valid_cfe_yield_at_k | 200 | 0.9105 | 0.8985 | 0.9200 | Có | 6.0748e-37 |
| MIMIC-IV | Genetic CFE | diversity | 197 | -0.0044 | -0.0086 | -0.0004 | Có | 0.4527 |
| MIMIC-IV | Genetic CFE | proximity | 197 | 0.0153 | 0.0111 | 0.0187 | Không | 3.6873e-08 |
| MIMIC-IV | Genetic CFE | robust_completeness_at_k | 200 | 0.0785 | 0.0525 | 0.1110 | Có | 0.0013 |
| MIMIC-IV | Genetic CFE | sparsity | 197 | 0.2920 | 0.2746 | 0.3056 | Có | 2.1533e-31 |
| MIMIC-IV | Genetic CFE | valid_cfe_yield_at_k | 200 | 0.0520 | 0.0260 | 0.0745 | Có | 0.0558 |
| MIMIC-IV | Uniform Random | diversity | 189 | -0.0512 | -0.0578 | -0.0453 | Có | 1.4525e-19 |
| MIMIC-IV | Uniform Random | proximity | 189 | -0.0117 | -0.0154 | -0.0085 | Không | 2.4985e-05 |
| MIMIC-IV | Uniform Random | robust_completeness_at_k | 200 | 0.2250 | 0.1859 | 0.2670 | Có | 1.6069e-19 |
| MIMIC-IV | Uniform Random | sparsity | 189 | 0.4241 | 0.4101 | 0.4380 | Có | 3.9270e-30 |
| MIMIC-IV | Uniform Random | valid_cfe_yield_at_k | 200 | 0.1540 | 0.1090 | 0.1961 | Có | 6.0638e-06 |
| MIMIC-IV | Wachter-style | diversity | 166 | 0.0797 | 0.0758 | 0.0832 | Có | 2.2544e-26 |
| MIMIC-IV | Wachter-style | proximity | 166 | 0.0149 | 0.0107 | 0.0200 | Không | 6.6086e-05 |
| MIMIC-IV | Wachter-style | robust_completeness_at_k | 200 | 0.8415 | 0.8235 | 0.8575 | Có | 3.8651e-33 |
| MIMIC-IV | Wachter-style | sparsity | 166 | 0.4947 | 0.4763 | 0.5127 | Có | 2.2544e-26 |
| MIMIC-IV | Wachter-style | valid_cfe_yield_at_k | 200 | 0.9110 | 0.8990 | 0.9200 | Có | 4.7125e-37 |

**Cách đọc kiểm định paired.** Mỗi delta dùng đúng cùng factual nên loại được khác biệt do chọn bệnh nhân. Với metric có `Higher is better=Yes`, delta dương có lợi cho proposed; với Proximity, delta âm mới có lợi. Khoảng bootstrap không chứa 0 và Holm p nhỏ cho bằng chứng khác biệt ở factual level. Tuy nhiên 200 factual không biến thành 200 lần train GAN: bảng này vẫn chỉ phản ánh một generator-training seed và cần multi-seed để kết luận độ ổn định qua huấn luyện.

## 4. Budget frontier của phương pháp primary DP ε=4

Bảng này cho phép đọc riêng một hướng mà không cần suy từ macro. Full-10 không bị ép bằng 1; khi budget/round giảm, Yield và Full-10 được phép giảm tự nhiên.

### 4.1 Leipzig ECG

| Budget B | Direction | Yield@10 ↑ | Coverage@1 ↑ | Full-10 ↑ | Robust Yield@10 ↑ | Sparsity ↑ | Diversity ↑ | Time to K-or-cap (s) ↓ | Candidates | Rounds ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 64 | Bệnh → không bệnh | 0.9450 | 0.9800 | 0.8100 | 0.7740 | 0.5776 | 0.0469 | 0.2375 | 64.0000 | 1.0000 |
| 64 | Không bệnh → bệnh | 0.9670 | 1.0000 | 0.9500 | 0.7490 | 0.5936 | 0.0597 | 0.2441 | 64.0000 | 1.0000 |
| 128 | Bệnh → không bệnh | 1.0000 | 1.0000 | 1.0000 | 0.7940 | 0.6470 | 0.0616 | 0.4816 | 128.0000 | 2.0000 |
| 128 | Không bệnh → bệnh | 1.0000 | 1.0000 | 1.0000 | 0.7330 | 0.6399 | 0.0714 | 0.4939 | 128.0000 | 2.0000 |
| 256 | Bệnh → không bệnh | 1.0000 | 1.0000 | 1.0000 | 0.8140 | 0.7141 | 0.0718 | 0.9846 | 256.0000 | 4.0000 |
| 256 | Không bệnh → bệnh | 1.0000 | 1.0000 | 1.0000 | 0.7390 | 0.7004 | 0.0759 | 0.9988 | 256.0000 | 4.0000 |
| 512 | Bệnh → không bệnh | 1.0000 | 1.0000 | 1.0000 | 0.8330 | 0.7761 | 0.0677 | 2.0121 | 512.0000 | 8.0000 |
| 512 | Không bệnh → bệnh | 1.0000 | 1.0000 | 1.0000 | 0.7360 | 0.7608 | 0.0740 | 2.0293 | 512.0000 | 8.0000 |

**Nhận xét budget.** Khi tăng B từ 64 lên 512, Yield@10 của hướng bệnh → không bệnh đổi từ 0.945 lên 1.000; hướng không bệnh → bệnh đổi từ 0.967 lên 1.000. B lớn hơn cho archive nhiều cơ hội đạt K và chọn MMR, nhưng tăng thời gian/candidate calls. Operating budget được khóa từ inner validation, không chọn bằng hàng có outer Yield đẹp nhất.

### 4.2 Heart+

| Budget B | Direction | Yield@10 ↑ | Coverage@1 ↑ | Full-10 ↑ | Robust Yield@10 ↑ | Sparsity ↑ | Diversity ↑ | Time to K-or-cap (s) ↓ | Candidates | Rounds ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 128 | Bệnh → không bệnh | 0.5780 | 0.8700 | 0.3300 | 0.5710 | 0.5261 | 0.0611 | 0.0205 | 128.0000 | 1.0000 |
| 128 | Không bệnh → bệnh | 0.1470 | 0.3200 | 0.0400 | 0.1410 | 0.5599 | 0.0412 | 0.0159 | 128.0000 | 1.0000 |
| 256 | Bệnh → không bệnh | 0.8400 | 0.9100 | 0.7600 | 0.8220 | 0.5798 | 0.0703 | 0.0522 | 256.0000 | 2.0000 |
| 256 | Không bệnh → bệnh | 0.2610 | 0.4000 | 0.1500 | 0.2500 | 0.6092 | 0.0492 | 0.0350 | 256.0000 | 2.0000 |
| 512 | Bệnh → không bệnh | 0.8690 | 0.9100 | 0.8400 | 0.8370 | 0.6412 | 0.0715 | 0.1245 | 512.0000 | 4.0000 |
| 512 | Không bệnh → bệnh | 0.4110 | 0.5000 | 0.3400 | 0.3970 | 0.6474 | 0.0502 | 0.0809 | 512.0000 | 4.0000 |
| 1,024 | Bệnh → không bệnh | 0.9070 | 0.9400 | 0.8800 | 0.8720 | 0.6472 | 0.0730 | 0.2766 | 1024.0000 | 8.0000 |
| 1,024 | Không bệnh → bệnh | 0.5460 | 0.7000 | 0.4700 | 0.5170 | 0.6390 | 0.0478 | 0.1871 | 1024.0000 | 8.0000 |

**Nhận xét budget.** Khi tăng B từ 128 lên 1024, Yield@10 của hướng bệnh → không bệnh đổi từ 0.578 lên 0.907; hướng không bệnh → bệnh đổi từ 0.147 lên 0.546. B lớn hơn cho archive nhiều cơ hội đạt K và chọn MMR, nhưng tăng thời gian/candidate calls. Operating budget được khóa từ inner validation, không chọn bằng hàng có outer Yield đẹp nhất.

### 4.3 MIMIC-IV

| Budget B | Direction | Yield@10 ↑ | Coverage@1 ↑ | Full-10 ↑ | Robust Yield@10 ↑ | Sparsity ↑ | Diversity ↑ | Time to K-or-cap (s) ↓ | Candidates | Rounds ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 64 | Bệnh → không bệnh | 0.8700 | 0.9400 | 0.8300 | 0.8010 | 0.2488 | 0.0503 | 0.3695 | 64.0000 | 1.0000 |
| 64 | Không bệnh → bệnh | 0.9830 | 0.9900 | 0.9600 | 0.9660 | 0.2793 | 0.0716 | 0.3644 | 64.0000 | 1.0000 |
| 128 | Bệnh → không bệnh | 0.9500 | 0.9900 | 0.9100 | 0.8600 | 0.3018 | 0.0683 | 0.7888 | 128.0000 | 2.0000 |
| 128 | Không bệnh → bệnh | 0.9900 | 0.9900 | 0.9900 | 0.9710 | 0.3448 | 0.0794 | 0.7506 | 128.0000 | 2.0000 |
| 256 | Bệnh → không bệnh | 1.0000 | 1.0000 | 1.0000 | 0.9090 | 0.3894 | 0.0835 | 1.6382 | 256.0000 | 4.0000 |
| 256 | Không bệnh → bệnh | 0.9900 | 0.9900 | 0.9900 | 0.9570 | 0.4529 | 0.0833 | 1.6085 | 256.0000 | 4.0000 |
| 512 | Bệnh → không bệnh | 1.0000 | 1.0000 | 1.0000 | 0.8920 | 0.5062 | 0.0918 | 3.4133 | 512.0000 | 8.0000 |
| 512 | Không bệnh → bệnh | 0.9900 | 0.9900 | 0.9900 | 0.9540 | 0.6203 | 0.0740 | 3.4471 | 512.0000 | 8.0000 |

**Nhận xét budget.** Khi tăng B từ 64 lên 512, Yield@10 của hướng bệnh → không bệnh đổi từ 0.870 lên 1.000; hướng không bệnh → bệnh đổi từ 0.983 lên 0.990. B lớn hơn cho archive nhiều cơ hội đạt K và chọn MMR, nhưng tăng thời gian/candidate calls. Operating budget được khóa từ inner validation, không chọn bằng hàng có outer Yield đẹp nhất.

## 5. Population/round sensitivity trên inner validation

Population là candidate mỗi round; B/P là số round tối đa. Đây là inner sensitivity đã dùng để freeze P trước outer test, không phải batch bệnh nhân và không được chọn lại bằng kết quả HE outer.

### 5.1 Leipzig ECG

| P | B | Max rounds | Yield@10 ↑ | Robust validity ↑ | Sparsity ↑ | Diversity ↑ | Rounds to K-or-cap ↓ | Candidates to K-or-cap ↓ | Selected |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16 | 512 | 32 | 1.0000 | 0.9313 | 0.7389 | 0.0699 | 3.0000 | 48.0000 | Không |
| 32 | 512 | 16 | 1.0000 | 0.9729 | 0.7243 | 0.0895 | 1.7500 | 56.0000 | Không |
| 64 | 512 | 8 | 1.0000 | 0.9542 | 0.7500 | 0.0812 | 1.2500 | 80.0000 | Có |
| 128 | 512 | 4 | 1.0000 | 0.9688 | 0.6458 | 0.0949 | 1.1250 | 144.0000 | Không |
| 256 | 512 | 2 | 1.0000 | 0.9604 | 0.6104 | 0.0829 | 1.1250 | 288.0000 | Không |
| 512 | 512 | 1 | 1.0000 | 0.9354 | 0.5424 | 0.0547 | 1.0000 | 512.0000 | Không |

**Nhận xét population.** P là số candidate được chấm trong một round, không phải số bệnh nhân và không phải batch training. Với B cố định, P lớn làm ít round hơn nhưng mỗi round HE mang population lớn hơn; P nhỏ tạo nhiều cơ hội feedback qua round nhưng phải trả thêm latency/communication theo round. Leipzig ECG chọn P=64 trước outer test. Dòng `Selected` là quyết định vận hành, không khẳng định P đó tối ưu tuyệt đối cho mọi metric.

### 5.2 Heart+

| P | B | Max rounds | Yield@10 ↑ | Robust validity ↑ | Sparsity ↑ | Diversity ↑ | Rounds to K-or-cap ↓ | Candidates to K-or-cap ↓ | Selected |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 1,024 | 32 | 0.9750 | 0.9870 | 0.6722 | 0.0592 | 9.8750 | 316.0000 | Không |
| 64 | 1,024 | 16 | 0.8625 | 0.9878 | 0.7411 | 0.0645 | 6.2500 | 400.0000 | Không |
| 128 | 1,024 | 8 | 0.9375 | 0.9979 | 0.6917 | 0.0658 | 3.6250 | 464.0000 | Có |
| 256 | 1,024 | 4 | 0.7875 | 0.9976 | 0.6899 | 0.0696 | 2.3750 | 608.0000 | Không |
| 512 | 1,024 | 2 | 0.7000 | 0.9972 | 0.6938 | 0.0739 | 1.5000 | 768.0000 | Không |
| 1,024 | 1,024 | 1 | 0.5625 | 1.0000 | 0.6067 | 0.0739 | 1.0000 | 1024.0000 | Không |

**Nhận xét population.** P là số candidate được chấm trong một round, không phải số bệnh nhân và không phải batch training. Với B cố định, P lớn làm ít round hơn nhưng mỗi round HE mang population lớn hơn; P nhỏ tạo nhiều cơ hội feedback qua round nhưng phải trả thêm latency/communication theo round. Heart+ chọn P=128 trước outer test. Dòng `Selected` là quyết định vận hành, không khẳng định P đó tối ưu tuyệt đối cho mọi metric.

### 5.3 MIMIC-IV

| P | B | Max rounds | Yield@10 ↑ | Robust validity ↑ | Sparsity ↑ | Diversity ↑ | Rounds to K-or-cap ↓ | Candidates to K-or-cap ↓ | Selected |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 512 | 16 | 1.0000 | 0.9792 | 0.5890 | 0.0915 | 1.0000 | 32.0000 | Không |
| 64 | 512 | 8 | 1.0000 | 0.9688 | 0.6221 | 0.0665 | 1.0000 | 64.0000 | Có |
| 128 | 512 | 4 | 1.0000 | 0.9917 | 0.4934 | 0.0809 | 1.0000 | 128.0000 | Không |
| 256 | 512 | 2 | 1.0000 | 0.9938 | 0.4096 | 0.0709 | 1.0000 | 256.0000 | Không |
| 512 | 512 | 1 | 1.0000 | 1.0000 | 0.3776 | 0.0668 | 1.0000 | 512.0000 | Không |

**Nhận xét population.** P là số candidate được chấm trong một round, không phải số bệnh nhân và không phải batch training. Với B cố định, P lớn làm ít round hơn nhưng mỗi round HE mang population lớn hơn; P nhỏ tạo nhiều cơ hội feedback qua round nhưng phải trả thêm latency/communication theo round. MIMIC-IV chọn P=64 trước outer test. Dòng `Selected` là quyết định vận hành, không khẳng định P đó tối ưu tuyệt đối cho mọi metric.

## 6. Differential privacy và đánh giá tấn công generator

DP là conditional generator-training DP khi frozen oracle và development-fitted preprocessing được coi là public auxiliary. Discriminator là training-only và không được release/query ở V5.5. MIA gần 0.5 là attacker gần random, không phải mục tiêu cần ép chính xác bằng 0.5; CI và cảnh báo phải được giữ.

### 6.1 DP accounting theo checkpoint

| Dataset | Variant | Requested ε | Achieved composed ε | δ | Noise G | Noise D | G steps | D steps | Secure RNG | Warm-up epochs | Ramp epochs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Leipzig ECG | DP ε=16 | 16.0000 | 15.9986 | 1.0000e-05 | 0.5915 | 0.5915 | 6,000 | 6,000 | Có | 30 | 180 |
| Leipzig ECG | DP ε=8 | 8.0000 | 7.9988 | 1.0000e-05 | 0.7307 | 0.7307 | 6,000 | 6,000 | Có | 30 | 180 |
| Leipzig ECG | DP ε=4 | 4.0000 | 3.9992 | 1.0000e-05 | 0.9506 | 0.9506 | 6,000 | 6,000 | Có | 30 | 180 |
| Leipzig ECG | DP ε=2 | 2.0000 | 1.9993 | 1.0000e-05 | 1.4136 | 1.4136 | 6,000 | 6,000 | Có | 30 | 180 |
| Heart+ | DP ε=16 | 16.0000 | 15.9988 | 1.0000e-05 | 0.4248 | 0.4248 | 6,000 | 6,000 | Có | 30 | 180 |
| Heart+ | DP ε=8 | 8.0000 | 7.9988 | 1.0000e-05 | 0.5203 | 0.5203 | 6,000 | 6,000 | Có | 30 | 180 |
| Heart+ | DP ε=4 | 4.0000 | 3.9990 | 1.0000e-05 | 0.6531 | 0.6531 | 6,000 | 6,000 | Có | 30 | 180 |
| Heart+ | DP ε=2 | 2.0000 | 1.9983 | 1.0000e-05 | 0.8459 | 0.8459 | 6,000 | 6,000 | Có | 30 | 180 |
| MIMIC-IV | DP ε=16 | 16.0000 | 15.9986 | 1.0000e-05 | 1.4104 | 1.4104 | 4,200 | 4,200 | Có | 21 | 126 |
| MIMIC-IV | DP ε=8 | 8.0000 | 7.9989 | 1.0000e-05 | 2.3419 | 2.3419 | 4,200 | 4,200 | Có | 21 | 126 |
| MIMIC-IV | DP ε=4 | 4.0000 | 3.9983 | 1.0000e-05 | 4.2041 | 4.2041 | 4,200 | 4,200 | Có | 21 | 126 |
| MIMIC-IV | DP ε=2 | 2.0000 | 1.9987 | 1.0000e-05 | 7.8369 | 7.8369 | 4,200 | 4,200 | Có | 21 | 126 |

`Requested ε` là trần khai báo cho một checkpoint; `Achieved composed ε` là ε thực tế sau khi cộng privacy loss của hai optimizer G và D theo contract huấn luyện. Noise multiplier lớn hơn thường làm gradient riêng tư hơn nhưng không bảo đảm utility hoặc attack AUC biến thiên đơn điệu. `Secure RNG=Có` nghĩa Opacus chạy `secure_mode=True` với nguồn số giả ngẫu nhiên mật mã để tạo nhiễu Gaussian và randomness liên quan DP, làm giảm khả năng kẻ tấn công dự đoán hoặc tái tạo chuỗi nhiễu từ trạng thái RNG thông thường. Secure RNG không mã hóa dữ liệu/checkpoint, không thay CKKS/HE, không tự tạo bảo đảm (ε,δ), không thay privacy accountant và không phải thủ thuật tăng Accuracy/F1. Nếu phát hành nhiều checkpoint từ nhiều training seed, privacy loss vẫn phải composition theo release policy; các seed không được coi là miễn phí.

### 6.2 Strongest generator-only MIA

| Dataset | Variant | Strongest attack | MIA AUC (target 0.5) | Absolute AUC gap from 0.5 ↓ | Signed MIA advantage (target 0) | Mean absolute MIA advantage ↓ | AUC SD | AUC 95% CI low | AUC 95% CI high | Attack seeds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Leipzig ECG | CounterGAN non-DP | calibrated_gaussian_reconstruction_likelihood | 0.5155 | 0.0155 | 0.0134 | 0.0214 | 0.0142 | 0.5069 | 0.5237 | 10 |
| Leipzig ECG | DP ε=16 | multi_draw_mean_objective | 0.4475 | 0.0525 | -0.0432 | 0.0484 | 0.0129 | 0.4401 | 0.4551 | 10 |
| Leipzig ECG | DP ε=2 | calibrated_gaussian_reconstruction_likelihood | 0.6054 | 0.1054 | 0.1412 | 0.1412 | 0.0184 | 0.5942 | 0.6155 | 10 |
| Leipzig ECG | DP ε=4 | multi_draw_mean_objective | 0.5016 | 0.0016 | 0.0508 | 0.0568 | 0.0139 | 0.4935 | 0.5095 | 10 |
| Leipzig ECG | DP ε=8 | multi_draw_mean_objective | 0.4538 | 0.0462 | -0.0155 | 0.0155 | 0.0103 | 0.4482 | 0.4602 | 10 |
| Heart+ | CounterGAN non-DP | multi_draw_reconstruction | 0.5045 | 0.0045 | -0.0018 | 0.0078 | 0.0109 | 0.4980 | 0.5111 | 10 |
| Heart+ | DP ε=16 | single_draw_loss_threshold | 0.4983 | 0.0017 | -0.0132 | 0.0182 | 0.0127 | 0.4908 | 0.5060 | 10 |
| Heart+ | DP ε=2 | multi_draw_reconstruction | 0.5031 | 0.0031 | 0.0071 | 0.0149 | 0.0125 | 0.4960 | 0.5106 | 10 |
| Heart+ | DP ε=4 | multi_draw_mean_objective | 0.4988 | 0.0012 | -0.0045 | 0.0207 | 0.0127 | 0.4914 | 0.5062 | 10 |
| Heart+ | DP ε=8 | single_draw_loss_threshold | 0.4997 | 0.0003 | -0.0047 | 0.0165 | 0.0118 | 0.4931 | 0.5069 | 10 |
| MIMIC-IV | CounterGAN non-DP | calibrated_gaussian_reconstruction_likelihood | 0.4993 | 0.0007 | 0.0027 | 0.0131 | 0.0107 | 0.4927 | 0.5054 | 10 |
| MIMIC-IV | DP ε=16 | single_draw_loss_threshold | 0.5042 | 0.0042 | 0.0070 | 0.0162 | 0.0143 | 0.4957 | 0.5128 | 10 |
| MIMIC-IV | DP ε=2 | multi_draw_reconstruction | 0.5026 | 0.0026 | 0.0011 | 0.0163 | 0.0147 | 0.4930 | 0.5106 | 10 |
| MIMIC-IV | DP ε=4 | multi_draw_reconstruction | 0.5052 | 0.0052 | 0.0021 | 0.0171 | 0.0119 | 0.4985 | 0.5123 | 10 |
| MIMIC-IV | DP ε=8 | multi_draw_reconstruction | 0.5098 | 0.0098 | 0.0166 | 0.0214 | 0.0114 | 0.5031 | 0.5167 | 10 |

AUC bằng 0.5 và advantage bằng 0 tương ứng attacker ngẫu nhiên. Không ghi `MIA AUC ↓` đơn thuần vì AUC dưới 0.5 vẫn có thể phản ánh orientation đảo; cột `|AUC−0.5|` tránh cách đọc sai đó. `Signed MIA advantage` là TPR−FPR tại threshold chọn trên calibration split; dấu có thể đổi trên evaluation split, nên `Mean |advantage|` mới là độ lớn rủi ro dễ so sánh. Heart+ và MIMIC nhìn chung gần random trong release surface này. ECG DP ε=2 có AUC 0.6054 với CI không chứa 0.5 và phải giữ như cảnh báo empirical; ε không được chọn bằng attack result trên outer data.

### 6.3 Memorization distance

| Dataset | Variant | Exact match rate ↓ | Near duplicate ≤0.01 ↓ | DCR mean ↑ | NNDR mean ↑ | Train/holdout DCR ratio |
| --- | --- | --- | --- | --- | --- | --- |
| Leipzig ECG | CounterGAN non-DP | 0.0000 | 0.0000 | 0.0783 | 0.9454 | 0.5708 |
| Leipzig ECG | DP ε=16 | 0.0000 | 0.0000 | 0.0772 | 0.9582 | 0.4791 |
| Leipzig ECG | DP ε=2 | 0.0000 | 0.0000 | 0.0847 | 0.9558 | 0.5140 |
| Leipzig ECG | DP ε=4 | 0.0000 | 0.0000 | 0.0844 | 0.9534 | 0.4937 |
| Leipzig ECG | DP ε=8 | 0.0000 | 0.0000 | 0.0775 | 0.9561 | 0.4453 |
| Heart+ | CounterGAN non-DP | 0.0000 | 0.0010 | 0.1289 | 0.8568 | 0.9813 |
| Heart+ | DP ε=16 | 0.0000 | 0.0353 | 0.1007 | 0.8076 | 0.9995 |
| Heart+ | DP ε=2 | 0.0000 | 0.0241 | 0.1020 | 0.8171 | 0.9847 |
| Heart+ | DP ε=4 | 0.0000 | 0.0404 | 0.1005 | 0.8050 | 1.0148 |
| Heart+ | DP ε=8 | 0.0000 | 0.0473 | 0.0925 | 0.8006 | 0.9935 |
| MIMIC-IV | CounterGAN non-DP | 0.0000 | 0.0000 | 0.1492 | 0.9473 | 0.8877 |
| MIMIC-IV | DP ε=16 | 0.0000 | 0.0000 | 0.1370 | 0.9455 | 0.8765 |
| MIMIC-IV | DP ε=2 | 0.0000 | 0.0000 | 0.1511 | 0.9524 | 0.8921 |
| MIMIC-IV | DP ε=4 | 0.0000 | 0.0000 | 0.1554 | 0.9538 | 0.8963 |
| MIMIC-IV | DP ε=8 | 0.0000 | 0.0000 | 0.1439 | 0.9498 | 0.8839 |

Exact match bằng 0 chỉ loại trừ sao chép chính xác; nó không chứng minh không có leakage. DCR lớn và NNDR gần 1 thường tốt hơn, nhưng phải so với holdout: ratio gần 1 nghĩa là generated samples không gần train hơn rõ rệt so với holdout. Heart+ có near-duplicate rate khác 0 ở các checkpoint DP, vì vậy không được tóm tắt bảng này bằng câu `không có memorization`.

### 6.4 Attribute Inversion / Conditional Attribute Inference

| Dataset | Variant | Sensitive attribute | Attack metric | Attack score | Trivial score | Attacker advantage ↓ | Diễn giải |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Leipzig ECG | CounterGAN non-DP | total_power | normalized MAE | 0.0633 | 0.1022 | 0.0389 | Cảnh báo |
| Leipzig ECG | DP ε=16 | total_power | normalized MAE | 0.0657 | 0.1022 | 0.0365 | Cảnh báo |
| Leipzig ECG | DP ε=2 | total_power | normalized MAE | 0.0649 | 0.1022 | 0.0373 | Cảnh báo |
| Leipzig ECG | DP ε=4 | total_power | normalized MAE | 0.0625 | 0.1022 | 0.0397 | Cảnh báo |
| Leipzig ECG | DP ε=8 | total_power | normalized MAE | 0.0732 | 0.1022 | 0.0290 | Cảnh báo |
| Heart+ | CounterGAN non-DP | Diabetic | balanced accuracy | 0.3827 | 0.2500 | 0.1327 | Cảnh báo |
| Heart+ | DP ε=16 | Diabetic | balanced accuracy | 0.3873 | 0.2500 | 0.1373 | Cảnh báo |
| Heart+ | DP ε=2 | Diabetic | balanced accuracy | 0.3964 | 0.2500 | 0.1464 | Cảnh báo |
| Heart+ | DP ε=4 | Diabetic | balanced accuracy | 0.4182 | 0.2500 | 0.1682 | Cảnh báo |
| Heart+ | DP ε=8 | Diabetic | balanced accuracy | 0.4003 | 0.2500 | 0.1503 | Cảnh báo |
| MIMIC-IV | CounterGAN non-DP | Creatinine_Mean | normalized MAE | 0.0439 | 0.1093 | 0.0654 | Cảnh báo |
| MIMIC-IV | DP ε=16 | Creatinine_Mean | normalized MAE | 0.0588 | 0.1093 | 0.0505 | Cảnh báo |
| MIMIC-IV | DP ε=2 | Creatinine_Mean | normalized MAE | 0.0484 | 0.1093 | 0.0609 | Cảnh báo |
| MIMIC-IV | DP ε=4 | Creatinine_Mean | normalized MAE | 0.0564 | 0.1093 | 0.0529 | Cảnh báo |
| MIMIC-IV | DP ε=8 | Creatinine_Mean | normalized MAE | 0.0576 | 0.1093 | 0.0517 | Cảnh báo |

`Attacker advantage` đưa hai loại metric về cùng chiều, càng thấp càng tốt. Với ECG/MIMIC, advantage = trivial normalized MAE − attack normalized MAE vì error thấp hơn nghĩa là attacker mạnh hơn. Với Heart+, advantage = attack balanced accuracy − trivial balanced accuracy vì accuracy cao hơn nghĩa là attacker mạnh hơn. Do đó Heart+ cũng là cảnh báo; cách diễn giải cũ coi 0.4182 là MAE và kết luận attacker kém trivial là sai loại metric. Attribute inversion đo rủi ro từ conditional query/released CFE, không phải training-membership guarantee của DP-SGD và không nên gộp với MIA.

## 7. Explanation-Linkage Attack theo hai hướng

Attack chạy trên toàn bộ valid CFE release của cohort, còn qualitative grid ở Mục 8 dùng 10 factual đại diện để minh họa. Numeric quasi-identifiers được chia quartile bằng development training data; untouched outer test đóng vai auxiliary population. Đây là mô phỏng re-identification/linkage trong điều kiện có bảng phụ, không phải khẳng định đã nhận diện một bệnh nhân thật vì dữ liệu không chứa direct identifier.

| Dataset | Quasi-identifiers + predicted label | Sensitive attribute | Auxiliary population |
| --- | --- | --- | --- |
| Leipzig ECG | age, gender, pre_rr, post_rr, predicted_label | total_power | untouched outer test |
| Heart+ | Sex, AgeCategory, Race, BMI, Smoking, predicted_label | Diabetic | untouched outer test |
| MIMIC-IV | anchor_age, gender, race_group, HeartRate_Mean, SysBP_Mean, predicted_label | Creatinine_Mean | untouched outer test |

`1-anonymity ↓` là tỷ lệ released CFE đứng một mình trong equivalence class của release. `1-diversity ↓` là tỷ lệ CFE thuộc class chỉ có một sensitive value. `One-MAP ↓` là tỷ lệ signature nối đúng một auxiliary row; `Small-map 1–4 ↓` là tỷ lệ thu hẹp còn 1–4 rows; `No auxiliary match ↑` là tỷ lệ không có match. `Median matches ↑` càng lớn thường càng khó cô lập, nhưng chỉ tính trên các release đã match nên phải đọc cùng No-match và số Released CFEs.

### 7.1 Leipzig ECG

| Method | Direction | Released CFEs | 1-anonymity ↓ | 1-diversity ↓ | One-MAP ↓ | Small-map 1–4 ↓ | No auxiliary match ↑ | Median matches ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Uniform Random | Bệnh → không bệnh | 1,000 | 0.0080 | 0.0400 | 0.1330 | 0.3550 | 0.0830 | 7.0000 |
| Uniform Random | Không bệnh → bệnh | 1,000 | 0.0070 | 0.0200 | 0.0830 | 0.1130 | 0.1830 | 38.0000 |
| Genetic CFE | Bệnh → không bệnh | 1,000 | 0.0100 | 0.0310 | 0.1470 | 0.3460 | 0.0710 | 7.0000 |
| Genetic CFE | Không bệnh → bệnh | 1,000 | 0.0040 | 0.0350 | 0.0790 | 0.1310 | 0.1600 | 38.0000 |
| DiCE-style gradient | Bệnh → không bệnh | 104 | 0.0865 | 0.3654 | 0.0288 | 0.2115 | 0.1058 | 26.0000 |
| DiCE-style gradient | Không bệnh → bệnh | 100 | 0.1300 | 0.5300 | 0.2400 | 0.2600 | 0.1100 | 28.0000 |
| Wachter-style | Bệnh → không bệnh | 103 | 0.1068 | 0.3786 | 0.0291 | 0.2039 | 0.1068 | 33.0000 |
| Wachter-style | Không bệnh → bệnh | 100 | 0.1400 | 0.5000 | 0.2500 | 0.2600 | 0.1100 | 28.0000 |
| CounterGAN one-shot | Bệnh → không bệnh | 944 | 0.0074 | 0.0339 | 0.0307 | 0.2786 | 0.0212 | 11.0000 |
| CounterGAN one-shot | Không bệnh → bệnh | 1,000 | 0.0030 | 0.2500 | 0.0050 | 0.0370 | 0.0060 | 81.0000 |
| CounterGAN iterative | Bệnh → không bệnh | 1,000 | 0.0110 | 0.0360 | 0.0400 | 0.2880 | 0.0240 | 11.0000 |
| CounterGAN iterative | Không bệnh → bệnh | 1,000 | 0.0070 | 0.2870 | 0.0140 | 0.0490 | 0.0120 | 89.0000 |
| CounterGAN-SD non-DP | Bệnh → không bệnh | 1,000 | 0.0060 | 0.0350 | 0.1100 | 0.3460 | 0.0960 | 8.0000 |
| CounterGAN-SD non-DP | Không bệnh → bệnh | 1,000 | 0.0080 | 0.1740 | 0.0430 | 0.0730 | 0.1440 | 88.0000 |
| DP-CounterGAN-SD ε=16 | Bệnh → không bệnh | 1,000 | 0.0070 | 0.0750 | 0.0960 | 0.3150 | 0.0820 | 8.0000 |
| DP-CounterGAN-SD ε=16 | Không bệnh → bệnh | 1,000 | 0.0050 | 0.2640 | 0.0500 | 0.0840 | 0.1850 | 43.0000 |
| DP-CounterGAN-SD ε=8 | Bệnh → không bệnh | 1,000 | 0.0030 | 0.0470 | 0.0880 | 0.3350 | 0.0810 | 8.0000 |
| DP-CounterGAN-SD ε=8 | Không bệnh → bệnh | 1,000 | 0.0070 | 0.2950 | 0.0480 | 0.0710 | 0.2000 | 60.5000 |
| DP-CounterGAN-SD ε=4 (primary) | Bệnh → không bệnh | 1,000 | 0.0050 | 0.0690 | 0.0860 | 0.3130 | 0.0890 | 8.0000 |
| DP-CounterGAN-SD ε=4 (primary) | Không bệnh → bệnh | 1,000 | 0.0080 | 0.1860 | 0.0520 | 0.0770 | 0.1410 | 81.0000 |
| DP-CounterGAN-SD ε=2 | Bệnh → không bệnh | 1,000 | 0.0070 | 0.0660 | 0.0970 | 0.3120 | 0.0870 | 8.0000 |
| DP-CounterGAN-SD ε=2 | Không bệnh → bệnh | 1,000 | 0.0100 | 0.1130 | 0.0440 | 0.0740 | 0.1610 | 63.0000 |

**Nhận xét ECG linkage.** Với DP ε=4, One-MAP là 0.0860 ở bệnh → không bệnh và 0.0520 ở chiều ngược lại; Small-map tương ứng 0.3130 và 0.0770. Như vậy rủi ro phụ thuộc mạnh vào hướng và tiêu chí: chiều đầu có nhiều small-map hơn dù One-MAP không phải lớn nhất bảng. Không có xu hướng đơn điệu theo ε.

### 7.2 Heart+

| Method | Direction | Released CFEs | 1-anonymity ↓ | 1-diversity ↓ | One-MAP ↓ | Small-map 1–4 ↓ | No auxiliary match ↑ | Median matches ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Uniform Random | Bệnh → không bệnh | 676 | 0.0533 | 0.4822 | 0.0000 | 0.0178 | 0.0000 | 183.0000 |
| Uniform Random | Không bệnh → bệnh | 254 | 0.1299 | 0.7598 | 0.0079 | 0.0591 | 0.0394 | 84.0000 |
| Genetic CFE | Bệnh → không bệnh | 534 | 0.0581 | 0.6049 | 0.0000 | 0.0150 | 0.0000 | 184.0000 |
| Genetic CFE | Không bệnh → bệnh | 261 | 0.1724 | 0.8391 | 0.0038 | 0.0651 | 0.0383 | 73.0000 |
| DiCE-style gradient | Bệnh → không bệnh | 328 | 0.0701 | 0.6433 | 0.0000 | 0.0152 | 0.0000 | 162.0000 |
| DiCE-style gradient | Không bệnh → bệnh | 187 | 0.1551 | 0.7807 | 0.0000 | 0.0481 | 0.0160 | 71.0000 |
| Wachter-style | Bệnh → không bệnh | 318 | 0.0692 | 0.6384 | 0.0000 | 0.0157 | 0.0000 | 163.0000 |
| Wachter-style | Không bệnh → bệnh | 178 | 0.1404 | 0.7697 | 0.0000 | 0.0506 | 0.0169 | 71.0000 |
| CounterGAN one-shot | Bệnh → không bệnh | 137 | 0.0657 | 0.4234 | 0.0000 | 0.0000 | 0.0000 | 268.0000 |
| CounterGAN one-shot | Không bệnh → bệnh | 274 | 0.1277 | 0.6277 | 0.0182 | 0.0912 | 0.0109 | 71.0000 |
| CounterGAN iterative | Bệnh → không bệnh | 894 | 0.0414 | 0.3009 | 0.0000 | 0.0045 | 0.0000 | 259.0000 |
| CounterGAN iterative | Không bệnh → bệnh | 627 | 0.0574 | 0.5917 | 0.0223 | 0.1005 | 0.0351 | 59.0000 |
| CounterGAN-SD non-DP | Bệnh → không bệnh | 914 | 0.0284 | 0.5635 | 0.0000 | 0.0142 | 0.0011 | 241.0000 |
| CounterGAN-SD non-DP | Không bệnh → bệnh | 661 | 0.0545 | 0.7141 | 0.0454 | 0.1256 | 0.0303 | 46.0000 |
| DP-CounterGAN-SD ε=16 | Bệnh → không bệnh | 908 | 0.0253 | 0.5617 | 0.0000 | 0.0110 | 0.0011 | 241.0000 |
| DP-CounterGAN-SD ε=16 | Không bệnh → bệnh | 551 | 0.0835 | 0.7187 | 0.0254 | 0.0744 | 0.0254 | 46.0000 |
| DP-CounterGAN-SD ε=8 | Bệnh → không bệnh | 912 | 0.0296 | 0.5066 | 0.0000 | 0.0110 | 0.0011 | 220.0000 |
| DP-CounterGAN-SD ε=8 | Không bệnh → bệnh | 586 | 0.0512 | 0.7355 | 0.0222 | 0.0717 | 0.0290 | 59.0000 |
| DP-CounterGAN-SD ε=4 (primary) | Bệnh → không bệnh | 907 | 0.0276 | 0.5083 | 0.0000 | 0.0132 | 0.0000 | 241.0000 |
| DP-CounterGAN-SD ε=4 (primary) | Không bệnh → bệnh | 546 | 0.0769 | 0.7546 | 0.0238 | 0.0733 | 0.0275 | 48.0000 |
| DP-CounterGAN-SD ε=2 | Bệnh → không bệnh | 904 | 0.0310 | 0.4989 | 0.0000 | 0.0144 | 0.0011 | 220.0000 |
| DP-CounterGAN-SD ε=2 | Không bệnh → bệnh | 606 | 0.0594 | 0.7162 | 0.0297 | 0.0792 | 0.0231 | 58.0000 |

**Nhận xét Heart+ linkage.** One-MAP gần 0 cho hướng bệnh → không bệnh ở hầu hết phương pháp, nhưng điều đó không đồng nghĩa release vô danh hoàn toàn: 1-diversity của DP ε=4 vẫn là 0.5083. Ở chiều không bệnh → bệnh, DP ε=4 có One-MAP 0.0238 và Small-map 0.0733. Auxiliary pool lớn tạo nhiều matches, nên kết quả không được so trị tuyệt đối với ECG/MIMIC như thể cùng độ khó.

### 7.3 MIMIC-IV

| Method | Direction | Released CFEs | 1-anonymity ↓ | 1-diversity ↓ | One-MAP ↓ | Small-map 1–4 ↓ | No auxiliary match ↑ | Median matches ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Uniform Random | Bệnh → không bệnh | 692 | 0.0925 | 0.1777 | 0.0910 | 0.3598 | 0.0766 | 5.0000 |
| Uniform Random | Không bệnh → bệnh | 990 | 0.0727 | 0.1505 | 0.2545 | 0.5646 | 0.2222 | 2.0000 |
| Genetic CFE | Bệnh → không bệnh | 896 | 0.0469 | 0.2344 | 0.0792 | 0.3594 | 0.0982 | 5.0000 |
| Genetic CFE | Không bệnh → bệnh | 990 | 0.0495 | 0.2111 | 0.2556 | 0.5545 | 0.2091 | 2.0000 |
| DiCE-style gradient | Bệnh → không bệnh | 79 | 0.6203 | 0.7342 | 0.0506 | 0.2658 | 0.1013 | 6.0000 |
| DiCE-style gradient | Không bệnh → bệnh | 90 | 0.6556 | 0.7222 | 0.1667 | 0.5222 | 0.1667 | 4.0000 |
| Wachter-style | Bệnh → không bệnh | 79 | 0.6203 | 0.7342 | 0.0506 | 0.2658 | 0.1013 | 6.0000 |
| Wachter-style | Không bệnh → bệnh | 89 | 0.6742 | 0.7191 | 0.1685 | 0.5169 | 0.1685 | 4.0000 |
| CounterGAN one-shot | Bệnh → không bệnh | 965 | 0.0093 | 0.3285 | 0.0611 | 0.2166 | 0.0694 | 8.0000 |
| CounterGAN one-shot | Không bệnh → bệnh | 990 | 0.0212 | 0.3859 | 0.2737 | 0.6071 | 0.1596 | 2.0000 |
| CounterGAN iterative | Bệnh → không bệnh | 1,000 | 0.0170 | 0.3020 | 0.0740 | 0.2270 | 0.0660 | 8.0000 |
| CounterGAN iterative | Không bệnh → bệnh | 990 | 0.0212 | 0.3616 | 0.2818 | 0.6111 | 0.1424 | 2.0000 |
| CounterGAN-SD non-DP | Bệnh → không bệnh | 1,000 | 0.0400 | 0.1830 | 0.0950 | 0.2770 | 0.0900 | 7.0000 |
| CounterGAN-SD non-DP | Không bệnh → bệnh | 990 | 0.0424 | 0.2939 | 0.2596 | 0.5293 | 0.2172 | 2.0000 |
| DP-CounterGAN-SD ε=16 | Bệnh → không bệnh | 1,000 | 0.0420 | 0.1460 | 0.0830 | 0.3260 | 0.0850 | 6.0000 |
| DP-CounterGAN-SD ε=16 | Không bệnh → bệnh | 990 | 0.0535 | 0.2061 | 0.2263 | 0.5192 | 0.1737 | 3.0000 |
| DP-CounterGAN-SD ε=8 | Bệnh → không bệnh | 1,000 | 0.0460 | 0.1600 | 0.0970 | 0.3200 | 0.0710 | 7.0000 |
| DP-CounterGAN-SD ε=8 | Không bệnh → bệnh | 990 | 0.0556 | 0.1646 | 0.2343 | 0.5121 | 0.1838 | 3.0000 |
| DP-CounterGAN-SD ε=4 (primary) | Bệnh → không bệnh | 1,000 | 0.0550 | 0.1470 | 0.0880 | 0.3030 | 0.0860 | 7.0000 |
| DP-CounterGAN-SD ε=4 (primary) | Không bệnh → bệnh | 990 | 0.0525 | 0.1646 | 0.2273 | 0.5010 | 0.1960 | 3.0000 |
| DP-CounterGAN-SD ε=2 | Bệnh → không bệnh | 1,000 | 0.0440 | 0.1450 | 0.0920 | 0.2870 | 0.0750 | 7.0000 |
| DP-CounterGAN-SD ε=2 | Không bệnh → bệnh | 990 | 0.0657 | 0.1687 | 0.2455 | 0.5586 | 0.1545 | 3.0000 |

**Nhận xét MIMIC linkage.** Chiều không bệnh → bệnh là cảnh báo rõ nhất: DP ε=4 có One-MAP 0.2273 và Small-map 0.5010, so với 0.0880 và 0.3030 ở chiều bệnh → không bệnh. Genetic/iterative CounterGAN cũng có cùng xu hướng, cho thấy phần lớn rủi ro đến từ release signature và auxiliary population của hướng này, không chỉ từ DP checkpoint. Không được che kết quả bằng equal-direction macro.

## 8. Qualitative counterfactual grid

| Dataset | Representative factuals | Bệnh → không bệnh | Không bệnh → bệnh | Artifact |
| --- | --- | --- | --- | --- |
| Leipzig ECG | 10 | 5 | 5 | 10_2_qualitative_grid_compact.csv; 10_1_qualitative_grid_long.csv |
| Heart+ | 10 | 5 | 5 | 10_2_qualitative_grid_compact.csv; 10_1_qualitative_grid_long.csv |
| MIMIC-IV | 10 | 5 | 5 | 10_2_qualitative_grid_compact.csv; 10_1_qualitative_grid_long.csv |

Mỗi hướng chọn năm factual theo các quantile 0.1–0.9 của predicted probability, thay vì chọn thủ công các case đẹp. Mỗi ô hiển thị factual gốc hoặc CFE hợp lệ xếp hạng đầu, xác suất và các raw feature đã đổi sau inverse transform. Grid chứa Uniform, Genetic, DiCE-style, Wachter-style, CounterGAN one-shot/iterative và DP-CounterGAN ở các ε. Ô `NO VALID CFE` được giữ nguyên, không xóa case khó. Đây là kiểm tra định tính về tính dễ hiểu và hợp lý lâm sàng; 10 case không thay thế các bảng định lượng trên 200 factual.

## 9. CKKS/HE population benchmark — final paper CPU đã audit

Phần này đọc trực tiếp ba artifact `Q1_V55_HE_POPULATION_PAPER` có `accepted=true` và `paper_numbers=true`, dùng real TenSEAL/CKKS ciphertext. Mỗi population có 15 timing runs = 3 candidate-cohort seeds × 5 repetitions; frontier báo median [Q1, Q3]. HE chạy CPU (4 logical/2 physical cores trên Kaggle), GPU không được bật. DP chỉ xác định provenance của candidate và không thay đổi frozen MLP/CKKS graph, vì vậy timing được đo theo population một lần, còn correctness được tách theo hướng và nguồn candidate.

| Dataset | HE paper kernel | Trạng thái lúc cập nhật | Final paper numbers |
| --- | --- | --- | --- |
| Leipzig ECG | [leipzig-ecg-v5-5-he-population-paper](https://www.kaggle.com/code/lamhuy8904/leipzig-ecg-v5-5-he-population-paper) | COMPLETE + accepted | Có |
| Heart+ | [heart-v5-5-he-population-paper](https://www.kaggle.com/code/huylmhuhu/heart-v5-5-he-population-paper) | COMPLETE + accepted | Có |
| MIMIC-IV | [mimic-iv-v5-5-he-population-paper](https://www.kaggle.com/code/buiquocviet/mimic-iv-v5-5-he-population-paper) | COMPLETE + accepted | Có |

### 9.1 Cấu hình mật mã và đóng gói

| Dataset | Population grid | Selected P | Profile | N | Slots | Modulus chain | Total bits | TC128 max | Scale bits | Input CT | Output CT | Paper accepted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Leipzig ECG | 16/32/64/128/256/512 | 64 | poly_n16384_scale45 | 16,384 | 8,192 | [55, 45, 45, 45, 45, 45, 55] | 335 | 438 | 45 | 20 | 1 | Có |
| Heart+ | 32/64/128/256/512/1024 | 128 | square_n8192_scale29 | 8,192 | 4,096 | [36, 29, 29, 29, 29, 29, 36] | 217 | 218 | 29 | 23 | 1 | Có |
| MIMIC-IV | 32/64/128/256/512 | 64 | poly_n16384_scale40 | 16,384 | 8,192 | [50, 40, 40, 40, 40, 40, 50] | 300 | 438 | 40 | 40 | 1 | Có |

`N` là polynomial modulus degree; CKKS có N/2 complex slots. Mỗi feature dùng một ciphertext và các slot của ciphertext chứa nhiều candidate trong cùng population (SIMD), nên `Input CT` bằng số processed inputs chứ không bằng P. Modulus chain phải đủ level cho hai activation đa thức và vẫn không vượt giới hạn TC128. Heart+ chain `[36,29,29,29,29,29,36]` có tổng **217 bits**, không phải 188; vẫn nằm dưới TC128 max 218. Client giữ secret key; server nhận public context + relinearization keys, không nhận secret/Galois keys. Threat model honest-but-curious; malicious-server integrity ngoài phạm vi.

### 9.2 Frontier population: latency, throughput, communication, RAM và correctness

| Dataset | P | Latency/round median [Q1,Q3] s ↓ | Time/candidate median [Q1,Q3] s ↓ | Throughput median [Q1,Q3] candidate/s ↑ | Communication median [Q1,Q3] MB ↓ | Context once MB ↓ | Peak RSS median [Q1,Q3] MB ↓ | Max logit error median [Q1,Q3] ↓ | Label agreement median [Q1,Q3] ↑ | Gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Leipzig ECG | 16 | 82.4627 [82.3040, 82.8926] | 5.1539 [5.1440, 5.1808] | 0.1940 [0.1930, 0.1944] | 27.2077 [27.2074, 27.2080] | 11.1638 | 1345.7203 [1321.3901, 1349.0586] | 0.000011 [0.000006, 0.000012] | 1.0000 [1.0000, 1.0000] | Có |
| Leipzig ECG | 32 | 82.3370 [81.9436, 82.8112] | 2.5730 [2.5607, 2.5879] | 0.3886 [0.3864, 0.3905] | 27.2074 [27.2072, 27.2076] | 11.1638 | 1345.7244 [1323.6982, 1349.0627] | 0.000012 [0.000008, 0.000015] | 1.0000 [1.0000, 1.0000] | Có |
| Leipzig ECG | 64 | 83.0892 [82.4413, 83.5877] | 1.2983 [1.2881, 1.3061] | 0.7703 [0.7657, 0.7763] | 27.2073 [27.2071, 27.2079] | 11.1638 | 1345.7244 [1325.2219, 1349.0627] | 0.000013 [0.000012, 0.000015] | 1.0000 [1.0000, 1.0000] | Có |
| Leipzig ECG | 128 | 82.7660 [82.2277, 83.0388] | 0.6466 [0.6424, 0.6487] | 1.5465 [1.5414, 1.5567] | 27.2071 [27.2069, 27.2078] | 11.1638 | 1345.7244 [1326.7354, 1349.0627] | 0.000015 [0.000013, 0.000016] | 1.0000 [1.0000, 1.0000] | Có |
| Leipzig ECG | 256 | 83.0222 [82.7768, 83.2416] | 0.3243 [0.3233, 0.3252] | 3.0835 [3.0754, 3.0927] | 27.2073 [27.2069, 27.2079] | 11.1638 | 1345.7244 [1327.3661, 1349.0668] | 0.000016 [0.000016, 0.000017] | 1.0000 [1.0000, 1.0000] | Có |
| Leipzig ECG | 512 | 83.3414 [82.9433, 83.5304] | 0.1628 [0.1620, 0.1631] | 6.1434 [6.1295, 6.1729] | 27.2075 [27.2073, 27.2078] | 11.1638 | 1345.7244 [1327.7430, 1349.0708] | 0.000017 [0.000016, 0.000017] | 1.0000 [1.0000, 1.0000] | Có |
| Heart+ | 32 | 28.1682 [27.9337, 28.8388] | 0.8803 [0.8729, 0.9012] | 1.1360 [1.1096, 1.1456] | 10.7910 [10.7906, 10.7915] | 3.8987 | 794.1980 [787.8779, 795.5292] | 0.005704 [0.002990, 0.006723] | 1.0000 [1.0000, 1.0000] | Có |
| Heart+ | 64 | 28.4017 [28.1358, 29.0047] | 0.4438 [0.4396, 0.4532] | 2.2534 [2.2065, 2.2747] | 10.7916 [10.7910, 10.7919] | 3.8987 | 794.1980 [788.6930, 795.5333] | 0.005971 [0.004057, 0.006770] | 1.0000 [1.0000, 1.0000] | Có |
| Heart+ | 128 | 28.0602 [27.9902, 28.7541] | 0.2192 [0.2187, 0.2246] | 4.5616 [4.4515, 4.5730] | 10.7910 [10.7907, 10.7915] | 3.8987 | 794.1980 [789.2828, 795.5333] | 0.006974 [0.005713, 0.007815] | 1.0000 [1.0000, 1.0000] | Có |
| Heart+ | 256 | 28.0675 [27.9637, 28.1552] | 0.1096 [0.1092, 0.1100] | 9.1209 [9.0925, 9.1548] | 10.7911 [10.7907, 10.7914] | 3.8987 | 794.1980 [790.0242, 795.5333] | 0.008926 [0.007928, 0.014803] | 1.0000 [1.0000, 1.0000] | Có |
| Heart+ | 512 | 28.0663 [27.8874, 28.3369] | 0.0548 [0.0545, 0.0553] | 18.2425 [18.0683, 18.3596] | 10.7913 [10.7912, 10.7915] | 3.8987 | 794.1980 [790.3273, 795.5333] | 0.009099 [0.009081, 0.014855] | 1.0000 [1.0000, 1.0000] | Có |
| Heart+ | 1,024 | 28.1336 [27.9197, 28.4772] | 0.0275 [0.0273, 0.0278] | 36.3978 [35.9586, 36.6766] | 10.7914 [10.7910, 10.7919] | 3.8987 | 794.1980 [790.9868, 799.7153] | 0.011994 [0.009096, 0.014812] | 1.0000 [1.0000, 1.0000] | Có |
| MIMIC-IV | 32 | 80.5023 [80.3600, 80.7735] | 2.5157 [2.5112, 2.5242] | 0.3975 [0.3962, 0.3982] | 49.2313 [49.2307, 49.2351] | 10.2319 | 1298.2559 [1286.9550, 1298.4443] | 0.000499 [0.000394, 0.002083] | 1.0000 [1.0000, 1.0000] | Có |
| MIMIC-IV | 64 | 80.6351 [80.3825, 82.6200] | 1.2599 [1.2560, 1.2909] | 0.7937 [0.7746, 0.7962] | 49.2318 [49.2298, 49.2334] | 10.2319 | 1299.3290 [1289.8038, 1301.8112] | 0.001909 [0.001280, 0.002083] | 1.0000 [1.0000, 1.0000] | Có |
| MIMIC-IV | 128 | 80.5375 [80.3099, 80.8458] | 0.6292 [0.6274, 0.6316] | 1.5893 [1.5833, 1.5938] | 49.2315 [49.2307, 49.2333] | 10.2319 | 1301.8112 [1293.3693, 1303.5151] | 0.002084 [0.001909, 0.002253] | 1.0000 [1.0000, 1.0000] | Có |
| MIMIC-IV | 256 | 80.5742 [80.3172, 81.7130] | 0.3147 [0.3137, 0.3192] | 3.1772 [3.1330, 3.1874] | 49.2303 [49.2297, 49.2330] | 10.2319 | 1301.8194 [1293.5782, 1303.5192] | 0.003389 [0.002084, 0.003881] | 1.0000 [1.0000, 1.0000] | Có |
| MIMIC-IV | 512 | 80.5639 [80.1884, 80.9803] | 0.1574 [0.1566, 0.1582] | 6.3552 [6.3225, 6.3850] | 49.2294 [49.2288, 49.2317] | 10.2319 | 1301.8194 [1293.9059, 1303.5233] | 0.003636 [0.002307, 0.003881] | 1.0000 [1.0000, 1.0000] | Có |

Mỗi ô `median [Q1,Q3]` dùng 15 timing runs; Q1 và Q3 là phân vị 25% và 75%, không phải confidence interval. Latency/round và communication gần như phẳng theo P vì cùng số ciphertext feature chứa nhiều slot hơn, trong khi time/candidate giảm gần tỷ lệ nghịch và throughput tăng theo P. Đây là lợi ích SIMD thực sự; không nên thay bảng bằng benchmark batch=1. `Context once` là public evaluation context truyền một lần và không được cộng lại cho mỗi round. `Communication/round` gồm ciphertext input upload + logit ciphertext download; không gồm model weights/public context. Peak RSS là absolute process RSS; không diễn giải RSS delta rất nhỏ như mức RAM thực của HE. Tất cả population đạt agreement gate, nhưng P vận hành vẫn được chọn từ trade-off inner search chứ không phải lấy P lớn nhất chỉ vì throughput cao.

### 9.3 Breakdown tại selected population

| Dataset | P | Encrypt (s) | Upload serialize (s) | Server deserialize (s) | Server HE inference (s) | Download serialize (s) | Decrypt (s) | Total (s) | Upload MB | Download MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Leipzig ECG | 64 | 0.4959 | 0.0856 | 0.0696 | 82.4327 | 0.0010 | 0.0021 | 83.0892 | 26.9604 | 0.2469 |
| Heart+ | 128 | 0.2962 | 0.2041 | 0.0637 | 27.4963 | 0.0019 | 0.0012 | 28.0602 | 10.7024 | 0.0886 |
| MIMIC-IV | 64 | 0.9516 | 0.6995 | 0.2113 | 78.7849 | 0.0011 | 0.0023 | 80.6351 | 48.9961 | 0.2357 |

Server HE inference chiếm gần toàn bộ total latency ở cả ba bộ; encryption, serialization và decryption nhỏ hơn nhiều. Vì vậy hướng tối ưu hệ thống nên tập trung vào polynomial evaluation, modulus/scale profile và packing, không nên tuyên bố cải thiện lớn chỉ từ việc tối ưu bước decrypt. MIMIC upload lớn nhất do có 40 ciphertext feature; Heart+ nhanh hơn nhờ square profile N=8192, nhưng chain 217 bits nằm sát giới hạn TC128 nên không còn nhiều dư địa tăng depth.

### 9.4 Agreement theo hướng và nguồn candidate

| Dataset | Direction | Candidate source | Candidates | Agreement ↑ | Mean logit error ↓ | Max logit error ↓ | Min absolute plaintext logit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Leipzig ECG | Bệnh → không bệnh | Baseline | 129 | 1.0000 | 4.0268e-06 | 5.8594e-06 | 0.1646 |
| Leipzig ECG | Bệnh → không bệnh | DP-CounterGAN eps=16 | 129 | 1.0000 | 3.6362e-06 | 5.5996e-06 | 0.1334 |
| Leipzig ECG | Bệnh → không bệnh | DP-CounterGAN eps=2 | 129 | 1.0000 | 3.3779e-06 | 5.6205e-06 | 0.1063 |
| Leipzig ECG | Bệnh → không bệnh | DP-CounterGAN eps=4 | 129 | 1.0000 | 3.8370e-06 | 5.9025e-06 | 0.1223 |
| Leipzig ECG | Bệnh → không bệnh | DP-CounterGAN eps=8 | 129 | 1.0000 | 3.6977e-06 | 5.8501e-06 | 0.1274 |
| Leipzig ECG | Bệnh → không bệnh | Non-DP CounterGAN | 129 | 1.0000 | 4.0951e-06 | 5.9790e-06 | 0.1235 |
| Leipzig ECG | Không bệnh → bệnh | Baseline | 129 | 1.0000 | 6.5726e-06 | 1.1407e-05 | 0.2539 |
| Leipzig ECG | Không bệnh → bệnh | DP-CounterGAN eps=16 | 129 | 1.0000 | 6.9367e-06 | 1.6712e-05 | 0.1380 |
| Leipzig ECG | Không bệnh → bệnh | DP-CounterGAN eps=2 | 126 | 1.0000 | 6.8515e-06 | 1.7019e-05 | 0.1034 |
| Leipzig ECG | Không bệnh → bệnh | DP-CounterGAN eps=4 | 126 | 1.0000 | 6.5204e-06 | 1.6324e-05 | 0.1469 |
| Leipzig ECG | Không bệnh → bệnh | DP-CounterGAN eps=8 | 126 | 1.0000 | 6.2595e-06 | 1.6465e-05 | 0.1263 |
| Leipzig ECG | Không bệnh → bệnh | Non-DP CounterGAN | 126 | 1.0000 | 6.9949e-06 | 1.6097e-05 | 0.3254 |
| Heart+ | Bệnh → không bệnh | Baseline | 258 | 1.0000 | 0.0013 | 0.0089 | 0.1001 |
| Heart+ | Bệnh → không bệnh | DP-CounterGAN eps=16 | 258 | 1.0000 | 0.0011 | 0.0097 | 0.1014 |
| Heart+ | Bệnh → không bệnh | DP-CounterGAN eps=2 | 258 | 1.0000 | 0.0010 | 0.0070 | 0.1007 |
| Heart+ | Bệnh → không bệnh | DP-CounterGAN eps=4 | 258 | 1.0000 | 0.0012 | 0.0112 | 0.1012 |
| Heart+ | Bệnh → không bệnh | DP-CounterGAN eps=8 | 255 | 1.0000 | 0.0012 | 0.0069 | 0.1004 |
| Heart+ | Bệnh → không bệnh | Non-DP CounterGAN | 255 | 1.0000 | 0.0023 | 0.0149 | 0.1002 |
| Heart+ | Không bệnh → bệnh | Baseline | 255 | 1.0000 | 0.0015 | 0.0068 | 0.1002 |
| Heart+ | Không bệnh → bệnh | DP-CounterGAN eps=16 | 255 | 1.0000 | 0.0013 | 0.0043 | 0.1005 |
| Heart+ | Không bệnh → bệnh | DP-CounterGAN eps=2 | 255 | 1.0000 | 0.0014 | 0.0046 | 0.1002 |
| Heart+ | Không bệnh → bệnh | DP-CounterGAN eps=4 | 255 | 1.0000 | 0.0012 | 0.0043 | 0.1008 |
| Heart+ | Không bệnh → bệnh | DP-CounterGAN eps=8 | 255 | 1.0000 | 0.0013 | 0.0068 | 0.1015 |
| Heart+ | Không bệnh → bệnh | Non-DP CounterGAN | 255 | 1.0000 | 0.0016 | 0.0059 | 0.1030 |
| MIMIC-IV | Bệnh → không bệnh | Baseline | 129 | 1.0000 | 2.8998e-05 | 0.0002 | 0.1036 |
| MIMIC-IV | Bệnh → không bệnh | DP-CounterGAN eps=16 | 129 | 1.0000 | 5.9291e-05 | 0.0007 | 0.1259 |
| MIMIC-IV | Bệnh → không bệnh | DP-CounterGAN eps=2 | 129 | 1.0000 | 8.8075e-05 | 0.0007 | 0.1453 |
| MIMIC-IV | Bệnh → không bệnh | DP-CounterGAN eps=4 | 129 | 1.0000 | 8.2874e-05 | 0.0012 | 0.1060 |
| MIMIC-IV | Bệnh → không bệnh | DP-CounterGAN eps=8 | 129 | 1.0000 | 6.9368e-05 | 0.0006 | 0.1459 |
| MIMIC-IV | Bệnh → không bệnh | Non-DP CounterGAN | 129 | 1.0000 | 9.5223e-05 | 0.0003 | 0.1224 |
| MIMIC-IV | Không bệnh → bệnh | Baseline | 129 | 1.0000 | 0.0001 | 0.0005 | 0.1027 |
| MIMIC-IV | Không bệnh → bệnh | DP-CounterGAN eps=16 | 129 | 1.0000 | 0.0002 | 0.0034 | 0.2011 |
| MIMIC-IV | Không bệnh → bệnh | DP-CounterGAN eps=2 | 126 | 1.0000 | 0.0005 | 0.0036 | 0.1352 |
| MIMIC-IV | Không bệnh → bệnh | DP-CounterGAN eps=4 | 126 | 1.0000 | 0.0002 | 0.0012 | 0.1899 |
| MIMIC-IV | Không bệnh → bệnh | DP-CounterGAN eps=8 | 126 | 1.0000 | 0.0002 | 0.0039 | 0.1294 |
| MIMIC-IV | Không bệnh → bệnh | Non-DP CounterGAN | 126 | 1.0000 | 0.0004 | 0.0019 | 0.3566 |

Agreement 1.0 được báo đúng như đo, không phải đóng góp utility. Correctness dùng ba max-population candidate cohorts; mọi candidate row đều được giữ và tách theo direction/provenance. Gate chỉ yêu cầu ≥0.99; stress cohort dưới đây được chọn trước HE bằng |plaintext logit| nhỏ nhất nên kiểm tra riêng các điểm sát biên hợp lệ.

### 9.5 Boundary stress

| Dataset | Population | Min absolute plaintext logit | Mean error ↓ | Max error ↓ | Agreement ↑ |
| --- | --- | --- | --- | --- | --- |
| Leipzig ECG | 512 | 0.1003 | 5.2012e-06 | 6.2662e-06 | 1.0000 |
| Heart+ | 1,024 | 0.1000 | 0.0005 | 0.0034 | 1.0000 |
| MIMIC-IV | 512 | 0.1003 | 3.3185e-05 | 7.7308e-05 | 1.0000 |

Khoảng cách plaintext logit nhỏ nhất xấp xỉ 0.10 vì CFE protocol đã yêu cầu target margin 0.10. Max CKKS error vẫn nhỏ hơn khoảng cách tới biên trên stress cohort, giải thích vì sao agreement đạt 1.0. Kết quả này xác nhận numerical correctness cho cohort đã kiểm; nó không chứng minh mọi ciphertext input có thể có đều giữ nhãn.

### 9.6 Liên kết inner utility với chi phí HE theo hai hướng

| Dataset | P | Direction | Inner Yield@10 ↑ | Robust validity ↑ | Sparsity ↑ | Diversity ↑ | Mean rounds to K-or-cap ↓ | Mean candidates to K-or-cap ↓ | Composed HE time (s) ↓ | Composed communication MB ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Leipzig ECG | 64 | Bệnh → không bệnh | 1.0000 | 0.9417 | 0.7792 | 0.0759 | 1.2500 | 80.0000 | 103.8615 | 34.0092 |
| Leipzig ECG | 64 | Không bệnh → bệnh | 1.0000 | 0.9667 | 0.7208 | 0.0865 | 1.2500 | 80.0000 | 103.8615 | 34.0092 |
| Heart+ | 128 | Bệnh → không bệnh | 1.0000 | 0.9958 | 0.7389 | 0.0807 | 1.5000 | 192.0000 | 42.0903 | 16.1865 |
| Heart+ | 128 | Không bệnh → bệnh | 0.8750 | 1.0000 | 0.6444 | 0.0509 | 5.7500 | 736.0000 | 161.3462 | 62.0483 |
| MIMIC-IV | 64 | Bệnh → không bệnh | 1.0000 | 0.9625 | 0.6176 | 0.0581 | 1.0000 | 64.0000 | 80.6351 | 49.2318 |
| MIMIC-IV | 64 | Không bệnh → bệnh | 1.0000 | 0.9750 | 0.6265 | 0.0750 | 1.0000 | 64.0000 | 80.6351 | 49.2318 |

`Composed HE time` = median measured CKKS latency/round × frozen inner rounds-to-K-or-cap; communication được ghép tương tự. Heart+ không bệnh → bệnh cần trung bình 5.75 rounds nên chi phí ghép cao hơn nhiều chiều còn lại, nhất quán với Yield thấp hơn trong plaintext. Đây là phép ghép giữa utility log và HE timing, không phải full-cohort encrypted end-to-end run; cách viết đúng là estimated composed cost.

## 10. Bản V5.5 mới nhất khác bản đầu tiên như thế nào

Đối chiếu ở mục này là hai notebook thật trong folder `Heart-plus/`: `cfe-heart-plus-table12346.ipynb` (protocol v4.7) và `cfe-heart-plus-table5.ipynb`, không phải template `bo_sung_first.md`. Bản đầu tiên đã có nhiều điểm tốt: exact-feature-group-disjoint split, K=10 với slot thiếu là failure, ba GAN seeds, DiCE chính thức và CKKS handoff riêng. Vì V5.5 đổi classifier, factual cohort, hướng, budget và định nghĩa khoảng cách, **không được trừ trực tiếp các con số cũ–mới để tuyên bố cải thiện**.

| Thành phần | Bản đầu tiên | Bản V5.5 hiện tại | Ý nghĩa |
| --- | --- | --- | --- |
| Phạm vi dữ liệu | Chỉ Heart+ | Leipzig ECG, Heart+ và MIMIC-IV cùng protocol | V5.5 mạnh hơn về external dataset breadth |
| Split Heart+ | 80/20 exact-feature-group-disjoint | Exact-feature-group-disjoint, cùng source SHA-256 | Cả hai đều tránh exact duplicate leakage |
| Classifier Heart+ | 23→160→80→1, activation x+0.125x²; 5 classifier seeds; canonical seed 11 | 23→128→64→1, square; classifier seed 55 | Khác oracle nên raw CFE metrics không phải paired old-vs-new |
| Classifier result | Canonical Accuracy 0.8850, F1 0.3534, ROC-AUC 0.8372 | Accuracy 0.8771, F1 0.3490, ROC-AUC 0.8392 | Không có bằng chứng classifier V5.5 tốt hơn; hai bản gần nhau và cùng chịu imbalance |
| Training seeds | GAN seeds {11,22,33}; baseline seeds {11,22,33} | Một GAN seed 11 và một search seed 11 | Bản đầu mạnh hơn về between-training-run evidence |
| Factual cohort | 50 correctly classified disease factuals | 200/dataset: 100 mỗi hướng, chia near/mid/far boundary | V5.5 lớn và khó đa dạng hơn nhưng chỉ một training seed |
| Hướng đổi nhãn | Chỉ Bệnh → không bệnh | Bệnh → không bệnh và Không bệnh → bệnh; báo riêng + equal-direction macro | V5.5 phù hợp câu hỏi bidirectional hiện tại |
| Validity/K | Strict Validity@10; missing slot failure; Coverage và Full-10 | Strict Valid-CFE Yield@10; missing slot failure; Coverage và Full-10 | Nguyên tắc tốt của bản đầu được giữ, không phải điểm mới |
| Population/budget | P=200, tối đa 20 iterations; sensitivity P=50…500 | Heart+ B={128,256,512,1024}; P=128, tối đa 8 rounds; P khóa trước outer | V5.5 dùng declared candidate cap để nối trực tiếp với HE cost |
| Search | Generator + anchors + mutation + crossover; black-box; chọn dần CFE | Full counted archive + sparse/group prune + MMR; no donor/rescue/train-kNN | V5.5 kiểm soát budget, sparsity và diversity rõ hơn |
| GAN schedule | Target 5,600 optimizer steps, khoảng 10 epochs; loss weights ramp toàn lịch; không warm-up riêng | Heart+ 6,000 exact steps: 300 epochs ×20; warm-up 30, ramp 180; direction-balanced loss | V5.5 thêm warm-up và tối ưu hai hướng; bản đầu có 3 retrainings |
| Sparse/diverse | Sparse loss 0.25→1.25, diversity loss 0.05; snap threshold 0.05 | Feature/group pruning thresholds 0.03/0.06, MMR=4.0, sparse/diverse ablation non-DP + DP | V5.5 đưa sparsity vào cả training và selection trước benchmark |
| DP implementation | Generator+Critic DP-SGD, ε split 80/20; SECURE_MODE=False | Exact step-aware accounting, secure RNG=True; generator-only deployment, discriminator training-only | V5.5 mạnh hơn về release contract và RNG; DP vẫn conditional on public frozen components |
| Baseline | Official DiCE random/genetic trên 50 factuals và native Wachter 350 steps | Main benchmark giữ custom counted-budget baselines; extension chạy official DiCE random/genetic trên đủ 200 factuals/dataset | V5.5 mới mạnh hơn về bidirectional official DiCE breadth; native Wachter vẫn chỉ là sanity panel |
| Metric geometry | Proximity=L1 chưa scale; Diversity=L2; Plausibility=LOF inlier rate ↑ | Normalized Proximity/Diversity; tách 5-NN distance ↓, LOF inlier ↑ và target support ↑ | V5.5 dễ diễn giải hơn; giá trị cũ–mới không so trực tiếp |
| Robustness | Chưa có perturbation robustness/changed-set stability | Robust Yield@10, Robust Full-10 và Changed-set Dice | V5.5 bổ sung độ bền giải thích |
| Privacy attacks | MIA, exact match, DCR, NNDR và BMI inversion qua 3 training seeds | 10 attack seeds cho generator-only MIA/memorization/attribute + Explanation-Linkage | V5.5 rộng hơn về attack surface; bản đầu mạnh hơn về retraining variance |
| Qualitative | Một example grid | 10 representative factuals/dataset, 5 mỗi hướng; giữ NO VALID CFE | V5.5 giảm cherry-picking |
| HE | Notebook Table 5 riêng, N=8192, P=200 và packing sensitivity 50…500 | Full dataset-specific P grid, phase/communication/RAM/error/agreement; 3 cohorts × 5 repeats/P | V5.5 đã có accepted HE paper trên cả ba dataset |
| Reproducibility | Config hash, manifests, atomic outputs và ZIP | Checksum-locked handoff, acceptance JSON, common RNG, failure audit và renderer | Cả hai tốt; V5.5 có machine-checked cross-dataset gates nhiều hơn |
| Thống kê | Mean±SD qua 3 GAN/baseline seeds trên 50 one-direction factuals | Paired factual bootstrap + Wilcoxon/Holm trên 200 bidirectional factuals, nhưng 1 training seed | Hai loại bất định bổ sung nhau; V5.5 chưa thay thế multi-seed |

### 10.1 Đánh giá tính công bằng và chuẩn học thuật

| Tiêu chí audit | Kết luận | Giải thích |
| --- | --- | --- |
| Cùng outer cohort và hướng | Đạt | Mọi method nhận cùng 200 factuals và cùng hai hướng; kết quả paired theo query_id |
| Cùng K và cách tính failure | Đạt | K=10; slot thiếu, constraint fail hoặc không đổi nhãn đều làm giảm Yield@10 |
| Cùng candidate budget cho population methods | Đạt | Uniform, Genetic và nhánh CounterGAN-SD tiêu thụ full cap; báo B frontier |
| Cùng immutable/actionability constraints | Đạt | Cùng projector, feature groups, frozen MLP và target semantics trong mỗi dataset |
| Kiểm soát randomness | Đạt | Paired base/search/robustness seeds; không loại factual thất bại |
| Chọn hyperparameter không nhìn outer test | Đạt theo contract | P và primary ε=4 được khóa prospectively từ inner/deployment contract; outer attack không dùng để tune |
| DP comparison | Đạt cho privacy–utility frontier | Cùng kiến trúc/schedule family và exact accountant; non-DP được ghi ε=∞, không gán bảo đảm DP |
| DiCE/Wachter so với black-box methods | Công bằng có phân tầng | Gradient methods có backward oracle nên không có access ngang black-box; vì vậy báo riêng oracle stratum thay vì tuyên bố cùng chi phí |
| Official baseline libraries | Đạt cho DiCE; giới hạn cho Wachter | Official DiCE random/genetic chạy đủ 100 factuals/hướng và giữ timeout/no-CF; native Wachter vẫn chỉ 2 factuals/hướng |
| Tối ưu riêng từng baseline | Giới hạn | Common deployment budget/P bảo đảm cùng tài nguyên, nhưng không chứng minh mọi baseline đang ở optimum riêng của nó |
| Patient-level independence | Đạt ECG; giới hạn MIMIC | ECG subject-disjoint; Heart+ duplicate-group-disjoint; MIMIC source thiếu patient ID nên chỉ stratified row split |
| Multi-seed training | Chưa đạt final confirmatory | Hiện đúng một GAN-training seed; 10 attack seeds không bù cho thiếu nhiều training seeds |
| HE final estimate | Đạt | Cả ba HE paper accepted; 15 timing runs/population, TC128 gate, boundary stress và label agreement gate đều đạt |


## 11. Bằng chứng, giới hạn và trạng thái paper

- Plaintext: 3 accepted **one-seed** paper artifacts, mỗi dataset 200 factuals × 11 methods × 4 budgets = 8,800 per-factual rows; cả hai hướng được giữ.
- Official DiCE: 3 accepted paper extensions, mỗi dataset 200 factuals × 2 official methods = 400 rows; không retrain MLP/GAN, giữ timeout/no-CF và không tuyên bố equal compute.
- Qualitative: 10 representative factuals/dataset và long/compact grid đã xuất.
- Privacy: generator-only MIA, memorization distance, attribute inference và Explanation-Linkage có lặp attack/resampling; các lần lặp attack không được mô tả là GAN-training seeds và attack không dùng để tune outer method.
- HE paper: cả ba artifact qua checksum, CPU-only, TC128, population, direction/source, boundary và selected-P gates; `accepted=true`, `paper_numbers=true`. Mỗi population có 3 candidate cohorts × 5 timing repetitions.
- HE correctness không đồng nghĩa CFE utility: agreement chỉ xác nhận encrypted MLP giữ nhãn plaintext trên supplied candidates; Yield/Sparsity/Diversity vẫn thuộc plaintext CFE evaluation.
- Plaintext vẫn là một generator-training seed; multi-seed confirmatory paper là bước riêng. Không viết `mean ± 0`, `ổn định qua seed`, hay `generalizes across runs` từ artifact hiện tại.

## 12. Nguồn artifact canonical

| Dataset | Canonical plaintext artifact | Accepted | Paper numbers | Per-factual rows | K | Queries/direction | Directions | GAN-training seeds | Accepted official DiCE paper | Accepted HE paper |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Leipzig ECG | experiments/cloud/results/ecg/q1_v55_generator_only_paper_v1/artifact_extracted | Có | Có | 8,800 | 10 | 100 | both | [11] | experiments/cloud/results/ecg/q1_v55_official_dice_extension_paper_v1/ecg_official_dice_extension_paper | experiments/cloud/results/ecg/q1_v55_he_population_paper_v1/ecg_q1_v55_he_population_paper |
| Heart+ | experiments/cloud/results/heartplus/q1_v55_generator_only_paper_v1/artifact_extracted | Có | Có | 8,800 | 10 | 100 | both | [11] | experiments/cloud/results/heartplus/q1_v55_official_dice_extension_paper_v1/heartplus_official_dice_extension_paper | experiments/cloud/results/heartplus/q1_v55_he_population_paper_v1/heartplus_q1_v55_he_population_paper |
| MIMIC-IV | experiments/cloud/results/mimic/q1_v55_generator_only_paper_v1/artifact_extracted | Có | Có | 8,800 | 10 | 100 | both | [11] | experiments/cloud/results/mimic/q1_v55_official_dice_extension_paper_v1/mimic_official_dice_extension_paper | experiments/cloud/results/mimic/q1_v55_he_population_paper_v1/mimic_q1_v55_he_population_paper |
