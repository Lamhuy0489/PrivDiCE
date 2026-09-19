#!/usr/bin/env python3
"""Audit the public PrivDiCE one-seed reproducibility release.

Uses only the Python standard library so reviewers can run it before installing
the experiment dependencies.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FAILURES: list[str] = []
WARNINGS: list[str] = []


def passed(message: str) -> None:
    print(f"PASS  {message}")


def failed(message: str) -> None:
    FAILURES.append(message)
    print(f"FAIL  {message}")


def warned(message: str) -> None:
    WARNINGS.append(message)
    print(f"WARN  {message}")


def require(condition: bool, message: str) -> None:
    passed(message) if condition else failed(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def close(actual: str | float, expected: float, atol: float = 5e-7) -> bool:
    return math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=atol)


def boolish(value: object) -> bool:
    return value is True or (isinstance(value, str) and value.lower() == "true")


DATASETS = {
    "ecg": {
        "budget": 512,
        "population": 64,
        "data": "data/ecg/ecg_beats_features_selected_20.csv",
        "size": 26_070_248,
        "sha": "ffa8482f426ca0fdb6055f745610e88befe7e0cb36889a0bd2678dfe7bca5a9b",
        "classifier": (0.904632152588556, 0.9248562919869796, 0.983285112661418),
        "directions": {
            "disease_to_no_disease": (1.0, 0.833, 0.0723591640777885, 0.7760555555555554),
            "no_disease_to_disease": (1.0, 0.736, 0.0673351811803877, 0.7607777777777776),
        },
        "mia_dp4": 0.5016,
        "he": (83.08918291300051, 27.207333, 1.2938767444126142e-05, 1.0),
    },
    "heartplus": {
        "budget": 1024,
        "population": 128,
        "data": "data/heartplus/merged_data.csv",
        "size": 75_465_590,
        "sha": "8c3611f95469b3a06fe33628c2d82283dc3571b57bd0ff9b8c5c63ec44dac27e",
        "classifier": (0.8771281553143172, 0.3490385618415105, 0.8391716737638617),
        "directions": {
            "disease_to_no_disease": (0.907, 0.872, 0.0988068385644162, 0.6472156553716838),
            "no_disease_to_disease": (0.546, 0.517, 0.0928364114037581, 0.6390388007054676),
        },
        "mia_dp4": 0.4988,
        "he": (28.06021373199997, 10.791014, 0.0069737130523876, 1.0),
    },
    "mimic": {
        "budget": 512,
        "population": 64,
        "data": None,
        "classifier": (0.8840236686390532, 0.8574199806013579, 0.9495991656100456),
        "directions": {
            "disease_to_no_disease": (1.0, 0.892, 0.1175101792067289, 0.5061764705882353),
            "no_disease_to_disease": (0.99, 0.954, 0.0885428777999348, 0.6202614379084966),
        },
        "mia_dp4": 0.5052,
        "he": (80.63507672599826, 49.231794, 0.0019088341176143, 1.0),
    },
}


def audit_acceptance(dataset: str) -> None:
    base = ROOT / "results" / dataset
    plain = json.loads((base / "plaintext/v55_paper_acceptance.json").read_text())
    require(plain.get("accepted") is True and plain.get("paper_numbers") is True,
            f"{dataset}: plaintext accepted as paper numbers")
    require(plain.get("generator_training_seeds") == [11]
            and plain.get("method_search_seeds") == [11]
            and plain.get("queries_per_direction") == 100
            and plain.get("target_directions") == "both"
            and plain.get("raw_rows") == 8800,
            f"{dataset}: one-seed plaintext contract and exact counts")
    require(all(boolish(v) for v in plain.get("checks", {}).values()),
            f"{dataset}: every plaintext acceptance gate passes")

    official = json.loads(
        (base / "official_dice/official_dice_extension_acceptance.json").read_text()
    )
    require(official.get("accepted") is True
            and official.get("n_queries") == 200
            and official.get("n_result_rows") == 400
            and official.get("queries_per_direction") == 100,
            f"{dataset}: Official DiCE accepted with 200 queries/400 rows")
    require(all(boolish(v) for v in official.get("checks", {}).values()),
            f"{dataset}: every Official DiCE acceptance gate passes")

    he = json.loads((base / "he/12_he_acceptance.json").read_text())
    require(he.get("accepted") is True and he.get("paper_numbers") is True,
            f"{dataset}: HE accepted as paper numbers")
    require(he.get("selected_population") == DATASETS[dataset]["population"]
            and he.get("candidate_budget") == DATASETS[dataset]["budget"]
            and he.get("candidate_cohort_seeds") == [101, 202, 303]
            and he.get("timing_repetitions_per_cohort") == 5,
            f"{dataset}: HE population/budget and 3x5 timing contract")
    require(all(boolish(v) for v in he.get("checks", {}).values()),
            f"{dataset}: every HE acceptance gate passes")


def audit_numbers(dataset: str) -> None:
    spec = DATASETS[dataset]
    base = ROOT / "results" / dataset

    classifier = read_csv(base / "plaintext/classifier_metrics_by_seed.csv")
    require(len(classifier) == 1 and classifier[0]["seed"] == "55",
            f"{dataset}: exactly classifier seed 55")
    acc, f1, auc = spec["classifier"]
    require(close(classifier[0]["accuracy"], acc)
            and close(classifier[0]["f1"], f1)
            and close(classifier[0]["roc_auc"], auc),
            f"{dataset}: classifier Accuracy/F1/ROC-AUC match canonical values")

    metrics = read_csv(base / "plaintext/03_2a_main_metrics_by_direction.csv")
    primary = [
        row for row in metrics
        if row["method"] == "proposed_dp_eps4_sparse_diverse"
        and int(float(row["candidate_budget"])) == spec["budget"]
    ]
    require(len(primary) == 2, f"{dataset}: two primary directional rows")
    for row in primary:
        expected = spec["directions"][row["direction"]]
        actual = (
            row["valid_cfe_yield_at_k_mean"],
            row["robust_completeness_at_k_mean"],
            row["proximity_mean"],
            row["sparsity_mean"],
        )
        require(all(close(a, e) for a, e in zip(actual, expected)),
                f"{dataset}/{row['direction']}: primary Yield/Robust/Proximity/Sparsity")

    mia = read_csv(base / "plaintext/06_3_strongest_generator_only_mia_v2.csv")
    dp4 = [row for row in mia if row["variant"] == "dp_eps4"]
    require(len(dp4) == 1 and close(dp4[0]["mean"], spec["mia_dp4"], atol=5e-5)
            and dp4[0]["attack_seeds"] == "10",
            f"{dataset}: primary DP4 MIA AUC and 10 attack seeds")

    he_rows = read_csv(base / "he/05_he_population_timing_summary.csv")
    selected = [row for row in he_rows if int(row["population"]) == spec["population"]]
    require(len(selected) == 1, f"{dataset}: selected HE population row exists")
    row = selected[0]
    expected_he = spec["he"]
    actual_he = (
        row["total_latency_seconds_per_round_median"],
        row["total_communication_mb_per_round_median"],
        row["max_abs_logit_error_median"],
        row["plaintext_he_label_agreement_median"],
    )
    require(all(close(a, e) for a, e in zip(actual_he, expected_he))
            and row["candidate_cohort_seeds"] == "3"
            and row["timing_repetitions_total"] == "15",
            f"{dataset}: HE latency/communication/error/agreement and 15 timings")


def audit_data() -> None:
    for dataset in ("ecg", "heartplus"):
        spec = DATASETS[dataset]
        path = ROOT / str(spec["data"])
        require(path.is_file(), f"{dataset}: public input exists")
        if path.is_file():
            require(path.stat().st_size == spec["size"] and sha256(path) == spec["sha"],
                    f"{dataset}: public input size and SHA-256")

    mimic = ROOT / "data/mimic/mimic_icu_mortality_cfe_12671.csv"
    require(not mimic.exists(), "mimic: access-controlled CSV is absent")
    require(not list((ROOT / "artifacts/mimic").glob("*.zip")),
            "mimic: complete artifact ZIP is absent")

    extracted_artifacts = {
        "ecg plaintext": (ROOT / "artifacts/ecg/ecg_q1v55_generator_only_paper_artifacts", 169),
        "ecg Official DiCE": (ROOT / "artifacts/ecg/ecg_official_dice_extension_paper_artifacts", 10),
        "ecg HE": (ROOT / "artifacts/ecg/ecg_q1_v55_he_population_paper_artifacts", 16),
        "heartplus plaintext": (ROOT / "artifacts/heartplus/heartplus_q1v55_generator_only_paper_artifacts", 169),
        "heartplus Official DiCE": (ROOT / "artifacts/heartplus/heartplus_official_dice_extension_paper_artifacts", 10),
        "heartplus HE": (ROOT / "artifacts/heartplus/heartplus_q1_v55_he_population_paper_artifacts", 16),
    }
    for label, (directory, expected_files) in extracted_artifacts.items():
        actual_files = sum(path.is_file() for path in directory.rglob("*"))
        require(directory.is_dir() and actual_files == expected_files,
                f"{label}: extracted artifact directory has {expected_files} files")
    require(not list((ROOT / "artifacts").rglob("*.zip")),
            "redistributable artifact ZIPs were removed after verified extraction")

    supplement = ROOT / "docs/EXPERIMENTAL_SUPPLEMENT_ONE_SEED.md"
    supplement_text = supplement.read_text()
    require("paper một seed" in supplement_text and "seed 11" in supplement_text,
            "public one-seed supplement retains the audited experimental scope")


def audit_notebooks() -> None:
    notebooks = sorted((ROOT / "notebooks").glob("*/*/*.ipynb"))
    metadata = sorted((ROOT / "notebooks").glob("*/*/kernel-metadata.json"))
    require(len(notebooks) == 9 and len(metadata) == 9,
            "nine executed notebook copies and nine metadata files")

    fully_executed_notebooks = []
    incomplete_notebooks = []
    notebooks_with_errors = []
    for path in notebooks:
        notebook = json.loads(path.read_text())
        code = [cell for cell in notebook.get("cells", []) if cell.get("cell_type") == "code"]
        has_saved_outputs = sum(len(cell.get("outputs", [])) for cell in code) > 0
        if code and all(cell.get("execution_count") is not None for cell in code) and has_saved_outputs:
            fully_executed_notebooks.append(path)
        else:
            incomplete_notebooks.append(path)
        if any(output.get("output_type") == "error"
               for cell in code for output in cell.get("outputs", [])):
            notebooks_with_errors.append(path)
    require(len(fully_executed_notebooks) == 9 and not incomplete_notebooks,
            "all nine paper notebooks retain execution counts and saved outputs")
    require(not notebooks_with_errors,
            "no accepted paper notebook contains an error output")

    private = []
    for path in metadata:
        record = json.loads(path.read_text())
        if record.get("is_private") is True:
            private.append(record.get("id", str(path)))
    if private:
        warned("Kaggle notebooks still private: " + ", ".join(private))
    else:
        passed("all Kaggle notebooks are public")


def audit_repository_safety() -> None:
    oversized = [
        str(path.relative_to(ROOT)) for path in ROOT.rglob("*")
        if path.is_file() and path.stat().st_size >= 100_000_000
    ]
    require(not oversized, "no GitHub file reaches the 100 MB hard limit")
    credential_names = {"kaggle.json", ".env", "credentials.json"}
    credentials = [
        str(path.relative_to(ROOT)) for path in ROOT.rglob("*")
        if path.is_file() and path.name.lower() in credential_names
    ]
    require(not credentials, "no obvious credential file is present")


def main() -> int:
    print("PrivDiCE public-release audit\n")
    audit_data()
    for dataset in DATASETS:
        audit_acceptance(dataset)
        audit_numbers(dataset)
    audit_notebooks()
    audit_repository_safety()
    print(f"\nSummary: {len(FAILURES)} failure(s), {len(WARNINGS)} warning(s)")
    for warning in WARNINGS:
        print(f"  warning: {warning}")
    if FAILURES:
        for failure in FAILURES:
            print(f"  failure: {failure}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
