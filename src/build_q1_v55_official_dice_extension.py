#!/usr/bin/env python3
"""Build no-retraining official DiCE extensions for accepted Q1 V5.5 runs.

The generated notebooks consume an accepted V5.5 Kaggle kernel output plus
the original raw dataset.  They checksum-audit and reuse the fitted
preprocessor, split, classifier and exact outer-test query manifest.  Only
dice-ml 0.12 Random and Genetic searches are executed; MLP/GAN training is
never repeated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[2]

CONFIG = {
    "ecg": {
        "owner": "REDACTED_KAGGLE_OWNER",
        "title": "Leipzig ECG V5.5 Official DiCE Fidelity-120 Extension {mode}",
        "slug": "leipzig-ecg-v5-5-official-dice-extension-{mode}",
        "dataset_source": "REDACTED_KAGGLE_OWNER/heart-ecg",
        "kernel_source": "REDACTED_KAGGLE_OWNER/leipzig-ecg-q1-v5-5-gen-only-one-seed-paper-t4",
        "operating_budget": 512,
    },
    "heartplus": {
        "owner": "REDACTED_KAGGLE_OWNER",
        "title": "Heart+ V5.5 Official DiCE Fidelity-120 Extension {mode}",
        "slug": "heart-v5-5-official-dice-extension-{mode}",
        "dataset_source": "REDACTED_KAGGLE_OWNER/heart-max",
        "kernel_source": "REDACTED_KAGGLE_OWNER/heart-q1-v5-5-gen-only-one-seed-paper-t4",
        "operating_budget": 1024,
    },
    "mimic": {
        "owner": "meanalways",
        "title": "MIMIC V5.5 DiCE Fidelity-120 {mode}",
        "slug": "mimic-v5-5-dice-fidelity-120-{mode}",
        "dataset_source": "meanalways/mimiciv-full",
        "kernel_source": "buiquocviet/mimic-iv-q1-v5-5-gen-only-one-seed-paper-t4",
        "operating_budget": 512,
    },
}


def code(text: str):
    return nbformat.v4.new_code_cell(text.strip() + "\n")


def markdown(text: str):
    return nbformat.v4.new_markdown_cell(text.strip() + "\n")


def build(dataset: str, mode: str) -> Path:
    cfg = CONFIG[dataset]
    paper = mode == "paper"
    per_direction = 100 if paper else 2
    output_dir = ROOT / f"kaggle/notebooks/{dataset}/q1_v55_official_dice_extension_{mode}"
    output_dir.mkdir(parents=True, exist_ok=True)

    nb = nbformat.v4.new_notebook()
    nb.metadata.kernelspec = {
        "display_name": "Python 3", "language": "python", "name": "python3"
    }
    nb.metadata.language_info = {"name": "python", "version": "3"}
    nb.cells = [
        markdown(f"""
# {cfg['title'].format(mode=mode.title())}

No-retraining extension of the accepted Q1 V5.5 plaintext paper run. This
notebook runs the **official `dice-ml==0.12` Random and Genetic methods** on
the frozen outer-test cohort. It reuses the checksum-locked split,
preprocessor and MLP checkpoint; neither the MLP nor any GAN is trained.

The full run evaluates 100 factuals per direction. `timeout`, `no_cf`, and
implementation failures stay in the denominator. Outcome/constraint metrics
are directly comparable with V5.5. Candidate-budget cost is not claimed to be
matched because official DiCE exposes native sample/iteration controls rather
than the proposed method's counted population budget.
"""),
        code("""
import sys, subprocess
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "scikit-learn==1.6.1", "dice-ml==0.12"
])
"""),
        code(f"""
from pathlib import Path
import hashlib, json, os, shutil, sys, time, zipfile
import joblib
import numpy as np
import pandas as pd
import torch

DATASET = {dataset!r}
RUN_MODE = {mode!r}
PROTOCOL = "Q1_V55_OFFICIAL_DICE_FIDELITY120_EXTENSION_" + RUN_MODE.upper()
PER_DIRECTION = {per_direction}
OPERATING_BUDGET = {cfg['operating_budget']}
K = 10
MARGIN = 0.10
BASELINE_SEED = 11
TIMEOUT_SECONDS = 120
RANDOM_SAMPLE_SIZE = 10_000
GENETIC_MAXITERATIONS = 300
ROBUSTNESS_REPETITIONS = 16
FIDELITY_SAMPLES = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT = Path("/kaggle/working") / f"{{DATASET}}_official_dice_extension_{{RUN_MODE}}"
OUTPUT.mkdir(parents=True, exist_ok=True)

print({{
    "protocol": PROTOCOL, "dataset": DATASET, "device": DEVICE,
    "queries_per_direction": PER_DIRECTION, "K": K, "margin": MARGIN,
    "baseline_seed": BASELINE_SEED, "timeout_seconds": TIMEOUT_SECONDS,
    "random_sample_size": RANDOM_SAMPLE_SIZE,
    "genetic_maxiterations": GENETIC_MAXITERATIONS,
    "fidelity_samples": FIDELITY_SAMPLES,
    "retraining": False,
}})
"""),
        markdown("""
## Frozen-source audit and exact reconstruction

The notebook rejects an unaccepted source or checksum mismatch. The raw data
are only transformed with the already fitted preprocessor so that
DomainProjector, constraints, plausibility references, and the test row
indices are identical to the accepted V5.5 run.
"""),
        code("""
def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

def discover_source():
    roots = []
    for acceptance in Path("/kaggle/input").rglob("v55_paper_acceptance.json"):
        root = acceptance.parent
        if (root / "paper_training_handoff/classifier_seed55.pt").is_file():
            roots.append(root)
    if not roots:
        extract = Path("/kaggle/working/v55_source_extract")
        extract.mkdir(exist_ok=True)
        for archive in Path("/kaggle/input").rglob("*artifacts.zip"):
            destination = extract / archive.stem
            destination.mkdir(exist_ok=True)
            with zipfile.ZipFile(archive) as zipped:
                zipped.extractall(destination)
        for acceptance in extract.rglob("v55_paper_acceptance.json"):
            root = acceptance.parent
            if (root / "paper_training_handoff/classifier_seed55.pt").is_file():
                roots.append(root)
    unique = [Path(value) for value in sorted(set(map(str, roots)))]
    if len(unique) != 1:
        raise RuntimeError(f"Expected exactly one V5.5 source root, found {unique}")
    return unique[0]

SOURCE = discover_source()
HANDOFF = SOURCE / "paper_training_handoff"
ACCEPTANCE = json.loads((SOURCE / "v55_paper_acceptance.json").read_text())
if not ACCEPTANCE.get("accepted") or not ACCEPTANCE.get("paper_numbers"):
    raise AssertionError("Source is not an accepted V5.5 paper artifact")
if ACCEPTANCE.get("dataset") != DATASET:
    raise AssertionError((ACCEPTANCE.get("dataset"), DATASET))

manifest_path = HANDOFF / "training_handoff_manifest.json"
manifest = json.loads(manifest_path.read_text())
checksum_rows = []
for relative, expected in manifest["files_sha256"].items():
    actual = sha256_file(HANDOFF / relative)
    checksum_rows.append({
        "file": relative, "expected_sha256": expected,
        "actual_sha256": actual, "checksum_match": actual == expected,
    })
checksum_audit = pd.DataFrame(checksum_rows)
if not checksum_audit.checksum_match.all():
    raise AssertionError(checksum_audit.loc[~checksum_audit.checksum_match])
checksum_audit.to_csv(OUTPUT / "01_source_checksum_audit.csv", index=False)

sys.path.insert(0, str(HANDOFF))
import plaintext_cfe_pipeline as pipeline
from plaintext_cfe_pipeline import (
    PreparedData, HEPolynomialMLP, DomainProjector, SearchConfig,
    build_official_dice_explainers, cfe_set_metrics, load_dataset,
    official_dice_generate, representative_query_subset, resolve_input,
)

raw = load_dataset(DATASET, resolve_input(DATASET))
bundle = joblib.load(HANDOFF / "preprocessor_and_split.joblib")
preprocessor, split = bundle["preprocessor"], bundle["split"]
y = raw.frame[raw.target_col].to_numpy(np.int8)
x_train = preprocessor.transform(raw.frame.iloc[split.train_idx]).astype(np.float32)
x_valid = preprocessor.transform(raw.frame.iloc[split.valid_idx]).astype(np.float32)
x_test = preprocessor.transform(raw.frame.iloc[split.test_idx]).astype(np.float32)
prepared = PreparedData(
    raw, split, preprocessor,
    x_train, y[split.train_idx], x_valid, y[split.valid_idx],
    x_test, y[split.test_idx], list(bundle["feature_names"]),
)

checkpoint = torch.load(
    HANDOFF / "classifier_seed55.pt", map_location="cpu", weights_only=False
)
classifier_cfg = checkpoint["classifier_config"]
oracle = HEPolynomialMLP(
    checkpoint["input_dim"], classifier_cfg["h1"], classifier_cfg["h2"],
    classifier_cfg["activation_kind"], classifier_cfg["alpha"],
    classifier_cfg["dropout"],
)
oracle.load_state_dict(checkpoint["state_dict"])
oracle = oracle.eval().to(DEVICE)
projector = DomainProjector(prepared)

all_queries = pd.read_csv(SOURCE / "outer_test_query_manifest_v5.csv")
if RUN_MODE == "paper":
    queries = all_queries.copy()
else:
    queries = representative_query_subset(all_queries, per_direction=PER_DIRECTION)
expected = PER_DIRECTION if RUN_MODE == "paper" else PER_DIRECTION
counts = queries.groupby("direction").size().to_dict()
if counts != {"disease_to_no_disease": expected, "no_disease_to_disease": expected}:
    raise AssertionError(counts)
if queries.query_id.duplicated().any():
    raise AssertionError("Duplicate query_id in evaluation cohort")
queries.to_csv(OUTPUT / "02_exact_query_cohort.csv", index=False)

source_audit = {
    "protocol": PROTOCOL, "dataset": DATASET,
    "source_acceptance_protocol": ACCEPTANCE.get("protocol"),
    "source_accepted": True, "source_paper_numbers": True,
    "source_manifest_sha256": sha256_file(manifest_path),
    "classifier_sha256": sha256_file(HANDOFF / "classifier_seed55.pt"),
    "preprocessor_sha256": sha256_file(HANDOFF / "preprocessor_and_split.joblib"),
    "query_manifest_sha256": sha256_file(SOURCE / "outer_test_query_manifest_v5.csv"),
    "all_handoff_checksums_match": bool(checksum_audit.checksum_match.all()),
    "mlp_retrained": False, "gan_retrained": False,
    "dp_training_or_accountant_changed": False,
}
(OUTPUT / "source_reuse_contract.json").write_text(json.dumps(source_audit, indent=2))
print(pd.DataFrame([source_audit]).to_string(index=False))
print(queries.groupby(["direction", "difficulty_stratum"]).size())
"""),
        markdown("""
## Official DiCE evaluation

Both methods receive the same factuals, desired labels, immutable/actionable
features, permitted raw ranges, (K=10), margin, classifier and metric code.
DiCE Random uses `sample_size=10,000`; DiCE Genetic uses at most 300 native
iterations. Each factual-method call has a 120-second wall-time cap. This is an
extended-time sensitivity revision; the accepted 20-second artifact remains an
immutable historical comparison. Local-surrogate Fidelity uses 64 evaluation
samples after generation and is not included in DiCE runtime or native search
budget. Results and returned CFE vectors are checkpointed after every call.
"""),
        code("""
dice_explainers, features_to_vary, permitted_range = build_official_dice_explainers(
    prepared, oracle, projector, DEVICE, reference_size=5000
)
metric_config = SearchConfig(population=32, max_rounds=16, k=K, margin=MARGIN)
raw_path = OUTPUT / "03_official_dice_factual_metrics_raw.csv"
cfe_path = OUTPUT / "03_1_official_dice_cfe_vectors.csv"
rows = []
cfe_rows = []
run_started = time.perf_counter()

for ordinal, record in enumerate(queries.itertuples(index=False), start=1):
    query = prepared.x_test[int(record.local_test_index)]
    desired = int(record.desired_class)
    for method in ["official_dice_random", "official_dice_genetic"]:
        run_seed = (
            8_000_000 + 100_000 * BASELINE_SEED
            + 1000 * int(record.query_id) + sum(map(ord, method))
        )
        cfes, outcome = official_dice_generate(
            method, query, desired, prepared, oracle, projector, DEVICE,
            dice_explainers[method], features_to_vary, permitted_range,
            k=K, seed=run_seed, timeout_seconds=TIMEOUT_SECONDS,
            random_sample_size=RANDOM_SAMPLE_SIZE,
            genetic_maxiterations=GENETIC_MAXITERATIONS,
        )
        metrics = cfe_set_metrics(
            cfes, query, desired, oracle, projector, metric_config, DEVICE,
            outcome["runtime_seconds"], audit=None,
            robustness_seed=run_seed + 7_000_000,
            robustness_repetitions=ROBUSTNESS_REPETITIONS,
            fidelity_samples=FIDELITY_SAMPLES,
        )
        rows.append({
            "query_id": int(record.query_id),
            "local_test_index": int(record.local_test_index),
            "query_class": int(record.query_class),
            "desired_class": desired,
            "direction": record.direction,
            "difficulty_stratum": record.difficulty_stratum,
            "baseline_seed": BASELINE_SEED,
            "run_seed": run_seed,
            "method": method,
            "implementation": "official dice-ml==0.12",
            "K": K, "margin": MARGIN,
            "native_random_sample_size": (
                RANDOM_SAMPLE_SIZE if method == "official_dice_random" else np.nan
            ),
            "native_genetic_maxiterations": (
                GENETIC_MAXITERATIONS if method == "official_dice_genetic" else np.nan
            ),
            **metrics, **outcome,
        })
        if len(cfes):
            raw_cfes = prepared.preprocessor.inverse(cfes)
            for rank, vector in enumerate(cfes, start=1):
                cfe_row = {
                    "query_id": int(record.query_id),
                    "local_test_index": int(record.local_test_index),
                    "direction": record.direction,
                    "desired_class": desired,
                    "method": method,
                    "baseline_seed": BASELINE_SEED,
                    "run_seed": run_seed,
                    "cfe_rank": rank,
                }
                cfe_row.update({
                    f"cfe__{name}": float(value)
                    for name, value in zip(prepared.feature_names, vector)
                })
                cfe_row.update({
                    f"raw__{name}": raw_cfes.iloc[rank - 1][name]
                    for name in raw_cfes.columns
                })
                cfe_rows.append(cfe_row)
        pd.DataFrame(rows).to_csv(raw_path, index=False)
        pd.DataFrame(cfe_rows).to_csv(cfe_path, index=False)
    if ordinal == 1 or ordinal % 10 == 0 or ordinal == len(queries):
        elapsed = time.perf_counter() - run_started
        print(f"{ordinal}/{len(queries)} factuals; elapsed={elapsed/60:.1f} min")

raw_results = pd.DataFrame(rows)
print(raw_results.groupby(["method", "direction", "algorithm_status"]).size())
"""),
        markdown("""
## Unconditional success and success-conditioned quality

Yield, Coverage, Full-K and Robust Yield use every predeclared factual,
including timeouts/no-CF. Proximity, Sparsity, Diversity and Plausibility are
defined only when at least one valid CFE exists, so their effective sample
count is printed explicitly.
"""),
        code("""
def summarize(group):
    return pd.Series({
        "n_factuals": int(group.query_id.nunique()),
        "n_ok": int((group.algorithm_status == "ok").sum()),
        "n_timeout": int((group.algorithm_status == "timeout").sum()),
        "n_no_cf": int((group.algorithm_status == "no_cf").sum()),
        "n_implementation_error": int((group.algorithm_status == "implementation_error").sum()),
        "n_quality_defined": int(group.proximity.notna().sum()),
        "n_fidelity_defined": int(group.fidelity.notna().sum()),
        "valid_cfe_yield_at_k": float(group.valid_cfe_yield_at_k.mean()),
        "coverage_at_1": float(group.coverage_at_1.mean()),
        "full_k_success": float(group.full_k_success.mean()),
        "robust_yield_at_k": float(group.robust_completeness_at_k.mean()),
        "robust_full_k_success": float(group.robust_full_k_success.mean()),
        "proximity": float(group.proximity.mean()),
        "sparsity": float(group.sparsity.mean()),
        "diversity": float(group.diversity.mean()),
        "plausibility": float(group.plausibility.mean()),
        "fidelity": float(group.fidelity.mean()),
        "fidelity_oracle_evaluations_mean": float(
            group.fidelity_oracle_evaluations.mean()
        ),
        "constraint_validity": float(group.constraint_validity.mean()),
        "runtime_seconds_mean": float(group.runtime_seconds.mean()),
        "runtime_seconds_total": float(group.runtime_seconds.sum()),
    })

summary_rows = []
for (method, direction), group in raw_results.groupby(
    ["method", "direction"], sort=True
):
    row = summarize(group).to_dict()
    row.update({"method": method, "direction": direction})
    summary_rows.append(row)
by_direction = pd.DataFrame(summary_rows)
by_direction.to_csv(OUTPUT / "04_official_dice_summary_by_direction.csv", index=False)

# Equal-direction macro: each clinical direction has weight 1/2.
macro_metrics = [
    "valid_cfe_yield_at_k", "coverage_at_1", "full_k_success",
    "robust_yield_at_k", "robust_full_k_success", "proximity", "sparsity",
    "diversity", "plausibility", "fidelity",
    "fidelity_oracle_evaluations_mean", "constraint_validity",
    "runtime_seconds_mean",
]
macro = by_direction.groupby("method", as_index=False)[macro_metrics].mean()
macro["aggregation"] = "equal-direction macro (0.5 per direction)"
macro.to_csv(OUTPUT / "05_official_dice_equal_direction_macro.csv", index=False)

outcomes = (
    raw_results.groupby(["method", "direction", "algorithm_status"], as_index=False)
    .size().rename(columns={"size": "n_factuals"})
)
outcomes.to_csv(OUTPUT / "06_official_dice_outcomes.csv", index=False)
display(by_direction)
display(macro)
display(outcomes)
"""),
        markdown("""
## Linked comparison with the accepted V5.5 table

This table links official DiCE to the accepted paper operating point. The
cohort and metrics are matched. The final columns explicitly prevent a false
claim of equal computational budget: V5.5 custom methods have counted
candidate budgets, while official DiCE has native samples/iterations and
measured wall time.
"""),
        code("""
main_path = SOURCE / "plaintext_manuscript_tables/03_2a_main_metrics_by_direction.csv"
main = pd.read_csv(main_path)
main = main.loc[main.candidate_budget.eq(OPERATING_BUDGET)].copy()
main_comparison = pd.DataFrame({
    "method": main.method,
    "method_label": main.method_label,
    "direction": main.direction,
    "n_factuals": main.n_factuals.astype(int),
    "valid_cfe_yield_at_k": main.valid_cfe_yield_at_k_mean,
    "coverage_at_1": main.coverage_at_1_mean,
    "full_k_success": main.full_k_success_mean,
    "robust_yield_at_k": main.robust_completeness_at_k_mean,
    "proximity": main.proximity_mean,
    "sparsity": main.sparsity_mean,
    "diversity": main.diversity_mean,
    "plausibility": main.plausibility_mean,
    "fidelity": main.fidelity_mean,
    "constraint_validity": main.constraint_validity_mean,
    "runtime_seconds_mean": main.generation_time_to_k_or_cap_seconds_mean,
    "access_stratum": "custom counted-candidate implementation",
    "candidate_budget": main.candidate_budget,
    "native_control": "counted candidate evaluations",
    "same_cohort_constraints_metrics": True,
    "equal_compute_budget_claimed": False,
})
official_comparison = by_direction.copy()
official_comparison["method_label"] = official_comparison.method.map({
    "official_dice_random": "Official DiCE Random",
    "official_dice_genetic": "Official DiCE Genetic",
})
official_comparison["access_stratum"] = "official dice-ml native implementation"
official_comparison["candidate_budget"] = np.nan
official_comparison["native_control"] = official_comparison.method.map({
    "official_dice_random": f"sample_size={RANDOM_SAMPLE_SIZE}",
    "official_dice_genetic": f"maxiterations={GENETIC_MAXITERATIONS}",
})
official_comparison["same_cohort_constraints_metrics"] = True
official_comparison["equal_compute_budget_claimed"] = False

columns = [
    "method", "method_label", "direction", "n_factuals",
    "valid_cfe_yield_at_k", "coverage_at_1", "full_k_success",
    "robust_yield_at_k", "proximity", "sparsity", "diversity",
    "plausibility", "fidelity", "constraint_validity", "runtime_seconds_mean",
    "access_stratum", "candidate_budget", "native_control",
    "same_cohort_constraints_metrics", "equal_compute_budget_claimed",
]
linked = pd.concat([
    main_comparison[columns], official_comparison[columns]
], ignore_index=True)
linked.to_csv(OUTPUT / "07_linked_fair_comparison_by_direction.csv", index=False)

fairness = pd.DataFrame([
    {"item": "outer-test factual cohort", "matched": True,
      "detail": f"same frozen query IDs; {PER_DIRECTION} per direction"},
    {"item": "desired-label directions", "matched": True,
      "detail": "disease->no disease and no disease->disease reported separately"},
    {"item": "classifier/preprocessor", "matched": True,
      "detail": "checksum-locked V5.5 checkpoint and fitted transform"},
    {"item": "K / margin / constraints", "matched": True,
      "detail": "K=10, logit margin=0.10, same DomainProjector"},
    {"item": "failure denominator", "matched": True,
      "detail": "timeout and no-CF remain zero-yield factuals"},
    {"item": "random seed count", "matched": True,
      "detail": "one predeclared baseline seed (11), matching one-seed V5.5 scope"},
    {"item": "candidate/query budget", "matched": False,
      "detail": "official DiCE native controls differ; runtime and controls reported"},
    {"item": "local-surrogate Fidelity", "matched": True,
      "detail": "64 post-generation oracle samples; excluded from generation runtime/budget"},
    {"item": "timeout revision", "matched": False,
      "detail": "120-second post-hoc sensitivity; preserve accepted 20-second result"},
    {"item": "model training", "matched": True,
      "detail": "none repeated; accepted frozen artifacts reused"},
])
fairness.to_csv(OUTPUT / "08_fairness_audit.csv", index=False)
display(linked)
display(fairness)
"""),
        code("""
expected_rows = len(queries) * 2
pair_counts = raw_results.groupby(["query_id", "method"]).size()
checks = {
    "source_accepted": bool(ACCEPTANCE.get("accepted")),
    "source_paper_numbers": bool(ACCEPTANCE.get("paper_numbers")),
    "all_source_checksums_match": bool(checksum_audit.checksum_match.all()),
    "exact_direction_counts": bool(
        queries.groupby("direction").size().eq(PER_DIRECTION).all()
    ),
    "exact_result_rows": len(raw_results) == expected_rows,
    "one_row_per_query_method": bool(pair_counts.eq(1).all()),
    "both_official_methods_present": set(raw_results.method) == {
        "official_dice_random", "official_dice_genetic"
    },
    "no_implementation_errors": not bool(
        raw_results.algorithm_status.eq("implementation_error").any()
    ),
    "timeouts_retained": True,
    "no_cf_retained": True,
    "no_mlp_retraining": True,
    "no_gan_retraining": True,
    "no_dp_accountant_change": True,
    "equal_compute_budget_not_claimed": True,
    "fidelity_enabled": FIDELITY_SAMPLES == 64,
    "fidelity_evaluated_when_cfe_available": bool(
        raw_results.loc[
            raw_results.coverage_at_1.eq(1), "fidelity_oracle_evaluations"
        ].gt(0).all()
        and raw_results.loc[
            raw_results.coverage_at_1.eq(0), "fidelity_oracle_evaluations"
        ].eq(0).all()
    ),
    "returned_cfe_vectors_saved": bool(
        len(cfe_rows) == int(raw_results.reference_valid_returned.sum())
    ),
}
acceptance = {
    "protocol": PROTOCOL, "dataset": DATASET, "run_mode": RUN_MODE,
    "accepted": bool(all(checks.values())), "checks": checks,
    "n_queries": int(len(queries)), "queries_per_direction": PER_DIRECTION,
    "n_result_rows": int(len(raw_results)), "K": K, "margin": MARGIN,
    "timeout_seconds": TIMEOUT_SECONDS,
    "fidelity_samples": FIDELITY_SAMPLES,
    "n_saved_cfe_vectors": int(len(cfe_rows)),
    "wall_time_seconds": float(time.perf_counter() - run_started),
}
(OUTPUT / "official_dice_extension_acceptance.json").write_text(
    json.dumps(acceptance, indent=2)
)

archive = shutil.make_archive(str(OUTPUT) + "_artifacts", "zip", OUTPUT)
print(json.dumps(acceptance, indent=2))
print("ARTIFACT_ZIP", archive)
if not acceptance["accepted"]:
    print("WARNING: acceptance checks failed; preserve the ZIP for diagnosis")
"""),
    ]

    notebook_name = f"{dataset}_q1_v55_official_dice_extension_{mode}_t4.ipynb"
    notebook_path = output_dir / notebook_name
    nbformat.write(nb, notebook_path)
    metadata = {
        "id": f"{cfg['owner']}/{cfg['slug'].format(mode=mode)}",
        "title": cfg["title"].format(mode=mode.title()),
        "code_file": notebook_name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": True,
        "keywords": ["healthcare"],
        "dataset_sources": [cfg["dataset_source"]],
        "kernel_sources": [cfg["kernel_source"]],
        "competition_sources": [],
        "model_sources": [],
    }
    (output_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(CONFIG), required=True)
    parser.add_argument("--mode", choices=["smoke", "paper"], required=True)
    args = parser.parse_args()
    print(build(args.dataset, args.mode))


if __name__ == "__main__":
    main()
