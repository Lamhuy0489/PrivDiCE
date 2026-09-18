#!/usr/bin/env python3
"""Build Q1-V5.5 generator-only plaintext smoke/paper notebooks.

V5.5 is a prospective protocol.  It starts from the completed V5.4
paper-depth notebook, removes the discriminator from deployment-time search,
uses common random seeds and paired robustness perturbations, adds a clean
non-DP sparse/diverse ablation, and freezes an HE-aware population selected
from the V5.4 inner-validation frontier.  Smoke must pass before paper is
submitted.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[2]


CONFIG = {
    "ecg": {
        "source": ROOT / "kaggle/notebooks/ecg/q1_v54_paper_depth_pilot/ecg_q1_v54_paper_depth_pilot_t4.ipynb",
        "owner": "REDACTED_KAGGLE_OWNER",
        "slug": "leipzig-ecg-cfe-q1-v5-5-generator-only-{mode}-t4",
        "title": "Leipzig ECG CFE Q1 V5.5 Generator-Only {mode_title} T4",
        "paper_slug": "leipzig-ecg-q1-v5-5-gen-only-one-seed-paper-t4",
        "paper_title": "Leipzig ECG Q1 V5.5 Gen-Only One-Seed Paper T4",
        "dataset_source": "REDACTED_KAGGLE_OWNER/heart-ecg",
        "population": 64,
        "operating_budget": 512,
        "budgets": [64, 128, 256, 512],
        "population_grid": [16, 32, 64, 128, 256, 512],
        "paper_gan": (300, 20, 30, 180),
    },
    "heartplus": {
        "source": ROOT / "kaggle/notebooks/heartplus/q1_v54_paper_depth_pilot/heartplus_q1_v54_paper_depth_pilot_t4.ipynb",
        "owner": "REDACTED_KAGGLE_OWNER",
        "slug": "heart-cfe-q1-v5-5-generator-only-{mode}-t4",
        "title": "Heart+ CFE Q1 V5.5 Generator-Only {mode_title} T4",
        "paper_slug": "heart-q1-v5-5-gen-only-one-seed-paper-t4",
        "paper_title": "Heart+ Q1 V5.5 Gen-Only One-Seed Paper T4",
        "dataset_source": "REDACTED_KAGGLE_OWNER/heart-max",
        "population": 128,
        "operating_budget": 1024,
        "budgets": [128, 256, 512, 1024],
        "population_grid": [32, 64, 128, 256, 512, 1024],
        "paper_gan": (300, 20, 30, 180),
    },
    "mimic": {
        "source": ROOT / "kaggle/notebooks/mimic/q1_v54_paper_depth_pilot/mimic_q1_v54_paper_depth_pilot_t4.ipynb",
        "owner": "buiquocviet",
        "slug": "mimic-iv-cfe-q1-v5-5-generator-only-{mode}-t4",
        "title": "MIMIC-IV CFE Q1 V5.5 Generator-Only {mode_title} T4",
        "paper_slug": "mimic-iv-q1-v5-5-gen-only-one-seed-paper-t4",
        "paper_title": "MIMIC-IV Q1 V5.5 Gen-Only One-Seed Paper T4",
        "dataset_source": "meanalways/mimiciv-full",
        "population": 64,
        "operating_budget": 512,
        "budgets": [64, 128, 256, 512],
        "population_grid": [32, 64, 128, 256, 512],
        "paper_gan": (210, 20, 21, 126),
    },
}


def cell_with(nb, marker: str):
    matches = [cell for cell in nb.cells if marker in getattr(cell, "source", "")]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one cell containing {marker!r}, got {len(matches)}")
    return matches[0]


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one occurrence, got {count}")
    return text.replace(old, new, 1)


def replace_all_cells(nb, old: str, new: str) -> None:
    for cell in nb.cells:
        if hasattr(cell, "source"):
            cell.source = cell.source.replace(old, new)


def patch_pipeline(source: str) -> str:
    """Patch the embedded V5.4 pipeline without modifying frozen notebooks."""
    source = replace_once(
        source,
        "    plausibility_weight: float = .20",
        "    plausibility_weight: float = 0.0",
        "data-free deployment default",
    )
    source = replace_once(
        source,
        "    if plausibility_cost is None:\n        plausible = projector.plausibility(candidate)\n    else:",
        "    if config.plausibility_weight <= 0:\n"
        "        plausible = np.zeros(len(candidate), dtype=float)\n"
        "    elif plausibility_cost is None:\n"
        "        raise RuntimeError(\n"
        "            'Nonzero search plausibility weight requires an explicit public scorer'\n"
        "        )\n"
        "    else:",
        "no implicit private kNN search",
    )
    source = replace_once(
        source,
        "    \"proposed_countergan\": \"CounterGAN + iterative black-box search\",",
        "    \"proposed_countergan\": \"CounterGAN + iterative black-box search\",\n"
        "    \"countergan_sparse_diverse_no_dp\": (\n"
        "        \"CounterGAN non-DP + identical counted sparse-diverse search\"\n"
        "    ),",
        "non-DP matched-search label",
    )
    source = replace_once(
        source,
        "    generator_methods: list[tuple[str, nn.Module]] = [\n"
        "        (\"countergan_one_shot\", generators[\"countergan\"]),\n"
        "        (\"proposed_countergan\", generators[\"countergan\"]),\n"
        "    ]",
        "    generator_methods: list[tuple[str, nn.Module]] = [\n"
        "        (\"countergan_one_shot\", generators[\"countergan\"]),\n"
        "        (\"proposed_countergan\", generators[\"countergan\"]),\n"
        "    ]\n"
        "    if (include_methods is not None\n"
        "            and \"countergan_sparse_diverse_no_dp\" in include_methods):\n"
        "        generator_methods.append((\n"
        "            \"countergan_sparse_diverse_no_dp\", generators[\"countergan\"]\n"
        "        ))",
        "matched-search registry",
    )
    source = replace_once(
        source,
        "                    method_hash = sum((i + 1) * ord(char)\n"
        "                                      for i, char in enumerate(method))\n"
        "                    run_seed = (seed + 1_000_003 * int(record.query_id)\n"
        "                                + 10_007 * budget + 101 * method_seed\n"
        "                                + method_hash)",
        "                    # Paired common random-number contract: the same factual,\n"
        "                    # budget and nominal search seed receive the same base seed\n"
        "                    # for every method. Algorithms may consume the stream\n"
        "                    # differently, but method identity never changes it.\n"
        "                    run_seed = (seed + 1_000_003 * int(record.query_id)\n"
        "                                + 10_007 * budget + 101 * method_seed)",
        "common random numbers",
    )
    source = replace_once(
        source,
        '                        "method_seed": int(method_seed),\n'
        '                        "generator_training_seed": (np.nan if generator is None',
        '                        "method_seed": int(method_seed),\n'
        '                        "run_seed": int(run_seed),\n'
        '                        "robustness_seed": int(run_seed + 7_000_000),\n'
        '                        "generator_training_seed": (np.nan if generator is None',
        "persist paired seeds in factual metrics",
    )
    source = replace_once(
        source,
        '                    audit["method_seed"] = int(method_seed)\n'
        '                    audit["generator_training_seed"] = (np.nan if generator is None',
        '                    audit["method_seed"] = int(method_seed)\n'
        '                    audit["run_seed"] = int(run_seed)\n'
        '                    audit["robustness_seed"] = int(run_seed + 7_000_000)\n'
        '                    audit["generator_training_seed"] = (np.nan if generator is None',
        "persist paired seeds in round audit",
    )
    source = replace_once(
        source,
        "                    elif (method.startswith(\"proposed_dp_eps\")\n"
        "                          and method.endswith(\"_sparse_diverse\")):",
        "                    elif ((method.startswith(\"proposed_dp_eps\")\n"
        "                           and method.endswith(\"_sparse_diverse\"))\n"
        "                          or method == \"countergan_sparse_diverse_no_dp\"):",
        "shared sparse-diverse branch",
    )
    source = replace_once(
        source,
        "                        \"search_plausibility_source\": (\n"
        "                            \"released_discriminator\"\n"
        "                            if method in plausibility_scorers\n"
        "                            else \"development_knn\"\n"
        "                        ),",
        "                        \"search_plausibility_source\": (\n"
        "                            \"explicit_public_scorer\"\n"
        "                            if method in plausibility_scorers\n"
        "                            else \"none_generator_only\"\n"
        "                        ),",
        "search provenance",
    )
    source = source.replace(
        "DP search requires released-discriminator plausibility scorers:",
        "DP search requires explicit public plausibility scorers:",
    )

    # Fixed-cap quality protocol for the two population baselines.  They keep
    # their native proposal mechanisms but, like the proposed sparse/diverse
    # method, consume the full declared cap and select K with the same MMR
    # routine.  Time-to-K milestones are still recoverable from the round log.
    uniform_start = source.index("def search_uniform_random(")
    genetic_start = source.index("def search_genetic(", uniform_start)
    uniform = source[uniform_start:genetic_start]
    uniform = replace_once(
        uniform,
        "rng = np.random.default_rng(seed); archive: list[np.ndarray] = []; audits = []",
        "rng = np.random.default_rng(seed); archive: list[np.ndarray] = []; "
        "archive_quality: list[float] = []; audits = []",
        "uniform archive quality",
    )
    uniform = replace_once(
        uniform,
        "        for index in np.where(valid)[0][np.argsort(score[valid])]:\n"
        "            if _distinct(candidate[index], archive, projector, config.distinctness):\n"
        "                archive.append(candidate[index].copy())\n"
        "                if len(archive) == config.k:\n"
        "                    break",
        "        _archive_update(\n"
        "            archive, archive_quality, candidate, score, valid, projector,\n"
        "            config.distinctness,\n"
        "        )",
        "uniform full archive",
    )
    uniform = replace_once(
        uniform,
        "        if len(archive) >= config.k:\n            break\n",
        "",
        "uniform full cap",
    )
    uniform = replace_once(
        uniform,
        "    returned = (np.vstack(archive) if archive\n"
        "                else np.empty((0, len(query)), np.float32))",
        "    returned = _mmr_select(\n"
        "        archive, archive_quality, projector, config.k,\n"
        "        diversity_weight=config.set_diversity_weight,\n"
        "    )",
        "uniform MMR",
    )
    source = source[:uniform_start] + uniform + source[genetic_start:]

    genetic_start = source.index("def search_genetic(")
    dice_start = source.index("def search_dice_gradient(", genetic_start)
    genetic = source[genetic_start:dice_start]
    genetic = replace_once(
        genetic,
        "rng = np.random.default_rng(seed); archive: list[np.ndarray] = []; audits = []",
        "rng = np.random.default_rng(seed); archive: list[np.ndarray] = []; "
        "archive_quality: list[float] = []; audits = []",
        "genetic archive quality",
    )
    genetic = replace_once(
        genetic,
        "        for index in np.where(valid)[0][np.argsort(score[valid])]:\n"
        "            if _distinct(candidate[index], archive, projector, config.distinctness):\n"
        "                archive.append(candidate[index].copy())\n"
        "                if len(archive) == config.k:\n"
        "                    break",
        "        _archive_update(\n"
        "            archive, archive_quality, candidate, score, valid, projector,\n"
        "            config.distinctness,\n"
        "        )",
        "genetic full archive",
    )
    genetic = replace_once(
        genetic,
        "        if len(archive) >= config.k:\n            break\n",
        "",
        "genetic full cap",
    )
    genetic = replace_once(
        genetic,
        "    returned = (np.vstack(archive) if archive\n"
        "                else np.empty((0, len(query)), np.float32))",
        "    returned = _mmr_select(\n"
        "        archive, archive_quality, projector, config.k,\n"
        "        diversity_weight=config.set_diversity_weight,\n"
        "    )",
        "genetic MMR",
    )
    source = source[:genetic_start] + genetic + source[dice_start:]
    return source


def patch_tuning(source: str, cfg: dict) -> str:
    source = source.replace("PRUNE_INNER_PER_DIRECTION = 10", "PRUNE_INNER_PER_DIRECTION = 4")
    source = re.sub(
        r"POPULATION_SENSITIVITY = \[[^\n]+\]",
        f"POPULATION_SENSITIVITY = {cfg['population_grid']!r}", source,
    )
    source = re.sub(
        r"POPULATION_SENSITIVITY_BUDGET = \d+",
        f"POPULATION_SENSITIVITY_BUDGET = {cfg['operating_budget']}", source,
    )
    source = re.sub(
        r"PRUNE_INNER_BUDGET = \d+",
        f"PRUNE_INNER_BUDGET = {cfg['operating_budget']}", source,
    )
    source = re.sub(
        r"SD_BUDGET = \d+",
        f"SD_BUDGET = {cfg['operating_budget']}", source,
    )
    start = source.index("def load_released_discriminator_scorers")
    end_marker = "inner_factuals = calibration_queries("
    end = source.index(end_marker, start)
    source = source[:start] + (
        "# V5.5 deployment releases and queries only the generator.  The GAN\n"
        "# discriminator remains a training-only component and is never loaded\n"
        "# by profile tuning, population selection or outer CFE search.\n"
        "canonical_plausibility_scorers = {}\n"
    ) + source[end:]
    source = source.replace("plausibility_weight=.15", "plausibility_weight=0.0")
    source = re.sub(
        r",?\n\s*plausibility_scorer=canonical_plausibility_scorers\[[^\n]+\]",
        "", source,
    )
    old_selection = '''selected_population = eligible_populations.sort_values(
    ["population_selection_score", "valid_cfe_yield_at_k", "population"],
    ascending=[False, False, True],
).iloc[0]
SD_POPULATION = int(selected_population.population)'''
    new_selection = f'''# Population is prospectively frozen from the completed V5.4 inner
# frontier with an explicit encrypted-round cost.  This V5.5 run reports the
# full sensitivity grid but cannot retune the deployment population on its
# own outer-test results.
PREDECLARED_HE_POPULATION = {cfg["population"]}
SD_POPULATION = PREDECLARED_HE_POPULATION
selected_population = population_round_summary.loc[
    population_round_summary.population.astype(int).eq(SD_POPULATION)
].iloc[0]'''
    source = replace_once(source, old_selection, new_selection, "frozen HE population")
    source = source.replace(
        '"population_selection_scope": "inner validation only",',
        '"population_selection_scope": "prospectively frozen from V5.4 inner validation plus HE round cost",',
    )
    source = source.replace(
        '"search_plausibility_source": "jointly-accounted released discriminator",',
        '"search_plausibility_source": "none; generator-only deployment",',
    )
    source = source.replace(
        '"raw_training_knn_used_during_dp_search": False,',
        '"raw_training_knn_used_during_dp_search": False,\n'
        '    "discriminator_released_or_queried_at_inference": False,',
    )
    source = source.replace(
        '["CGI", "proposed_countergan", "fixed counted forward-score budget", True],',
        '["CGI", "proposed_countergan", "native early-stop iterative generator baseline", True],\n'
        '    ["CG-SD", "countergan_sparse_diverse_no_dp", "non-DP generator + identical fixed-cap prune/MMR", True],',
    )
    return source


def build(dataset: str, mode: str) -> Path:
    cfg = CONFIG[dataset]
    nb = nbformat.read(cfg["source"], as_version=4)
    mode_title = "Smoke" if mode == "smoke" else "One-Seed Paper"
    protocol = "Q1_V55_GENERATOR_ONLY_SMOKE" if mode == "smoke" else "Q1_V55_GENERATOR_ONLY_ONE_SEED_PAPER"
    title = (cfg["title"].format(mode_title=mode_title)
             if mode == "smoke" else cfg["paper_title"])
    # New Kaggle kernel titles are limited to 50 characters. Paper uses a
    # shorter display title and its exact independently frozen slug.
    slug = (cfg["slug"].format(mode="smoke")
            if mode == "smoke" else cfg["paper_slug"])
    folder = ROOT / f"kaggle/notebooks/{dataset}/q1_v55_generator_only_{mode}"
    notebook_name = f"{dataset}_q1_v55_generator_only_{mode}_t4.ipynb"

    replace_all_cells(nb, "Q1_V54_PAPER_DEPTH_PILOT", protocol)
    replace_all_cells(nb, "Q1-V5.4", "Q1-V5.5")
    replace_all_cells(nb, "q1v54_paper_depth_pilot", f"q1v55_generator_only_{mode}")
    replace_all_cells(nb, "paper_depth_pilot_acceptance.json", f"v55_{mode}_acceptance.json")
    # The V5.4 source described a discriminator-release deployment. V5.5 keeps
    # the discriminator only inside the private training/audit workspace. Its
    # attack is a hypothetical release ablation, never part of the primary
    # generator-only attack surface. Formal accounting deliberately remains
    # conservative over both optimized GAN components.
    replace_all_cells(
        nb,
        '"dp_scope": ("record-level conditional GAN training; generator and "\n'
        '                     "discriminator are jointly released under basic composed "\n'
        '                     "accounting; oracle and preprocessing are public auxiliary"),',
        '"dp_scope": ("record-level conditional GAN training; only the generator "\n'
        '                     "is deployed, while generator and discriminator optimizer "\n'
        '                     "steps remain conservatively covered by basic composed "\n'
        '                     "accounting; oracle and preprocessing are public auxiliary"),',
    )
    replace_all_cells(
        nb,
        '"released_discriminator_attack_included": True,',
        '"deployment_release": "generator_only",\n'
        '        "released_discriminator_primary_attack_surface": False,\n'
        '        "discriminator_release_ablation_included": True,',
    )
    replace_all_cells(
        nb,
        '"discriminator_release_role": (\n'
        '                    "bounded realism scorer used only as post-processing of the "\n'
        '                    "jointly accounted generator+discriminator DP release"\n'
        '                ),',
        '"discriminator_release_role": (\n'
        '                    "training-only internal audit state; excluded from the "\n'
        '                    "generator-only deployment package"\n'
        '                ),',
    )
    nb.cells[0].source = f'''# {title}

Prospective generator-only protocol.  The discriminator is training-only and
is neither released nor queried during CFE search.  All methods use paired
base seeds and paired robustness perturbations.  The HE-aware deployment
population is frozen before outer evaluation; missing K slots remain failures.'''

    setup = cell_with(nb, "NOTEBOOK_STARTED =")
    setup.source = re.sub(r'RUN_MODE = "[^"]+"', f'RUN_MODE = "{mode}"', setup.source)

    embedded = cell_with(nb, "%%writefile /kaggle/working/plaintext_cfe_pipeline.py")
    prefix, pipeline = embedded.source.split("\n", 1)
    embedded.source = prefix + "\n" + patch_pipeline(pipeline)

    classifier = cell_with(nb, "max_epochs =")
    if mode == "smoke":
        classifier.source = re.sub(
            r'max_epochs = \{"ecg": 35, "heartplus": 60, "mimic": 60\}\[DATASET\]',
            'max_epochs = {"ecg": 4, "heartplus": 6, "mimic": 4}[DATASET]',
            classifier.source,
        )
        classifier.source = classifier.source.replace("patience=8", "patience=3")

    training = cell_with(nb, "SCHEDULE = dict(")
    if mode == "smoke":
        start = training.source.index("SCHEDULE = dict(")
        end = training.source.index("\nprivacy_rows =", start)
        training.source = training.source[:start] + '''SCHEDULE = dict(
    epochs=12, batch_size=256, steps_per_epoch=10,
    secure_rng=True,
    warmup_epochs=2, ramp_epochs=7,
    classification_weight_start=8.0,
    classification_weight_end=12.0,
    proximity_weight_start=.04, proximity_weight_end=.20,
    sparsity_weight_start=.01, sparsity_weight_end=.10,
    group_sparsity_weight_start=0.0, group_sparsity_weight_end=.15,
    diversity_weight_start=0.0, diversity_weight_end=.08,
    direction_balance_weighting=True,
)''' + training.source[end:]
    else:
        epochs, steps, warmup, ramp = cfg["paper_gan"]
        training.source = re.sub(r"epochs=\d+, batch_size=256, steps_per_epoch=\d+", f"epochs={epochs}, batch_size=256, steps_per_epoch={steps}", training.source)
        training.source = re.sub(r"warmup_epochs=\d+, ramp_epochs=\d+", f"warmup_epochs={warmup}, ramp_epochs={ramp}", training.source)
    training.source = training.source.replace(
        '"v54_paper_depth_direction_balanced_warmup_ramp"',
        '"v55_generator_only_direction_balanced_warmup_ramp"',
    )

    handoff = cell_with(nb, "HANDOFF = OUTPUT")
    handoff.source = re.sub(
        r'"paper_budgets": \[[^\]]+\]',
        f'"paper_budgets": {cfg["budgets"]!r}', handoff.source,
    )
    handoff.source = re.sub(
        r'"paper_queries_per_direction": \d+',
        f'"paper_queries_per_direction": {6 if mode == "smoke" else 100}', handoff.source,
    )

    cohort = cell_with(nb, "K_PRIMARY = 10")
    cohort.source = re.sub(r"BUDGETS = \[[^\n]+\]", f"BUDGETS = {cfg['budgets']!r}", cohort.source)
    cohort.source = re.sub(r"PER_DIRECTION = \d+", f"PER_DIRECTION = {6 if mode == 'smoke' else 100}", cohort.source)
    cohort.source = re.sub(r"BASE_POPULATION = \d+", f"BASE_POPULATION = {cfg['population']}", cohort.source)
    cohort.source = cohort.source.replace(
        '"proposed_countergan",',
        '"proposed_countergan", "countergan_sparse_diverse_no_dp",',
    )

    tuning = cell_with(nb, "PRUNE_INNER_PER_DIRECTION =")
    tuning.source = patch_tuning(tuning.source, cfg)

    contract = cell_with(nb, "sd_contract = {")
    contract.source = contract.source.replace(
        '"population_selection_scope": "inner validation only",',
        '"population_selection_scope": "prospectively frozen from V5.4 inner validation plus HE round cost",',
    ).replace(
        '"search_plausibility_source": "jointly-accounted released discriminator",',
        '"search_plausibility_source": "none; generator-only deployment",',
    ).replace(
        '"raw_training_knn_used_during_dp_search": False,',
        '"raw_training_knn_used_during_dp_search": False,\n'
        '    "discriminator_released_or_queried_at_inference": False,',
    )

    primary = cell_with(nb, "detail_parts, round_parts = [], []")
    scorer_start = primary.source.index("    discriminator_scorers = (")
    scorer_end = primary.source.index("    learned_detail, _, learned_rounds", scorer_start)
    primary.source = primary.source[:scorer_start] + primary.source[scorer_end:]
    primary.source = re.sub(
        r"\n\s*plausibility_scorers=method_plausibility_scorers,\n\s*require_private_reference_free_dp_search=True,",
        "", primary.source,
    )
    primary.source = primary.source.replace(
        "robustness_repetitions=8", f"robustness_repetitions={3 if mode == 'smoke' else 8}"
    )

    release = cell_with(nb, "# Recompute the selected B")
    release.source = re.sub(
        r"\n\s*plausibility_scorers=\{.*?\n\s*\},\n\s*require_private_reference_free_dp_search=True,",
        "", release.source, flags=re.S,
    )
    release.source = release.source.replace(
        "robustness_repetitions=8", f"robustness_repetitions={3 if mode == 'smoke' else 8}"
    )
    release.source = release.source.replace(
        "representatives_per_direction=3",
        f"representatives_per_direction={3 if mode == 'smoke' else 5}",
    )
    release.source = release.source.replace(
        "sample_size=100", f"sample_size={100 if mode == 'smoke' else 1000}"
    )

    privacy = cell_with(nb, "security_v2_contract = AttackContract(")
    if mode == "paper":
        start = privacy.source.index("security_v2_contract = AttackContract(")
        end = privacy.source.index("\nrun_privacy_v2(", start)
        privacy.source = privacy.source[:start] + "security_v2_contract = AttackContract()" + privacy.source[end:]
    old = '''mia_auc = mia_v2_summary[mia_v2_summary.metric.eq("evaluation_auc")].copy()
strongest_mia_v2 = (
    mia_auc.sort_values(["variant", "mean"], ascending=[True, False])
    .groupby("variant", as_index=False).head(1).reset_index(drop=True)
)'''
    new = '''mia_auc = mia_v2_summary[mia_v2_summary.metric.eq("evaluation_auc")].copy()
generator_only_mia_auc = mia_auc[
    ~mia_auc.attack.eq("released_discriminator_score")
].copy()
strongest_mia_v2 = (
    generator_only_mia_auc.sort_values(
        ["variant", "mean"], ascending=[True, False]
    ).groupby("variant", as_index=False).head(1).reset_index(drop=True)
)
discriminator_release_ablation = mia_auc[
    mia_auc.attack.eq("released_discriminator_score")
].reset_index(drop=True)'''
    privacy.source = replace_once(privacy.source, old, new, "generator-only MIA summary")
    privacy.source = privacy.source.replace(
        "display(strongest_mia_v2)",
        "display(strongest_mia_v2)\n"
        "display(discriminator_release_ablation)",
    )

    manuscript = cell_with(nb, "MANUSCRIPT_DIR =")
    manuscript.source = manuscript.source.replace(
        'save_show("06_3", "strongest_mia_v2", strongest_mia_v2)',
        'save_show("06_3", "strongest_generator_only_mia_v2", strongest_mia_v2)\n'
        'save_show("06_3b", "discriminator_release_mia_ablation", discriminator_release_ablation)',
    ).replace(
        '["DP-CounterGAN", "released generator/checkpoint observer",',
        '["DP-CounterGAN", "released generator/checkpoint observer; discriminator training-only",',
    )

    gate = cell_with(nb, "EXPECTED_METHODS =")
    gate.source = re.sub(
        r"PRUNE_INNER_BUDGET == SD_BUDGET == \d+",
        f"PRUNE_INNER_BUDGET == SD_BUDGET == {cfg['operating_budget']}", gate.source,
    )
    gate.source = re.sub(
        r"set\(population_round_summary.population.astype\(int\)\) == \{[^\}]+\}",
        f"set(population_round_summary.population.astype(int)) == {set(cfg['population_grid'])!r}", gate.source,
    )
    gate.source = re.sub(
        r"population_round_summary.candidate_budget.eq\(\d+\)",
        f"population_round_summary.candidate_budget.eq({cfg['operating_budget']})", gate.source,
    )
    gate.source = gate.source.replace(
        "and bool(population_round_summary.loc[\n"
        "            population_round_summary.selected_for_outer, \"yield_noninferior\"\n"
        "        ].iloc[0])",
        f"and int(BASE_POPULATION) == {cfg['population']}",
    )
    if mode == "paper":
        gate.source = gate.source.replace(
            "len(qualitative_grid) == 6", "len(qualitative_grid) == 10"
        )
        gate.source = gate.source.replace(
            "len(mia_v2_raw) == 50", "len(mia_v2_raw) == 250"
        ).replace(
            "mia_v2_raw.attack_seed.nunique() == 2", "mia_v2_raw.attack_seed.nunique() == 10"
        ).replace(
            'security_v2_report["paper_numbers"] is False',
            'security_v2_report["paper_numbers"] is True',
        )
    gate.source = gate.source.replace(
        '"paper_numbers": False,',
        f'"paper_numbers": {str(mode == "paper")},',
    )
    gate.source = gate.source.replace(
        '"privacy_all_five_generators": (',
        '"generator_only_search_contract": (\n'
        '        detail.search_plausibility_source.eq("none_generator_only").all()\n'
        '        and not sd_contract["discriminator_released_or_queried_at_inference"]\n'
        '    ),\n'
        '    "common_base_seed_and_paired_robustness_contract": (\n'
        '        {"run_seed", "robustness_seed"}.issubset(detail.columns)\n'
        '        and detail.groupby(\n'
        '            ["query_id", "method_seed", "candidate_budget"]\n'
        '        ).run_seed.nunique().eq(1).all()\n'
        '        and detail.groupby(\n'
        '            ["query_id", "method_seed", "candidate_budget"]\n'
        '        ).robustness_seed.nunique().eq(1).all()\n'
        '    ),\n'
        '    "fixed_cap_methods_consume_full_budget": (\n'
        '        detail.loc[\n'
        '            detail.method.isin({\n'
        '                "uniform_random", "genetic_cfe",\n'
        '                "countergan_sparse_diverse_no_dp",\n'
        '                "proposed_dp_eps16_sparse_diverse",\n'
        '                "proposed_dp_eps8_sparse_diverse",\n'
        '                "proposed_dp_eps4_sparse_diverse",\n'
        '                "proposed_dp_eps2_sparse_diverse",\n'
        '            }), "candidate_evaluations"\n'
        '        ].eq(detail.loc[\n'
        '            detail.method.isin({\n'
        '                "uniform_random", "genetic_cfe",\n'
        '                "countergan_sparse_diverse_no_dp",\n'
        '                "proposed_dp_eps16_sparse_diverse",\n'
        '                "proposed_dp_eps8_sparse_diverse",\n'
        '                "proposed_dp_eps4_sparse_diverse",\n'
        '                "proposed_dp_eps2_sparse_diverse",\n'
        '            }), "candidate_budget"\n'
        '        ]).all()\n'
        '    ),\n'
        '    "privacy_all_five_generators": (',
    )
    gate.source = gate.source.replace(
        'and not security_v2_report["evaluation_labels_used_for_attack_selection"]\n'
        '    ),',
        'and not security_v2_report["evaluation_labels_used_for_attack_selection"]\n'
        '        and security_v2_report["deployment_release"] == "generator_only"\n'
        '        and not security_v2_report[\n'
        '            "released_discriminator_primary_attack_surface"\n'
        '        ]\n'
        '        and security_v2_report[\n'
        '            "discriminator_release_ablation_included"\n'
        '        ]\n'
        '    ),',
    )
    gate.source = gate.source.replace(
        "paper_depth_pilot_acceptance", f"v55_{mode}_acceptance"
    )

    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.execution_count = None
            cell.outputs = []

    folder.mkdir(parents=True, exist_ok=True)
    target = folder / notebook_name
    nbformat.write(nb, target)
    metadata = {
        "id": f"{cfg['owner']}/{slug}",
        "title": title,
        "code_file": notebook_name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "keywords": ["healthcare"],
        "dataset_sources": [cfg["dataset_source"]],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
    }
    (folder / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "paper"), required=True)
    parser.add_argument("--dataset", choices=tuple(CONFIG), action="append")
    args = parser.parse_args()
    datasets = args.dataset or list(CONFIG)
    for dataset in datasets:
        print(build(dataset, args.mode))


if __name__ == "__main__":
    main()
