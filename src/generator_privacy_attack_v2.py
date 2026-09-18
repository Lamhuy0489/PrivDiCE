#!/usr/bin/env python3
"""Calibrated, repeated privacy attacks for frozen conditional CFE generators.

This program evaluates attack strength; it never tunes or retrains a released
generator.  Calibration members/nonmembers use outer-development train/valid
rows, while final attack evaluation uses disjoint train/outer-test rows.  The
outer-test attack labels never select score orientation or thresholds.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    roc_auc_score,
    roc_curve,
)
from sklearn.neighbors import NearestNeighbors


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ROOT / "experiments/cloud"))
import plaintext_cfe_pipeline as pipe  # noqa: E402


GENERATOR_FILES = {
    "countergan": "generator_seed11_countergan.pt",
    "dp_eps16": "generator_seed11_dp_eps16.pt",
    "dp_eps8": "generator_seed11_dp_eps8.pt",
    "dp_eps4": "generator_seed11_dp_eps4.pt",
    "dp_eps2": "generator_seed11_dp_eps2.pt",
}
SENSITIVE_FEATURE = {
    "ecg": "total_power",
    "heartplus": "Diabetic",
    "mimic": "Creatinine_Mean",
}


@dataclass(frozen=True)
class AttackContract:
    calibration_rows_per_group: int = 1000
    evaluation_rows_per_group: int = 1000
    latent_draws: int = 16
    attack_seeds: tuple[int, ...] = (
        101, 202, 303, 404, 505, 606, 707, 808, 909, 1010,
    )
    low_fpr_levels: tuple[float, ...] = (.001, .01, .05)
    reference_rows: int = 20_000


def _load_frozen(dataset: str, raw_csv: Path, handoff: Path):
    manifest = json.loads((handoff / "training_handoff_manifest.json").read_text())
    if manifest["dataset"] != dataset:
        raise RuntimeError("handoff dataset mismatch")
    payload = joblib.load(handoff / "preprocessor_and_split.joblib")
    raw = pipe.load_dataset(dataset, raw_csv)
    split, pre = payload["split"], payload["preprocessor"]
    y = raw.frame[raw.target_col].to_numpy(np.int8)
    prepared = pipe.PreparedData(
        raw=raw, split=split, preprocessor=pre,
        x_train=pre.transform(raw.frame.iloc[split.train_idx]),
        y_train=y[split.train_idx],
        x_valid=pre.transform(raw.frame.iloc[split.valid_idx]),
        y_valid=y[split.valid_idx],
        x_test=pre.transform(raw.frame.iloc[split.test_idx]),
        y_test=y[split.test_idx],
        feature_names=list(payload["feature_names"]),
    )
    canonical_seed = int(manifest["canonical_classifier_seed"])
    classifier_checkpoint = torch.load(
        handoff / f"classifier_seed{canonical_seed}.pt",
        map_location="cpu", weights_only=False,
    )
    cfg = classifier_checkpoint["classifier_config"]
    oracle = pipe.HEPolynomialMLP(
        classifier_checkpoint["input_dim"], cfg["h1"], cfg["h2"],
        cfg["activation_kind"], cfg["alpha"], cfg["dropout"],
    )
    oracle.load_state_dict(classifier_checkpoint["state_dict"])
    generators, discriminators = {}, {}
    for variant, filename in GENERATOR_FILES.items():
        checkpoint = torch.load(handoff / filename, map_location="cpu", weights_only=False)
        generator = pipe.ResidualGenerator(checkpoint["input_dim"])
        generator.load_state_dict(checkpoint["state_dict"])
        generators[variant] = generator
        if "discriminator_state_dict" not in checkpoint:
            raise RuntimeError(
                f"{filename} predates V5.2 and has no released discriminator"
            )
        discriminator = pipe.PlausibilityDiscriminator(checkpoint["input_dim"])
        discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        discriminators[variant] = discriminator
    return (prepared, oracle, generators, discriminators,
            pipe.DomainProjector(prepared), manifest)


def _matched_indices(member_y: np.ndarray, nonmember_y: np.ndarray, number: int,
                     rng: np.random.Generator,
                     excluded_member: set[int] | None = None):
    excluded_member = excluded_member or set()
    member_available = {
        label: np.asarray([
            index for index in np.where(member_y == label)[0]
            if int(index) not in excluded_member
        ], dtype=int)
        for label in (0, 1)
    }
    nonmember_available = {
        label: np.where(nonmember_y == label)[0] for label in (0, 1)
    }
    prevalence = .5 * (float(member_y.mean()) + float(nonmember_y.mean()))
    desired_one = int(round(number * prevalence))
    low_one = max(0, number - min(
        len(member_available[0]), len(nonmember_available[0]),
    ))
    high_one = min(
        number, len(member_available[1]), len(nonmember_available[1]),
    )
    one = int(np.clip(desired_one, low_one, high_one))
    counts = {0: number - one, 1: one}
    if any(counts[label] > min(
        len(member_available[label]), len(nonmember_available[label]),
    ) for label in (0, 1)):
        raise RuntimeError("insufficient matched attack rows")
    member = np.concatenate([
        rng.choice(member_available[label], counts[label], replace=False)
        for label in (0, 1)
    ])
    nonmember = np.concatenate([
        rng.choice(nonmember_available[label], counts[label], replace=False)
        for label in (0, 1)
    ])
    rng.shuffle(member); rng.shuffle(nonmember)
    return member, nonmember, counts


def _objectives(generator: torch.nn.Module, oracle: torch.nn.Module,
                projector: pipe.DomainProjector, x: np.ndarray, y: np.ndarray,
                device: str, seed: int, draws: int, batch_size: int = 8192):
    generator.eval(); oracle.eval()
    rng = np.random.default_rng(seed)
    repeated_x = np.repeat(np.asarray(x, np.float32), draws, axis=0)
    repeated_y = np.repeat(np.asarray(y, np.int8), draws)
    generated_parts = []
    latent_dim = generator.latent_dim
    with torch.no_grad():
        for start in range(0, len(repeated_x), batch_size):
            xb_np = repeated_x[start:start + batch_size]
            yb_np = repeated_y[start:start + batch_size]
            xb = torch.as_tensor(xb_np, dtype=torch.float32, device=device)
            desired = torch.as_tensor(1 - yb_np, dtype=torch.float32, device=device)
            z = torch.as_tensor(
                rng.normal(size=(len(xb_np), latent_dim)),
                dtype=torch.float32, device=device,
            )
            generated_parts.append(generator(xb, desired, z).cpu().numpy())
    generated = projector.project(np.vstack(generated_parts), repeated_x)
    logits = pipe.logits_numpy(oracle, generated, device)
    desired = 1 - repeated_y
    target_penalty = np.where(
        desired == 1, np.maximum(0., .10 - logits),
        np.maximum(0., .10 + logits),
    )
    objective = (
        projector.distance(generated, repeated_x) + target_penalty
    ).reshape(len(x), draws)
    cube = generated.reshape(len(x), draws, generated.shape[1])
    best_index = np.argmin(objective, axis=1)
    best = cube[np.arange(len(x)), best_index]
    return {
        "single": objective[:, 0],
        "minimum": objective.min(axis=1),
        "mean": objective.mean(axis=1),
        "best_candidate": best,
    }


def _discriminator_scores(discriminator: torch.nn.Module, x: np.ndarray,
                          device: str, batch_size: int = 8192) -> np.ndarray:
    """LOGAN-style released-discriminator membership score on factual rows."""
    discriminator.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            values = torch.as_tensor(
                np.asarray(x[start:start + batch_size], np.float32),
                dtype=torch.float32, device=device,
            )
            parts.append(discriminator(values).cpu().numpy())
    return np.concatenate(parts).astype(np.float64)


def _threshold_metrics(cal_member_score: np.ndarray,
                       cal_nonmember_score: np.ndarray,
                       eval_member_score: np.ndarray,
                       eval_nonmember_score: np.ndarray,
                       fpr_levels: tuple[float, ...]):
    cal_labels = np.r_[np.ones(len(cal_member_score)),
                       np.zeros(len(cal_nonmember_score))]
    cal_score = np.r_[cal_member_score, cal_nonmember_score]
    raw_cal_auc = float(roc_auc_score(cal_labels, cal_score))
    orientation = 1 if raw_cal_auc >= .5 else -1
    cal_score *= orientation
    fpr_cal, tpr_cal, threshold_cal = roc_curve(cal_labels, cal_score)
    finite = np.isfinite(threshold_cal)
    youden = tpr_cal - fpr_cal
    youden[~finite] = -np.inf
    selected = int(np.argmax(youden))
    threshold = float(threshold_cal[selected])

    eval_labels = np.r_[np.ones(len(eval_member_score)),
                        np.zeros(len(eval_nonmember_score))]
    eval_score = orientation * np.r_[eval_member_score, eval_nonmember_score]
    prediction = (eval_score >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(eval_labels, prediction, labels=[0, 1]).ravel()
    eval_auc = float(roc_auc_score(eval_labels, eval_score))
    fpr_eval, tpr_eval, _ = roc_curve(eval_labels, eval_score)
    result = {
        "calibration_raw_auc": raw_cal_auc,
        "orientation_selected_on_calibration": int(orientation),
        "calibration_threshold": threshold,
        "evaluation_auc": eval_auc,
        "evaluation_accuracy": float(accuracy_score(eval_labels, prediction)),
        "evaluation_balanced_accuracy": float(
            balanced_accuracy_score(eval_labels, prediction)
        ),
        "evaluation_tpr_at_calibrated_threshold": float(tp / max(tp + fn, 1)),
        "evaluation_fpr_at_calibrated_threshold": float(fp / max(fp + tn, 1)),
    }
    result["evaluation_membership_advantage"] = (
        result["evaluation_tpr_at_calibrated_threshold"]
        - result["evaluation_fpr_at_calibrated_threshold"]
    )
    for level in fpr_levels:
        permitted = tpr_eval[fpr_eval <= level + 1e-12]
        result[f"tpr_at_fpr_{level:g}"] = (
            float(permitted.max()) if len(permitted) else 0.0
        )
    return result


def _gaussian_llr(member: np.ndarray, nonmember: np.ndarray,
                  values: np.ndarray) -> np.ndarray:
    def logpdf(reference: np.ndarray):
        mean = float(np.mean(reference))
        sd = max(float(np.std(reference, ddof=1)), 1e-6)
        return -np.log(sd) - .5 * ((values - mean) / sd) ** 2
    return logpdf(member) - logpdf(nonmember)


def _distance_audit(candidate: np.ndarray, projector: pipe.DomainProjector,
                    train_reference: np.ndarray, holdout_reference: np.ndarray):
    normalized_candidate = candidate / projector.scale
    train_nn = NearestNeighbors(n_neighbors=2).fit(
        train_reference / projector.scale
    )
    holdout_nn = NearestNeighbors(n_neighbors=1).fit(
        holdout_reference / projector.scale
    )
    train_distance, _ = train_nn.kneighbors(normalized_candidate)
    holdout_distance, _ = holdout_nn.kneighbors(normalized_candidate)
    dcr = train_distance[:, 0] / math.sqrt(candidate.shape[1])
    holdout_dcr = holdout_distance[:, 0] / math.sqrt(candidate.shape[1])
    nndr = train_distance[:, 0] / np.maximum(train_distance[:, 1], 1e-12)
    return {
        "exact_match_rate": float(np.mean(dcr <= 1e-6)),
        "near_duplicate_rate_0_001": float(np.mean(dcr <= .001)),
        "near_duplicate_rate_0_01": float(np.mean(dcr <= .01)),
        "dcr_mean": float(np.mean(dcr)),
        "dcr_median": float(np.median(dcr)),
        "dcr_p01": float(np.quantile(dcr, .01)),
        "dcr_p05": float(np.quantile(dcr, .05)),
        "nndr_mean": float(np.mean(nndr)),
        "nndr_median": float(np.median(nndr)),
        "train_to_holdout_dcr_ratio_median": float(np.median(
            dcr / np.maximum(holdout_dcr, 1e-12)
        )),
    }


def _attribute_attack(feature_names: list[str], sensitive: str,
                      cal_candidate: np.ndarray, cal_factual: np.ndarray,
                      eval_candidate: np.ndarray, eval_factual: np.ndarray):
    index = feature_names.index(sensitive)
    keep = [i for i in range(len(feature_names)) if i != index]
    y_cal = cal_factual[:, index]
    y_eval = eval_factual[:, index]
    unique = np.unique(np.round(y_cal, 6))
    if len(unique) <= 10:
        cal_label = np.abs(y_cal[:, None] - unique[None, :]).argmin(axis=1)
        eval_label = np.abs(y_eval[:, None] - unique[None, :]).argmin(axis=1)
        model = LogisticRegression(
            max_iter=1000, class_weight="balanced", solver="lbfgs",
        ).fit(cal_candidate[:, keep], cal_label)
        prediction = model.predict(eval_candidate[:, keep])
        majority = np.full_like(eval_label, np.bincount(cal_label).argmax())
        return {
            "sensitive_feature": sensitive,
            "attribute_kind": "categorical_or_ordinal",
            "attack_metric": "balanced_accuracy",
            "attack_value": float(balanced_accuracy_score(eval_label, prediction)),
            "trivial_baseline_value": float(
                balanced_accuracy_score(eval_label, majority)
            ),
        }
    model = Ridge(alpha=1.0).fit(cal_candidate[:, keep], y_cal)
    prediction = model.predict(eval_candidate[:, keep])
    baseline = np.full_like(y_eval, np.median(y_cal))
    scale = max(float(np.ptp(y_cal)), .10)
    return {
        "sensitive_feature": sensitive,
        "attribute_kind": "continuous",
        "attack_metric": "normalized_mae",
        "attack_value": float(mean_absolute_error(y_eval, prediction) / scale),
        "trivial_baseline_value": float(
            mean_absolute_error(y_eval, baseline) / scale
        ),
    }


def _seed_summary(frame: pd.DataFrame, group_columns: list[str],
                  metric_columns: list[str], bootstrap_seed: int):
    rng = np.random.default_rng(bootstrap_seed)
    rows = []
    for keys, group in frame.groupby(group_columns):
        keys = keys if isinstance(keys, tuple) else (keys,)
        base = dict(zip(group_columns, keys))
        for metric in metric_columns:
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy()
            if not len(values):
                continue
            samples = rng.choice(values, size=(5000, len(values)), replace=True).mean(axis=1)
            rows.append({
                **base, "metric": metric, "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.,
                "ci95_low": float(np.quantile(samples, .025)),
                "ci95_high": float(np.quantile(samples, .975)),
                "attack_seeds": int(len(values)),
            })
    return pd.DataFrame(rows)


def run(dataset: str, raw_csv: Path, handoff: Path, output: Path,
        device: str, contract: AttackContract):
    output.mkdir(parents=True, exist_ok=True)
    prepared, oracle, generators, discriminators, projector, manifest = _load_frozen(
        dataset, raw_csv, handoff
    )
    oracle.to(device).eval()
    generators = {name: model.to(device).eval() for name, model in generators.items()}
    discriminators = {
        name: model.to(device).eval() for name, model in discriminators.items()
    }
    reference_rng = np.random.default_rng(88001)
    train_reference = prepared.x_train[reference_rng.choice(
        len(prepared.x_train), min(contract.reference_rows, len(prepared.x_train)),
        replace=False,
    )]
    holdout_reference = prepared.x_valid[reference_rng.choice(
        len(prepared.x_valid), min(contract.reference_rows, len(prepared.x_valid)),
        replace=False,
    )]
    attack_rows, distance_rows, attribute_rows, sample_rows = [], [], [], []

    for attack_seed in contract.attack_seeds:
        rng = np.random.default_rng(attack_seed)
        cal_member_ix, cal_nonmember_ix, cal_counts = _matched_indices(
            prepared.y_train, prepared.y_valid,
            contract.calibration_rows_per_group, rng,
        )
        eval_member_ix, eval_nonmember_ix, eval_counts = _matched_indices(
            prepared.y_train, prepared.y_test,
            contract.evaluation_rows_per_group, rng,
            excluded_member=set(map(int, cal_member_ix)),
        )
        sets = {
            "cal_member": (prepared.x_train[cal_member_ix], prepared.y_train[cal_member_ix]),
            "cal_nonmember": (prepared.x_valid[cal_nonmember_ix], prepared.y_valid[cal_nonmember_ix]),
            "eval_member": (prepared.x_train[eval_member_ix], prepared.y_train[eval_member_ix]),
            "eval_nonmember": (prepared.x_test[eval_nonmember_ix], prepared.y_test[eval_nonmember_ix]),
        }
        sample_rows.extend([
            {"attack_seed": attack_seed, "stage": "calibration", "membership": name,
             "rows": contract.calibration_rows_per_group, "class_0": cal_counts[0],
             "class_1": cal_counts[1]}
            for name in ("member", "nonmember")
        ] + [
            {"attack_seed": attack_seed, "stage": "evaluation", "membership": name,
             "rows": contract.evaluation_rows_per_group, "class_0": eval_counts[0],
             "class_1": eval_counts[1]}
            for name in ("member", "nonmember")
        ])

        for variant, generator in generators.items():
            scored = {}
            for set_number, (set_name, (x, y)) in enumerate(sets.items()):
                scored[set_name] = _objectives(
                    generator, oracle, projector, x, y, device,
                    # Common random numbers pair epsilon checkpoints and reduce
                    # Monte-Carlo noise without using evaluation labels.
                    seed=attack_seed * 1009 + set_number,
                    draws=contract.latent_draws,
                )
            score_types = {
                "single_draw_loss_threshold": "single",
                "multi_draw_reconstruction": "minimum",
                "multi_draw_mean_objective": "mean",
            }
            multi_draw_auc = np.nan
            for attack_name, score_key in score_types.items():
                metric = _threshold_metrics(
                    -scored["cal_member"][score_key],
                    -scored["cal_nonmember"][score_key],
                    -scored["eval_member"][score_key],
                    -scored["eval_nonmember"][score_key],
                    contract.low_fpr_levels,
                )
                attack_rows.append({
                    "dataset": dataset, "variant": variant,
                    "attack": attack_name, "attack_seed": attack_seed,
                    "calibration_rows_per_group": contract.calibration_rows_per_group,
                    "evaluation_rows_per_group": contract.evaluation_rows_per_group,
                    "latent_draws": contract.latent_draws, **metric,
                })
                if attack_name == "multi_draw_reconstruction":
                    multi_draw_auc = metric["evaluation_auc"]

            cal_member_score = -scored["cal_member"]["minimum"]
            cal_nonmember_score = -scored["cal_nonmember"]["minimum"]
            cal_llr = _gaussian_llr(
                cal_member_score, cal_nonmember_score,
                np.r_[cal_member_score, cal_nonmember_score],
            )
            eval_member_score = -scored["eval_member"]["minimum"]
            eval_nonmember_score = -scored["eval_nonmember"]["minimum"]
            eval_llr = _gaussian_llr(
                cal_member_score, cal_nonmember_score,
                np.r_[eval_member_score, eval_nonmember_score],
            )
            metric = _threshold_metrics(
                cal_llr[:len(cal_member_score)], cal_llr[len(cal_member_score):],
                eval_llr[:len(eval_member_score)], eval_llr[len(eval_member_score):],
                contract.low_fpr_levels,
            )
            attack_rows.append({
                "dataset": dataset, "variant": variant,
                "attack": "calibrated_gaussian_reconstruction_likelihood",
                "attack_seed": attack_seed,
                "calibration_rows_per_group": contract.calibration_rows_per_group,
                "evaluation_rows_per_group": contract.evaluation_rows_per_group,
                "latent_draws": contract.latent_draws, **metric,
            })

            discriminator_score = {
                name: _discriminator_scores(
                    discriminators[variant], values[0], device,
                )
                for name, values in sets.items()
            }
            metric = _threshold_metrics(
                discriminator_score["cal_member"],
                discriminator_score["cal_nonmember"],
                discriminator_score["eval_member"],
                discriminator_score["eval_nonmember"],
                contract.low_fpr_levels,
            )
            attack_rows.append({
                "dataset": dataset, "variant": variant,
                "attack": "released_discriminator_score",
                "attack_seed": attack_seed,
                "calibration_rows_per_group": contract.calibration_rows_per_group,
                "evaluation_rows_per_group": contract.evaluation_rows_per_group,
                "latent_draws": 0, **metric,
            })

            distance_rows.append({
                "dataset": dataset, "variant": variant,
                "attack_seed": attack_seed,
                **_distance_audit(
                    scored["eval_nonmember"]["best_candidate"], projector,
                    train_reference, holdout_reference,
                ),
            })
            attribute_rows.append({
                "dataset": dataset, "variant": variant,
                "attack_seed": attack_seed,
                **_attribute_attack(
                    prepared.feature_names, SENSITIVE_FEATURE[dataset],
                    scored["cal_member"]["best_candidate"],
                    sets["cal_member"][0],
                    scored["eval_nonmember"]["best_candidate"],
                    sets["eval_nonmember"][0],
                ),
            })
            print({
                "dataset": dataset, "variant": variant, "seed": attack_seed,
                "multi_draw_auc": round(float(multi_draw_auc), 4),
            }, flush=True)

    attack = pd.DataFrame(attack_rows)
    distance = pd.DataFrame(distance_rows)
    attribute = pd.DataFrame(attribute_rows)
    samples = pd.DataFrame(sample_rows)
    attack_metrics = [
        "evaluation_auc", "evaluation_accuracy", "evaluation_balanced_accuracy",
        "evaluation_membership_advantage", "evaluation_tpr_at_calibrated_threshold",
        "evaluation_fpr_at_calibrated_threshold",
        *[f"tpr_at_fpr_{level:g}" for level in contract.low_fpr_levels],
    ]
    attack_summary = _seed_summary(
        attack, ["dataset", "variant", "attack"], attack_metrics, 424242,
    )
    distance_summary = _seed_summary(
        distance, ["dataset", "variant"],
        [column for column in distance.columns if column not in {
            "dataset", "variant", "attack_seed",
        }], 434343,
    )
    attribute_summary = _seed_summary(
        attribute, ["dataset", "variant", "attribute_kind", "attack_metric"],
        ["attack_value", "trivial_baseline_value"], 444444,
    )
    attack.to_csv(output / "mia_v2_by_attack_seed.csv", index=False)
    attack_summary.to_csv(output / "mia_v2_summary.csv", index=False)
    distance.to_csv(output / "memorization_distance_by_seed.csv", index=False)
    distance_summary.to_csv(output / "memorization_distance_summary.csv", index=False)
    attribute.to_csv(output / "attribute_inference_by_seed.csv", index=False)
    attribute_summary.to_csv(output / "attribute_inference_summary.csv", index=False)
    samples.to_csv(output / "attack_sample_accounting.csv", index=False)
    privacy = pd.read_csv(handoff / "privacy_accounting.csv")
    privacy.to_csv(output / "frozen_privacy_accounting.csv", index=False)
    full_paper_contract = (
        contract.calibration_rows_per_group >= 1000
        and contract.evaluation_rows_per_group >= 1000
        and contract.latent_draws >= 16
        and len(contract.attack_seeds) >= 10
    )
    report = {
        "protocol": "GENERATOR_PRIVACY_ATTACK_V2",
        "dataset": dataset,
        "paper_numbers": bool(full_paper_contract),
        "outer_test_used_for_attack_evaluation_only": True,
        "generator_training_seed_count": len(manifest["generator_training_seeds"]),
        "generator_training_seeds": manifest["generator_training_seeds"],
        "contract": asdict(contract),
        "score_orientation_selection": "calibration only",
        "threshold_selection": "calibration Youden J only",
        "evaluation_labels_used_for_attack_selection": False,
        "dp_scope": ("record-level conditional GAN training; generator and "
                     "discriminator are jointly released under basic composed "
                     "accounting; oracle and preprocessing are public auxiliary"),
        "mia_common_random_numbers_across_variants": True,
        "released_discriminator_attack_included": True,
        "attribute_inference_scope": (
            "query-attribute leakage through conditional CFE outputs; this is "
            "not evidence about training membership and is not guaranteed by DP-SGD"
        ),
        "secure_rng_limitation": bool(not privacy["secure_rng"].fillna(False).all()),
        "patient_level_dp": False,
        "formal_dp_is_primary": True,
        "empirical_attack_interpretation": "attack-specific evidence; never a universal privacy proof",
        "all_expected_rows": len(attack) == (
            len(contract.attack_seeds) * len(GENERATOR_FILES) * 5
        ),
    }
    (output / "privacy_attack_v2_contract.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8",
    )
    print(attack_summary.to_string(index=False))
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["ecg", "heartplus", "mimic"], required=True)
    parser.add_argument("--raw-csv", type=Path, required=True)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--calibration-rows", type=int, default=1000)
    parser.add_argument("--evaluation-rows", type=int, default=1000)
    parser.add_argument("--latent-draws", type=int, default=16)
    parser.add_argument("--attack-seeds", type=int, nargs="+", default=list(AttackContract().attack_seeds))
    args = parser.parse_args()
    contract = AttackContract(
        calibration_rows_per_group=args.calibration_rows,
        evaluation_rows_per_group=args.evaluation_rows,
        latent_draws=args.latent_draws,
        attack_seeds=tuple(args.attack_seeds),
    )
    run(args.dataset, args.raw_csv, args.handoff, args.output, args.device, contract)


if __name__ == "__main__":
    main()
