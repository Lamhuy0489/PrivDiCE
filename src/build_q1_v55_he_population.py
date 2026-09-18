"""Build checksum-locked CPU CKKS population notebooks for Q1 V5.5.

The notebooks consume accepted plaintext paper outputs as Kaggle kernel
sources.  They never retrain the classifier or generator.  Smoke runs one
timing repetition for one deterministic candidate cohort; paper runs three
candidate cohorts and five timing repetitions per population.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_SOURCE = ROOT / "experiments/cloud/he_population_benchmark.py"

CONFIG = {
    "ecg": {
        "title": "Leipzig ECG V5.5 HE Population {mode}",
        "slug": "leipzig-ecg-v5-5-he-population-{mode}",
        "owner": "REDACTED_KAGGLE_OWNER",
        "kernel_source": "REDACTED_KAGGLE_OWNER/leipzig-ecg-q1-v5-5-gen-only-one-seed-paper-t4",
        "profile": "poly_n16384_scale45",
        "populations": [16, 32, 64, 128, 256, 512],
        "selected_population": 64,
        "candidate_budget": 512,
    },
    "heartplus": {
        "title": "Heart+ V5.5 HE Population {mode}",
        "slug": "heart-v5-5-he-population-{mode}",
        "owner": "REDACTED_KAGGLE_OWNER",
        "kernel_source": "REDACTED_KAGGLE_OWNER/heart-q1-v5-5-gen-only-one-seed-paper-t4",
        "profile": "square_n8192_scale29",
        "populations": [32, 64, 128, 256, 512, 1024],
        "selected_population": 128,
        "candidate_budget": 1024,
    },
    "mimic": {
        "title": "MIMIC-IV V5.5 HE Population {mode}",
        "slug": "mimic-iv-v5-5-he-population-{mode}",
        "owner": "buiquocviet",
        "kernel_source": "buiquocviet/mimic-iv-q1-v5-5-gen-only-one-seed-paper-t4",
        "profile": "poly_n16384_scale40",
        "populations": [32, 64, 128, 256, 512],
        "selected_population": 64,
        "candidate_budget": 512,
    },
}


def code_cell(text: str):
    return nbformat.v4.new_code_cell(text.strip() + "\n")


def markdown_cell(text: str):
    return nbformat.v4.new_markdown_cell(text.strip() + "\n")


def build(dataset: str, mode: str) -> Path:
    cfg = CONFIG[dataset]
    paper = mode == "paper"
    repetitions = 5 if paper else 1
    cohort_seeds = [101, 202, 303] if paper else [101]
    output_dir = ROOT / f"kaggle/notebooks/{dataset}/q1_v55_he_population_{mode}"
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_source = BENCHMARK_SOURCE.read_text(encoding="utf-8")
    benchmark_source = benchmark_source.rsplit('\nif __name__ == "__main__":', 1)[0]

    nb = nbformat.v4.new_notebook()
    nb.metadata.kernelspec = {
        "display_name": "Python 3", "language": "python", "name": "python3"
    }
    nb.metadata.language_info = {"name": "python", "version": "3"}
    nb.cells = [
        markdown_cell(f"""
# {cfg['title'].format(mode=mode.title())}

CPU-only TenSEAL/CKKS benchmark linked to the accepted Q1 V5.5 one-seed
plaintext paper artifact. The classifier, preprocessing contract, factual
cohort and CFE release are checksum-locked. No model is retrained and no
outer-test result is used to tune the cryptographic profile.

Population means CFE candidates packed into CKKS SIMD slots in one encrypted
round, not patients per batch. The server computes only the dropout-free
add/multiply polynomial MLP logit. The client owns the secret key, decrypts the
logit and applies the already-absorbed zero-logit decision threshold.
"""),
        code_cell("""
import sys, subprocess
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "tenseal==0.3.16", "psutil>=5.9"
])
"""),
        code_cell(f"""
from pathlib import Path
import hashlib, json, math, os, platform, shutil, time, zipfile
import numpy as np
import pandas as pd
import psutil
import torch
import tenseal as ts

DATASET = {dataset!r}
RUN_MODE = {mode!r}
PROTOCOL = "Q1_V55_HE_POPULATION_" + RUN_MODE.upper()
PROFILE_NAME = {cfg['profile']!r}
POPULATIONS = {cfg['populations']!r}
PREDECLARED_SELECTED_POPULATION = {cfg['selected_population']}
CANDIDATE_BUDGET = {cfg['candidate_budget']}
COHORT_SEEDS = {cohort_seeds!r}
TIMING_REPETITIONS = {repetitions}
MIN_LABEL_AGREEMENT = 0.99
MAX_ABS_LOGIT_ERROR = 0.025
OUTPUT = Path('/kaggle/working') / f"{{DATASET}}_q1_v55_he_population_{{RUN_MODE}}"
OUTPUT.mkdir(parents=True, exist_ok=True)

print({{
    "protocol": PROTOCOL, "dataset": DATASET, "run_mode": RUN_MODE,
    "profile": PROFILE_NAME, "populations": POPULATIONS,
    "candidate_cohort_seeds": COHORT_SEEDS,
    "timing_repetitions": TIMING_REPETITIONS,
    "torch_cuda_available": torch.cuda.is_available(),
}})
"""),
        markdown_cell("""
## Frozen plaintext handoff audit

The notebook rejects an unaccepted paper source or any checksum mismatch. The
MIMIC legacy protocol label inside its original training manifest is retained
as provenance metadata; the accepted V5.5 paper acceptance record is the
authoritative wrapper and the original file is never edited.
"""),
        code_cell("""
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def discover_artifact_root():
    roots = []
    for classifier in Path('/kaggle/input').rglob('classifier_seed55.pt'):
        if classifier.parent.name != 'paper_training_handoff':
            continue
        root = classifier.parent.parent
        if (root / 'frozen_release_cfe_values.csv').is_file() and (root / 'v55_paper_acceptance.json').is_file():
            roots.append(root)
    if not roots:
        extract = Path('/kaggle/working/source_extract')
        extract.mkdir(exist_ok=True)
        for archive in Path('/kaggle/input').rglob('*artifacts.zip'):
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(extract / archive.stem)
        for classifier in extract.rglob('classifier_seed55.pt'):
            root = classifier.parent.parent
            if (root / 'frozen_release_cfe_values.csv').is_file() and (root / 'v55_paper_acceptance.json').is_file():
                roots.append(root)
    unique = sorted(set(map(str, roots)))
    if len(unique) != 1:
        raise RuntimeError(f'Expected exactly one V5.5 paper artifact root, found {unique}')
    return Path(unique[0])

SOURCE = discover_artifact_root()
ACCEPTANCE = json.loads((SOURCE / 'v55_paper_acceptance.json').read_text())
if not ACCEPTANCE.get('accepted') or not ACCEPTANCE.get('paper_numbers'):
    raise AssertionError('Plaintext source is not an accepted paper artifact')
if ACCEPTANCE.get('dataset') != DATASET:
    raise AssertionError((ACCEPTANCE.get('dataset'), DATASET))

HANDOFF = SOURCE / 'paper_training_handoff'
ORIGINAL_MANIFEST_PATH = HANDOFF / 'training_handoff_manifest.json'
ORIGINAL_MANIFEST = json.loads(ORIGINAL_MANIFEST_PATH.read_text())
checksum_rows = []
for relative, expected in ORIGINAL_MANIFEST['files_sha256'].items():
    path = HANDOFF / relative
    actual = sha256_file(path)
    checksum_rows.append({
        'file': relative, 'expected_sha256': expected, 'actual_sha256': actual,
        'checksum_match': actual == expected,
    })
checksum_audit = pd.DataFrame(checksum_rows)
if not checksum_audit.checksum_match.all():
    raise AssertionError(checksum_audit.loc[~checksum_audit.checksum_match])
checksum_audit.to_csv(OUTPUT / '01_v55_source_handoff_checksum_audit.csv', index=False)

wrapper = {
    'protocol': PROTOCOL,
    'dataset': DATASET,
    'source_acceptance_protocol': ACCEPTANCE.get('protocol'),
    'source_accepted': True,
    'source_paper_numbers': True,
    'source_training_manifest_protocol_verbatim': ORIGINAL_MANIFEST.get('protocol'),
    'original_training_manifest_sha256': sha256_file(ORIGINAL_MANIFEST_PATH),
    'classifier_sha256': sha256_file(HANDOFF / 'classifier_seed55.pt'),
    'frozen_release_sha256': sha256_file(SOURCE / 'frozen_release_cfe_values.csv'),
    'all_original_handoff_checksums_match': bool(checksum_audit.checksum_match.all()),
    'mimic_legacy_manifest_label_disclosed': bool(
        DATASET != 'mimic' or ORIGINAL_MANIFEST.get('protocol') == 'Q1_V5_UNIFIED_WARMUP_RAMP'
    ),
    'source_was_modified': False,
}
(OUTPUT / 'source_handoff_wrapper_manifest.json').write_text(json.dumps(wrapper, indent=2))
print(pd.DataFrame([wrapper]).to_string(index=False))
"""),
        markdown_cell("""
## Dropout-free HE graph and deterministic candidate cohorts

The saved threshold has already been absorbed into the final bias. Therefore,
the HE graph compares the decrypted logit with zero; sigmoid is optional
client-side presentation and is not evaluated homomorphically. Candidate
cohorts are nested and round-robin balanced across direction and method family.
"""),
        code_cell("""
checkpoint = torch.load(HANDOFF / 'classifier_seed55.pt', map_location='cpu', weights_only=False)
state = checkpoint['state_dict']
cfg = checkpoint['classifier_config']
if not checkpoint.get('canonical_for_cfe'):
    raise AssertionError('Classifier is not marked canonical_for_cfe')

parameter_path = OUTPUT / 'he_add_mul_parameters_v55.npz'
np.savez(
    parameter_path,
    fc1_weight=state['fc1.weight'].numpy(), fc1_bias=state['fc1.bias'].numpy(),
    fc2_weight=state['fc2.weight'].numpy(), fc2_bias=state['fc2.bias'].numpy(),
    fc3_weight=state['fc3.weight'].numpy(), fc3_bias=state['fc3.bias'].numpy(),
    activation_kind=np.asarray(cfg['activation_kind']),
    activation_alpha=np.asarray(cfg['alpha'], dtype=np.float64),
)

release = pd.read_csv(SOURCE / 'frozen_release_cfe_values.csv')
model_columns = [c for c in release.columns if c.startswith('model_input__')]
if len(model_columns) != int(checkpoint['input_dim']):
    raise AssertionError((len(model_columns), checkpoint['input_dim']))

def method_family(method):
    method = str(method)
    for eps in [16, 8, 4, 2]:
        if f'dp_eps{eps}' in method:
            return f'DP-CounterGAN eps={eps}'
    if 'countergan' in method:
        return 'Non-DP CounterGAN'
    return 'Baseline'

release['method_family'] = release.method.map(method_family)
max_budget = int(release.candidate_budget.max())
candidate_pool = release.loc[release.candidate_budget.eq(max_budget)].copy()

def nested_balanced_cohort(seed, size):
    rng = np.random.default_rng(seed)
    queues = []
    for _, group in candidate_pool.groupby(['direction', 'method_family'], sort=True):
        indices = group.index.to_numpy().copy()
        rng.shuffle(indices)
        queues.append(list(indices))
    chosen = []
    while len(chosen) < size and any(queues):
        for queue in queues:
            if queue and len(chosen) < size:
                chosen.append(queue.pop())
    if len(chosen) != size:
        raise AssertionError(f'Only {len(chosen)} rows available for cohort size {size}')
    return release.loc[chosen].reset_index(drop=True)

cohorts = {seed: nested_balanced_cohort(seed, max(POPULATIONS)) for seed in COHORT_SEEDS}
cohort_manifest = pd.concat([
    frame.assign(candidate_cohort_seed=seed, cohort_position=np.arange(len(frame)))
    [['candidate_cohort_seed','cohort_position','query_id','method','method_family','direction','candidate_budget','rank','cfe_logit']]
    for seed, frame in cohorts.items()
], ignore_index=True)
cohort_manifest.to_csv(OUTPUT / '02_candidate_cohort_manifest.csv', index=False)

graph_contract = pd.DataFrame([{
    'dataset': DATASET, 'input_features': len(model_columns),
    'hidden_layer_1': cfg['h1'], 'hidden_layer_2': cfg['h2'],
    'output_units': 1, 'activation': cfg['activation_kind'],
    'activation_alpha': cfg['alpha'], 'training_dropout': cfg['dropout'],
    'dropout_active_during_plaintext_reference': False,
    'dropout_present_in_he_graph': False, 'sigmoid_present_in_he_graph': False,
    'client_label_rule': 'decrypted_logit >= 0 (absorbed threshold)',
    'server_operations': 'ciphertext/plaintext add; ciphertext/plaintext multiply; ciphertext square',
}])
graph_contract.to_csv(OUTPUT / '03_he_graph_contract.csv', index=False)
print(graph_contract.to_string(index=False))
"""),
        code_cell(benchmark_source),
        markdown_cell("""
## Real CKKS population frontier

Each cell below performs real encryption, serialization, public-context server
deserialization, encrypted MLP inference, output serialization and client
decryption. Context construction/key generation is timed separately once per
candidate cohort and is not added repeatedly to query latency.
"""),
        code_cell("""
raw_frames = []
per_seed_summaries = []
for cohort_seed, frame in cohorts.items():
    candidates = frame[model_columns].to_numpy(np.float64)
    raw, summary = benchmark_population_grid(
        candidates, parameter_path, populations=POPULATIONS,
        repeats=TIMING_REPETITIONS, warmup=True, profile_name=PROFILE_NAME,
    )
    raw.insert(0, 'candidate_cohort_seed', cohort_seed)
    summary.insert(0, 'candidate_cohort_seed', cohort_seed)
    raw_frames.append(raw); per_seed_summaries.append(summary)

timing_raw = pd.concat(raw_frames, ignore_index=True)
timing_raw.to_csv(OUTPUT / '04_he_population_timing_raw.csv', index=False)
pd.concat(per_seed_summaries, ignore_index=True).to_csv(
    OUTPUT / '04b_he_population_timing_by_cohort.csv', index=False
)

metric_columns = [
    'encryption_seconds','upload_serialization_seconds',
    'server_deserialization_seconds','server_inference_seconds_per_round',
    'download_serialization_seconds','decryption_seconds',
    'client_total_seconds_per_round','server_total_seconds_per_round',
    'total_latency_seconds_per_round','time_per_candidate_seconds',
    'throughput_candidates_per_second',
    'server_inference_throughput_candidates_per_second',
    'upload_ciphertext_mb','download_ciphertext_mb',
    'total_communication_mb_per_round','public_context_mb_one_time',
    'peak_rss_mb','peak_rss_delta_mb','mean_abs_logit_error',
    'max_abs_logit_error','plaintext_he_label_agreement',
    'plaintext_he_margin_state_agreement','minimum_abs_plaintext_logit',
]
summary_rows = []
for population, group in timing_raw.groupby('population', sort=True):
    row = {
        'population': int(population), 'candidate_cohort_seeds': group.candidate_cohort_seed.nunique(),
        'timing_repetitions_total': len(group), 'profile_name': PROFILE_NAME,
        'poly_modulus_degree': int(group.poly_modulus_degree.iloc[0]),
        'slot_count': int(group.slot_count.iloc[0]),
        'slot_utilization': float(group.slot_utilization.iloc[0]),
        'coeff_mod_bit_sizes': group.coeff_mod_bit_sizes.iloc[0],
        'coeff_mod_total_bits': int(group.coeff_mod_total_bits.iloc[0]),
        'tc128_max_coeff_mod_bits': int(group.tc128_max_coeff_mod_bits.iloc[0]),
        'global_scale_bits': int(group.global_scale_bits.iloc[0]),
        'feature_ciphertexts_uploaded': int(group.feature_ciphertexts_uploaded.iloc[0]),
        'logit_ciphertexts_downloaded': int(group.logit_ciphertexts_downloaded.iloc[0]),
    }
    for metric in metric_columns:
        values = group[metric].astype(float)
        row[metric + '_median'] = float(values.median())
        row[metric + '_q25'] = float(values.quantile(.25))
        row[metric + '_q75'] = float(values.quantile(.75))
    row['all_repetitions_pass_gate'] = bool(
        (group.plaintext_he_label_agreement >= MIN_LABEL_AGREEMENT).all()
        and (group.max_abs_logit_error <= MAX_ABS_LOGIT_ERROR).all()
    )
    summary_rows.append(row)
timing_summary = pd.DataFrame(summary_rows)
timing_summary.to_csv(OUTPUT / '05_he_population_timing_summary.csv', index=False)

display_columns = [
    'population','slot_utilization','total_latency_seconds_per_round_median',
    'time_per_candidate_seconds_median','throughput_candidates_per_second_median',
    'max_abs_logit_error_median','plaintext_he_label_agreement_median',
    'total_communication_mb_per_round_median','peak_rss_mb_median',
    'all_repetitions_pass_gate',
]
print(timing_summary[display_columns].to_string(index=False))
"""),
        markdown_cell("""
## Correctness stratified by direction and candidate provenance

Timing is not duplicated for each DP epsilon because every candidate is scored
by the same frozen encrypted MLP. Correctness is nevertheless stratified by
DP/non-DP candidate provenance to detect input-distribution-specific CKKS
errors. A separate stress cohort uses the smallest absolute plaintext logits;
it is not filtered after seeing HE labels.
"""),
        code_cell("""
correctness_rows = []
for cohort_seed, frame in cohorts.items():
    candidates = frame[model_columns].to_numpy(np.float64)
    p = load_parameters(parameter_path)
    client, server, public_bytes, profile, context_metrics = create_contexts(
        str(p['activation_kind']), PROFILE_NAME
    )
    metrics, he_logits = evaluate_population(
        candidates, p, client, server, public_bytes, profile
    )
    plain_logits = plaintext_logits(candidates, p)
    detail = frame[['direction','method','method_family']].copy()
    detail['candidate_cohort_seed'] = cohort_seed
    detail['plaintext_logit'] = plain_logits
    detail['he_logit'] = he_logits
    detail['abs_logit_error'] = np.abs(plain_logits - he_logits)
    detail['label_agreement'] = (plain_logits >= 0) == (he_logits >= 0)
    correctness_rows.append(detail)
correctness = pd.concat(correctness_rows, ignore_index=True)
correctness.to_csv(OUTPUT / '06_he_correctness_per_candidate.csv', index=False)
correctness_summary = correctness.groupby(
    ['direction','method_family'], as_index=False
).agg(
    candidates=('label_agreement','size'),
    label_agreement=('label_agreement','mean'),
    mean_abs_logit_error=('abs_logit_error','mean'),
    max_abs_logit_error=('abs_logit_error','max'),
    minimum_abs_plaintext_logit=('plaintext_logit', lambda x: float(np.min(np.abs(x)))),
)
correctness_summary.to_csv(OUTPUT / '07_he_correctness_by_direction_and_source.csv', index=False)
print(correctness_summary.to_string(index=False))

stress_size = max(POPULATIONS)
stress = release.assign(abs_plaintext_logit=release.cfe_logit.abs()).sort_values(
    ['abs_plaintext_logit','direction','method','query_id','rank']
).head(stress_size).reset_index(drop=True)
stress_candidates = stress[model_columns].to_numpy(np.float64)
p = load_parameters(parameter_path)
client, server, public_bytes, profile, context_metrics = create_contexts(
    str(p['activation_kind']), PROFILE_NAME
)
stress_metrics, stress_he = evaluate_population(
    stress_candidates, p, client, server, public_bytes, profile
)
stress_plain = plaintext_logits(stress_candidates, p)
stress_detail = stress[['direction','method','method_family','cfe_logit']].copy()
stress_detail['plaintext_logit_recomputed'] = stress_plain
stress_detail['he_logit'] = stress_he
stress_detail['abs_logit_error'] = np.abs(stress_plain - stress_he)
stress_detail['label_agreement'] = (stress_plain >= 0) == (stress_he >= 0)
stress_detail.to_csv(OUTPUT / '08_he_boundary_stress_per_candidate.csv', index=False)
stress_table = pd.DataFrame([{
    'dataset': DATASET, 'population': stress_size,
    'selection': 'smallest absolute plaintext CFE logits; selected before HE evaluation',
    'minimum_abs_plaintext_logit': float(np.min(np.abs(stress_plain))),
    'mean_abs_logit_error': float(np.mean(np.abs(stress_plain - stress_he))),
    'max_abs_logit_error': float(np.max(np.abs(stress_plain - stress_he))),
    'label_agreement': float(np.mean((stress_plain >= 0) == (stress_he >= 0))),
}])
stress_table.to_csv(OUTPUT / '08b_he_boundary_stress_summary.csv', index=False)
print(stress_table.to_string(index=False))
"""),
        markdown_cell("""
## Link HE population cost to the frozen plaintext search frontier

This table does not claim a new encrypted utility experiment. It composes the
measured median ciphertext latency per round with the already-frozen inner
population sensitivity rows to estimate encrypted time-to-K and time-to-cap.
Both flip directions remain separate. The selected P remains the prospective
V5.5 choice unless its HE correctness gate fails.
"""),
        code_cell("""
sensitivity = pd.read_csv(
    SOURCE / 'q1_v5_tables/00_population_round_sensitivity_raw.csv'
)
link = sensitivity.groupby(['population','direction'], as_index=False).agg(
    inner_queries=('query_number','size'),
    valid_cfe_yield_at_k=('valid_cfe_yield_at_k','mean'),
    robust_validity=('robust_validity','mean'),
    sparsity=('sparsity','mean'), diversity=('diversity','mean'),
    mean_rounds_to_k10_or_cap=('rounds_to_k10_or_cap','mean'),
    mean_candidates_to_k10_or_cap=('candidates_to_k10_or_cap','mean'),
    maximum_rounds=('maximum_rounds','max'),
    candidate_budget=('candidate_budget','max'),
)
he_link_columns = timing_summary[[
    'population','total_latency_seconds_per_round_median',
    'time_per_candidate_seconds_median','throughput_candidates_per_second_median',
    'total_communication_mb_per_round_median','peak_rss_mb_median',
    'max_abs_logit_error_median','plaintext_he_label_agreement_median',
    'all_repetitions_pass_gate',
]]
link = link.merge(he_link_columns, on='population', how='inner', validate='many_to_one')
link['composed_he_time_to_k10_or_cap_seconds'] = (
    link.mean_rounds_to_k10_or_cap * link.total_latency_seconds_per_round_median
)
link['composed_he_time_to_candidate_cap_seconds'] = (
    link.maximum_rounds * link.total_latency_seconds_per_round_median
)
link['composed_communication_to_k10_or_cap_mb'] = (
    link.mean_rounds_to_k10_or_cap * link.total_communication_mb_per_round_median
)
link['selected_for_v55_outer'] = link.population.eq(PREDECLARED_SELECTED_POPULATION)
link.to_csv(OUTPUT / '09_he_population_utility_latency_link_by_direction.csv', index=False)
print(link.to_string(index=False))
"""),
        markdown_cell("""
## Cryptographic deployment contract and acceptance

CKKS protects confidentiality against the stated honest-but-curious server but
does not provide malicious-server integrity. Public evaluation context and
relinearization keys are sent to the server; the secret key remains client
side. DP protects generator-training records under its separate accounting and
does not change the CKKS graph, ciphertext size or per-round inference cost.
"""),
        code_cell("""
selected_timing = timing_summary.loc[
    timing_summary.population.eq(PREDECLARED_SELECTED_POPULATION)
].copy()
if len(selected_timing) != 1:
    raise AssertionError('Selected population missing from HE grid')

environment = pd.DataFrame([{
    'dataset': DATASET, 'protocol': PROTOCOL, 'cpu_only': not torch.cuda.is_available(),
    'cpu_logical_count': psutil.cpu_count(logical=True),
    'cpu_physical_count': psutil.cpu_count(logical=False),
    'python_version': platform.python_version(), 'torch_version': torch.__version__,
    'tenseal_version': ts.__version__, 'platform': platform.platform(),
    'secret_key_owner': 'client',
    'server_context': 'public key plus relinearization keys; no secret key; no Galois keys',
    'server_model': 'honest-but-curious',
    'malicious_server_integrity': 'out of scope; CKKS confidentiality does not prove integrity',
    'context_reuse': 'one public context reused across queries; size reported once',
    'dp_relation_to_he_cost': 'candidate provenance only; no change to frozen MLP CKKS graph',
}])
environment.to_csv(OUTPUT / '10_he_environment_threat_model.csv', index=False)

crypto = timing_summary[[
    'population','profile_name','poly_modulus_degree','slot_count','slot_utilization',
    'coeff_mod_bit_sizes','coeff_mod_total_bits','tc128_max_coeff_mod_bits',
    'global_scale_bits','feature_ciphertexts_uploaded','logit_ciphertexts_downloaded',
    'max_abs_logit_error_median','plaintext_he_label_agreement_median',
    'all_repetitions_pass_gate',
]].copy()
crypto['security_validation'] = 'SEAL TC128 default validation; total modulus bits within bound'
crypto.to_csv(OUTPUT / '11_he_crypto_correctness_config.csv', index=False)

all_timing_gates = bool(timing_summary.all_repetitions_pass_gate.all())
all_strata_gates = bool(
    (correctness_summary.label_agreement >= MIN_LABEL_AGREEMENT).all()
    and (correctness_summary.max_abs_logit_error <= MAX_ABS_LOGIT_ERROR).all()
)
stress_gate = bool(
    stress_table.label_agreement.iloc[0] >= MIN_LABEL_AGREEMENT
    and stress_table.max_abs_logit_error.iloc[0] <= MAX_ABS_LOGIT_ERROR
)
acceptance = {
    'protocol': PROTOCOL, 'dataset': DATASET, 'run_mode': RUN_MODE,
    'accepted': bool(all_timing_gates and all_strata_gates and stress_gate),
    'paper_numbers': RUN_MODE == 'paper',
    'checks': {
        'accepted_plaintext_v55_source': bool(ACCEPTANCE.get('accepted')),
        'paper_plaintext_source': bool(ACCEPTANCE.get('paper_numbers')),
        'all_handoff_checksums_match': bool(checksum_audit.checksum_match.all()),
        'cpu_only_kernel': not torch.cuda.is_available(),
        'all_population_timing_gates': all_timing_gates,
        'all_direction_and_source_correctness_gates': all_strata_gates,
        'boundary_stress_gate': stress_gate,
        'selected_population_present': len(selected_timing) == 1,
        'secret_key_client_only': True,
        'dropout_absent_from_he_graph': True,
        'sigmoid_client_side_only': True,
    },
    'profile_name': PROFILE_NAME, 'population_grid': POPULATIONS,
    'selected_population': PREDECLARED_SELECTED_POPULATION,
    'candidate_budget': CANDIDATE_BUDGET,
    'candidate_cohort_seeds': COHORT_SEEDS,
    'timing_repetitions_per_cohort': TIMING_REPETITIONS,
    'minimum_label_agreement': MIN_LABEL_AGREEMENT,
    'maximum_abs_logit_error': MAX_ABS_LOGIT_ERROR,
}
(OUTPUT / '12_he_acceptance.json').write_text(json.dumps(acceptance, indent=2))
print(environment.to_string(index=False))
print(crypto.to_string(index=False))
print(json.dumps(acceptance, indent=2))
if not acceptance['accepted']:
    raise AssertionError('HE acceptance gate failed')

archive = shutil.make_archive(
    str(Path('/kaggle/working') / f"{DATASET}_q1_v55_he_population_{RUN_MODE}_artifacts"),
    'zip', root_dir=OUTPUT,
)
print('ARTIFACT', archive)
"""),
    ]

    notebook_path = output_dir / f"{dataset}_q1_v55_he_population_{mode}_cpu.ipynb"
    nbformat.write(nb, notebook_path)
    metadata = {
        "id": f"{cfg['owner']}/{cfg['slug'].format(mode=mode)}",
        "title": cfg["title"].format(mode=mode.title()),
        "code_file": notebook_path.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": True,
        "keywords": ["healthcare"],
        "dataset_sources": [],
        "kernel_sources": [cfg["kernel_source"]],
        "competition_sources": [],
        "model_sources": [],
    }
    (output_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return notebook_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=[*CONFIG, "all"], default="all")
    parser.add_argument("--mode", choices=["smoke", "paper", "both"], default="smoke")
    args = parser.parse_args()
    datasets = CONFIG if args.dataset == "all" else [args.dataset]
    modes = ["smoke", "paper"] if args.mode == "both" else [args.mode]
    for dataset in datasets:
        for mode in modes:
            print(build(dataset, mode))


if __name__ == "__main__":
    main()
