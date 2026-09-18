"""TenSEAL CKKS population-wise SIMD benchmark for the frozen HE MLP.

One ciphertext is created per model input.  Its slots hold the same feature for
an entire CFE candidate population, so a single encrypted graph evaluation
returns one ciphertext containing all candidate logits.  This module measures
real ciphertext operations; it never substitutes plaintext timing.
"""

from __future__ import annotations

import argparse
import json
import platform
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import tenseal as ts


# Six multiplicative/rescale transitions are needed by the straightforward
# TenSEAL graph (three plaintext-weight linear layers and two squares, plus
# library level alignment).  The previous 280-bit chain is rejected at N=8192.
# Heart+'s square graph fits at N=8192. ECG's folded polynomial and MIMIC's
# real iterative square-search candidates use N=16384 after their N=8192
# precision diagnostics proved insufficient. Both profiles stay below SEAL's
# TC128 maximum coefficient-modulus bit count (218 and 438 bits, respectively).
HE_PROFILES = {
    "square_n8192": {
        "poly_modulus_degree": 8192,
        "coeff_mod_bit_sizes": [40, 27, 27, 27, 27, 27, 40],
        "global_scale_bits": 27,  # legacy diagnostic only
        "status": "legacy_precision_diagnostic",
    },
    "square_n8192_scale29": {
        "poly_modulus_degree": 8192,
        "coeff_mod_bit_sizes": [36, 29, 29, 29, 29, 29, 36],
        "global_scale_bits": 29,
        "status": "paper_candidate",
    },
    "poly_n16384_scale40": {
        "poly_modulus_degree": 16384,
        "coeff_mod_bit_sizes": [50, 40, 40, 40, 40, 40, 50],
        "global_scale_bits": 40,
        "status": "paper_candidate",
    },
    "poly_n16384_scale45": {
        "poly_modulus_degree": 16384,
        "coeff_mod_bit_sizes": [55, 45, 45, 45, 45, 45, 55],
        "global_scale_bits": 45,
        "status": "paper_precision_profile",
    },
}
DEFAULT_PROFILE = {
    "square": "square_n8192_scale29",
    "poly": "poly_n16384_scale40",
}
TC128_MAX_COEFF_MOD_BITS = {8192: 218, 16384: 438}
POPULATION_GRID = [32, 64, 128, 256, 512]
POLY_PREACTIVATION_SCALE = .25
# Agreement is reported exactly for every run. The acceptance floor is 99%;
# candidates closer to the plaintext decision boundary than the CKKS error are
# not hidden or discarded. Dataset profiles may still achieve 100% agreement.
MIN_LABEL_AGREEMENT = .99
MAX_LOGIT_ERROR_FRACTION_OF_MARGIN = .25
DEFAULT_CFE_MARGIN = .10


class PeakRSS:
    def __enter__(self):
        self.process = psutil.Process()
        self.start = self.process.memory_info().rss
        self.peak = self.start
        self.stop = threading.Event()

        def sample():
            while not self.stop.wait(.01):
                self.peak = max(self.peak, self.process.memory_info().rss)

        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set(); self.thread.join()
        self.peak = max(self.peak, self.process.memory_info().rss)

    @property
    def peak_delta_mb(self) -> float:
        return max(0, self.peak - self.start) / 1_000_000

    @property
    def baseline_mb(self) -> float:
        return self.start / 1_000_000

    @property
    def peak_mb(self) -> float:
        return self.peak / 1_000_000


def profile_for(activation_kind: str, profile_name: str | None = None) -> dict:
    kind = "square" if activation_kind == "square" else "poly"
    name = profile_name or DEFAULT_PROFILE[kind]
    if name not in HE_PROFILES:
        raise KeyError(f"Unknown HE profile: {name}")
    profile = {"profile_name": name, **HE_PROFILES[name]}
    total = sum(profile["coeff_mod_bit_sizes"])
    maximum = TC128_MAX_COEFF_MOD_BITS[profile["poly_modulus_degree"]]
    if total > maximum:
        raise ValueError(f"{name}: coeff modulus {total} exceeds TC128 maximum {maximum}")
    return profile


def create_contexts(activation_kind: str, profile_name: str | None = None):
    profile = profile_for(activation_kind, profile_name)
    with PeakRSS() as memory:
        start = time.perf_counter()
        client = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=profile["poly_modulus_degree"],
            coeff_mod_bit_sizes=profile["coeff_mod_bit_sizes"],
        )
        client.global_scale = 2 ** profile["global_scale_bits"]
        client.auto_relin = True
        client.auto_rescale = True
        client.generate_relin_keys()
        public_blob = client.serialize(
            save_public_key=True, save_secret_key=False,
            save_galois_keys=False, save_relin_keys=True,
        )
        server = ts.context_from(public_blob)
        construction_seconds = time.perf_counter() - start
    context_metrics = {
        "context_construction_seconds_one_time": construction_seconds,
        "context_peak_rss_delta_mb": memory.peak_delta_mb,
        "public_context_mb_one_time": len(public_blob) / 1_000_000,
        "tenseal_version": getattr(ts, "__version__", "unknown"),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu_logical_count": psutil.cpu_count(logical=True),
        "cpu_physical_count": psutil.cpu_count(logical=False),
        "secret_key_location": "client_only",
        "server_context_contains_secret_key": False,
        "server_context_contains_relinearization_keys": True,
        "server_context_contains_galois_keys": False,
        "context_reuse_policy": "one context per benchmark run; reuse across queries",
    }
    return client, server, len(public_blob), profile, context_metrics


def load_parameters(path: str | Path) -> dict[str, np.ndarray | float | str]:
    raw = np.load(path)
    result = {key: raw[key] for key in raw.files}
    result["activation_kind"] = str(result["activation_kind"])
    result["activation_alpha"] = float(result["activation_alpha"])
    return result


def plaintext_logits(x: np.ndarray, p: dict) -> np.ndarray:
    def activate(value):
        if p["activation_kind"] == "square":
            return value * value
        return value + p["activation_alpha"] * value * value

    value = activate(x @ p["fc1_weight"].T + p["fc1_bias"])
    value = activate(value @ p["fc2_weight"].T + p["fc2_bias"])
    return (value @ p["fc3_weight"].T + p["fc3_bias"]).reshape(-1)


def benchmark_plaintext_population(x: np.ndarray, p: dict,
                                   warmup: int = 20,
                                   repeats: int = 500
                                   ) -> tuple[pd.DataFrame, dict[str, float]]:
    """Time the exact dropout-free plaintext graph on one candidate batch.

    The input and graph are identical to the CKKS reference.  Warm-up calls are
    excluded, every measured repetition is retained, and both batch latency
    and per-candidate latency are reported.  This is deliberately separate
    from GAN generation/search time.
    """
    x = np.asarray(x, np.float64)
    if warmup < 0 or repeats < 1:
        raise ValueError("warmup must be non-negative and repeats must be positive")
    for _ in range(warmup):
        plaintext_logits(x, p)
    rows = []
    for repeat in range(1, repeats + 1):
        started = time.perf_counter()
        logits = plaintext_logits(x, p)
        elapsed = time.perf_counter() - started
        rows.append({
            "repeat": repeat,
            "population": int(len(x)),
            "plaintext_inference_seconds": float(elapsed),
            "plaintext_time_per_candidate_seconds": float(elapsed / len(x)),
            "plaintext_throughput_candidates_per_second": float(len(x) / elapsed),
            "logit_checksum": float(np.sum(logits)),
        })
    raw = pd.DataFrame(rows)
    values = raw.plaintext_inference_seconds
    per_candidate = raw.plaintext_time_per_candidate_seconds
    throughput = raw.plaintext_throughput_candidates_per_second
    summary = {
        "population": int(len(x)), "warmup": int(warmup),
        "repeats": int(repeats),
        "plaintext_inference_seconds_mean": float(values.mean()),
        "plaintext_inference_seconds_std": float(values.std(ddof=1)),
        "plaintext_inference_seconds_median": float(values.median()),
        "plaintext_inference_seconds_q25": float(values.quantile(.25)),
        "plaintext_inference_seconds_q75": float(values.quantile(.75)),
        "plaintext_time_per_candidate_seconds_median": float(per_candidate.median()),
        "plaintext_throughput_candidates_per_second_median": float(throughput.median()),
    }
    return raw, summary


def encrypted_linear(inputs, weight: np.ndarray, bias: np.ndarray):
    outputs = []
    for row, intercept in zip(weight, bias):
        value = inputs[0] * float(row[0])
        for index in range(1, len(inputs)):
            value += inputs[index] * float(row[index])
        outputs.append(value + float(intercept))
    return outputs


def encrypted_activation(values, kind: str, alpha: float):
    if kind == "square":
        return [value.square() for value in values]
    raise ValueError("Polynomial activation is folded into the following linear layer")


def encrypted_graph(inputs, p: dict):
    if p["activation_kind"] == "square":
        value = encrypted_linear(inputs, p["fc1_weight"], p["fc1_bias"])
        value = encrypted_activation(value, "square", 0.0)
        value = encrypted_linear(value, p["fc2_weight"], p["fc2_bias"])
        value = encrypted_activation(value, "square", 0.0)
        return encrypted_linear(value, p["fc3_weight"], p["fc3_bias"])[0]

    # Exact depth-saving identity for a(z)=z+alpha*z^2:
    # a(z)=alpha*(z+1/(2*alpha))^2 - 1/(4*alpha).
    # Absorb alpha and the final constant into the following linear layer.
    # This avoids an additional ciphertext/plaintext multiplication level per
    # activation while preserving the original plaintext logit exactly.
    alpha = float(p["activation_alpha"])
    if alpha <= 0:
        raise ValueError("Polynomial activation requires positive alpha")
    lam = POLY_PREACTIVATION_SCALE
    # Work with t=lam*z so the square shift is 1 rather than 4 for ECG's
    # alpha=.125. The following linear weights absorb alpha/lam^2 exactly.
    value = encrypted_linear(inputs, p["fc1_weight"] * lam, p["fc1_bias"] * lam)
    shift = lam / (2.0 * alpha)
    constant = 1.0 / (4.0 * alpha)
    value = [(item + shift).square() for item in value]
    folded_w2 = p["fc2_weight"] * (alpha / (lam * lam))
    folded_b2 = p["fc2_bias"] - constant * p["fc2_weight"].sum(axis=1)
    # Pre-scale the second pre-activation before its square as well.
    folded_w2 = folded_w2 * lam
    folded_b2 = folded_b2 * lam
    value = encrypted_linear(value, folded_w2, folded_b2)
    value = [(item + shift).square() for item in value]
    folded_w3 = p["fc3_weight"] * (alpha / (lam * lam))
    folded_b3 = p["fc3_bias"] - constant * p["fc3_weight"].sum(axis=1)
    return encrypted_linear(value, folded_w3, folded_b3)[0]


def evaluate_population(x: np.ndarray, p: dict, client, server,
                        public_context_bytes: int, profile: dict,
                        cfe_margin: float = DEFAULT_CFE_MARGIN
                        ) -> tuple[dict[str, float | int | bool | str], np.ndarray]:
    x = np.asarray(x, np.float64)
    population, input_dim = x.shape
    slots = profile["poly_modulus_degree"] // 2
    if population > slots:
        raise ValueError(f"population={population} exceeds CKKS slots={slots}")
    if input_dim != p["fc1_weight"].shape[1]:
        raise ValueError("candidate input dimension does not match HE parameters")
    plaintext_started = time.perf_counter()
    plain = plaintext_logits(x, p)
    plaintext_reference_seconds = time.perf_counter() - plaintext_started

    with PeakRSS() as memory:
        start = time.perf_counter()
        encrypted_client = [ts.ckks_vector(client, x[:, j].tolist())
                            for j in range(input_dim)]
        encryption_seconds = time.perf_counter() - start

        start = time.perf_counter()
        uploads = [value.serialize() for value in encrypted_client]
        upload_serialization_seconds = time.perf_counter() - start

        start = time.perf_counter()
        encrypted_server = [ts.ckks_vector_from(server, value) for value in uploads]
        server_deserialization_seconds = time.perf_counter() - start

        start = time.perf_counter()
        encrypted_output = encrypted_graph(encrypted_server, p)
        server_inference_seconds = time.perf_counter() - start

        start = time.perf_counter()
        download = encrypted_output.serialize()
        download_serialization_seconds = time.perf_counter() - start

        start = time.perf_counter()
        client_output = ts.ckks_vector_from(client, download)
        decrypted = np.asarray(client_output.decrypt()[:population], np.float64)
        decryption_seconds = time.perf_counter() - start

    client_seconds = (encryption_seconds + upload_serialization_seconds
                      + download_serialization_seconds + decryption_seconds)
    server_seconds = server_deserialization_seconds + server_inference_seconds
    total_seconds = client_seconds + server_seconds
    absolute = np.abs(decrypted - plain)
    agreement = float(np.mean((plain >= 0) == (decrypted >= 0)))
    plain_state = np.where(plain >= cfe_margin, 1,
                           np.where(plain <= -cfe_margin, -1, 0))
    he_state = np.where(decrypted >= cfe_margin, 1,
                        np.where(decrypted <= -cfe_margin, -1, 0))
    margin_state_agreement = float(np.mean(plain_state == he_state))
    plain_probability = 1 / (1 + np.exp(-np.clip(plain, -40, 40)))
    he_probability = 1 / (1 + np.exp(-np.clip(decrypted, -40, 40)))
    probability_error = np.abs(plain_probability - he_probability)
    maximum_allowed_error = cfe_margin * MAX_LOGIT_ERROR_FRACTION_OF_MARGIN
    gate = (agreement >= MIN_LABEL_AGREEMENT
            and float(absolute.max()) <= maximum_allowed_error)
    result = {
        "population": population,
        "packed_slots_used": population,
        "slot_utilization": population / slots,
        "feature_ciphertexts_uploaded": input_dim,
        "logit_ciphertexts_downloaded": 1,
        # Correctness-audit reference only. It is not part of the deployed HE
        # client/server latency accumulated below.
        "plaintext_reference_inference_seconds": plaintext_reference_seconds,
        "encryption_seconds": encryption_seconds,
        "upload_serialization_seconds": upload_serialization_seconds,
        "server_deserialization_seconds": server_deserialization_seconds,
        "server_inference_seconds_per_round": server_inference_seconds,
        "download_serialization_seconds": download_serialization_seconds,
        "decryption_seconds": decryption_seconds,
        "total_latency_seconds_per_round": total_seconds,
        "time_per_candidate_seconds": total_seconds / population,
        "throughput_candidates_per_second": population / total_seconds,
        "server_inference_throughput_candidates_per_second":
            population / server_inference_seconds,
        "client_total_seconds_per_round": client_seconds,
        "server_total_seconds_per_round": server_seconds,
        "upload_ciphertext_mb": sum(map(len, uploads)) / 1_000_000,
        "download_ciphertext_mb": len(download) / 1_000_000,
        "total_communication_mb_per_round":
            (sum(map(len, uploads)) + len(download)) / 1_000_000,
        "public_context_mb_one_time": public_context_bytes / 1_000_000,
        "peak_rss_delta_mb": memory.peak_delta_mb,
        "baseline_rss_mb": memory.baseline_mb,
        "peak_rss_mb": memory.peak_mb,
        "mean_abs_logit_error": float(absolute.mean()),
        "max_abs_logit_error": float(absolute.max()),
        "mean_abs_probability_error_client_side": float(probability_error.mean()),
        "max_abs_probability_error_client_side": float(probability_error.max()),
        "plaintext_he_label_agreement": agreement,
        "plaintext_he_margin_state_agreement": margin_state_agreement,
        "agreement_and_error_gate_on_supplied_candidates": gate,
        "required_label_agreement": MIN_LABEL_AGREEMENT,
        "maximum_allowed_abs_logit_error": maximum_allowed_error,
        "minimum_abs_plaintext_logit": float(np.min(np.abs(plain))),
        "profile_name": profile["profile_name"],
        "profile_status": profile["status"],
        "poly_modulus_degree": profile["poly_modulus_degree"],
        "slot_count": slots,
        "coeff_mod_bit_sizes": json.dumps(profile["coeff_mod_bit_sizes"]),
        "coeff_mod_total_bits": sum(profile["coeff_mod_bit_sizes"]),
        "tc128_max_coeff_mod_bits": TC128_MAX_COEFF_MOD_BITS[profile["poly_modulus_degree"]],
        "global_scale_bits": profile["global_scale_bits"],
        "security_validation": "SEAL default TC128 parameter validation",
        "parameter_construction_passed": True,
    }
    return result, decrypted


def benchmark_population(x: np.ndarray, p: dict, client, server,
                         public_context_bytes: int, profile: dict,
                         cfe_margin: float = DEFAULT_CFE_MARGIN
                         ) -> dict[str, float | int | bool | str]:
    return evaluate_population(
        x, p, client, server, public_context_bytes, profile, cfe_margin
    )[0]


class HEPopulationScorer:
    """Reusable client/server context for an actual encrypted CFE search."""

    def __init__(self, parameters: str | Path | dict,
                 profile_name: str | None = None,
                 cfe_margin: float = DEFAULT_CFE_MARGIN):
        self.parameters = (load_parameters(parameters)
                           if isinstance(parameters, (str, Path)) else parameters)
        (self.client, self.server, self.public_context_bytes, self.profile,
         self.context_metrics) = create_contexts(
            str(self.parameters["activation_kind"]), profile_name
        )
        self.cfe_margin = cfe_margin

    def __call__(self, candidates: np.ndarray):
        metrics, logits = evaluate_population(
            candidates, self.parameters, self.client, self.server,
            self.public_context_bytes, self.profile, self.cfe_margin,
        )
        return logits.astype(np.float32), metrics


def benchmark_population_grid(candidates: np.ndarray, parameters: str | Path,
                              populations=POPULATION_GRID, repeats: int = 3,
                              warmup: bool = True,
                              profile_name: str | None = None
                              ) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = np.asarray(candidates, np.float64)
    if len(candidates) < max(populations):
        raise ValueError(f"Need at least {max(populations)} candidates")
    p = load_parameters(parameters)
    (client, server, public_context_bytes, profile,
     context_metrics) = create_contexts(str(p["activation_kind"]), profile_name)
    if warmup:
        benchmark_population(candidates[:min(populations)], p, client, server,
                             public_context_bytes, profile)
    rows = []
    for population in populations:
        for repeat in range(1, repeats + 1):
            metric = benchmark_population(
                candidates[:population], p, client, server,
                public_context_bytes, profile,
            )
            rows.append({"repeat": repeat, **context_metrics, **metric})
    raw = pd.DataFrame(rows)
    numeric = [column for column in raw.select_dtypes(include=[np.number, "bool"]).columns
               if column not in {"population", "repeat"}]
    constants = [column for column in raw.columns
                 if column not in numeric + ["population", "repeat"]]
    summary_rows = []
    for population, group in raw.groupby("population", sort=True):
        row = {"population": int(population), "repeats": int(len(group))}
        row.update({column: group[column].iloc[0] for column in constants})
        for column in numeric:
            values = group[column].astype(float)
            row[f"{column}_median"] = float(values.median())
            row[f"{column}_q25"] = float(values.quantile(.25))
            row[f"{column}_q75"] = float(values.quantile(.75))
        summary_rows.append(row)
    return raw, pd.DataFrame(summary_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parameters", required=True)
    parser.add_argument("--candidates-npy", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--no-warmup", action="store_true")
    args = parser.parse_args()
    candidates = np.load(args.candidates_npy)
    raw, table = benchmark_population_grid(
        candidates, args.parameters, repeats=args.repeats,
        warmup=not args.no_warmup, profile_name=args.profile,
    )
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    raw.to_csv(output.with_name(output.stem + "_raw.csv"), index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
