#!/usr/bin/env python3
"""Build a leakage-auditable beat-level table from the Leipzig ECG records.

The source annotations define 113,924 classifiable beats.  Rhythm markers
(`+`), signal-quality markers (`~`) and unclassifiable beats (`Q`) are not
classification examples.  A beat is normal only when its symbol is `N` and
it has no auxiliary label; in particular, `N` + `N-Prex` is abnormal.

The output intentionally retains patient/record/annotation metadata.  Model
code must exclude META_COLUMNS and split by subject_id before fitting any
preprocessor or feature selector.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.signal as signal
import scipy.stats as stats
import wfdb


FS_EXPECTED = 977.0
WINDOW_BEFORE_SECONDS = 0.2
WINDOW_AFTER_SECONDS = 0.4

# Exactly the beat symbols documented by the Leipzig database.  The database
# reports 113,924 beats after excluding Q, + and ~ annotations.
CLASSIFIABLE_BEAT_SYMBOLS = {
    "N", "R", "L", "b", "j", "X", "A", "a", "V", "F", "J", "/", "f"
}
NON_BEAT_SYMBOLS = {"+", "~"}

META_COLUMNS = [
    "subject_id",
    "record_id",
    "cohort",
    "diagnosis",
    "beat_sample",
    "beat_symbol",
    "beat_aux",
]


def _natural_record_key(name: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", name)
    return (int(match.group(1)) if match else 10**9, name)


def _parse_age(value: object, record_id: str) -> float:
    text = str(value).strip()
    # The source has one obvious typo for x007: ".14.3" instead of "14.3".
    if text.startswith(".") and text.count(".") == 2:
        text = text[1:]
    try:
        age = float(text)
    except ValueError as exc:
        raise ValueError(f"Invalid age {value!r} for {record_id}") from exc
    if not 0.0 <= age <= 120.0:
        raise ValueError(f"Age outside [0, 120]: {age} for {record_id}")
    return age


def load_demographics(raw_dir: Path) -> pd.DataFrame:
    frames = []
    for filename, cohort in (
        ("children-subject-info.csv", "child"),
        ("adults-subject-info.csv", "adult"),
    ):
        frame = pd.read_csv(raw_dir / filename, dtype=str)
        frame["cohort"] = cohort
        frames.append(frame)
    demo = pd.concat(frames, ignore_index=True)
    needed = {"subject_id", "file_name", "gender", "age", "diagnosis"}
    missing = sorted(needed - set(demo.columns))
    if missing:
        raise ValueError(f"Missing demographic columns: {missing}")
    demo["file_name"] = demo["file_name"].str.strip()
    demo["subject_id"] = pd.to_numeric(demo["subject_id"], errors="raise").astype(int)
    demo["age"] = [
        _parse_age(value, record_id)
        for value, record_id in zip(demo["age"], demo["file_name"])
    ]
    demo["gender"] = demo["gender"].str.strip().str.upper()
    if not set(demo["gender"]).issubset({"F", "M"}):
        raise ValueError(f"Unexpected genders: {sorted(set(demo['gender']))}")
    if demo["file_name"].duplicated().any() or demo["subject_id"].duplicated().any():
        raise ValueError("Demographic record and subject IDs must be unique")
    return demo.set_index("file_name", drop=False)


def _discrete_entropy_32(segments: np.ndarray) -> np.ndarray:
    """Scale-invariant 32-bin Shannon entropy for each row."""
    lo = segments.min(axis=1, keepdims=True)
    span = segments.max(axis=1, keepdims=True) - lo
    scaled = (segments - lo) / np.maximum(span, 1e-12)
    bins = np.minimum((scaled * 32.0).astype(np.int16), 31)
    counts = np.zeros((segments.shape[0], 32), dtype=np.float64)
    rows = np.repeat(np.arange(segments.shape[0]), segments.shape[1])
    np.add.at(counts, (rows, bins.ravel()), 1.0)
    probs = counts / segments.shape[1]
    log_probs = np.zeros_like(probs)
    np.log2(probs, out=log_probs, where=probs > 0)
    return -np.sum(probs * log_probs, axis=1)


def _extract_chunk(segments: np.ndarray, fs: float) -> dict[str, np.ndarray]:
    """Compute the original feature family in vectorized chunks."""
    x = np.asarray(segments, dtype=np.float64)
    mean = x.mean(axis=1)
    median = np.median(x, axis=1)
    std = x.std(axis=1)
    var = std**2
    maximum = x.max(axis=1)
    minimum = x.min(axis=1)
    rms = np.sqrt(np.mean(x**2, axis=1))
    mean_abs = np.mean(np.abs(x), axis=1)
    diff1 = np.diff(x, axis=1)
    diff2 = np.diff(diff1, axis=1)
    std_diff1 = diff1.std(axis=1)
    std_diff2 = diff2.std(axis=1)
    mobility = np.divide(std_diff1, std, out=np.zeros_like(std), where=std > 0)
    mobility_diff = np.divide(
        std_diff2, std_diff1, out=np.zeros_like(std), where=std_diff1 > 0
    )
    complexity = np.divide(
        mobility_diff, mobility, out=np.zeros_like(std), where=mobility > 0
    )

    freqs, psd = signal.welch(x, fs=fs, nperseg=x.shape[1], axis=1)
    total_power = psd.sum(axis=1)
    psd_prob = np.divide(
        psd, total_power[:, None], out=np.zeros_like(psd), where=total_power[:, None] > 0
    )
    log_psd_prob = np.zeros_like(psd_prob)
    np.log2(psd_prob, out=log_psd_prob, where=psd_prob > 0)
    spectral_entropy = -np.sum(psd_prob * log_psd_prob, axis=1)

    def band_power(low: float, high: float) -> np.ndarray:
        mask = (freqs >= low) & (freqs < high)
        return psd[:, mask].sum(axis=1)

    return {
        "mean": mean,
        "median": median,
        "std": std,
        "var": var,
        "skew": stats.skew(x, axis=1, bias=True),
        "kurtosis": stats.kurtosis(x, axis=1, bias=True),
        "max": maximum,
        "min": minimum,
        "p2p": maximum - minimum,
        "rms": rms,
        "crest_factor": np.divide(
            maximum, rms, out=np.zeros_like(rms), where=rms > 0
        ),
        "form_factor": np.divide(
            rms, mean_abs, out=np.zeros_like(rms), where=mean_abs > 0
        ),
        "zcr": np.sum(np.diff(np.signbit(x), axis=1) != 0, axis=1) / x.shape[1],
        "iqr": np.percentile(x, 75, axis=1) - np.percentile(x, 25, axis=1),
        "mad": np.mean(np.abs(x - mean[:, None]), axis=1),
        "energy": np.sum(x**2, axis=1),
        "entropy": _discrete_entropy_32(x),
        "p25": np.percentile(x, 25, axis=1),
        "p75": np.percentile(x, 75, axis=1),
        "hjorth_mob": mobility,
        "hjorth_comp": complexity,
        "total_power": total_power,
        "peak_freq": freqs[np.argmax(psd, axis=1)],
        "spectral_entropy": spectral_entropy,
        "power_vlf": band_power(0, 4),
        "power_lf": band_power(4, 8),
        "power_hf": band_power(8, 15),
        "power_vhf": band_power(15, 50),
    }


def extract_record(
    raw_dir: Path,
    record_id: str,
    demographic: pd.Series,
    chunk_size: int,
) -> pd.DataFrame:
    annotation = wfdb.rdann(str(raw_dir / record_id), "atr")
    ecg, fields = wfdb.rdsamp(str(raw_dir / record_id), channels=[0])
    ecg = np.asarray(ecg[:, 0], dtype=np.float64)
    fs = float(fields["fs"])
    if not np.isclose(fs, FS_EXPECTED):
        raise ValueError(f"Unexpected sampling rate {fs} for {record_id}")

    symbols = np.asarray(annotation.symbol, dtype=object)
    aux = np.asarray([(value or "").strip() for value in annotation.aux_note], dtype=object)
    samples = np.asarray(annotation.sample, dtype=np.int64)

    # Q is a real but unclassifiable beat and is therefore retained when
    # finding RR neighbours. + and ~ are not beats.
    rr_beat_positions = np.flatnonzero(~np.isin(symbols, list(NON_BEAT_SYMBOLS)))
    rr_order = {position: order for order, position in enumerate(rr_beat_positions)}
    class_positions = np.flatnonzero(np.isin(symbols, list(CLASSIFIABLE_BEAT_SYMBOLS)))
    before = int(round(WINDOW_BEFORE_SECONDS * fs))
    after = int(round(WINDOW_AFTER_SECONDS * fs))

    usable = []
    pre_rr = []
    post_rr = []
    for position in class_positions:
        order = rr_order[position]
        if order == 0 or order == len(rr_beat_positions) - 1:
            continue
        sample = samples[position]
        if sample - before < 0 or sample + after > len(ecg):
            continue
        usable.append(position)
        pre_rr.append((sample - samples[rr_beat_positions[order - 1]]) / fs)
        post_rr.append((samples[rr_beat_positions[order + 1]] - sample) / fs)

    usable = np.asarray(usable, dtype=np.int64)
    pre_rr = np.asarray(pre_rr, dtype=np.float64)
    post_rr = np.asarray(post_rr, dtype=np.float64)
    output_parts = []
    offsets = np.arange(-before, after, dtype=np.int64)
    for start in range(0, len(usable), chunk_size):
        stop = min(start + chunk_size, len(usable))
        positions = usable[start:stop]
        beat_samples = samples[positions]
        segments = ecg[beat_samples[:, None] + offsets[None, :]]
        values = _extract_chunk(segments, fs)
        frame = pd.DataFrame(values)
        frame.insert(0, "post_rr", post_rr[start:stop])
        frame.insert(0, "pre_rr", pre_rr[start:stop])
        frame.insert(0, "gender", str(demographic["gender"]))
        frame.insert(0, "age", float(demographic["age"]))
        frame.insert(0, "beat_aux", aux[positions])
        frame.insert(0, "beat_symbol", symbols[positions])
        frame.insert(0, "beat_sample", beat_samples)
        frame.insert(0, "diagnosis", str(demographic["diagnosis"]))
        frame.insert(0, "cohort", str(demographic["cohort"]))
        frame.insert(0, "record_id", record_id)
        frame.insert(0, "subject_id", int(demographic["subject_id"]))
        is_pure_normal = (symbols[positions] == "N") & (aux[positions] == "")
        frame["Target_Abnormal"] = (~is_pure_normal).astype(np.int8)
        output_parts.append(frame)
    if not output_parts:
        raise ValueError(f"No usable beats in {record_id}")
    return pd.concat(output_parts, ignore_index=True)


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_dataset(raw_dir: Path, output: Path, chunk_size: int, limit: int | None) -> None:
    demographics = load_demographics(raw_dir)
    records = sorted(demographics.index.tolist(), key=_natural_record_key)
    available = [
        record_id
        for record_id in records
        if all((raw_dir / f"{record_id}.{suffix}").exists() for suffix in ("hea", "dat", "atr"))
    ]
    if len(available) != len(records):
        missing = sorted(set(records) - set(available), key=_natural_record_key)
        raise FileNotFoundError(f"Records missing hea/dat/atr files: {missing}")
    if limit is not None:
        available = available[:limit]

    frames = []
    for number, record_id in enumerate(available, start=1):
        frame = extract_record(raw_dir, record_id, demographics.loc[record_id], chunk_size)
        frames.append(frame)
        counts = frame["Target_Abnormal"].value_counts().to_dict()
        print(
            f"[{number:02d}/{len(available):02d}] {record_id}: {len(frame):,} beats "
            f"(normal={counts.get(0, 0):,}, abnormal={counts.get(1, 0):,})",
            flush=True,
        )

    dataset = pd.concat(frames, ignore_index=True)
    numeric = dataset.select_dtypes(include=[np.number]).columns
    if not np.isfinite(dataset[numeric].to_numpy(dtype=np.float64)).all():
        raise ValueError("Non-finite numeric values found in extracted dataset")

    expected = 113_924 if limit is None else None
    # At most two boundary beats per record can be removed because a complete
    # RR context or signal window is unavailable.
    if expected is not None and not (expected - 2 * len(available) <= len(dataset) <= expected):
        raise ValueError(
            f"Expected approximately {expected:,} usable beats, found {len(dataset):,}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    dataset.to_csv(temporary, index=False, float_format="%.8g")
    temporary.replace(output)

    per_subject = (
        dataset.groupby(["subject_id", "record_id", "cohort"], as_index=False)
        .agg(n_beats=("Target_Abnormal", "size"), n_abnormal=("Target_Abnormal", "sum"))
    )
    per_subject["abnormal_rate"] = per_subject["n_abnormal"] / per_subject["n_beats"]
    summary = {
        "source_dir": str(raw_dir.resolve()),
        "output_file": str(output.resolve()),
        "sha256": sha256sum(output),
        "n_rows": int(len(dataset)),
        "n_subjects": int(dataset["subject_id"].nunique()),
        "n_features_before_encoding": int(
            len(dataset.columns) - len(META_COLUMNS) - 1
        ),
        "class_counts": {
            str(key): int(value)
            for key, value in dataset["Target_Abnormal"].value_counts().sort_index().items()
        },
        "classifiable_symbols": sorted(CLASSIFIABLE_BEAT_SYMBOLS),
        "normal_definition": "beat_symbol == 'N' and beat_aux is empty",
        "excluded_annotation_symbols": ["Q", "+", "~"],
        "lead": "I (channel 0)",
        "sampling_rate_hz": FS_EXPECTED,
        "window_seconds": [-WINDOW_BEFORE_SECONDS, WINDOW_AFTER_SECONDS],
        "age_correction": {"record_id": "x007", "source": ".14.3", "parsed": 14.3},
        "per_subject": per_subject.to_dict(orient="records"),
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(dataset):,} rows to {output}")
    print(f"Saved audit summary to {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("ecg_beats_features_full.csv"),
    )
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--limit-records", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_dataset(args.raw_dir, args.output, args.chunk_size, args.limit_records)
