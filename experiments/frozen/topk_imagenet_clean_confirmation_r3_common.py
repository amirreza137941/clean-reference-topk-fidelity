#!/usr/bin/env python3
"""Frozen protocol and dependency-light statistics for ImageNet R3.

R3 is a clean-confirmation stage.  It evaluates the single operating point
selected by ImageNet R2 on the disjoint 45,000-image ``clean_confirmation``
partition.  No hyperparameter selection or attack is permitted here.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np


PROTOCOL_ID = "imagenet_topk_clean_confirmation_r3_20260911"
PARENT_R0_PROTOCOL_ID = "imagenet_topk_preflight_r0_20260910"
PARENT_R1_PROTOCOL_ID = "imagenet_topk_robust_anchor_r1_20260910"
PARENT_R2_PROTOCOL_ID = "imagenet_topk_defense_calibration_dev_r2_20260911"

FROZEN_SPLIT_SHA256 = "829b84ffac9f2fe4d97f812b42bf91d8d6254662a8dcfdee99ec85345ca88619"
STANDARD_CHECKPOINT_SHA256 = "c377ee6a665d9cb868b071b2623dff729084f6b3fe0faad6b79c0ab51b99a33f"
ROBUST_CHECKPOINT_SHA256 = "29d4db62dfc2635ca364c1b98c1437af84b10620c99407a3c6ca6967c9e6bfc6"
ROBUST_STATE_FINGERPRINT = "ab15ff6b4fe3d75542686d7138ce127bca3467fcff86820f5049b1540e50251d"

R2_CONFIG_HASH = "e09f3e09e5279957e6da1c7a4f05ac5a5812e3e14c4134cbaea99fd90d9503c4"
R2_SELECTION_HASH = "e078d36935326bac17b810e5c888a201d801957a4552ea788442eeff899d0386"
EXPECTED_R2_ARTIFACT_HASHES = {
    "run_config.json": "241ad790bc89e206c8a03a078237def96b11a0c1967f46b196c53b06311ead14",
    "defense_selection_r2.json": "65ffaaa870825ffca7c8895963d1f0c6aba39f49a781cca16af0c1a6cdd1658d",
    "final_summary.json": "7db1b3aa49800c8473720ee83ef9d9ed5077fb45f427cab34f038e12707766fe",
    "selected_dev_summary.csv": "f8207a924e654554f0cc1c96091c67f1e9f7d455501d9aa0186d47923ab78d7c",
    "selected_dev_paired_bootstrap.csv": "7875a22944736827129bde0e71a64f876a98fdc6660cbd1b13b2172b86c299e5",
}
EXPECTED_R2_SOURCE_HASHES = {
    "topk_imagenet_defense_calibration_r2_common.py": "1331d9718a3e762e9646b0920b864e50657d0d0610d5149bf3fc44f95691c1b6",
    "calibrate_topk_imagenet_defense_r2.py": "f885ab00531612b038457a369c7c95bb25dd58c16dcfb6ea9367f967717a1527",
    "audit_topk_imagenet_defense_calibration_r2.py": "d50bb0f980df6892b5e71b5f76188ac8d5a9480d80b69b21edb3f8851ecd76d1",
    "test_topk_imagenet_defense_calibration_r2.py": "465c4aadc3b8f128c9036d38a59591a92302f75ac3c5c0f13cba21f160b1c17a",
    "run_topk_imagenet_defense_calibration_r2.py": "8b52b2d93c6cd0dbcde8f80406e1be3911b360735173681a9e769e1a2574f1ea",
}

PARTITION = "clean_confirmation"
EXAMPLES = 45_000
CLASSES = 1_000
EXAMPLES_PER_CLASS = 45
SHARDS = 90
EXAMPLES_PER_SHARD = 500
PRIMARY_K = 5
SECONDARY_K = 2
MC_DRAWS = 50
SIGMA = 0.025
NONINFERIORITY_MARGIN_PP = 2.0
BOOTSTRAP_RESAMPLES = 50_000
BOOTSTRAP_SEED = 2026091106
EVALUATION_SEED = 2026091105

FROZEN_STRENGTH = {
    "candidate_id": "global_a1_s4_w4",
    "gate_mode": "global",
    "alpha_px": 1.0,
    "epsilon_px": 4.0,
    "steps": 4,
    "top1_weight": 1.0,
    "set_weight": 4.0,
}

METHODS = ("standard", "robust", "full_sigma0", "noise_only", "full")
METRIC_KEYS = {
    "top1": "top1_accuracy_probability",
    "top5": "label_coverage_k5_probability",
    "exact_k5": "exact_set_k5_probability",
    "exact_k2": "exact_set_k2_probability",
}
ALL_PROBABILITY_KEYS = (
    "top1_accuracy_probability",
    "reference_top1_match_probability",
    "label_coverage_k5_probability",
    "exact_set_k5_probability",
    "label_coverage_k2_probability",
    "exact_set_k2_probability",
)

# The eight non-inferiority checks reproduce the R2 sigma-selection contract.
# Two superiority checks confirm that the full defense improves Exact-Top5
# alignment over both the robust anchor and randomization alone.
GATE_DEFINITIONS = (
    ("noise_top1_noninferiority", "noise_only_minus_robust_top1", "noninferiority"),
    ("noise_top5_noninferiority", "noise_only_minus_robust_top5", "noninferiority"),
    ("noise_exact_k5_noninferiority", "noise_only_minus_robust_exact_k5", "noninferiority"),
    ("full_top1_vs_robust_noninferiority", "full_minus_robust_top1", "noninferiority"),
    ("full_top5_vs_robust_noninferiority", "full_minus_robust_top5", "noninferiority"),
    ("full_top1_vs_sigma0_noninferiority", "full_minus_full_sigma0_top1", "noninferiority"),
    ("full_top5_vs_sigma0_noninferiority", "full_minus_full_sigma0_top5", "noninferiority"),
    ("full_exact_k5_vs_sigma0_noninferiority", "full_minus_full_sigma0_exact_k5", "noninferiority"),
    ("full_exact_k5_vs_robust_superiority", "full_minus_robust_exact_k5", "superiority"),
    ("full_exact_k5_vs_noise_superiority", "full_minus_noise_only_exact_k5", "superiority"),
)

PAIRWISE_METHODS = (
    ("noise_only", "robust"),
    ("full_sigma0", "robust"),
    ("full", "robust"),
    ("full", "noise_only"),
    ("full", "full_sigma0"),
)

NEW_SCRIPT_NAMES = (
    "topk_imagenet_clean_confirmation_r3_common.py",
    "evaluate_topk_imagenet_clean_confirmation_r3.py",
    "audit_topk_imagenet_clean_confirmation_r3.py",
    "test_topk_imagenet_clean_confirmation_r3.py",
    "run_topk_imagenet_clean_confirmation_r3.py",
)


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def object_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path = Path(path)
    materialized = [dict(row) for row in rows]
    if not materialized:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in materialized:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(materialized)
    os.replace(temporary, path)


def atomic_gzip_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path = Path(path)
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=3, mtime=0) as zipped:
            import io

            with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
    os.replace(temporary, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    return rows


def iter_csv(path: Path) -> Iterator[dict[str, str]]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle)


def shard_id(index: int) -> str:
    if not 0 <= int(index) < SHARDS:
        raise ValueError(f"Shard index outside [0,{SHARDS - 1}]: {index}")
    return f"shard_{int(index):02d}"


def stable_seed(base_seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{int(base_seed)}|{label}".encode("utf-8")).hexdigest()
    return int(digest[:15], 16) % (2**63 - 1)


def frozen_protocol_payload() -> dict[str, Any]:
    payload = {
        "protocol_id": PROTOCOL_ID,
        "parent_protocols": [
            PARENT_R0_PROTOCOL_ID,
            PARENT_R1_PROTOCOL_ID,
            PARENT_R2_PROTOCOL_ID,
        ],
        "partition": PARTITION,
        "examples": EXAMPLES,
        "classes": CLASSES,
        "examples_per_class": EXAMPLES_PER_CLASS,
        "primary_k": PRIMARY_K,
        "secondary_k": SECONDARY_K,
        "methods": list(METHODS),
        "strength": dict(FROZEN_STRENGTH),
        "sigma": SIGMA,
        "mc_draws": MC_DRAWS,
        "common_random_numbers": True,
        "bootstrap": {
            "unit": "ImageNet class cluster",
            "clusters": CLASSES,
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "interval": "paired percentile 95%",
        },
        "noninferiority_margin_pp": NONINFERIORITY_MARGIN_PP,
        "gate_definitions": [
            {"gate_id": gate, "comparison_id": comparison, "kind": kind}
            for gate, comparison, kind in GATE_DEFINITIONS
        ],
        "selection_or_retuning_allowed": False,
        "attack_allowed": False,
        "training_allowed": False,
        "attack_partitions_read": False,
        "next_stage_if_all_gates_pass": "attack_family_pilot_on_attack_pilot_partition",
        "next_stage_if_any_gate_fails": "stop_and_report_without_holdout_retuning",
    }
    payload["protocol_hash"] = object_sha256(payload)
    return payload


def rows_by_sample(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    output: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        path = str(row["relative_path"])
        if path in output:
            raise ValueError(f"Duplicate sample: {path}")
        output[path] = row
    return output


def class_cluster_delta(
    left_rows: Sequence[Mapping[str, Any]],
    right_rows: Sequence[Mapping[str, Any]],
    metric_key: str,
) -> tuple[np.ndarray, int]:
    left = rows_by_sample(left_rows)
    right = rows_by_sample(right_rows)
    paths = sorted(left)
    if paths != sorted(right):
        raise ValueError("Paired methods do not share identical samples")
    by_class: dict[int, list[float]] = {}
    for path in paths:
        left_row = left[path]
        right_row = right[path]
        if int(left_row["label"]) != int(right_row["label"]):
            raise ValueError(f"Paired labels differ for {path}")
        target = int(left_row["label"])
        by_class.setdefault(target, []).append(
            float(left_row[metric_key]) - float(right_row[metric_key])
        )
    if sorted(by_class) != list(range(CLASSES)):
        raise ValueError("Class-cluster bootstrap requires all 1,000 ImageNet classes")
    if {len(values) for values in by_class.values()} != {EXAMPLES_PER_CLASS}:
        raise ValueError("Every class cluster must contain exactly 45 paired examples")
    return np.asarray([np.mean(by_class[index]) for index in range(CLASSES)]), len(paths)


def paired_class_bootstrap_many(
    comparisons: Sequence[Mapping[str, Any]],
    resamples: int,
    seed: int,
    chunk_size: int = 100,
) -> list[dict[str, Any]]:
    if not comparisons or resamples <= 0 or chunk_size <= 0:
        raise ValueError("Bootstrap requires comparisons and positive sizes")
    vectors = [np.asarray(item["class_delta"], dtype=np.float64) for item in comparisons]
    cluster_count = len(vectors[0])
    if cluster_count <= 0 or any(len(vector) != cluster_count for vector in vectors):
        raise ValueError("Class-delta vectors must be non-empty and aligned")
    matrix = np.column_stack(vectors)
    rng = np.random.default_rng(int(seed))
    boot = np.empty((int(resamples), matrix.shape[1]), dtype=np.float64)
    cursor = 0
    while cursor < int(resamples):
        count = min(int(chunk_size), int(resamples) - cursor)
        sampled = rng.integers(0, cluster_count, size=(count, cluster_count))
        boot[cursor : cursor + count] = matrix[sampled].mean(axis=1)
        cursor += count
    output: list[dict[str, Any]] = []
    for position, (source, delta) in enumerate(zip(comparisons, vectors)):
        low, high = np.quantile(boot[:, position], [0.025, 0.975])
        row = {key: value for key, value in source.items() if key != "class_delta"}
        row.update(
            {
                "difference_pp": 100.0 * float(delta.mean()),
                "ci95_low_pp": 100.0 * float(low),
                "ci95_high_pp": 100.0 * float(high),
                "examples": int(source["examples"]),
                "clusters": cluster_count,
                "cluster_unit": "ImageNet class",
                "resamples": int(resamples),
                "seed": int(seed),
            }
        )
        row["status"] = (
            "positive"
            if row["ci95_low_pp"] > 0.0
            else "negative"
            if row["ci95_high_pp"] < 0.0
            else "statistically_indistinguishable"
        )
        output.append(row)
    return output


def apply_preregistered_gates(
    comparison_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    lookup = {str(row["comparison_id"]): row for row in comparison_rows}
    if len(lookup) != len(comparison_rows):
        raise ValueError("Duplicate comparison IDs")
    output: list[dict[str, Any]] = []
    for gate_id, comparison_id, kind in GATE_DEFINITIONS:
        if comparison_id not in lookup:
            raise ValueError(f"Missing preregistered comparison: {comparison_id}")
        source = lookup[comparison_id]
        lower = float(source["ci95_low_pp"])
        threshold = -NONINFERIORITY_MARGIN_PP if kind == "noninferiority" else 0.0
        passed = lower >= threshold if kind == "noninferiority" else lower > threshold
        output.append(
            {
                "protocol_id": PROTOCOL_ID,
                "gate_id": gate_id,
                "comparison_id": comparison_id,
                "kind": kind,
                "threshold_pp": threshold,
                "difference_pp": float(source["difference_pp"]),
                "ci95_low_pp": lower,
                "ci95_high_pp": float(source["ci95_high_pp"]),
                "passed": bool(passed),
                "status": "PASS" if passed else "FAIL",
            }
        )
    return output
