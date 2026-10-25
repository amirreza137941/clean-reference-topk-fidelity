#!/usr/bin/env python3
"""Dependency-light helpers for ViT-reference ImageNet calibration VR1.

The protocol deliberately uses only the outcome-blind ``calibration_dev``
partition frozen in ImageNet R0.  Selection rules live in this module so they
can be tested without loading PyTorch, ImageNet, or either checkpoint.

The standard reference is the user's local torchvision ViT-B/16 checkpoint;
the defended backbone remains the independently audited robust ResNet-50.
This is therefore a cross-architecture reference experiment, not a claim that
the local ViT checkpoint is adversarially robust.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PROTOCOL_ID = "imagenet_topk_vit_reference_crossarch_calibration_vr1_20260916"
PARENT_R0_PROTOCOL_ID = "imagenet_topk_preflight_r0_20260910"
PARENT_R1_PROTOCOL_ID = "imagenet_topk_robust_anchor_r1_20260910"

FROZEN_SPLIT_SHA256 = "829b84ffac9f2fe4d97f812b42bf91d8d6254662a8dcfdee99ec85345ca88619"
STANDARD_CHECKPOINT_SHA256 = "e58e1215db231ba85279d1c1f11ab9dd028bc19c2216165e0b3215386b92e4f5"
ROBUST_CHECKPOINT_SHA256 = "29d4db62dfc2635ca364c1b98c1437af84b10620c99407a3c6ca6967c9e6bfc6"
ROBUST_STATE_FINGERPRINT = "ab15ff6b4fe3d75542686d7138ce127bca3467fcff86820f5049b1540e50251d"
R1_HANDOFF_SHA256 = "a550ba917489e313cb2a15bf3aac90679d20cf1b36d8945d8615fa53b3907b55"
R1_REPORT_SHA256 = "b19cb50ffc548d29c86929135bb77cb6676b301734b46b06bd3eb99dd7939c6f"

# These hashes were recorded by the completed R1 run.  VR1 refuses to mix
# outputs with modified parent loaders or split helpers.
EXPECTED_PARENT_SOURCE_HASHES = {
    "topk_imagenet_preflight_common.py": "a8584d7a654bd6b1737c6c3116c5a4acedbfa98a79ad21c47440b83b1249019d",
    "preflight_topk_imagenet_r0.py": "daade1a93656558e8eee47608f611838317a1f5bdbb592fc8b5e31151160649f",
    "topk_imagenet_robust_anchor_common_r1.py": "a68dca4e176e6a344354fdb0434e8c845eb7eae993903d903991a1bff9a269d4",
    "prepare_topk_imagenet_robust_anchor_r1.py": "61370abb0853fdd0c9c5cc6cf036b465f3c63d311a20b2513d88c24ff327c788",
}

PRIMARY_K = 5
SECONDARY_K = 2
CALIBRATION_EXAMPLES = 2000
CALIBRATION_SHARDS = 20
EXAMPLES_PER_SHARD = 100
STRENGTH_EPSILON_PX = 4.0
GATE_MODES = ("global", "disagreement")

# ImageNet-specific factorial grid.  No CIFAR-selected alpha, step count, or
# set-objective weight is fixed as the ImageNet operating point.
STRENGTH_GRID = tuple(
    {
        "alpha_px": float(alpha_px),
        "epsilon_px": STRENGTH_EPSILON_PX,
        "steps": int(steps),
        "top1_weight": 1.0,
        "set_weight": float(set_weight),
    }
    for alpha_px in (0.25, 0.5, 1.0)
    for set_weight in (1.0, 2.0, 4.0)
    for steps in (1, 2, 4)
)

SIGMA_GRID = tuple(float(round(value, 7)) for value in np.arange(0.0, 0.0300001, 0.0025))

PROBABILITY_METRICS = {
    "top1": "top1_accuracy_probability",
    "reference_top1": "reference_top1_match_probability",
    "top5": "label_coverage_k5_probability",
    "exact_k5": "exact_set_k5_probability",
    "exact_k2": "exact_set_k2_probability",
    "top2": "label_coverage_k2_probability",
}

STRENGTH_CONSTRAINTS = ("top1", "top5")
SIGMA_CONSTRAINTS = (
    "noise_top1_vs_robust",
    "noise_top5_vs_robust",
    "noise_exact_k5_vs_robust",
    "full_top1_vs_robust",
    "full_top5_vs_robust",
    "full_top1_vs_full_sigma0",
    "full_top5_vs_full_sigma0",
    "full_exact_k5_vs_full_sigma0",
)

NEW_SCRIPT_NAMES = (
    "topk_imagenet_vit_reference_calibration_vr1_common.py",
    "calibrate_topk_imagenet_vit_reference_vr1.py",
    "audit_topk_imagenet_vit_reference_calibration_vr1.py",
    "test_topk_imagenet_vit_reference_calibration_vr1.py",
    "run_topk_imagenet_vit_reference_calibration_vr1.py",
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    return rows


def shard_id(index: int) -> str:
    if not 0 <= int(index) < CALIBRATION_SHARDS:
        raise ValueError(f"Shard index outside [0,{CALIBRATION_SHARDS - 1}]: {index}")
    return f"shard_{int(index):02d}"


def sigma_tag(sigma: float) -> str:
    return (f"{float(sigma):.7f}".rstrip("0").rstrip(".") or "0").replace(".", "p")


def strength_candidate_id(gate_mode: str, candidate: Mapping[str, Any]) -> str:
    if gate_mode not in GATE_MODES:
        raise ValueError(f"Unknown gate mode: {gate_mode}")
    alpha = (f"{float(candidate['alpha_px']):.3f}".rstrip("0").rstrip(".")).replace(".", "p")
    weight = (f"{float(candidate['set_weight']):.1f}".rstrip("0").rstrip(".")).replace(".", "p")
    return f"{gate_mode}_a{alpha}_s{int(candidate['steps'])}_w{weight}"


def rows_by_sample(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    output: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        key = str(row["relative_path"])
        if key in output:
            raise ValueError(f"Duplicate sample: {key}")
        output[key] = row
    return output


def aligned_delta(
    left_rows: Sequence[Mapping[str, Any]],
    right_rows: Sequence[Mapping[str, Any]],
    metric_key: str,
) -> np.ndarray:
    left = rows_by_sample(left_rows)
    right = rows_by_sample(right_rows)
    keys = sorted(left)
    if keys != sorted(right):
        raise ValueError("Paired methods do not share identical relative paths")
    return np.asarray(
        [float(left[key][metric_key]) - float(right[key][metric_key]) for key in keys],
        dtype=np.float64,
    )


def paired_bootstrap_many(
    comparisons: Sequence[Mapping[str, Any]],
    resamples: int,
    seed: int,
    chunk_size: int = 10,
) -> list[dict[str, Any]]:
    """Paired percentile bootstrap with shared resamples across comparisons."""

    if resamples <= 0 or chunk_size <= 0 or not comparisons:
        raise ValueError("Bootstrap requires comparisons and positive sizes")
    vectors = [np.asarray(item["delta"], dtype=np.float64) for item in comparisons]
    sample_count = len(vectors[0])
    if sample_count <= 0 or any(len(vector) != sample_count for vector in vectors):
        raise ValueError("Bootstrap delta vectors must be non-empty and aligned")
    matrix = np.column_stack(vectors)
    rng = np.random.default_rng(int(seed))
    boot = np.empty((int(resamples), matrix.shape[1]), dtype=np.float64)
    cursor = 0
    while cursor < int(resamples):
        count = min(int(chunk_size), int(resamples) - cursor)
        sampled = rng.integers(0, sample_count, size=(count, sample_count))
        boot[cursor : cursor + count] = matrix[sampled].mean(axis=1)
        cursor += count
    output: list[dict[str, Any]] = []
    for position, (source, delta) in enumerate(zip(comparisons, vectors)):
        low, high = np.quantile(boot[:, position], [0.025, 0.975])
        row = {key: value for key, value in source.items() if key != "delta"}
        row.update(
            {
                "difference_pp": 100.0 * float(delta.mean()),
                "ci95_low_pp": 100.0 * float(low),
                "ci95_high_pp": 100.0 * float(high),
                "examples": sample_count,
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


def _bootstrap_lookup(
    bootstrap_rows: Sequence[Mapping[str, Any]], group_id: str, metric: str
) -> Mapping[str, Any]:
    matches = [
        row
        for row in bootstrap_rows
        if str(row["group_id"]) == str(group_id) and str(row["metric"]) == str(metric)
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one bootstrap row for {group_id}/{metric}; got {len(matches)}")
    return matches[0]


def select_strength_candidate(
    candidate_rows: Sequence[Mapping[str, Any]],
    bootstrap_rows: Sequence[Mapping[str, Any]],
    max_drop_pp: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Maximize deterministic Exact-Top5 under Top-1 and Top-5 CI floors."""

    annotated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in candidate_rows:
        row = dict(source)
        candidate_id = str(row["candidate_id"])
        if candidate_id in seen:
            raise ValueError(f"Duplicate strength candidate: {candidate_id}")
        seen.add(candidate_id)
        feasible = True
        for metric in STRENGTH_CONSTRAINTS:
            ci = _bootstrap_lookup(bootstrap_rows, candidate_id, metric)
            row[f"{metric}_difference_pp"] = float(ci["difference_pp"])
            row[f"{metric}_ci95_low_pp"] = float(ci["ci95_low_pp"])
            row[f"{metric}_ci95_high_pp"] = float(ci["ci95_high_pp"])
            feasible &= float(ci["ci95_low_pp"]) >= -float(max_drop_pp) - 1e-12
        row["ci_noninferiority_margin_pp"] = float(max_drop_pp)
        row["jointly_ci_feasible"] = bool(feasible)
        annotated.append(row)
    feasible_rows = [row for row in annotated if row["jointly_ci_feasible"]]
    if not feasible_rows:
        raise RuntimeError("Identity should make the strength frontier non-empty")
    gate_tiebreak = {"none": 2, "global": 1, "disagreement": 0}
    selected = max(
        feasible_rows,
        key=lambda row: (
            float(row["exact_set_k5_probability_percent"]),
            float(row["top1_accuracy_percent"]),
            float(row["label_coverage_k5_probability_percent"]),
            gate_tiebreak.get(str(row["gate_mode"]), -1),
            -int(row["strength_steps"]),
            -float(row["strength_alpha_px"]),
            -float(row["set_weight"]),
        ),
    )
    return dict(selected), annotated


def select_sigma_candidate(
    combined_rows: Sequence[Mapping[str, Any]],
    bootstrap_rows: Sequence[Mapping[str, Any]],
    max_drop_pp: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Select the largest sigma satisfying all preregistered CI constraints."""

    annotated: list[dict[str, Any]] = []
    seen: set[float] = set()
    for source in sorted(combined_rows, key=lambda row: float(row["sigma"])):
        row = dict(source)
        sigma = round(float(row["sigma"]), 7)
        if sigma in seen:
            raise ValueError(f"Duplicate sigma: {sigma}")
        seen.add(sigma)
        feasible = True
        group_id = "sigma_" + sigma_tag(sigma)
        for constraint in SIGMA_CONSTRAINTS:
            ci = _bootstrap_lookup(bootstrap_rows, group_id, constraint)
            row[f"{constraint}_difference_pp"] = float(ci["difference_pp"])
            row[f"{constraint}_ci95_low_pp"] = float(ci["ci95_low_pp"])
            row[f"{constraint}_ci95_high_pp"] = float(ci["ci95_high_pp"])
            feasible &= float(ci["ci95_low_pp"]) >= -float(max_drop_pp) - 1e-12
        row["ci_noninferiority_margin_pp"] = float(max_drop_pp)
        row["jointly_ci_feasible"] = bool(feasible)
        annotated.append(row)
    feasible_rows = [row for row in annotated if row["jointly_ci_feasible"]]
    if not feasible_rows:
        raise RuntimeError("Sigma zero should make the sigma frontier non-empty")
    return dict(max(feasible_rows, key=lambda row: float(row["sigma"]))), annotated
