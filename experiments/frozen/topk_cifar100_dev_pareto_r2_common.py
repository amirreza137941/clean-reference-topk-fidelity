#!/usr/bin/env python3
"""Dependency-light helpers for the CIFAR-100 clean Pareto dev calibration R2."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PROTOCOL_ID = "cifar100_actual_topk_defense_dev_pareto_r2_20260904"
PARENT_R1_PROTOCOL_ID = "cifar100_actual_topk_defense_calibration_r1_20260904"
PARENT_R2_PROTOCOL_ID = "cifar100_resnet20_external_validity_r2_20260903"

EXPECTED_TRAIN_HASH = "1cb2f22bc264aefbd3fef202b6e9c19d84fa0a558445e801be9ffe68efc525a0"
EXPECTED_VALIDATION_HASH = "558249ed46b9a6183521c42c506bca72a8b63b10fd85c5c294f0794c6c00b5a4"
EXPECTED_STANDARD_SHA256 = "72e1c894a2b2deefd2c071faec19ebc5ab79597b4bbba6d513fcd7ff88e6e1f2"
EXPECTED_ANCHOR_SHA256 = "f5f9de6d009853a65a557a1c0e2642e92815e0c392151a8433911ac6c1134ac7"
EXPECTED_R2_COMMON_SHA256 = "fa99c679cba67130c197c6f9d115b71b63cb6866e8636c08bfc63e53088d9d2b"
EXPECTED_R2_ANCHOR_SOURCE_SHA256 = "7b73c4fba6fdce69fc83b476bcd563b0068fefdbbb220dfb6af4fb372b2d7d4a"
EXPECTED_R1_EVALUATOR_SHA256 = "32084429f007f3a8eaf0a5f3167abec6277e45ed77a5c1b75718ef0f2c3c148d"
EXPECTED_R1_COMMON_SHA256 = "e84514afd66f62f33282af8f06ec125445b48d76054799f5960bba07cbadba15"
EXPECTED_R1_FINAL_SUMMARY_SHA256 = "6c38c7a684e097d3dfbe0b99e50c163781640eba3d591e60722d249618821039"

GATE_MODES = ("global", "disagreement")
STRENGTH_GRID = tuple(
    {
        "alpha_px": float(alpha_px),
        "epsilon_px": 8.0,
        "steps": int(steps),
        "top1_weight": 1.0,
        "set_weight": 4.0,
    }
    for alpha_px in (0.5, 1.0, 1.275)
    for steps in (1, 2, 4)
)
SIGMA_GRID = tuple(float(round(value, 7)) for value in np.arange(0.0, 0.0300001, 0.0025))

PROBABILITY_METRICS = {
    "top1": "top1_accuracy_probability",
    "reference_top1": "reference_top1_match_probability",
    "exact_set": "exact_set_probability",
    "label_coverage": "label_coverage_probability",
}

SIGMA_CONSTRAINTS = (
    "noise_top1_vs_robust",
    "noise_exact_set_vs_robust",
    "full_top1_vs_robust",
    "full_top1_vs_full_sigma0",
    "full_exact_set_vs_full_sigma0",
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


def indices_sha256(indices: Sequence[int]) -> str:
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes(order="C")).hexdigest()


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
        raise ValueError(f"Refusing to write an empty CSV: {path}")
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


def strength_candidate_id(gate_mode: str, candidate: Mapping[str, Any]) -> str:
    if gate_mode not in GATE_MODES:
        raise ValueError(f"Unknown gate mode: {gate_mode}")
    alpha = (f"{float(candidate['alpha_px']):.3f}".rstrip("0").rstrip(".")).replace(".", "p")
    return f"{gate_mode}_a{alpha}_s{int(candidate['steps'])}"


def strength_condition_id(candidate: Mapping[str, Any]) -> str:
    alpha = (f"{float(candidate['alpha_px']):.3f}".rstrip("0").rstrip(".")).replace(".", "p")
    return f"strength_a{alpha}_s{int(candidate['steps'])}"


def sigma_tag(sigma: float) -> str:
    return (f"{float(sigma):.7f}".rstrip("0").rstrip(".") or "0").replace(".", "p")


def sigma_condition_id(sigma: float) -> str:
    return "sigma_" + sigma_tag(sigma)


def rows_by_index(rows: Sequence[Mapping[str, Any]]) -> dict[int, Mapping[str, Any]]:
    output: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        index = int(row["sample_index"])
        if index in output:
            raise ValueError(f"Duplicate sample index: {index}")
        output[index] = row
    return output


def aligned_delta(
    left_rows: Sequence[Mapping[str, Any]],
    right_rows: Sequence[Mapping[str, Any]],
    metric_key: str,
) -> np.ndarray:
    left = rows_by_index(left_rows)
    right = rows_by_index(right_rows)
    indices = sorted(left)
    if indices != sorted(right):
        raise ValueError("Paired methods do not share identical sample indices")
    return np.asarray(
        [float(left[index][metric_key]) - float(right[index][metric_key]) for index in indices],
        dtype=np.float64,
    )


def paired_bootstrap_many(
    comparisons: Sequence[Mapping[str, Any]],
    resamples: int,
    seed: int,
    chunk_size: int = 20,
) -> list[dict[str, Any]]:
    """Paired percentile bootstrap using the same resamples for every column."""

    if resamples <= 0:
        raise ValueError("Bootstrap resamples must be positive")
    if chunk_size <= 0:
        raise ValueError("Bootstrap chunk size must be positive")
    if not comparisons:
        raise ValueError("At least one bootstrap comparison is required")
    vectors = [np.asarray(item["delta"], dtype=np.float64) for item in comparisons]
    sample_count = len(vectors[0])
    if sample_count <= 0 or any(len(vector) != sample_count for vector in vectors):
        raise ValueError("Bootstrap vectors must be non-empty and aligned")
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
        point = float(delta.mean())
        row = {key: value for key, value in source.items() if key != "delta"}
        row.update(
            {
                "difference_pp": 100.0 * point,
                "ci95_low_pp": 100.0 * float(low),
                "ci95_high_pp": 100.0 * float(high),
                "examples": sample_count,
                "resamples": int(resamples),
                "seed": int(seed),
            }
        )
        row["status"] = (
            "positive"
            if row["ci95_low_pp"] > 0
            else "negative"
            if row["ci95_high_pp"] < 0
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
    max_top1_drop_pp: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Maximize deterministic Exact-Set2 under a conservative paired-CI Top-1 floor."""

    annotated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in candidate_rows:
        row = dict(source)
        candidate_id = str(row["candidate_id"])
        if candidate_id in seen:
            raise ValueError(f"Duplicate strength candidate: {candidate_id}")
        seen.add(candidate_id)
        ci = _bootstrap_lookup(bootstrap_rows, candidate_id, "top1")
        row["top1_difference_pp"] = float(ci["difference_pp"])
        row["top1_ci95_low_pp"] = float(ci["ci95_low_pp"])
        row["top1_ci95_high_pp"] = float(ci["ci95_high_pp"])
        row["top1_noninferiority_margin_pp"] = float(max_top1_drop_pp)
        row["ci_feasible"] = bool(
            float(ci["ci95_low_pp"]) >= -float(max_top1_drop_pp) - 1e-12
        )
        annotated.append(row)
    feasible = [row for row in annotated if row["ci_feasible"]]
    if not feasible:
        raise RuntimeError("No strength candidate, including identity, satisfies the CI constraint")
    gate_simplicity = {"none": 2, "global": 1, "disagreement": 0}
    selected = max(
        feasible,
        key=lambda row: (
            float(row["exact_set_probability_percent"]),
            float(row["top1_accuracy_percent"]),
            gate_simplicity.get(str(row["gate_mode"]), -1),
            -int(row["strength_steps"]),
            -float(row["strength_alpha_px"]),
        ),
    )
    return dict(selected), annotated


def select_sigma_candidate(
    combined_rows: Sequence[Mapping[str, Any]],
    bootstrap_rows: Sequence[Mapping[str, Any]],
    max_drop_pp: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Choose the largest sigma whose five paired-CI utility constraints all pass."""

    annotated: list[dict[str, Any]] = []
    seen: set[float] = set()
    for source in sorted(combined_rows, key=lambda row: float(row["sigma"])):
        row = dict(source)
        sigma = round(float(row["sigma"]), 7)
        if sigma in seen:
            raise ValueError(f"Duplicate combined sigma row: {sigma}")
        seen.add(sigma)
        group_id = sigma_condition_id(sigma)
        feasible = True
        for constraint in SIGMA_CONSTRAINTS:
            ci = _bootstrap_lookup(bootstrap_rows, group_id, constraint)
            row[f"{constraint}_difference_pp"] = float(ci["difference_pp"])
            row[f"{constraint}_ci95_low_pp"] = float(ci["ci95_low_pp"])
            row[f"{constraint}_ci95_high_pp"] = float(ci["ci95_high_pp"])
            if float(ci["ci95_low_pp"]) < -float(max_drop_pp) - 1e-12:
                feasible = False
        row["ci_noninferiority_margin_pp"] = float(max_drop_pp)
        row["jointly_ci_feasible"] = bool(feasible)
        annotated.append(row)
    feasible_rows = [row for row in annotated if row["jointly_ci_feasible"]]
    if not feasible_rows:
        raise RuntimeError("Internal error: sigma=0 should always satisfy all paired constraints")
    return dict(max(feasible_rows, key=lambda row: float(row["sigma"]))), annotated

