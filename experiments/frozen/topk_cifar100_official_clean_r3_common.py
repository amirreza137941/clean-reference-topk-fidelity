#!/usr/bin/env python3
"""Dependency-light protocol helpers for CIFAR-100 official clean confirmation R3."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PROTOCOL_ID = "cifar100_official_clean_confirmation_r3_20260904"
PARENT_DEV_PROTOCOL_ID = "cifar100_actual_topk_defense_dev_pareto_r2_20260904"
PARENT_MODEL_PROTOCOL_ID = "cifar100_resnet20_external_validity_r2_20260903"

EXPECTED_DEV_CONFIG_HASH = (
    "16f280f1e8c7ea5cfc3a349dda4a99f64999c2f72bb9ecb9911bb0f8d57048fe"
)
EXPECTED_DEV_SELECTION_HASH = (
    "bcf56411164b659b65d2856f5c01e264d3ec037585800db31345508de2938089"
)
EXPECTED_STANDARD_SHA256 = (
    "72e1c894a2b2deefd2c071faec19ebc5ab79597b4bbba6d513fcd7ff88e6e1f2"
)
EXPECTED_ANCHOR_SHA256 = (
    "f5f9de6d009853a65a557a1c0e2642e92815e0c392151a8433911ac6c1134ac7"
)

EXPECTED_SELECTION = {
    "candidate_id": "global_a1p275_s2",
    "gate_mode": "global",
    "alpha_px": 1.275,
    "epsilon_px": 8.0,
    "steps": 2,
    "top1_weight": 1.0,
    "set_weight": 4.0,
    "sigma": 0.015,
}

METHODS = ("standard", "robust", "full_sigma0", "noise_only", "full")
PROBABILITY_METRICS = {
    "top1": "top1_accuracy_probability",
    "reference_top1": "reference_top1_match_probability",
    "exact_set": "exact_set_probability",
    "label_coverage": "label_coverage_probability",
}

# These gates are frozen before the selected defense is evaluated on the official
# test set.  Non-inferiority gates use the same two-percentage-point margin as R2.
GATE_RULES = (
    {
        "gate": "noise_top1_noninferiority",
        "comparison": "noise_only_minus_robust_top1",
        "rule": "ci95_low_pp_gte_minus_margin",
        "role": "component_utility",
    },
    {
        "gate": "noise_exact_noninferiority",
        "comparison": "noise_only_minus_robust_exact_set",
        "rule": "ci95_low_pp_gte_minus_margin",
        "role": "component_fidelity",
    },
    {
        "gate": "full_top1_noninferiority",
        "comparison": "full_minus_robust_top1",
        "rule": "ci95_low_pp_gte_minus_margin",
        "role": "primary_utility",
    },
    {
        "gate": "full_noise_top1_noninferiority",
        "comparison": "full_minus_full_sigma0_top1",
        "rule": "ci95_low_pp_gte_minus_margin",
        "role": "randomization_cost",
    },
    {
        "gate": "full_noise_exact_noninferiority",
        "comparison": "full_minus_full_sigma0_exact_set",
        "rule": "ci95_low_pp_gte_minus_margin",
        "role": "randomization_cost",
    },
    {
        "gate": "full_exact_superiority_over_robust",
        "comparison": "full_minus_robust_exact_set",
        "rule": "ci95_low_pp_gt_zero",
        "role": "primary_clean_fidelity",
    },
    {
        "gate": "full_exact_superiority_over_noise_only",
        "comparison": "full_minus_noise_only_exact_set",
        "rule": "ci95_low_pp_gt_zero",
        "role": "strengthening_ablation",
    },
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


def sequence_sha256(values: Sequence[int]) -> str:
    array = np.asarray([int(value) for value in values], dtype="<i8")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


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
    fields: list[str] = []
    seen: set[str] = set()
    for row in materialized:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(materialized)
    os.replace(temporary, path)


def write_or_validate_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    if path.is_file():
        if read_json(path) != payload:
            raise AssertionError(f"Frozen file differs from this invocation: {path}")
    else:
        atomic_json(path, payload)


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
    chunk_size: int = 50,
) -> list[dict[str, Any]]:
    """Paired percentile bootstrap with shared resamples across comparisons."""

    if not comparisons:
        raise ValueError("At least one comparison is required")
    if resamples <= 0 or chunk_size <= 0:
        raise ValueError("Bootstrap sizes must be positive")
    vectors = [np.asarray(item["delta"], dtype=np.float64) for item in comparisons]
    examples = len(vectors[0])
    if examples <= 0 or any(len(vector) != examples for vector in vectors):
        raise ValueError("Bootstrap vectors must be non-empty and aligned")
    matrix = np.column_stack(vectors)
    rng = np.random.default_rng(int(seed))
    boot = np.empty((int(resamples), matrix.shape[1]), dtype=np.float64)
    cursor = 0
    while cursor < int(resamples):
        count = min(int(chunk_size), int(resamples) - cursor)
        sampled = rng.integers(0, examples, size=(count, examples))
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
                "examples": examples,
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


def validate_frozen_selection(
    selection: Mapping[str, Any],
    dev_final: Mapping[str, Any],
    dev_config: Mapping[str, Any],
) -> None:
    if selection.get("protocol_id") != PARENT_DEV_PROTOCOL_ID:
        raise AssertionError("Unexpected parent dev protocol")
    if selection.get("config_hash") != EXPECTED_DEV_CONFIG_HASH:
        raise AssertionError("Unexpected parent dev config hash")
    claimed_selection_hash = str(selection.get("selection_hash"))
    hash_payload = dict(selection)
    hash_payload.pop("selection_hash", None)
    if object_sha256(hash_payload) != claimed_selection_hash:
        raise AssertionError("Parent selection self-hash mismatch")
    if claimed_selection_hash != EXPECTED_DEV_SELECTION_HASH:
        raise AssertionError("The frozen R2 operating point changed")
    if dev_config.get("protocol_id") != PARENT_DEV_PROTOCOL_ID:
        raise AssertionError("Parent run_config protocol mismatch")
    if dev_config.get("config_hash") != EXPECTED_DEV_CONFIG_HASH:
        raise AssertionError("Parent run_config hash mismatch")
    if dev_final.get("protocol_id") != PARENT_DEV_PROTOCOL_ID:
        raise AssertionError("Parent final_summary protocol mismatch")
    if dev_final.get("official_test_stage_ready") is not True:
        raise AssertionError("Parent dev stage did not authorize official clean confirmation")
    if dev_final.get("attack_stage_ready") is not False:
        raise AssertionError("Parent dev stage unexpectedly authorized attacks")
    if dev_final.get("selection_hash") != EXPECTED_DEV_SELECTION_HASH:
        raise AssertionError("Parent final_summary selection hash mismatch")

    strength = selection.get("strength", {})
    randomization = selection.get("randomization", {})
    observed = {
        "candidate_id": str(strength.get("candidate_id")),
        "gate_mode": str(strength.get("gate_mode")),
        "alpha_px": float(strength.get("alpha_px")),
        "epsilon_px": float(strength.get("epsilon_px")),
        "steps": int(strength.get("steps")),
        "top1_weight": float(strength.get("top1_weight")),
        "set_weight": float(strength.get("set_weight")),
        "sigma": float(randomization.get("sigma")),
    }
    for key, expected in EXPECTED_SELECTION.items():
        value = observed[key]
        if isinstance(expected, float):
            if abs(float(value) - expected) > 1e-12:
                raise AssertionError(f"Frozen selection changed: {key}={value}")
        elif value != expected:
            raise AssertionError(f"Frozen selection changed: {key}={value}")
    if strength.get("positive_strength_selected") is not True:
        raise AssertionError("Parent did not select positive strengthening")
    if randomization.get("positive_sigma_selected") is not True:
        raise AssertionError("Parent did not select positive randomization")


def balanced_attack_partitions(
    labels: Sequence[int],
    seed: int,
    pilot_per_class: int = 2,
    confirmatory_per_class: int = 10,
) -> dict[str, list[int]]:
    """Create label-only, disjoint future attack splits before attack outcomes exist."""

    array = np.asarray([int(value) for value in labels], dtype=np.int64)
    if len(array) != 10000 or set(array.tolist()) != set(range(100)):
        raise ValueError("Expected the complete CIFAR-100 official test labels")
    counts = np.bincount(array, minlength=100)
    if not np.all(counts == 100):
        raise ValueError("Official CIFAR-100 test must contain 100 examples per class")
    if min(pilot_per_class, confirmatory_per_class) <= 0:
        raise ValueError("Per-class split sizes must be positive")
    if pilot_per_class + confirmatory_per_class > 100:
        raise ValueError("Requested attack partitions exceed available class examples")
    rng = np.random.default_rng(int(seed))
    pilot: list[int] = []
    confirmatory: list[int] = []
    for class_id in range(100):
        candidates = np.flatnonzero(array == class_id).astype(np.int64)
        rng.shuffle(candidates)
        pilot.extend(int(value) for value in candidates[:pilot_per_class])
        start = pilot_per_class
        stop = start + confirmatory_per_class
        confirmatory.extend(int(value) for value in candidates[start:stop])
    rng.shuffle(pilot)
    rng.shuffle(confirmatory)
    if set(pilot) & set(confirmatory):
        raise AssertionError("Future attack partitions overlap")
    return {"pilot": pilot, "confirmatory": confirmatory}


def evaluate_gate_decisions(
    comparison_rows: Sequence[Mapping[str, Any]],
    noninferiority_margin_pp: float = 2.0,
) -> tuple[list[dict[str, Any]], bool]:
    if noninferiority_margin_pp < 0:
        raise ValueError("Non-inferiority margin must be non-negative")
    by_name = {str(row["comparison"]): row for row in comparison_rows}
    if len(by_name) != len(comparison_rows):
        raise ValueError("Duplicate comparison names")
    decisions: list[dict[str, Any]] = []
    for spec in GATE_RULES:
        name = str(spec["comparison"])
        if name not in by_name:
            raise ValueError(f"Missing preregistered comparison: {name}")
        source = by_name[name]
        lower = float(source["ci95_low_pp"])
        if spec["rule"] == "ci95_low_pp_gte_minus_margin":
            threshold = -float(noninferiority_margin_pp)
            passed = lower >= threshold - 1e-12
            criterion = f"CI95 lower >= {threshold:+.3f} pp"
        elif spec["rule"] == "ci95_low_pp_gt_zero":
            threshold = 0.0
            passed = lower > 0.0
            criterion = "CI95 lower > 0 pp"
        else:
            raise ValueError(f"Unknown gate rule: {spec['rule']}")
        decisions.append(
            {
                "protocol_id": PROTOCOL_ID,
                "gate": spec["gate"],
                "role": spec["role"],
                "comparison": name,
                "criterion": criterion,
                "difference_pp": float(source["difference_pp"]),
                "ci95_low_pp": lower,
                "ci95_high_pp": float(source["ci95_high_pp"]),
                "threshold_pp": threshold,
                "passed": bool(passed),
            }
        )
    return decisions, all(bool(row["passed"]) for row in decisions)
