#!/usr/bin/env python3
"""Pure, dependency-light helpers for CIFAR-100 defense calibration R1."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PROTOCOL_ID = "cifar100_actual_topk_defense_calibration_r1_20260904"
PARENT_PROTOCOL_ID = "cifar100_resnet20_external_validity_r2_20260903"
EXPECTED_TRAIN_HASH = "1cb2f22bc264aefbd3fef202b6e9c19d84fa0a558445e801be9ffe68efc525a0"
EXPECTED_VALIDATION_HASH = "558249ed46b9a6183521c42c506bca72a8b63b10fd85c5c294f0794c6c00b5a4"
EXPECTED_STANDARD_SHA256 = "72e1c894a2b2deefd2c071faec19ebc5ab79597b4bbba6d513fcd7ff88e6e1f2"
EXPECTED_ANCHOR_SHA256 = "f5f9de6d009853a65a557a1c0e2642e92815e0c392151a8433911ac6c1134ac7"
EXPECTED_R2_COMMON_SHA256 = "fa99c679cba67130c197c6f9d115b71b63cb6866e8636c08bfc63e53088d9d2b"
EXPECTED_R2_ANCHOR_SOURCE_SHA256 = "7b73c4fba6fdce69fc83b476bcd563b0068fefdbbb220dfb6af4fb372b2d7d4a"

STRENGTH_GRID = tuple(
    {"alpha_px": alpha_px, "steps": steps, "epsilon_px": 8.0}
    for alpha_px in (0.5, 1.0, 1.275)
    for steps in (1, 2, 4)
)
SIGMA_GRID = tuple(
    float(round(float(value), 7))
    for value in np.arange(0.0, 0.0300001, 0.0025)
)

PRIMARY_METRICS = (
    "top1_accuracy_percent",
    "exact_set_probability_percent",
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
    rows = list(rows)
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def read_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def strength_candidate_id(candidate: Mapping[str, Any]) -> str:
    alpha = str(candidate["alpha_px"]).replace(".", "p")
    return f"strength_a{alpha}_s{int(candidate['steps'])}"


def sigma_candidate_id(sigma: float) -> str:
    return "sigma_" + (f"{float(sigma):.7f}".rstrip("0").rstrip(".") or "0").replace(".", "p")


def select_strength_candidate(
    rows: Sequence[Mapping[str, Any]],
    robust_row: Mapping[str, Any],
    max_top1_drop_pp: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Select max Exact-Set2 under a predeclared Top-1 utility constraint."""

    floor = float(robust_row["top1_accuracy_percent"]) - float(max_top1_drop_pp)
    annotated: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        row["top1_floor_percent"] = floor
        row["feasible"] = bool(float(row["top1_accuracy_percent"]) >= floor - 1e-12)
        annotated.append(row)
    feasible = [row for row in annotated if row["feasible"]]
    if not feasible:
        raise RuntimeError("No strengthening candidate satisfies the clean Top-1 constraint")
    selected = max(
        feasible,
        key=lambda row: (
            float(row["exact_set_probability_percent"]),
            float(row["top1_accuracy_percent"]),
            -int(row["strength_steps"]),
            -float(row["strength_alpha_px"]),
        ),
    )
    return dict(selected), annotated


def sigma_feasibility(
    noise_row: Mapping[str, Any],
    full_row: Mapping[str, Any],
    robust_row: Mapping[str, Any],
    full_sigma0_row: Mapping[str, Any],
    max_drop_pp: float,
) -> tuple[bool, dict[str, float]]:
    constraints = {
        "noise_top1_drop_pp": float(noise_row["top1_accuracy_percent"])
        - float(robust_row["top1_accuracy_percent"]),
        "noise_exact_set_drop_pp": float(noise_row["exact_set_probability_percent"])
        - float(robust_row["exact_set_probability_percent"]),
        "full_top1_drop_pp": float(full_row["top1_accuracy_percent"])
        - float(full_sigma0_row["top1_accuracy_percent"]),
        "full_exact_set_drop_pp": float(full_row["exact_set_probability_percent"])
        - float(full_sigma0_row["exact_set_probability_percent"]),
    }
    feasible = all(value >= -float(max_drop_pp) - 1e-12 for value in constraints.values())
    return feasible, constraints


def select_sigma_candidate(
    rows: Sequence[Mapping[str, Any]],
    robust_row: Mapping[str, Any],
    full_sigma0_row: Mapping[str, Any],
    max_drop_pp: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Choose the largest jointly feasible sigma for Full and Noise-only."""

    by_sigma: dict[float, dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        sigma = round(float(row["sigma"]), 7)
        method = str(row["method"])
        if method not in {"noise_only", "full"}:
            raise ValueError(f"Unexpected sigma-calibration method: {method}")
        if method in by_sigma.setdefault(sigma, {}):
            raise ValueError(f"Duplicate {method} row at sigma={sigma}")
        by_sigma[sigma][method] = row

    annotated: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for sigma in sorted(by_sigma):
        pair = by_sigma[sigma]
        if set(pair) != {"noise_only", "full"}:
            raise ValueError(f"Incomplete method pair at sigma={sigma}: {sorted(pair)}")
        feasible, constraints = sigma_feasibility(
            pair["noise_only"], pair["full"], robust_row, full_sigma0_row, max_drop_pp
        )
        combined = {
            "sigma": sigma,
            "feasible": feasible,
            **constraints,
            "noise_top1_accuracy_percent": float(pair["noise_only"]["top1_accuracy_percent"]),
            "noise_exact_set_probability_percent": float(
                pair["noise_only"]["exact_set_probability_percent"]
            ),
            "full_top1_accuracy_percent": float(pair["full"]["top1_accuracy_percent"]),
            "full_exact_set_probability_percent": float(
                pair["full"]["exact_set_probability_percent"]
            ),
        }
        annotated.append(combined)
        if feasible:
            candidates.append(combined)
    if not candidates:
        raise RuntimeError("Internal error: sigma=0 must be jointly feasible")
    return dict(max(candidates, key=lambda row: float(row["sigma"]))), annotated


def gate_locked_phase(
    phase_rows: Sequence[Mapping[str, Any]],
    target_drop_pp: float,
    mc_tolerance_pp: float,
) -> dict[str, Any]:
    by_method = {str(row["method"]): row for row in phase_rows}
    required = {"robust", "noise_only", "full_sigma0", "full"}
    missing = sorted(required - set(by_method))
    if missing:
        raise ValueError(f"Locked phase misses methods: {missing}")
    limit = float(target_drop_pp) + float(mc_tolerance_pp)
    checks = {
        "noise_top1_drop_pp": float(by_method["noise_only"]["top1_accuracy_percent"])
        - float(by_method["robust"]["top1_accuracy_percent"]),
        "noise_exact_set_drop_pp": float(
            by_method["noise_only"]["exact_set_probability_percent"]
        ) - float(by_method["robust"]["exact_set_probability_percent"]),
        "full_top1_drop_pp": float(by_method["full"]["top1_accuracy_percent"])
        - float(by_method["full_sigma0"]["top1_accuracy_percent"]),
        "full_exact_set_drop_pp": float(
            by_method["full"]["exact_set_probability_percent"]
        ) - float(by_method["full_sigma0"]["exact_set_probability_percent"]),
    }
    return {
        "pass": all(value >= -limit - 1e-12 for value in checks.values()),
        "allowed_drop_pp": limit,
        **checks,
    }
