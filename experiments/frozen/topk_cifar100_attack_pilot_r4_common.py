#!/usr/bin/env python3
"""Frozen protocol and dependency-light helpers for CIFAR-100 attack pilot R4."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PROTOCOL_ID = "cifar100_adaptive_blackbox_pilot_r4_20260904"
PARENT_PROTOCOL_ID = "cifar100_official_clean_confirmation_r3_20260904"

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = Path("./datasets/cifar-100-python")
DEFAULT_R2_ROOT = Path(
    "./models/resnet20/"
    "paper_topk_cifar100_external_validity_r2"
)
DEFAULT_DEV_DIR = Path(
    "./models/resnet20/"
    "paper_topk_cifar100_actual_defense_dev_pareto_r2"
)
DEFAULT_PARENT_DIR = Path(
    "./models/resnet20/"
    "paper_topk_cifar100_official_clean_confirmation_k2_mc100_r3"
)
DEFAULT_OUTPUT_ROOT = Path(
    "./models/resnet20/"
    "paper_topk_cifar100_adaptive_attack_pilot_k2_q5k_r4"
)

EXPECTED_STANDARD_SHA256 = (
    "72e1c894a2b2deefd2c071faec19ebc5ab79597b4bbba6d513fcd7ff88e6e1f2"
)
EXPECTED_ANCHOR_SHA256 = (
    "f5f9de6d009853a65a557a1c0e2642e92815e0c392151a8433911ac6c1134ac7"
)
EXPECTED_FROZEN_DEFENSE = {
    "candidate_id": "global_a1p275_s2",
    "gate_mode": "global",
    "alpha_px": 1.275,
    "epsilon_px": 8.0,
    "steps": 2,
    "top1_weight": 1.0,
    "set_weight": 4.0,
    "sigma": 0.015,
}

METHODS = ("robust", "full_sigma0", "noise_only", "full")
ATTACKS = ("SquareAdaptive", "SquareRacing")
LIGHT_METHODS = ("robust", "noise_only")
FULL_METHODS = ("full_sigma0", "full")
SEARCH_SEEDS = (41041, 51051, 61061)
CONFIRMATORY_SEARCH_SEEDS = (71071, 82073, 93083)

TOP_K = 2
EPSILON_PX = 2.0
API_BUDGET = 5000
SQUARE_P_INIT = 0.30
REEVAL_DRAWS = 1000
REEVAL_SEED = 2026090601
REEVAL_BATCH_SIZE = 5
REEVAL_MC_CHUNK = 20
BOOTSTRAP_RESAMPLES = 50000
BOOTSTRAP_SEED = 2026090602
SEARCH_BATCH_SIZE_BY_METHOD = {
    "robust": 20,
    "noise_only": 20,
    "full_sigma0": 5,
    "full": 5,
}

ADAPTIVE_CONFIG = {
    "search_eot": 1,
    "refresh_eot": 1,
    "confirm_candidates": 5,
    "confirm_fraction": 0.20,
    "objective": "exact_set_margin_of_mean_logits",
}
RACING_CONFIG = {
    "search_eot": 5,
    "refresh_eot": 5,
    "pool_size": 8,
    "race_interval": 25,
    "race_eot": 5,
    "confirm_fraction": 0.25,
    "score_temperature": 0.10,
    "objective": "mean_per_draw_sigmoid_exact_set_margin",
}

SCRIPT_NAMES = (
    "topk_cifar100_attack_pilot_r4_common.py",
    "topk_cifar100_square_attack_core_r4.py",
    "search_topk_cifar100_attack_pilot_r4.py",
    "reevaluate_topk_cifar100_attack_pilot_r4_mc1000.py",
    "audit_topk_cifar100_attack_pilot_r4.py",
    "test_topk_cifar100_attack_pilot_r4.py",
    "topk_cifar100_attack_pilot_r4_launcher.py",
    "run_topk_cifar100_attack_pilot_r4_light.py",
    "run_topk_cifar100_attack_pilot_r4_full.py",
    "run_topk_cifar100_attack_pilot_r4_reeval.py",
)


@dataclass(frozen=True)
class MethodSpec:
    name: str
    sigma: float
    strength_steps: int
    strength_alpha_px: float = 0.0
    strength_epsilon_px: float = 0.0
    top1_weight: float = 1.0
    set_weight: float = 4.0

    @property
    def stochastic(self) -> bool:
        return self.sigma > 0.0


def method_spec(name: str) -> MethodSpec:
    if name not in METHODS:
        raise ValueError(f"Unknown method: {name}")
    strength = name in {"full_sigma0", "full"}
    noise = name in {"noise_only", "full"}
    return MethodSpec(
        name=name,
        sigma=EXPECTED_FROZEN_DEFENSE["sigma"] if noise else 0.0,
        strength_steps=EXPECTED_FROZEN_DEFENSE["steps"] if strength else 0,
        strength_alpha_px=(EXPECTED_FROZEN_DEFENSE["alpha_px"] if strength else 0.0),
        strength_epsilon_px=(EXPECTED_FROZEN_DEFENSE["epsilon_px"] if strength else 0.0),
        top1_weight=EXPECTED_FROZEN_DEFENSE["top1_weight"],
        set_weight=EXPECTED_FROZEN_DEFENSE["set_weight"],
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


def tensor_sha256(tensor: Any) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
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
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(materialized)
    os.replace(temporary, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_or_validate_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    if path.is_file():
        if read_json(path) != payload:
            raise AssertionError(f"Frozen JSON differs from this invocation: {path}")
    else:
        atomic_json(path, payload)


def _validate_self_hash(payload: Mapping[str, Any], field: str, label: str) -> str:
    claimed = str(payload.get(field, ""))
    base = dict(payload)
    base.pop(field, None)
    if not claimed or object_sha256(base) != claimed:
        raise AssertionError(f"{label} self-hash mismatch")
    return claimed


def load_parent_authorization(parent_dir: Path) -> dict[str, Any]:
    parent_dir = Path(parent_dir).expanduser().resolve()
    paths = {
        "handoff": parent_dir / "attack_handoff.json",
        "config": parent_dir / "run_config.json",
        "manifest": parent_dir / "official_test_manifest.json",
        "final": parent_dir / "final_summary.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing R3 authorization artifacts:\n  " + "\n  ".join(missing))
    handoff = read_json(paths["handoff"])
    config = read_json(paths["config"])
    manifest = read_json(paths["manifest"])
    final = read_json(paths["final"])
    handoff_hash = _validate_self_hash(handoff, "handoff_hash", "R3 attack handoff")
    manifest_hash = _validate_self_hash(manifest, "manifest_hash", "R3 test manifest")
    if handoff.get("protocol_id") != PARENT_PROTOCOL_ID:
        raise AssertionError("Unexpected R3 handoff protocol")
    if config.get("protocol_id") != PARENT_PROTOCOL_ID:
        raise AssertionError("Unexpected R3 run-config protocol")
    if manifest.get("protocol_id") != PARENT_PROTOCOL_ID:
        raise AssertionError("Unexpected R3 manifest protocol")
    if final.get("protocol_id") != PARENT_PROTOCOL_ID:
        raise AssertionError("Unexpected R3 final-summary protocol")
    config_hash = str(config.get("config_hash", ""))
    if not config_hash or handoff.get("config_hash") != config_hash:
        raise AssertionError("R3 handoff/config identity mismatch")
    if final.get("config_hash") != config_hash:
        raise AssertionError("R3 final/config identity mismatch")
    if handoff.get("authorized") is not True or final.get("attack_stage_ready") is not True:
        raise PermissionError("R3 did not authorize the attack stage")
    if final.get("official_clean_confirmation") != "PASS":
        raise PermissionError("R3 official clean confirmation did not pass")
    if handoff.get("frozen_defense") != EXPECTED_FROZEN_DEFENSE:
        raise AssertionError("Frozen R3 defense differs from the declared R4 defense")
    partitions = handoff.get("future_attack_partitions", {})
    if partitions != manifest.get("future_attack_partitions"):
        raise AssertionError("R3 handoff and manifest partitions differ")
    pilot = partitions.get("pilot", {})
    confirmatory = partitions.get("confirmatory", {})
    pilot_indices = [int(value) for value in pilot.get("indices", [])]
    confirmatory_indices = [int(value) for value in confirmatory.get("indices", [])]
    if len(pilot_indices) != 200 or len(set(pilot_indices)) != 200:
        raise AssertionError("R3 pilot partition is not 200 unique examples")
    if len(confirmatory_indices) != 1000 or len(set(confirmatory_indices)) != 1000:
        raise AssertionError("R3 confirmatory partition is not 1,000 unique examples")
    if set(pilot_indices) & set(confirmatory_indices):
        raise AssertionError("R3 pilot and confirmatory partitions overlap")
    if sequence_sha256(pilot_indices) != pilot.get("indices_sha256"):
        raise AssertionError("R3 pilot-index hash mismatch")
    if sequence_sha256(confirmatory_indices) != confirmatory.get("indices_sha256"):
        raise AssertionError("R3 confirmatory-index hash mismatch")
    checkpoints = handoff.get("checkpoints", {})
    for name, expected_hash in (
        ("standard", EXPECTED_STANDARD_SHA256),
        ("anchor", EXPECTED_ANCHOR_SHA256),
    ):
        item = checkpoints.get(name, {})
        path = Path(str(item.get("path", ""))).expanduser().resolve()
        if item.get("sha256") != expected_hash:
            raise AssertionError(f"R3 {name} checkpoint declaration changed")
        if not path.is_file() or file_sha256(path) != expected_hash:
            raise AssertionError(f"R3 {name} checkpoint bytes do not match")
    return {
        "parent_dir": parent_dir,
        "paths": paths,
        "handoff": handoff,
        "handoff_hash": handoff_hash,
        "config": config,
        "config_hash": config_hash,
        "manifest": manifest,
        "manifest_hash": manifest_hash,
        "final": final,
        "pilot_indices": pilot_indices,
        "confirmatory_indices": confirmatory_indices,
        "checkpoints": checkpoints,
    }


def validate_dataset_against_parent(dataset: Any, parent: Mapping[str, Any]) -> None:
    if len(dataset) != 10000 or len(dataset.targets) != 10000:
        raise AssertionError("Expected the complete 10,000-example CIFAR-100 test set")
    labels = [int(value) for value in dataset.targets]
    manifest = parent["manifest"]
    if sequence_sha256(labels) != manifest.get("labels_sha256"):
        raise AssertionError("Local CIFAR-100 labels differ from the frozen R3 manifest")
    if Counter(labels) != Counter({class_id: 100 for class_id in range(100)}):
        raise AssertionError("Local CIFAR-100 test set is not class-balanced")
    for key, count in (("pilot_indices", 2), ("confirmatory_indices", 10)):
        indices = [int(value) for value in parent[key]]
        if Counter(labels[index] for index in indices) != Counter(
            {class_id: count for class_id in range(100)}
        ):
            raise AssertionError(f"Frozen {key} are not class-balanced")


def load_frozen_pilot_references(
    parent_dir: Path, pilot_indices: Sequence[int]
) -> tuple[list[int], list[list[int]]]:
    """Load the exact ranked references recorded by the audited R3 run.

    Reusing these rows prevents batch-size-dependent floating-point boundary
    changes from redefining the attack target across R4 method groups.
    """

    path = Path(parent_dir) / "official_clean_per_example.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing R3 official clean rows: {path}")
    rows = [row for row in read_csv(path) if row.get("method") == "standard"]
    if len(rows) != 10000:
        raise AssertionError("R3 must contain 10,000 standard official-test rows")
    by_index: dict[int, Mapping[str, str]] = {}
    for row in rows:
        index = int(row["sample_index"])
        if index in by_index:
            raise AssertionError(f"Duplicate R3 standard row: {index}")
        by_index[index] = row
    labels: list[int] = []
    references: list[list[int]] = []
    for index in (int(value) for value in pilot_indices):
        row = by_index.get(index)
        if row is None:
            raise AssertionError(f"R3 standard rows miss pilot index {index}")
        members = [int(value) for value in str(row["reference_set"]).split("|")]
        if len(members) != 2 or len(set(members)) != 2:
            raise AssertionError(f"Invalid R3 reference set at index {index}")
        top1 = int(row["reference_top1"])
        if top1 not in members:
            raise AssertionError(f"R3 reference Top-1 is outside Top-2 at index {index}")
        second = members[0] if members[1] == top1 else members[1]
        labels.append(int(row["label"]))
        references.append([top1, second])
    return labels, references


def build_protocol_payload(
    *,
    parent: Mapping[str, Any],
    data_dir: Path,
    r2_root: Path,
    dev_dir: Path,
    output_root: Path,
) -> dict[str, Any]:
    scripts = {name: file_sha256(PROJECT_DIR / name) for name in SCRIPT_NAMES}
    payload: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "parent_protocol_id": PARENT_PROTOCOL_ID,
        "parent_config_hash": parent["config_hash"],
        "parent_handoff_hash": parent["handoff_hash"],
        "parent_manifest_hash": parent["manifest_hash"],
        "dataset": "CIFAR-100",
        "architecture": "ResNet-20",
        "data_dir": str(Path(data_dir).expanduser().resolve()),
        "r2_root": str(Path(r2_root).expanduser().resolve()),
        "dev_dir": str(Path(dev_dir).expanduser().resolve()),
        "parent_dir": str(Path(parent["parent_dir"])),
        "output_root": str(Path(output_root).expanduser().resolve()),
        "split": "predeclared_official_test_pilot_only",
        "pilot_examples": 200,
        "pilot_per_class": 2,
        "pilot_indices_sha256": parent["manifest"]["future_attack_partitions"]["pilot"]["indices_sha256"],
        "confirmatory_examples": 1000,
        "confirmatory_per_class": 10,
        "confirmatory_indices_sha256": parent["manifest"]["future_attack_partitions"]["confirmatory"]["indices_sha256"],
        "confirmatory_outcomes_read": False,
        "methods": list(METHODS),
        "attacks": list(ATTACKS),
        "search_seeds": list(SEARCH_SEEDS),
        "confirmatory_search_seeds_predeclared": list(CONFIRMATORY_SEARCH_SEEDS),
        "top_k": TOP_K,
        "epsilon_px": EPSILON_PX,
        "api_budget_per_example": API_BUDGET,
        "square_p_init": SQUARE_P_INIT,
        "adaptive": dict(ADAPTIVE_CONFIG),
        "racing": dict(RACING_CONFIG),
        "candidate_reevaluation": {
            "draws_stochastic": REEVAL_DRAWS,
            "draws_deterministic": 1,
            "seed": REEVAL_SEED,
            "common_random_numbers": True,
            "not_charged_to_attack_budget": True,
            "batch_size": REEVAL_BATCH_SIZE,
            "mc_chunk": REEVAL_MC_CHUNK,
        },
        "search_batch_size_by_method": dict(SEARCH_BATCH_SIZE_BY_METHOD),
        "selection_rule": {
            "unit": "method",
            "criterion": "lowest_three_seed_mean_common_mc1000_exact_set_probability",
            "tie_break": "SquareAdaptive",
            "no_per_example_candidate_selection": True,
            "pilot_only": True,
        },
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "frozen_defense": dict(EXPECTED_FROZEN_DEFENSE),
        "checkpoints": parent["handoff"]["checkpoints"],
        "script_sha256": scripts,
        "no_training": True,
        "no_defense_retuning": True,
        "budgets_run_independently": True,
        "every_eot_refresh_race_confirmation_draw_charged": True,
        "deterministic_preprocessing_cached_within_repeated_draw_scores": True,
        "candidate_files_include_clean_inputs": True,
        "attack_reference_source": "audited_R3_standard_rows",
    }
    payload["protocol_hash"] = object_sha256(payload)
    return payload


def validate_protocol(payload: Mapping[str, Any]) -> str:
    claimed = str(payload.get("protocol_hash", ""))
    base = dict(payload)
    base.pop("protocol_hash", None)
    if not claimed or object_sha256(base) != claimed:
        raise AssertionError("R4 protocol hash mismatch")
    expected_scalars = {
        "protocol_id": PROTOCOL_ID,
        "parent_protocol_id": PARENT_PROTOCOL_ID,
        "pilot_examples": 200,
        "confirmatory_examples": 1000,
        "top_k": TOP_K,
        "epsilon_px": EPSILON_PX,
        "api_budget_per_example": API_BUDGET,
        "square_p_init": SQUARE_P_INIT,
        "confirmatory_outcomes_read": False,
        "no_training": True,
        "no_defense_retuning": True,
    }
    for key, expected in expected_scalars.items():
        if payload.get(key) != expected:
            raise AssertionError(f"Frozen R4 field changed: {key}")
    if payload.get("methods") != list(METHODS):
        raise AssertionError("R4 method set changed")
    if payload.get("attacks") != list(ATTACKS):
        raise AssertionError("R4 attack set changed")
    if payload.get("search_seeds") != list(SEARCH_SEEDS):
        raise AssertionError("R4 search seeds changed")
    if payload.get("confirmatory_search_seeds_predeclared") != list(CONFIRMATORY_SEARCH_SEEDS):
        raise AssertionError("R4 confirmatory search seeds changed")
    if payload.get("adaptive") != ADAPTIVE_CONFIG or payload.get("racing") != RACING_CONFIG:
        raise AssertionError("R4 attack configuration changed")
    if payload.get("search_batch_size_by_method") != SEARCH_BATCH_SIZE_BY_METHOD:
        raise AssertionError("R4 search batch-size map changed")
    expected_reeval = {
        "draws_stochastic": REEVAL_DRAWS,
        "draws_deterministic": 1,
        "seed": REEVAL_SEED,
        "common_random_numbers": True,
        "not_charged_to_attack_budget": True,
        "batch_size": REEVAL_BATCH_SIZE,
        "mc_chunk": REEVAL_MC_CHUNK,
    }
    if payload.get("candidate_reevaluation") != expected_reeval:
        raise AssertionError("R4 candidate-reevaluation configuration changed")
    if payload.get("frozen_defense") != EXPECTED_FROZEN_DEFENSE:
        raise AssertionError("R4 defense configuration changed")
    return claimed


def cell_dir(output_root: Path, method: str, attack: str, seed: int) -> Path:
    return Path(output_root) / "search" / method / attack / f"seed_{int(seed)}"


def expected_cells() -> list[tuple[str, str, int]]:
    return [
        (method, attack, seed)
        for method in METHODS
        for attack in ATTACKS
        for seed in SEARCH_SEEDS
    ]


def aligned_values(
    rows: Sequence[Mapping[str, Any]],
    *,
    method: str,
    attack: str,
    metric: str,
) -> tuple[list[int], np.ndarray]:
    selected = [
        row for row in rows
        if str(row["method"]) == method and str(row["attack"]) == attack
    ]
    by_seed: dict[int, dict[int, float]] = {}
    for row in selected:
        seed = int(row["search_seed"])
        index = int(row["sample_index"])
        if index in by_seed.setdefault(seed, {}):
            raise ValueError(f"Duplicate row for {method}/{attack}/{seed}/{index}")
        by_seed[seed][index] = float(row[metric])
    if set(by_seed) != set(SEARCH_SEEDS):
        raise ValueError(f"Missing search seeds for {method}/{attack}")
    indices = sorted(next(iter(by_seed.values())))
    if any(sorted(values) != indices for values in by_seed.values()):
        raise ValueError(f"Seed rows are not aligned for {method}/{attack}")
    matrix = np.asarray(
        [[by_seed[seed][index] for index in indices] for seed in SEARCH_SEEDS],
        dtype=np.float64,
    )
    return indices, matrix.mean(axis=0)


def paired_bootstrap(
    delta: np.ndarray,
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    chunk_size: int = 250,
) -> dict[str, float | int | str]:
    delta = np.asarray(delta, dtype=np.float64)
    if delta.ndim != 1 or len(delta) == 0:
        raise ValueError("Bootstrap delta must be a non-empty vector")
    rng = np.random.default_rng(int(seed))
    boot = np.empty(int(resamples), dtype=np.float64)
    cursor = 0
    while cursor < int(resamples):
        count = min(int(chunk_size), int(resamples) - cursor)
        sampled = rng.integers(0, len(delta), size=(count, len(delta)))
        boot[cursor : cursor + count] = delta[sampled].mean(axis=1)
        cursor += count
    low, high = np.quantile(boot, [0.025, 0.975])
    return {
        "difference_pp": 100.0 * float(delta.mean()),
        "ci95_low_pp": 100.0 * float(low),
        "ci95_high_pp": 100.0 * float(high),
        "examples": len(delta),
        "resamples": int(resamples),
        "seed": int(seed),
        "status": (
            "positive" if low > 0 else "negative" if high < 0 else "statistically_indistinguishable"
        ),
    }
