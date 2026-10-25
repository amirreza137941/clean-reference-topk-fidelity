#!/usr/bin/env python3
"""Frozen protocol and dependency-light helpers for CIFAR-100 confirmatory R5."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import topk_cifar100_attack_pilot_r4_common as r4


PROTOCOL_ID = "cifar100_confirmatory_blackbox_r5_20260905"
PARENT_PROTOCOL_ID = r4.PARENT_PROTOCOL_ID
R4_PROTOCOL_ID = r4.PROTOCOL_ID

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = r4.DEFAULT_DATA_DIR
DEFAULT_R2_ROOT = r4.DEFAULT_R2_ROOT
DEFAULT_DEV_DIR = r4.DEFAULT_DEV_DIR
DEFAULT_PARENT_DIR = r4.DEFAULT_PARENT_DIR
DEFAULT_R4_ROOT = r4.DEFAULT_OUTPUT_ROOT
DEFAULT_OUTPUT_ROOT = Path(
    "./models/resnet20/"
    "paper_topk_cifar100_confirmatory_attack_k2_q5k_r5"
)

EXPECTED_R4_PROTOCOL_HASH = (
    "492bb3a72ea5d67f94c5623dbdceeb26b07774446073e4b6f062f740b64da6e9"
)
EXPECTED_R4_SELECTION_HASH = (
    "acda4c5abb3774d4bffdc499260f2856d9facb41fa300fd2907b9c944ccdfa84"
)
EXPECTED_CONFIRMATORY_INDICES_SHA256 = (
    "cd6d7ee5af704482876b00e4f3644373c10e0562985cbe3bfe4034d8e1e57b8c"
)

METHODS = r4.METHODS
METHOD_ATTACK = {
    "robust": "SquareRacing",
    "full_sigma0": "SquareRacing",
    "noise_only": "SquareAdaptive",
    "full": "SquareAdaptive",
}
SEARCH_SEEDS = r4.CONFIRMATORY_SEARCH_SEEDS
PILOT_SEARCH_SEEDS = r4.SEARCH_SEEDS
TOP_K = r4.TOP_K
EPSILON_PX = r4.EPSILON_PX
API_BUDGET = r4.API_BUDGET
SQUARE_P_INIT = r4.SQUARE_P_INIT
ADAPTIVE_CONFIG = dict(r4.ADAPTIVE_CONFIG)
RACING_CONFIG = dict(r4.RACING_CONFIG)
REEVAL_DRAWS = r4.REEVAL_DRAWS
REEVAL_SEED = r4.REEVAL_SEED
REEVAL_BATCH_SIZE = r4.REEVAL_BATCH_SIZE
REEVAL_MC_CHUNK = r4.REEVAL_MC_CHUNK
BOOTSTRAP_RESAMPLES = 50000
BOOTSTRAP_SEED = 2026090702
SEARCH_BATCH_SIZE_BY_METHOD = dict(r4.SEARCH_BATCH_SIZE_BY_METHOD)
EXPECTED_FROZEN_DEFENSE = dict(r4.EXPECTED_FROZEN_DEFENSE)

SCRIPT_NAMES = (
    "topk_cifar100_confirmatory_r5_common.py",
    "search_topk_cifar100_confirmatory_r5.py",
    "reevaluate_topk_cifar100_confirmatory_r5_method.py",
    "finalize_topk_cifar100_confirmatory_r5.py",
    "audit_topk_cifar100_confirmatory_r5.py",
    "test_topk_cifar100_confirmatory_r5.py",
    "topk_cifar100_confirmatory_r5_launcher.py",
    "run_topk_cifar100_confirmatory_r5_robust.py",
    "run_topk_cifar100_confirmatory_r5_full_sigma0.py",
    "run_topk_cifar100_confirmatory_r5_noise_only.py",
    "run_topk_cifar100_confirmatory_r5_full.py",
    "run_topk_cifar100_confirmatory_r5_finalize.py",
)

DEPENDENCY_NAMES = (
    "topk_cifar100_attack_pilot_r4_common.py",
    "topk_cifar100_square_attack_core_r4.py",
    "search_topk_cifar100_attack_pilot_r4.py",
    "reevaluate_topk_cifar100_attack_pilot_r4_mc1000.py",
    "calibrate_topk_cifar100_actual_defense_r1.py",
    "topk_cifar100_defense_calibration_common.py",
    "topk_cifar100_r2_common.py",
    "train_resnet20_teacher_top2_adversarial_cifar100_r2.py",
)

PRIMARY_COMPARISON = "full_minus_robust_attacked_exact_set"
SECONDARY_COMPARISONS = (
    "noise_only_minus_robust_attacked_exact_set",
    "full_sigma0_minus_robust_attacked_exact_set",
    "full_minus_noise_only_attacked_exact_set",
    "full_minus_full_sigma0_attacked_exact_set",
)


method_spec = r4.method_spec
canonical_json_bytes = r4.canonical_json_bytes
object_sha256 = r4.object_sha256
file_sha256 = r4.file_sha256
sequence_sha256 = r4.sequence_sha256
tensor_sha256 = r4.tensor_sha256
read_json = r4.read_json
read_csv = r4.read_csv
atomic_json = r4.atomic_json
atomic_csv = r4.atomic_csv
write_or_validate_json = r4.write_or_validate_json
load_parent_authorization = r4.load_parent_authorization
validate_dataset_against_parent = r4.validate_dataset_against_parent


def _validate_self_hash(payload: Mapping[str, Any], field: str, label: str) -> str:
    claimed = str(payload.get(field, ""))
    base = dict(payload)
    base.pop(field, None)
    if not claimed or object_sha256(base) != claimed:
        raise AssertionError(f"{label} self-hash mismatch")
    return claimed


def load_r4_freeze(r4_root: Path, parent: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the pilot result that is allowed to determine R5 attack families."""

    r4_root = Path(r4_root).expanduser().resolve()
    paths = {
        "protocol": r4_root / "pilot_protocol.json",
        "freeze": r4_root / "common_mc1000" / "confirmatory_attack_freeze.json",
        "final": r4_root / "common_mc1000" / "final_summary.json",
        "selected": r4_root / "common_mc1000" / "selected_attack_envelope.csv",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing audited R4 freeze artifacts:\n  " + "\n  ".join(missing))

    protocol = read_json(paths["protocol"])
    protocol_hash = r4.validate_protocol(protocol)
    if protocol_hash != EXPECTED_R4_PROTOCOL_HASH:
        raise AssertionError("R4 protocol hash differs from the audited pilot")
    if Path(protocol["output_root"]).expanduser().resolve() != r4_root:
        raise AssertionError("R4 protocol output root differs from the selected pilot root")
    if protocol.get("parent_handoff_hash") != parent["handoff_hash"]:
        raise AssertionError("R4 pilot and R3 parent handoff identities differ")
    if protocol.get("confirmatory_indices_sha256") != EXPECTED_CONFIRMATORY_INDICES_SHA256:
        raise AssertionError("R4 confirmatory split hash changed")
    if protocol.get("confirmatory_search_seeds_predeclared") != list(SEARCH_SEEDS):
        raise AssertionError("R4 predeclared confirmatory seeds changed")

    for name in r4.SCRIPT_NAMES:
        path = r4.PROJECT_DIR / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing R4 source dependency: {path}")
        if file_sha256(path) != protocol.get("script_sha256", {}).get(name):
            raise AssertionError(f"R4 source changed after the pilot freeze: {name}")

    freeze = read_json(paths["freeze"])
    selection_hash = _validate_self_hash(freeze, "selection_hash", "R4 selection freeze")
    if selection_hash != EXPECTED_R4_SELECTION_HASH:
        raise AssertionError("R4 attack-family selection hash differs from the audited result")
    if freeze.get("protocol_hash") != protocol_hash:
        raise AssertionError("R4 freeze/protocol identity mismatch")
    if freeze.get("status") != "PASS":
        raise PermissionError("R4 selection freeze did not pass")
    if freeze.get("authorized_next_stage") != "disjoint_1000_example_confirmatory_attack":
        raise PermissionError("R4 freeze does not authorize this confirmatory stage")
    if freeze.get("paper_claim_authorized_by_pilot") is not False:
        raise AssertionError("R4 pilot authorization flag is invalid")
    partition = freeze.get("confirmatory_partition", {})
    if partition != {
        "examples": 1000,
        "indices_sha256": EXPECTED_CONFIRMATORY_INDICES_SHA256,
        "outcomes_read": False,
    }:
        raise AssertionError("R4 confirmatory partition declaration changed")
    if freeze.get("confirmatory_search_seeds") != list(SEARCH_SEEDS):
        raise AssertionError("R4 freeze confirmatory seeds changed")
    if freeze.get("defense_frozen") != EXPECTED_FROZEN_DEFENSE:
        raise AssertionError("R4 freeze defense differs from the locked R3 defense")
    if freeze.get("attack_configs") != {
        "SquareAdaptive": ADAPTIVE_CONFIG,
        "SquareRacing": RACING_CONFIG,
    }:
        raise AssertionError("R4 frozen attack configurations changed")
    observed_map = {
        method: str(freeze.get("method_attack_selection", {}).get(method, {}).get("attack", ""))
        for method in METHODS
    }
    if observed_map != METHOD_ATTACK:
        raise AssertionError(f"Unexpected R4 method/attack freeze: {observed_map}")
    if freeze.get("no_per_example_candidate_selection") is not True:
        raise AssertionError("R4 freeze permits per-example candidate selection")

    selected_rows = read_csv(paths["selected"])
    if len(selected_rows) != len(METHODS):
        raise AssertionError("R4 selected attack table must have four method rows")
    selected_map = {row["method"]: row["selected_attack"] for row in selected_rows}
    if selected_map != METHOD_ATTACK:
        raise AssertionError("R4 selected attack CSV differs from its JSON freeze")
    if any(str(row.get("paper_claim_authorized", "")).lower() not in {"false", "0"} for row in selected_rows):
        raise AssertionError("R4 pilot CSV incorrectly authorizes paper claims")

    final = read_json(paths["final"])
    if final.get("status") != "PASS":
        raise PermissionError("R4 pilot final summary did not pass")
    if final.get("confirmatory_attack_freeze_hash") != selection_hash:
        raise AssertionError("R4 final/freeze hash mismatch")
    if final.get("confirmatory_partition_outcomes_read") is not False:
        raise AssertionError("R4 final summary reports confirmatory outcome access")
    if final.get("paper_claim_status") != "pilot_is_exploratory_only":
        raise AssertionError("R4 final summary overstates the pilot")
    return {
        "r4_root": r4_root,
        "paths": paths,
        "protocol": protocol,
        "protocol_hash": protocol_hash,
        "freeze": freeze,
        "selection_hash": selection_hash,
        "final": final,
    }


def load_frozen_confirmatory_references(
    parent_dir: Path, confirmatory_indices: Sequence[int]
) -> tuple[list[int], list[list[int]]]:
    """Load exact ranked Top-2 references previously recorded by audited R3."""

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
    for index in (int(value) for value in confirmatory_indices):
        row = by_index.get(index)
        if row is None:
            raise AssertionError(f"R3 standard rows miss confirmatory index {index}")
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
    r4_info: Mapping[str, Any],
    data_dir: Path,
    r2_root: Path,
    dev_dir: Path,
    output_root: Path,
) -> dict[str, Any]:
    scripts = {name: file_sha256(PROJECT_DIR / name) for name in SCRIPT_NAMES}
    dependencies = {name: file_sha256(PROJECT_DIR / name) for name in DEPENDENCY_NAMES}
    payload: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "parent_protocol_id": PARENT_PROTOCOL_ID,
        "r4_protocol_id": R4_PROTOCOL_ID,
        "parent_config_hash": parent["config_hash"],
        "parent_handoff_hash": parent["handoff_hash"],
        "parent_manifest_hash": parent["manifest_hash"],
        "r4_protocol_hash": r4_info["protocol_hash"],
        "r4_selection_hash": r4_info["selection_hash"],
        "dataset": "CIFAR-100",
        "architecture": "ResNet-20",
        "data_dir": str(Path(data_dir).expanduser().resolve()),
        "r2_root": str(Path(r2_root).expanduser().resolve()),
        "dev_dir": str(Path(dev_dir).expanduser().resolve()),
        "parent_dir": str(Path(parent["parent_dir"])),
        "r4_root": str(Path(r4_info["r4_root"])),
        "output_root": str(Path(output_root).expanduser().resolve()),
        "split": "predeclared_disjoint_official_test_confirmatory_only",
        "examples": 1000,
        "examples_per_class": 10,
        "indices_sha256": EXPECTED_CONFIRMATORY_INDICES_SHA256,
        "pilot_indices_sha256": r4_info["protocol"]["pilot_indices_sha256"],
        "pilot_and_confirmatory_disjoint": True,
        "methods": list(METHODS),
        "method_attack": dict(METHOD_ATTACK),
        "search_seeds": list(SEARCH_SEEDS),
        "pilot_search_seeds": list(PILOT_SEARCH_SEEDS),
        "top_k": TOP_K,
        "epsilon_px": EPSILON_PX,
        "api_budget_per_example": API_BUDGET,
        "square_p_init": SQUARE_P_INIT,
        "attack_configs": {
            "SquareAdaptive": dict(ADAPTIVE_CONFIG),
            "SquareRacing": dict(RACING_CONFIG),
        },
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
        "frozen_defense": dict(EXPECTED_FROZEN_DEFENSE),
        "checkpoints": parent["handoff"]["checkpoints"],
        "analysis_plan": {
            "unit": "example",
            "seed_aggregation": "arithmetic_mean_over_three_fixed_attack_search_seeds",
            "primary_comparison": PRIMARY_COMPARISON,
            "primary_decision_rule": "paired_bootstrap_95pct_lower_bound_gt_0",
            "secondary_comparisons": list(SECONDARY_COMPARISONS),
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "no_attack_family_selection": True,
            "no_seed_selection": True,
            "no_per_example_candidate_selection": True,
            "primary_claim_scope": (
                "Frozen full defense versus robust anchor on the disjoint CIFAR-100 "
                "ResNet-20 k=2, epsilon=2/255, Q=5000 confirmatory split"
            ),
        },
        "script_sha256": scripts,
        "dependency_sha256": dependencies,
        "no_training": True,
        "no_defense_retuning": True,
        "no_attack_retuning": True,
        "no_attack_family_selection": True,
        "no_seed_selection": True,
        "budgets_run_independently": True,
        "every_eot_refresh_race_confirmation_draw_charged": True,
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
        raise AssertionError("R5 protocol hash mismatch")
    expected = {
        "protocol_id": PROTOCOL_ID,
        "parent_protocol_id": PARENT_PROTOCOL_ID,
        "r4_protocol_id": R4_PROTOCOL_ID,
        "r4_protocol_hash": EXPECTED_R4_PROTOCOL_HASH,
        "r4_selection_hash": EXPECTED_R4_SELECTION_HASH,
        "dataset": "CIFAR-100",
        "architecture": "ResNet-20",
        "split": "predeclared_disjoint_official_test_confirmatory_only",
        "examples": 1000,
        "examples_per_class": 10,
        "indices_sha256": EXPECTED_CONFIRMATORY_INDICES_SHA256,
        "pilot_and_confirmatory_disjoint": True,
        "methods": list(METHODS),
        "method_attack": dict(METHOD_ATTACK),
        "search_seeds": list(SEARCH_SEEDS),
        "pilot_search_seeds": list(PILOT_SEARCH_SEEDS),
        "top_k": TOP_K,
        "epsilon_px": EPSILON_PX,
        "api_budget_per_example": API_BUDGET,
        "square_p_init": SQUARE_P_INIT,
        "frozen_defense": EXPECTED_FROZEN_DEFENSE,
        "no_training": True,
        "no_defense_retuning": True,
        "no_attack_retuning": True,
        "no_attack_family_selection": True,
        "no_seed_selection": True,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise AssertionError(f"Frozen R5 field changed: {key}")
    if payload.get("attack_configs") != {
        "SquareAdaptive": ADAPTIVE_CONFIG,
        "SquareRacing": RACING_CONFIG,
    }:
        raise AssertionError("R5 attack configurations changed")
    if payload.get("search_batch_size_by_method") != SEARCH_BATCH_SIZE_BY_METHOD:
        raise AssertionError("R5 search batch-size map changed")
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
        raise AssertionError("R5 candidate-reevaluation configuration changed")
    plan = payload.get("analysis_plan", {})
    if plan.get("primary_comparison") != PRIMARY_COMPARISON:
        raise AssertionError("R5 primary comparison changed")
    if plan.get("secondary_comparisons") != list(SECONDARY_COMPARISONS):
        raise AssertionError("R5 secondary comparisons changed")
    if plan.get("bootstrap_resamples") != BOOTSTRAP_RESAMPLES:
        raise AssertionError("R5 bootstrap count changed")
    if plan.get("bootstrap_seed") != BOOTSTRAP_SEED:
        raise AssertionError("R5 bootstrap seed changed")
    return claimed


def cell_dir(output_root: Path, method: str, seed: int) -> Path:
    return (
        Path(output_root)
        / "search"
        / method
        / METHOD_ATTACK[method]
        / f"seed_{int(seed)}"
    )


def method_reeval_dir(output_root: Path, method: str) -> Path:
    return Path(output_root) / "reevaluation" / method


def expected_cells() -> list[tuple[str, str, int]]:
    return [
        (method, METHOD_ATTACK[method], seed)
        for method in METHODS
        for seed in SEARCH_SEEDS
    ]


def aligned_values(
    rows: Sequence[Mapping[str, Any]], *, method: str, metric: str
) -> tuple[list[int], np.ndarray]:
    selected = [row for row in rows if str(row["method"]) == method]
    by_seed: dict[int, dict[int, float]] = {}
    for row in selected:
        attack = str(row["attack"])
        if attack != METHOD_ATTACK[method]:
            raise ValueError(f"Unexpected attack row for {method}: {attack}")
        seed = int(row["search_seed"])
        index = int(row["sample_index"])
        if index in by_seed.setdefault(seed, {}):
            raise ValueError(f"Duplicate row for {method}/{seed}/{index}")
        by_seed[seed][index] = float(row[metric])
    if set(by_seed) != set(SEARCH_SEEDS):
        raise ValueError(f"Missing confirmatory search seeds for {method}")
    indices = sorted(next(iter(by_seed.values())))
    if any(sorted(values) != indices for values in by_seed.values()):
        raise ValueError(f"Seed rows are not aligned for {method}")
    matrix = np.asarray(
        [[by_seed[seed][index] for index in indices] for seed in SEARCH_SEEDS],
        dtype=np.float64,
    )
    return indices, matrix.mean(axis=0)


def clean_values(
    rows: Sequence[Mapping[str, Any]], *, method: str, metric: str
) -> tuple[list[int], np.ndarray]:
    first_seed = SEARCH_SEEDS[0]
    selected = [
        row for row in rows
        if str(row["method"]) == method and int(row["search_seed"]) == first_seed
    ]
    by_index = {int(row["sample_index"]): float(row[metric]) for row in selected}
    if len(by_index) != 1000:
        raise ValueError(f"Expected 1,000 clean values for {method}")
    indices = sorted(by_index)
    return indices, np.asarray([by_index[index] for index in indices], dtype=np.float64)


def paired_bootstrap(
    delta: np.ndarray,
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    chunk_size: int = 250,
) -> dict[str, float | int | str]:
    delta = np.asarray(delta, dtype=np.float64)
    if delta.ndim != 1 or len(delta) == 0 or not np.isfinite(delta).all():
        raise ValueError("Bootstrap delta must be a finite non-empty vector")
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
            "positive" if low > 0
            else "negative" if high < 0
            else "statistically_indistinguishable"
        ),
    }


def assert_probability(value: float, label: str) -> None:
    if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
        raise AssertionError(f"Probability outside [0,1] for {label}: {value}")


def validate_confirmatory_labels(labels: Sequence[int]) -> None:
    if len(labels) != 1000:
        raise AssertionError("Confirmatory split must contain exactly 1,000 labels")
    if Counter(int(value) for value in labels) != Counter(
        {class_id: 10 for class_id in range(100)}
    ):
        raise AssertionError("Confirmatory split is not balanced at 10 examples per class")

