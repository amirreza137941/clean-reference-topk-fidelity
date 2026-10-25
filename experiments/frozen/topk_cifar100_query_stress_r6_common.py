#!/usr/bin/env python3
"""Frozen helpers for the post-confirmatory CIFAR-100 Q=5k/20k stress test."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

import topk_cifar100_confirmatory_r5_common as r5


PROTOCOL_ID = "cifar100_query_budget_stress_r6_20260908"
PARENT_PROTOCOL_ID = r5.PROTOCOL_ID

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = r5.DEFAULT_DATA_DIR
DEFAULT_R2_ROOT = r5.DEFAULT_R2_ROOT
DEFAULT_DEV_DIR = r5.DEFAULT_DEV_DIR
DEFAULT_R3_PARENT_DIR = r5.DEFAULT_PARENT_DIR
DEFAULT_R4_ROOT = r5.DEFAULT_R4_ROOT
DEFAULT_R5_ROOT = r5.DEFAULT_OUTPUT_ROOT
DEFAULT_OUTPUT_ROOT = Path(
    "./models/resnet20/"
    "paper_topk_cifar100_query_stress_k2_q5k_q20k_r6"
)

EXPECTED_R5_PROTOCOL_HASH = (
    "5cea83fc79fa5f1c1b4ee2e3674a87aa6a7172483d2f428e191ca2cdf953bfba"
)
EXPECTED_R5_FINAL_SUMMARY_HASH = (
    "4327b1e20c51bb1fa7b2a4c44c5d82d1a7a1b39e7a0c65f042821bb862b8d778"
)
EXPECTED_R5_HANDOFF_HASH = (
    "c3c7c94c92273cd8317701a306b75dd1c05d44ea91dca331753c3fbcebee70fd"
)

METHODS = r5.METHODS
METHOD_ATTACK = dict(r5.METHOD_ATTACK)
BUDGETS = (5000, 20000)
SEARCH_SEEDS = (111071, 122073, 133083)
STRESS_SELECTION_SEED = 2026090801
STRESS_EXAMPLES = 200
STRESS_PER_CLASS = 2
TOP_K = r5.TOP_K
EPSILON_PX = r5.EPSILON_PX
SQUARE_P_INIT = r5.SQUARE_P_INIT
ADAPTIVE_CONFIG = dict(r5.ADAPTIVE_CONFIG)
RACING_CONFIG = dict(r5.RACING_CONFIG)
REEVAL_DRAWS = 1000
REEVAL_SEED = 2026090802
REEVAL_BATCH_SIZE = r5.REEVAL_BATCH_SIZE
REEVAL_MC_CHUNK = r5.REEVAL_MC_CHUNK
BOOTSTRAP_RESAMPLES = 50000
BOOTSTRAP_SEED = 2026090803
SEARCH_BATCH_SIZE_BY_METHOD = dict(r5.SEARCH_BATCH_SIZE_BY_METHOD)
EXPECTED_FROZEN_DEFENSE = dict(r5.EXPECTED_FROZEN_DEFENSE)

SCRIPT_NAMES = (
    "topk_cifar100_query_stress_r6_common.py",
    "search_topk_cifar100_query_stress_r6.py",
    "reevaluate_topk_cifar100_query_stress_r6_method.py",
    "finalize_topk_cifar100_query_stress_r6.py",
    "audit_topk_cifar100_query_stress_r6.py",
    "test_topk_cifar100_query_stress_r6.py",
    "topk_cifar100_query_stress_r6_launcher.py",
    "run_topk_cifar100_query_stress_r6_light.py",
    "run_topk_cifar100_query_stress_r6_full.py",
    "run_topk_cifar100_query_stress_r6_finalize.py",
)

DEPENDENCY_NAMES = (
    "topk_cifar100_confirmatory_r5_common.py",
    "topk_cifar100_attack_pilot_r4_common.py",
    "topk_cifar100_square_attack_core_r4.py",
    "search_topk_cifar100_attack_pilot_r4.py",
    "reevaluate_topk_cifar100_attack_pilot_r4_mc1000.py",
    "calibrate_topk_cifar100_actual_defense_r1.py",
    "topk_cifar100_defense_calibration_common.py",
    "topk_cifar100_r2_common.py",
    "train_resnet20_teacher_top2_adversarial_cifar100_r2.py",
)

method_spec = r5.method_spec
canonical_json_bytes = r5.canonical_json_bytes
object_sha256 = r5.object_sha256
file_sha256 = r5.file_sha256
sequence_sha256 = r5.sequence_sha256
tensor_sha256 = r5.tensor_sha256
read_json = r5.read_json
read_csv = r5.read_csv
atomic_json = r5.atomic_json
atomic_csv = r5.atomic_csv
write_or_validate_json = r5.write_or_validate_json
assert_probability = r5.assert_probability


def _validate_self_hash(payload: Mapping[str, Any], field: str, label: str) -> str:
    claimed = str(payload.get(field, ""))
    base = dict(payload)
    base.pop(field, None)
    if not claimed or object_sha256(base) != claimed:
        raise AssertionError(f"{label} self-hash mismatch")
    return claimed


def load_r5_completion(r5_root: Path) -> dict[str, Any]:
    """Validate the exact R5 result that motivates, but does not tune, R6."""

    r5_root = Path(r5_root).expanduser().resolve()
    analysis = r5_root / "confirmatory_analysis"
    paths = {
        "protocol": r5_root / "confirmatory_protocol.json",
        "final": analysis / "final_summary.json",
        "handoff": analysis / "paper_claim_handoff.json",
        "methods": analysis / "confirmatory_method_summary.csv",
        "comparisons": analysis / "confirmatory_paired_bootstrap.csv",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing audited R5 artifacts:\n  " + "\n  ".join(missing))

    protocol = read_json(paths["protocol"])
    protocol_hash = r5.validate_protocol(protocol)
    if protocol_hash != EXPECTED_R5_PROTOCOL_HASH:
        raise AssertionError("R5 protocol differs from the audited confirmatory run")
    if Path(protocol["output_root"]).expanduser().resolve() != r5_root:
        raise AssertionError("R5 protocol output root differs from the selected result root")

    # R6 executes the same audited attack/defense dependencies. Refuse to run if
    # those bytes changed after R5 was frozen.
    for name in r5.SCRIPT_NAMES:
        path = r5.PROJECT_DIR / name
        if not path.is_file() or file_sha256(path) != protocol.get("script_sha256", {}).get(name):
            raise AssertionError(f"Frozen R5 source changed: {name}")
    for name in r5.DEPENDENCY_NAMES:
        path = r5.PROJECT_DIR / name
        if not path.is_file() or file_sha256(path) != protocol.get("dependency_sha256", {}).get(name):
            raise AssertionError(f"Frozen R5 dependency changed: {name}")

    final = read_json(paths["final"])
    final_hash = _validate_self_hash(final, "final_summary_hash", "R5 final summary")
    if final_hash != EXPECTED_R5_FINAL_SUMMARY_HASH:
        raise AssertionError("R5 final summary differs from the audited result")
    if final.get("protocol_hash") != protocol_hash or final.get("status") != "PASS":
        raise AssertionError("R5 final/protocol identity or status mismatch")
    if final.get("paper_claim_authorized") is not True:
        raise PermissionError("R5 primary confirmation did not pass")

    handoff = read_json(paths["handoff"])
    handoff_hash = _validate_self_hash(handoff, "handoff_hash", "R5 paper handoff")
    if handoff_hash != EXPECTED_R5_HANDOFF_HASH:
        raise AssertionError("R5 paper handoff differs from the audited result")
    if handoff.get("protocol_hash") != protocol_hash or handoff.get("authorized") is not True:
        raise AssertionError("R5 handoff/protocol identity or authorization mismatch")
    if final.get("paper_claim_handoff_hash") != handoff_hash:
        raise AssertionError("R5 final and handoff hashes disagree")

    method_rows = read_csv(paths["methods"])
    comparison_rows = read_csv(paths["comparisons"])
    if len(method_rows) != 4 or len(comparison_rows) != 16:
        raise AssertionError("R5 summary table dimensions changed")
    primary = next(
        (row for row in comparison_rows if row.get("comparison") == r5.PRIMARY_COMPARISON),
        None,
    )
    if primary is None or primary.get("status") != "positive":
        raise AssertionError("R5 primary result is missing or no longer positive")
    if abs(float(primary["difference_pp"]) - 21.2887) > 1e-9:
        raise AssertionError("R5 primary estimate changed")
    return {
        "root": r5_root,
        "paths": paths,
        "protocol": protocol,
        "protocol_hash": protocol_hash,
        "final": final,
        "final_hash": final_hash,
        "handoff": handoff,
        "handoff_hash": handoff_hash,
        "method_rows": method_rows,
        "comparison_rows": comparison_rows,
    }


def _selection_rank(class_id: int, sample_index: int) -> bytes:
    text = f"{STRESS_SELECTION_SEED}|class={int(class_id)}|index={int(sample_index)}"
    return hashlib.sha256(text.encode("ascii")).digest()


def official_standard_rows(parent_dir: Path) -> dict[int, dict[str, str]]:
    path = Path(parent_dir).expanduser().resolve() / "official_clean_per_example.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing audited R3 official rows: {path}")
    rows = [row for row in read_csv(path) if row.get("method") == "standard"]
    if len(rows) != 10000:
        raise AssertionError("R3 must contain exactly 10,000 standard official-test rows")
    by_index: dict[int, dict[str, str]] = {}
    for row in rows:
        index = int(row["sample_index"])
        if index in by_index:
            raise AssertionError(f"Duplicate R3 standard sample index: {index}")
        by_index[index] = dict(row)
    if set(by_index) != set(range(10000)):
        raise AssertionError("R3 standard rows do not cover CIFAR-100 test indices 0..9999")
    return by_index


def derive_stress_indices(parent: Mapping[str, Any]) -> list[int]:
    """Select two unused examples per class using labels only and a fixed hash seed."""

    by_index = official_standard_rows(Path(parent["parent_dir"]))
    excluded = set(int(value) for value in parent["pilot_indices"])
    excluded.update(int(value) for value in parent["confirmatory_indices"])
    selected: list[int] = []
    for class_id in range(100):
        candidates = [
            index
            for index, row in by_index.items()
            if index not in excluded and int(row["label"]) == class_id
        ]
        if len(candidates) != 88:
            raise AssertionError(
                f"Expected 88 unused examples for class {class_id}, found {len(candidates)}"
            )
        candidates.sort(key=lambda index: _selection_rank(class_id, index))
        selected.extend(candidates[:STRESS_PER_CLASS])
    validate_stress_indices(selected, parent, by_index)
    return selected


def validate_stress_indices(
    indices: Sequence[int],
    parent: Mapping[str, Any],
    by_index: Mapping[int, Mapping[str, str]] | None = None,
) -> None:
    indices = [int(value) for value in indices]
    if len(indices) != STRESS_EXAMPLES or len(set(indices)) != STRESS_EXAMPLES:
        raise AssertionError("R6 stress split must contain 200 unique examples")
    excluded = set(int(value) for value in parent["pilot_indices"])
    excluded.update(int(value) for value in parent["confirmatory_indices"])
    if set(indices) & excluded:
        raise AssertionError("R6 stress split overlaps R4 or R5")
    rows = by_index if by_index is not None else official_standard_rows(Path(parent["parent_dir"]))
    labels = [int(rows[index]["label"]) for index in indices]
    if Counter(labels) != Counter({class_id: STRESS_PER_CLASS for class_id in range(100)}):
        raise AssertionError("R6 stress split is not balanced at two examples per class")


def load_stress_references(
    parent: Mapping[str, Any], indices: Sequence[int]
) -> tuple[list[int], list[list[int]]]:
    by_index = official_standard_rows(Path(parent["parent_dir"]))
    validate_stress_indices(indices, parent, by_index)
    labels: list[int] = []
    references: list[list[int]] = []
    for index in (int(value) for value in indices):
        row = by_index[index]
        members = [int(value) for value in str(row["reference_set"]).split("|")]
        if len(members) != 2 or len(set(members)) != 2:
            raise AssertionError(f"Invalid R3 Top-2 reference set at index {index}")
        top1 = int(row["reference_top1"])
        if top1 not in members:
            raise AssertionError(f"R3 Top-1 is outside Top-2 at index {index}")
        second = members[0] if members[1] == top1 else members[1]
        labels.append(int(row["label"]))
        references.append([top1, second])
    return labels, references


def build_protocol_payload(
    *,
    parent: Mapping[str, Any],
    r5_info: Mapping[str, Any],
    stress_indices: Sequence[int],
    data_dir: Path,
    r2_root: Path,
    dev_dir: Path,
    output_root: Path,
) -> dict[str, Any]:
    validate_stress_indices(stress_indices, parent)
    scripts = {name: file_sha256(PROJECT_DIR / name) for name in SCRIPT_NAMES}
    dependencies = {name: file_sha256(PROJECT_DIR / name) for name in DEPENDENCY_NAMES}
    payload: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "parent_protocol_id": PARENT_PROTOCOL_ID,
        "r3_parent_protocol_id": r5.PARENT_PROTOCOL_ID,
        "r3_parent_config_hash": parent["config_hash"],
        "r3_parent_handoff_hash": parent["handoff_hash"],
        "r3_parent_manifest_hash": parent["manifest_hash"],
        "r5_protocol_hash": r5_info["protocol_hash"],
        "r5_final_summary_hash": r5_info["final_hash"],
        "r5_paper_handoff_hash": r5_info["handoff_hash"],
        "dataset": "CIFAR-100",
        "architecture": "ResNet-20",
        "data_dir": str(Path(data_dir).expanduser().resolve()),
        "r2_root": str(Path(r2_root).expanduser().resolve()),
        "dev_dir": str(Path(dev_dir).expanduser().resolve()),
        "r3_parent_dir": str(Path(parent["parent_dir"])),
        "r4_root": str(DEFAULT_R4_ROOT.expanduser().resolve()),
        "r5_root": str(Path(r5_info["root"])),
        "output_root": str(Path(output_root).expanduser().resolve()),
        "split": "post_confirmatory_disjoint_query_budget_stress",
        "stress_examples": STRESS_EXAMPLES,
        "stress_per_class": STRESS_PER_CLASS,
        "stress_selection_seed": STRESS_SELECTION_SEED,
        "stress_indices": [int(value) for value in stress_indices],
        "stress_indices_sha256": sequence_sha256(stress_indices),
        "excluded_r4_indices_sha256": sequence_sha256(parent["pilot_indices"]),
        "excluded_r5_indices_sha256": sequence_sha256(parent["confirmatory_indices"]),
        "stress_is_disjoint_from_r4_r5": True,
        "selection_uses_labels_only": True,
        "methods": list(METHODS),
        "method_attack": dict(METHOD_ATTACK),
        "budgets": list(BUDGETS),
        "search_seeds": list(SEARCH_SEEDS),
        "top_k": TOP_K,
        "epsilon_px": EPSILON_PX,
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
            "seed_aggregation": "arithmetic_mean_over_three_fixed_search_seeds",
            "budget_comparison": "paired_Q20000_minus_Q5000_on_identical_examples",
            "cap_envelope": "method_level_lowest_three_seed_mean_budget_not_per_example",
            "cross_method_comparisons": [
                "full_minus_robust",
                "noise_only_minus_robust",
                "full_sigma0_minus_robust",
                "full_minus_noise_only",
                "full_minus_full_sigma0",
            ],
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "exploratory_post_confirmatory_stress": True,
            "paper_primary_claim_authorized": False,
            "no_per_example_budget_selection": True,
            "no_attack_family_selection": True,
            "no_seed_selection": True,
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
        "r5_results_used_only_to_motivate_budget_stress": True,
    }
    payload["protocol_hash"] = object_sha256(payload)
    return payload


def validate_protocol(payload: Mapping[str, Any]) -> str:
    claimed = str(payload.get("protocol_hash", ""))
    base = dict(payload)
    base.pop("protocol_hash", None)
    if not claimed or object_sha256(base) != claimed:
        raise AssertionError("R6 protocol hash mismatch")
    expected = {
        "protocol_id": PROTOCOL_ID,
        "parent_protocol_id": PARENT_PROTOCOL_ID,
        "r5_protocol_hash": EXPECTED_R5_PROTOCOL_HASH,
        "r5_final_summary_hash": EXPECTED_R5_FINAL_SUMMARY_HASH,
        "r5_paper_handoff_hash": EXPECTED_R5_HANDOFF_HASH,
        "dataset": "CIFAR-100",
        "architecture": "ResNet-20",
        "split": "post_confirmatory_disjoint_query_budget_stress",
        "stress_examples": STRESS_EXAMPLES,
        "stress_per_class": STRESS_PER_CLASS,
        "stress_selection_seed": STRESS_SELECTION_SEED,
        "stress_is_disjoint_from_r4_r5": True,
        "selection_uses_labels_only": True,
        "methods": list(METHODS),
        "method_attack": dict(METHOD_ATTACK),
        "budgets": list(BUDGETS),
        "search_seeds": list(SEARCH_SEEDS),
        "top_k": TOP_K,
        "epsilon_px": EPSILON_PX,
        "square_p_init": SQUARE_P_INIT,
        "frozen_defense": EXPECTED_FROZEN_DEFENSE,
        "no_training": True,
        "no_defense_retuning": True,
        "no_attack_retuning": True,
        "no_attack_family_selection": True,
        "no_seed_selection": True,
        "budgets_run_independently": True,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise AssertionError(f"Frozen R6 protocol field changed: {key}")
    indices = [int(value) for value in payload.get("stress_indices", [])]
    if len(indices) != STRESS_EXAMPLES or sequence_sha256(indices) != payload.get(
        "stress_indices_sha256"
    ):
        raise AssertionError("R6 stress index list/hash mismatch")
    if payload.get("attack_configs") != {
        "SquareAdaptive": ADAPTIVE_CONFIG,
        "SquareRacing": RACING_CONFIG,
    }:
        raise AssertionError("R6 attack configurations changed")
    if payload.get("search_batch_size_by_method") != SEARCH_BATCH_SIZE_BY_METHOD:
        raise AssertionError("R6 search batch-size map changed")
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
        raise AssertionError("R6 candidate-reevaluation settings changed")
    plan = payload.get("analysis_plan", {})
    if plan.get("paper_primary_claim_authorized") is not False:
        raise AssertionError("R6 must remain an exploratory budget stress test")
    if plan.get("no_per_example_budget_selection") is not True:
        raise AssertionError("R6 protocol permits forbidden per-example budget selection")
    return claimed


def cell_dir(output_root: Path, method: str, budget: int, seed: int) -> Path:
    return (
        Path(output_root)
        / "search"
        / method
        / METHOD_ATTACK[method]
        / f"q{int(budget)}"
        / f"seed_{int(seed)}"
    )


def method_reeval_dir(output_root: Path, method: str) -> Path:
    return Path(output_root) / "reevaluation" / method


def expected_cells() -> list[tuple[str, str, int, int]]:
    return [
        (method, METHOD_ATTACK[method], budget, seed)
        for method in METHODS
        for budget in BUDGETS
        for seed in SEARCH_SEEDS
    ]


def aligned_values(
    rows: Sequence[Mapping[str, Any]], *, method: str, budget: int, metric: str
) -> tuple[list[int], np.ndarray]:
    selected = [
        row
        for row in rows
        if str(row["method"]) == method and int(row["api_budget"]) == int(budget)
    ]
    by_seed: dict[int, dict[int, float]] = {}
    for row in selected:
        if str(row["attack"]) != METHOD_ATTACK[method]:
            raise ValueError(f"Unexpected R6 attack row for {method}")
        seed = int(row["search_seed"])
        index = int(row["sample_index"])
        if index in by_seed.setdefault(seed, {}):
            raise ValueError(f"Duplicate R6 row for {method}/Q{budget}/{seed}/{index}")
        by_seed[seed][index] = float(row[metric])
    if set(by_seed) != set(SEARCH_SEEDS):
        raise ValueError(f"Missing R6 seeds for {method}/Q{budget}")
    indices = sorted(next(iter(by_seed.values())))
    if len(indices) != STRESS_EXAMPLES:
        raise ValueError(f"Expected {STRESS_EXAMPLES} aligned R6 examples")
    if any(sorted(values) != indices for values in by_seed.values()):
        raise ValueError(f"R6 seed rows are not aligned for {method}/Q{budget}")
    matrix = np.asarray(
        [[by_seed[seed][index] for index in indices] for seed in SEARCH_SEEDS],
        dtype=np.float64,
    )
    return indices, matrix.mean(axis=0)


def clean_values(
    rows: Sequence[Mapping[str, Any]], *, method: str, metric: str
) -> tuple[list[int], np.ndarray]:
    first_budget = BUDGETS[0]
    first_seed = SEARCH_SEEDS[0]
    selected = [
        row
        for row in rows
        if str(row["method"]) == method
        and int(row["api_budget"]) == first_budget
        and int(row["search_seed"]) == first_seed
    ]
    by_index = {int(row["sample_index"]): float(row[metric]) for row in selected}
    if len(by_index) != STRESS_EXAMPLES:
        raise ValueError(f"Expected {STRESS_EXAMPLES} clean R6 values for {method}")
    indices = sorted(by_index)
    return indices, np.asarray([by_index[index] for index in indices], dtype=np.float64)


def paired_bootstrap(
    delta: np.ndarray,
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    chunk_size: int = 1000,
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
            "positive"
            if low > 0
            else "negative"
            if high < 0
            else "statistically_indistinguishable"
        ),
    }


def validate_script_inventory() -> None:
    missing = [
        str(PROJECT_DIR / name)
        for name in (*SCRIPT_NAMES, *DEPENDENCY_NAMES)
        if not (PROJECT_DIR / name).is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing R6/new or existing dependency files:\n  " + "\n  ".join(missing))

