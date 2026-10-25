#!/usr/bin/env python3
"""Immutable contracts for the ImageNet ResNet-50 post-hoc Q=20k stress test.

This diagnostic is deliberately downstream of the frozen R5 confirmatory run.
It may read R5 artifacts, but it must never modify them, retune the defense,
select a seed, or authorize/revise the preregistered R5 paper claim.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import topk_imagenet_confirmatory_r5_common as r5


PROTOCOL_ID = "imagenet_topk_posthoc_q20k_eot_stress_d1_20260918"
PARENT_PROTOCOL_ID = r5.PROTOCOL_ID
PROJECT_DIR = Path(__file__).resolve().parent

DEFAULT_R5_ROOT = Path(
    "./models/imagenet_topk/"
    "paper_topk_imagenet_confirmatory_attack_k5_q5k_r5"
)
DEFAULT_OUTPUT_ROOT = Path(
    "./models/imagenet_topk/"
    "paper_topk_imagenet_posthoc_q20k_eot_stress_d1"
)

EXPECTED_R5_PROTOCOL_HASH = (
    "d1f54fc176932e3148d0465ecd2b44e54629d783af6b7ce5b0ec60272b6e6c14"
)
EXPECTED_R5_PROTOCOL_FILE_SHA256 = (
    "ee16a19f0d4723ba9f99f34ece1da06473627d779a734deabaa0d65c391aee9b"
)
EXPECTED_R5_FINAL_SUMMARY_HASH = (
    "130cecf8c7cd2e8d91c852f5751d33825e44abda9331cc683ab61d4e031ac360"
)
EXPECTED_R5_FINAL_SUMMARY_FILE_SHA256 = (
    "26ef7868ff7ce4de3d492d92d40b298e4ec9b2ef58210b422e691f1dc61a1493"
)
EXPECTED_R5_MANIFEST_FILE_SHA256 = (
    "8bd2f20bb54aa668c8119bcd89ecf80841ed6900ae54b53200a11ae000a2ad3f"
)
EXPECTED_R5_INPUTS_FILE_SHA256 = (
    "366c0f2d7f30c0db1399958e7657d2e68d767a4d674093b9284ef10d8a7205ac"
)
EXPECTED_R5_PREPARE_FILE_SHA256 = (
    "a093995bc86af9540aaffe106c43bf0ce623f5c37501f5ba43b7ce8fa6620d34"
)
EXPECTED_R5_INPUTS_PAYLOAD_HASH = (
    "56c95ef6b0b4e76eb7ecf1b9ab08654d1725689dc37d5027f5290893dba413aa"
)
EXPECTED_R5_CLEAN_TENSOR_SHA256 = (
    "6700ef1d2638ca2e7874907e1981eb8dba3fab2810c2ae9bf9f23174eb268914"
)
EXPECTED_R5_PER_EXAMPLE_SHA256 = (
    "417469fdce738b3d7c6d174c77ccd04176aaa600ae5bd67d453645ff7534ae18"
)
EXPECTED_R5_HANDOFF_FILE_SHA256 = (
    "f788a1abb9d65071ab5f6d11fc23d040271535e3845dfe12714df25bc609148f"
)
EXPECTED_R5_HANDOFF_HASH = (
    "9a90b0125be77f275f476b2c2e0bbee8e1cc497fc163ad572e60c6ea9aad1153"
)
EXPECTED_SUBSET_ROWS_SHA256 = (
    "516d2cfd84e4ec9431c619171fc4125b4d82b7302796051beee8e96d71f91aac"
)
EXPECTED_ATTACK_CORE_SHA256 = (
    "0dc459984ce69173ab83bfceb0b8fb824a5d38b565550c4a6b4ea515c30dded7"
)

METHODS = ("robust", "full")
METHOD_ATTACK = {"robust": "SquareAdaptive", "full": "SquareAdaptive"}
SEARCH_SEEDS = (807019, 918031, 1029043)
DIAGNOSTIC_EXAMPLES = 100
PRIMARY_K = 5
SECONDARY_K = 2
EPSILON_PX = 4.0
API_BUDGET = 20_000
SQUARE_P_INIT = 0.05
REFERENCE_BATCH_SIZE = 16
SEARCH_BATCH_SIZE_BY_METHOD = {"robust": 16, "full": 8}

# The deterministic anchor does not benefit from repeated identical draws.
# Full receives the explicitly strengthened EOT=5 search/refresh schedule.
ADAPTIVE_CONFIG_BY_METHOD = {
    "robust": {
        "search_eot": 1,
        "refresh_eot": 1,
        "confirm_candidates": 5,
        "confirm_fraction": 0.20,
        "objective": "exact_top5_set_margin_of_mean_logits",
    },
    "full": {
        "search_eot": 5,
        "refresh_eot": 5,
        "confirm_candidates": 5,
        "confirm_fraction": 0.20,
        "objective": "exact_top5_set_margin_of_mean_logits_eot5",
    },
}

# Kept only because the audited generic core requires both configurations.
RACING_CONFIG = {
    "search_eot": 5,
    "refresh_eot": 5,
    "pool_size": 8,
    "race_interval": 25,
    "race_eot": 5,
    "confirm_fraction": 0.25,
    "score_temperature": 0.10,
    "objective": "mean_per_draw_sigmoid_exact_top5_set_margin",
}

FROZEN_DEFENSE = dict(r5.FROZEN_DEFENSE)
REEVAL_DRAWS = 1000
REEVAL_SEED = 2026091811
REEVAL_BATCH_SIZE = 4
REEVAL_MC_CHUNK = 10
BOOTSTRAP_RESAMPLES = 50_000
BOOTSTRAP_SEED = 2026091812

SCRIPT_NAMES = (
    "topk_cifar100_square_attack_core_r4.py",
    "topk_imagenet_posthoc_q20k_eot_d1_common.py",
    "prepare_topk_imagenet_posthoc_q20k_eot_d1.py",
    "search_topk_imagenet_posthoc_q20k_eot_d1.py",
    "reevaluate_topk_imagenet_posthoc_q20k_eot_d1.py",
    "finalize_topk_imagenet_posthoc_q20k_eot_d1.py",
    "audit_topk_imagenet_posthoc_q20k_eot_d1.py",
    "test_topk_imagenet_posthoc_q20k_eot_d1.py",
    "topk_imagenet_posthoc_q20k_eot_d1_launcher.py",
    "run_topk_imagenet_posthoc_q20k_eot_d1_prepare.py",
    "run_topk_imagenet_posthoc_q20k_eot_d1_robust.py",
    "run_topk_imagenet_posthoc_q20k_eot_d1_full.py",
    "run_topk_imagenet_posthoc_q20k_eot_d1_finalize.py",
)

DEPENDENCY_NAMES = (
    "topk_imagenet_confirmatory_r5_common.py",
    "calibrate_topk_imagenet_defense_r2.py",
)

method_spec = r5.method_spec
canonical_json_bytes = r5.canonical_json_bytes
object_sha256 = r5.object_sha256
file_sha256 = r5.file_sha256
tensor_sha256 = r5.tensor_sha256
read_json = r5.read_json
read_csv = r5.read_csv
atomic_json = r5.atomic_json
atomic_csv = r5.atomic_csv
write_or_validate_json = r5.write_or_validate_json


def stable_seed(base_seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{int(base_seed)}|{label}".encode("utf-8")).hexdigest()
    return int(digest[:15], 16) % (2**31 - 1)


def subset_rank(relative_path: str) -> str:
    return hashlib.sha256(
        f"{PROTOCOL_ID}|path-only-fixed-subset|{relative_path}".encode("utf-8")
    ).hexdigest()


def select_subset_rows(
    manifest_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if len(manifest_rows) != r5.CONFIRMATORY_EXAMPLES:
        raise AssertionError("The parent R5 manifest must contain exactly 1,000 rows")
    canonical: list[dict[str, Any]] = []
    for row in manifest_rows:
        position = int(row["confirmatory_position"])
        canonical.append(
            {
                "source_confirmatory_position": position,
                "partition": "attack_confirmatory",
                "target": int(row["target"]),
                "class_name": str(row["class_name"]),
                "relative_path": str(row["relative_path"]),
                "selection_rank_sha256": subset_rank(str(row["relative_path"])),
            }
        )
    if sorted(row["source_confirmatory_position"] for row in canonical) != list(
        range(r5.CONFIRMATORY_EXAMPLES)
    ):
        raise AssertionError("The parent R5 positions are not canonical")
    if len({row["relative_path"] for row in canonical}) != len(canonical):
        raise AssertionError("The parent R5 manifest contains duplicate paths")
    selected = sorted(canonical, key=lambda row: row["selection_rank_sha256"])[
        :DIAGNOSTIC_EXAMPLES
    ]
    for diagnostic_position, row in enumerate(selected):
        row["diagnostic_position"] = diagnostic_position
    if len({row["target"] for row in selected}) != DIAGNOSTIC_EXAMPLES:
        raise AssertionError("The diagnostic subset must contain distinct classes")
    return selected


def subset_rows_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    canonical = [
        {
            "diagnostic_position": int(row["diagnostic_position"]),
            "source_confirmatory_position": int(row["source_confirmatory_position"]),
            "target": int(row["target"]),
            "class_name": str(row["class_name"]),
            "relative_path": str(row["relative_path"]),
            "selection_rank_sha256": str(row["selection_rank_sha256"]),
        }
        for row in rows
    ]
    return object_sha256(canonical)


def _require_sha(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    observed = file_sha256(path)
    if observed != expected:
        raise AssertionError(
            f"{label} SHA256 changed: expected {expected}, observed {observed}"
        )


def load_parent_r5(r5_root: Path) -> dict[str, Any]:
    root = Path(r5_root).expanduser().resolve()
    paths = {
        "protocol": root / "confirmatory_protocol.json",
        "manifest": root / "confirmatory_manifest.csv",
        "inputs": root / "confirmatory_inputs.pt",
        "prepare": root / "prepare_summary.json",
        "final": root / "confirmatory_analysis" / "final_summary.json",
        "handoff": root / "confirmatory_analysis" / "paper_claim_handoff.json",
        "per_example": root
        / "confirmatory_analysis"
        / "confirmatory_per_example.csv",
    }
    _require_sha(paths["protocol"], EXPECTED_R5_PROTOCOL_FILE_SHA256, "R5 protocol")
    _require_sha(paths["manifest"], EXPECTED_R5_MANIFEST_FILE_SHA256, "R5 manifest")
    _require_sha(paths["inputs"], EXPECTED_R5_INPUTS_FILE_SHA256, "R5 inputs")
    _require_sha(paths["prepare"], EXPECTED_R5_PREPARE_FILE_SHA256, "R5 preparation")
    _require_sha(paths["final"], EXPECTED_R5_FINAL_SUMMARY_FILE_SHA256, "R5 final summary")
    _require_sha(
        paths["per_example"], EXPECTED_R5_PER_EXAMPLE_SHA256, "R5 per-example table"
    )
    _require_sha(paths["handoff"], EXPECTED_R5_HANDOFF_FILE_SHA256, "R5 handoff")

    protocol = read_json(paths["protocol"])
    protocol_hash = r5.validate_protocol(protocol)
    if protocol_hash != EXPECTED_R5_PROTOCOL_HASH:
        raise AssertionError("This diagnostic is anchored to a different R5 protocol")
    if Path(protocol["paths"]["output_root"]).expanduser().resolve() != root:
        raise AssertionError("The R5 protocol points to another output root")

    final = read_json(paths["final"])
    final_base = dict(final)
    final_base.pop("final_summary_hash", None)
    if (
        final.get("status") != "PASS"
        or final.get("paper_claim_authorized") is not True
        or final.get("protocol_hash") != protocol_hash
        or final.get("final_summary_hash") != EXPECTED_R5_FINAL_SUMMARY_HASH
        or object_sha256(final_base) != EXPECTED_R5_FINAL_SUMMARY_HASH
    ):
        raise PermissionError("The frozen R5 confirmatory result is not the audited PASS")
    if final.get("files", {}).get("confirmatory_per_example.csv") != (
        EXPECTED_R5_PER_EXAMPLE_SHA256
    ):
        raise AssertionError("R5 final summary points to another per-example table")
    prepare = read_json(paths["prepare"])
    if (
        prepare.get("status") != "PASS"
        or prepare.get("protocol_hash") != protocol_hash
        or prepare.get("confirmatory_inputs_payload_hash")
        != EXPECTED_R5_INPUTS_PAYLOAD_HASH
        or prepare.get("clean_tensor_sha256") != EXPECTED_R5_CLEAN_TENSOR_SHA256
        or prepare.get("confirmatory_inputs_file_sha256")
        != EXPECTED_R5_INPUTS_FILE_SHA256
    ):
        raise AssertionError("The R5 preparation identity changed")
    primary = final.get("primary_result", {})
    if (
        primary.get("comparison")
        != "full_minus_robust_attacked_exact_set_k5_probability"
        or primary.get("paper_claim_authorized") is not True
    ):
        raise AssertionError("The R5 primary result identity changed")

    handoff = read_json(paths["handoff"])
    handoff_base = dict(handoff)
    handoff_base.pop("handoff_hash", None)
    if (
        handoff.get("authorized") is not True
        or handoff.get("protocol_hash") != protocol_hash
        or handoff.get("no_attack_family_selection_on_confirmatory_data") is not True
        or handoff.get("handoff_hash") != EXPECTED_R5_HANDOFF_HASH
        or object_sha256(handoff_base) != EXPECTED_R5_HANDOFF_HASH
        or final.get("paper_claim_handoff_hash") != EXPECTED_R5_HANDOFF_HASH
        or final.get("files", {}).get("paper_claim_handoff.json")
        != EXPECTED_R5_HANDOFF_FILE_SHA256
    ):
        raise AssertionError("The R5 paper-claim handoff is not the audited artifact")

    manifest = read_csv(paths["manifest"])
    if len(manifest) != r5.CONFIRMATORY_EXAMPLES:
        raise AssertionError("The R5 manifest row count changed")
    if any(
        row.get("protocol_id") != r5.PROTOCOL_ID
        or row.get("protocol_hash") != protocol_hash
        for row in manifest
    ):
        raise AssertionError("The R5 manifest identity changed")
    subset = select_subset_rows(manifest)
    if subset_rows_sha256(subset) != EXPECTED_SUBSET_ROWS_SHA256:
        raise AssertionError("The frozen D1 path-only subset changed")
    return {
        "root": root,
        "paths": paths,
        "protocol": protocol,
        "protocol_hash": protocol_hash,
        "final": final,
        "prepare": prepare,
        "handoff": handoff,
        "manifest": manifest,
        "subset": subset,
    }


def build_protocol_payload(
    *, parent: Mapping[str, Any], output_root: Path
) -> dict[str, Any]:
    scripts = {name: file_sha256(PROJECT_DIR / name) for name in SCRIPT_NAMES}
    dependencies = {
        name: file_sha256(PROJECT_DIR / name) for name in DEPENDENCY_NAMES
    }
    if scripts["topk_cifar100_square_attack_core_r4.py"] != (
        EXPECTED_ATTACK_CORE_SHA256
    ):
        raise AssertionError("The audited Square core bytes changed")
    subset = list(parent["subset"])
    if subset_rows_sha256(subset) != EXPECTED_SUBSET_ROWS_SHA256:
        raise AssertionError("Cannot build D1 from a different subset")
    payload: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "parent_protocol_id": PARENT_PROTOCOL_ID,
        "parent_r5_protocol_hash": parent["protocol_hash"],
        "parent_r5_protocol_file_sha256": EXPECTED_R5_PROTOCOL_FILE_SHA256,
        "parent_r5_final_summary_hash": EXPECTED_R5_FINAL_SUMMARY_HASH,
        "parent_r5_final_summary_file_sha256": EXPECTED_R5_FINAL_SUMMARY_FILE_SHA256,
        "parent_r5_per_example_sha256": EXPECTED_R5_PER_EXAMPLE_SHA256,
        "parent_r5_inputs_payload_hash": EXPECTED_R5_INPUTS_PAYLOAD_HASH,
        "parent_r5_clean_tensor_sha256": EXPECTED_R5_CLEAN_TENSOR_SHA256,
        "dataset": "ImageNet-1k validation",
        "architecture": "torchvision ResNet-50",
        "scope": "post-hoc query-budget/EOT masking stress diagnostic",
        "paths": {
            "data_root": parent["protocol"]["paths"]["data_root"],
            "r5_root": str(parent["root"]),
            "output_root": str(Path(output_root).expanduser().resolve()),
        },
        "checkpoints": parent["protocol"]["checkpoints"],
        "source_partition": "attack_confirmatory",
        "source_examples": r5.CONFIRMATORY_EXAMPLES,
        "diagnostic_examples": DIAGNOSTIC_EXAMPLES,
        "subset_rule": (
            "lowest SHA256(PROTOCOL_ID|path-only-fixed-subset|relative_path) "
            "over all 1,000 frozen R5 confirmatory paths"
        ),
        "subset_rows_sha256": EXPECTED_SUBSET_ROWS_SHA256,
        "methods": list(METHODS),
        "method_attack": dict(METHOD_ATTACK),
        "search_seeds": list(SEARCH_SEEDS),
        "primary_k": PRIMARY_K,
        "secondary_k": SECONDARY_K,
        "epsilon_px": EPSILON_PX,
        "api_budget_per_example": API_BUDGET,
        "square_p_init": SQUARE_P_INIT,
        "adaptive_config_by_method": ADAPTIVE_CONFIG_BY_METHOD,
        "racing_config_unused": RACING_CONFIG,
        "frozen_defense": FROZEN_DEFENSE,
        "search_batch_size_by_method": SEARCH_BATCH_SIZE_BY_METHOD,
        "candidate_reevaluation": {
            "draws_stochastic": REEVAL_DRAWS,
            "draws_deterministic": 1,
            "seed": REEVAL_SEED,
            "batch_size": REEVAL_BATCH_SIZE,
            "mc_chunk": REEVAL_MC_CHUNK,
            "common_random_numbers": True,
            "not_charged_to_attack_budget": True,
        },
        "analysis_plan": {
            "unit": "fixed diagnostic image after averaging three fixed seeds",
            "comparison": "full_minus_robust_attacked_exact_set_k5_probability",
            "r5_budget_reference": 5000,
            "r5_search_eot_by_method": {"robust": 1, "full": 1},
            "stress_budget": API_BUDGET,
            "stress_search_eot_by_method": {
                method: ADAPTIVE_CONFIG_BY_METHOD[method]["search_eot"]
                for method in METHODS
            },
            "budget_and_eot_effects_separately_identifiable": False,
            "r5_vs_d1_reevaluation_common_random_numbers": False,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "diagnostic_only": True,
        },
        "script_sha256": scripts,
        "dependency_sha256": dependencies,
        "post_hoc": True,
        "paper_claim_authorized": False,
        "may_modify_parent_r5": False,
        "no_training": True,
        "no_defense_retuning": True,
        "no_attack_family_selection": True,
        "no_seed_selection": True,
        "no_per_example_candidate_selection": True,
        "every_eot_refresh_confirmation_draw_charged": True,
    }
    payload["protocol_hash"] = object_sha256(payload)
    return payload


def validate_protocol(payload: Mapping[str, Any]) -> str:
    claimed = str(payload.get("protocol_hash", ""))
    base = dict(payload)
    base.pop("protocol_hash", None)
    if not claimed or object_sha256(base) != claimed:
        raise AssertionError("Post-hoc Q20k protocol self-hash mismatch")
    expected = {
        "protocol_id": PROTOCOL_ID,
        "parent_protocol_id": PARENT_PROTOCOL_ID,
        "parent_r5_protocol_hash": EXPECTED_R5_PROTOCOL_HASH,
        "parent_r5_final_summary_hash": EXPECTED_R5_FINAL_SUMMARY_HASH,
        "parent_r5_per_example_sha256": EXPECTED_R5_PER_EXAMPLE_SHA256,
        "parent_r5_inputs_payload_hash": EXPECTED_R5_INPUTS_PAYLOAD_HASH,
        "parent_r5_clean_tensor_sha256": EXPECTED_R5_CLEAN_TENSOR_SHA256,
        "diagnostic_examples": DIAGNOSTIC_EXAMPLES,
        "subset_rows_sha256": EXPECTED_SUBSET_ROWS_SHA256,
        "methods": list(METHODS),
        "method_attack": METHOD_ATTACK,
        "search_seeds": list(SEARCH_SEEDS),
        "primary_k": PRIMARY_K,
        "secondary_k": SECONDARY_K,
        "epsilon_px": EPSILON_PX,
        "api_budget_per_example": API_BUDGET,
        "square_p_init": SQUARE_P_INIT,
        "adaptive_config_by_method": ADAPTIVE_CONFIG_BY_METHOD,
        "frozen_defense": FROZEN_DEFENSE,
        "post_hoc": True,
        "paper_claim_authorized": False,
        "may_modify_parent_r5": False,
        "no_training": True,
        "no_defense_retuning": True,
        "no_attack_family_selection": True,
        "no_seed_selection": True,
        "no_per_example_candidate_selection": True,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise AssertionError(f"Frozen post-hoc field changed: {key}")
    reeval = payload.get("candidate_reevaluation", {})
    if reeval != {
        "draws_stochastic": REEVAL_DRAWS,
        "draws_deterministic": 1,
        "seed": REEVAL_SEED,
        "batch_size": REEVAL_BATCH_SIZE,
        "mc_chunk": REEVAL_MC_CHUNK,
        "common_random_numbers": True,
        "not_charged_to_attack_budget": True,
    }:
        raise AssertionError("Post-hoc reevaluation configuration changed")
    analysis = payload.get("analysis_plan", {})
    if analysis != {
        "unit": "fixed diagnostic image after averaging three fixed seeds",
        "comparison": "full_minus_robust_attacked_exact_set_k5_probability",
        "r5_budget_reference": 5000,
        "r5_search_eot_by_method": {"robust": 1, "full": 1},
        "stress_budget": API_BUDGET,
        "stress_search_eot_by_method": {"robust": 1, "full": 5},
        "budget_and_eot_effects_separately_identifiable": False,
        "r5_vs_d1_reevaluation_common_random_numbers": False,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "diagnostic_only": True,
    }:
        raise AssertionError("Post-hoc analysis plan changed")
    for name, expected_hash in payload.get("script_sha256", {}).items():
        path = PROJECT_DIR / name
        if name not in SCRIPT_NAMES or not path.is_file() or file_sha256(path) != expected_hash:
            raise AssertionError(f"Post-hoc source changed after freeze: {name}")
    if set(payload.get("script_sha256", {})) != set(SCRIPT_NAMES):
        raise AssertionError("Post-hoc source manifest is incomplete")
    for name, expected_hash in payload.get("dependency_sha256", {}).items():
        path = PROJECT_DIR / name
        if name not in DEPENDENCY_NAMES or not path.is_file() or file_sha256(path) != expected_hash:
            raise AssertionError(f"Post-hoc dependency changed after freeze: {name}")
    if set(payload.get("dependency_sha256", {})) != set(DEPENDENCY_NAMES):
        raise AssertionError("Post-hoc dependency manifest is incomplete")
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


def paired_bootstrap(
    delta: np.ndarray,
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    chunk_size: int = 500,
) -> dict[str, float | int | str]:
    delta = np.asarray(delta, dtype=np.float64)
    if (
        delta.ndim != 1
        or len(delta) != DIAGNOSTIC_EXAMPLES
        or not np.isfinite(delta).all()
    ):
        raise ValueError("Expected one finite paired value per diagnostic image")
    rng = np.random.default_rng(int(seed))
    boot = np.empty(int(resamples), dtype=np.float64)
    cursor = 0
    while cursor < int(resamples):
        count = min(int(chunk_size), int(resamples) - cursor)
        indices = rng.integers(0, len(delta), size=(count, len(delta)))
        boot[cursor : cursor + count] = delta[indices].mean(axis=1)
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
