#!/usr/bin/env python3
"""Frozen contracts for the ImageNet ViT-reference VR4 confirmatory black-box experiment.

VR4 is allowed to open the predeclared 1,000-image ``attack_confirmatory``
partition only after the VR3 attack-family freeze has been validated.  No
defense, attack-family, seed, or per-example candidate selection is permitted
on VR4 outcomes.
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import topk_imagenet_vit_reference_attack_pilot_vr3_common as vr3


PROTOCOL_ID = (
    "imagenet_topk_vit_reference_crossarch_confirmatory_vr4_20260919"
)
VR3_PROTOCOL_ID = vr3.PROTOCOL_ID
PARENT_PROTOCOL_ID = vr3.PARENT_PROTOCOL_ID

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = vr3.DEFAULT_DATA_ROOT
DEFAULT_R0_DIR = vr3.DEFAULT_R0_DIR
DEFAULT_CALIBRATION_DIR = vr3.DEFAULT_CALIBRATION_DIR
DEFAULT_CLEAN_DIR = vr3.DEFAULT_CLEAN_DIR
DEFAULT_VR3_ROOT = vr3.DEFAULT_OUTPUT_ROOT
DEFAULT_OUTPUT_ROOT = Path(
    "./models/imagenet_topk/"
    "paper_topk_imagenet_vit_reference_crossarch_confirmatory_attack_"
    "k5_q5k_vr4"
)

EXPECTED_VR3_PROTOCOL_HASH = (
    "1ecd17d0945decd9d3dc6c4a829fb44ee68ebbc297a2937a643de528c0704330"
)
EXPECTED_VR3_SELECTION_HASH = (
    "ebe8acd871f8fc4c49f6dd047fa0b22fdbb90b0a503f5a76cddb1dd303ec40cb"
)
EXPECTED_CONFIRMATORY_ROWS_SHA256 = (
    "b4fec7303ad50ee992c65ae2cea4f4f69c0912f69a44f7cafc600b61562114a1"
)

METHODS = vr3.METHODS
METHOD_ATTACK = {
    "robust": "SquareAdaptive",
    "full_sigma0": "SquareRacing",
    "noise_only": "SquareAdaptive",
    "full": "SquareAdaptive",
}
SEARCH_SEEDS = vr3.CONFIRMATORY_SEARCH_SEEDS
PILOT_SEARCH_SEEDS = vr3.SEARCH_SEEDS
PRIMARY_K = vr3.PRIMARY_K
SECONDARY_K = vr3.SECONDARY_K
EPSILON_PX = vr3.EPSILON_PX
API_BUDGET = vr3.API_BUDGET
SQUARE_P_INIT = vr3.SQUARE_P_INIT
ADAPTIVE_CONFIG = dict(vr3.ADAPTIVE_CONFIG)
RACING_CONFIG = dict(vr3.RACING_CONFIG)
FROZEN_DEFENSE = dict(vr3.FROZEN_DEFENSE)
SEARCH_BATCH_SIZE_BY_METHOD = dict(vr3.SEARCH_BATCH_SIZE_BY_METHOD)
CONFIRMATORY_EXAMPLES = vr3.CONFIRMATORY_EXAMPLES
REFERENCE_BATCH_SIZE = vr3.REFERENCE_BATCH_SIZE

# These seeds are new, fixed before VR4 outcomes are read, and deliberately
# distinct from the VR3 pilot reevaluation/bootstrap seeds.
REEVAL_DRAWS = 1000
REEVAL_SEED = 2026091911
REEVAL_BATCH_SIZE = 4
REEVAL_MC_CHUNK = 10
BOOTSTRAP_RESAMPLES = 50_000
BOOTSTRAP_SEED = 2026091912

PRIMARY_COMPARISON = "full_minus_robust_attacked_exact_set_k5_probability"
SECONDARY_COMPARISONS = (
    "noise_only_minus_robust_attacked_exact_set_k5_probability",
    "full_sigma0_minus_robust_attacked_exact_set_k5_probability",
    "full_minus_noise_only_attacked_exact_set_k5_probability",
    "full_minus_full_sigma0_attacked_exact_set_k5_probability",
)

SCRIPT_NAMES = (
    "topk_imagenet_vit_reference_confirmatory_vr4_common.py",
    "prepare_topk_imagenet_vit_reference_confirmatory_vr4.py",
    "search_topk_imagenet_vit_reference_confirmatory_vr4.py",
    "reevaluate_topk_imagenet_vit_reference_confirmatory_vr4_method.py",
    "finalize_topk_imagenet_vit_reference_confirmatory_vr4.py",
    "audit_topk_imagenet_vit_reference_confirmatory_vr4.py",
    "test_topk_imagenet_vit_reference_confirmatory_vr4.py",
    "topk_imagenet_vit_reference_confirmatory_vr4_launcher.py",
    "run_topk_imagenet_vit_reference_confirmatory_vr4_prepare.py",
    "run_topk_imagenet_vit_reference_confirmatory_vr4_robust.py",
    "run_topk_imagenet_vit_reference_confirmatory_vr4_full_sigma0.py",
    "run_topk_imagenet_vit_reference_confirmatory_vr4_noise_only.py",
    "run_topk_imagenet_vit_reference_confirmatory_vr4_full.py",
    "run_topk_imagenet_vit_reference_confirmatory_vr4_finalize.py",
    "pack_topk_imagenet_vit_reference_confirmatory_vr4.py",
)

DEPENDENCY_NAMES = (
    "topk_imagenet_preflight_common.py",
    "preflight_topk_imagenet_r0.py",
    "topk_imagenet_robust_anchor_common_r1.py",
    "prepare_topk_imagenet_robust_anchor_r1.py",
    "topk_imagenet_vit_reference_calibration_vr1_common.py",
    "calibrate_topk_imagenet_vit_reference_vr1.py",
    "topk_imagenet_vit_reference_attack_pilot_vr3_common.py",
    "topk_imagenet_vit_reference_square_attack_core_vr3.py",
    "topk_cifar100_square_attack_core_r4.py",
    "prepare_topk_imagenet_vit_reference_attack_pilot_vr3.py",
    "search_topk_imagenet_vit_reference_attack_pilot_vr3.py",
    "reevaluate_topk_imagenet_vit_reference_attack_pilot_vr3_mc1000.py",
    "audit_topk_imagenet_vit_reference_attack_pilot_vr3.py",
    "test_topk_imagenet_vit_reference_attack_pilot_vr3.py",
    "pack_topk_imagenet_vit_reference_attack_pilot_vr3.py",
    "topk_imagenet_vit_reference_attack_pilot_vr3_launcher.py",
    "run_topk_imagenet_vit_reference_attack_pilot_vr3_light.py",
    "run_topk_imagenet_vit_reference_attack_pilot_vr3_full.py",
    "run_topk_imagenet_vit_reference_attack_pilot_vr3_reeval.py",
)


method_spec = vr3.method_spec
canonical_json_bytes = vr3.canonical_json_bytes
object_sha256 = vr3.object_sha256
file_sha256 = vr3.file_sha256
tensor_sha256 = vr3.tensor_sha256
read_json = vr3.read_json
read_csv = vr3.read_csv
atomic_json = vr3.atomic_json
atomic_csv = vr3.atomic_csv
write_or_validate_json = vr3.write_or_validate_json
path_rows_sha256 = vr3.path_rows_sha256


def _validate_self_hash(payload: Mapping[str, Any], field: str, label: str) -> str:
    claimed = str(payload.get(field, ""))
    base = dict(payload)
    base.pop(field, None)
    if not claimed or object_sha256(base) != claimed:
        raise AssertionError(f"{label} self-hash mismatch")
    return claimed


def canonical_confirmatory_rows(parent: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = [
        {
            "confirmatory_position": position,
            "partition": "attack_confirmatory",
            "target": int(row["target"]),
            "class_name": str(row["class_name"]),
            "relative_path": str(row["relative_path"]),
        }
        for position, row in enumerate(parent["confirmatory"])
    ]
    if len(rows) != CONFIRMATORY_EXAMPLES:
        raise AssertionError("ImageNet ViT-reference VR4 requires exactly 1,000 confirmatory images")
    if len({row["relative_path"] for row in rows}) != CONFIRMATORY_EXAMPLES:
        raise AssertionError("ImageNet ViT-reference VR4 confirmatory paths are duplicated")
    validate_confirmatory_labels([int(row["target"]) for row in rows])
    if path_rows_sha256(rows) != EXPECTED_CONFIRMATORY_ROWS_SHA256:
        raise AssertionError("ImageNet ViT-reference VR4 confirmatory-row hash changed")
    return rows


def load_parent(r0_dir: Path, calibration_dir: Path, clean_dir: Path) -> dict[str, Any]:
    parent = vr3.validate_parent(r0_dir, calibration_dir, clean_dir)
    canonical_confirmatory_rows(parent)
    return parent


def load_vr3_freeze(vr3_root: Path, parent: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the only VR3 artifacts allowed to determine VR4 attacks."""

    vr3_root = Path(vr3_root).expanduser().resolve()
    paths = {
        "protocol": vr3_root / "attack_pilot_protocol.json",
        "freeze": vr3_root / "common_mc1000" / "confirmatory_attack_freeze.json",
        "selected": vr3_root / "common_mc1000" / "selected_attack_envelope.csv",
        "final": vr3_root / "common_mc1000" / "final_summary.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing audited ImageNet VR3 artifacts:\n  " + "\n  ".join(missing))

    protocol = read_json(paths["protocol"])
    protocol_hash = vr3.validate_protocol(protocol)
    if protocol_hash != EXPECTED_VR3_PROTOCOL_HASH:
        raise AssertionError("ImageNet VR3 protocol hash differs from the audited pilot")
    if Path(protocol["paths"]["output_root"]).expanduser().resolve() != vr3_root:
        raise AssertionError("ImageNet VR3 protocol points to another output root")
    expected_parent_paths = {
        "r0_dir": Path(parent["r0_dir"]).resolve(),
        "calibration_dir": Path(parent["calibration_dir"]).resolve(),
        "clean_dir": Path(parent["clean_dir"]).resolve(),
    }
    for key, expected in expected_parent_paths.items():
        if Path(protocol["paths"][key]).expanduser().resolve() != expected:
            raise AssertionError(f"ImageNet VR3/VR4 parent path mismatch: {key}")
    if protocol.get("confirmatory_rows_sha256") != EXPECTED_CONFIRMATORY_ROWS_SHA256:
        raise AssertionError("ImageNet VR3 confirmatory split hash changed")
    if protocol.get("confirmatory_outcomes_read") is not False:
        raise AssertionError("ImageNet VR3 reports confirmatory-outcome access")
    if protocol.get("confirmatory_search_seeds_predeclared") != list(SEARCH_SEEDS):
        raise AssertionError("ImageNet VR3 predeclared VR4 seeds changed")

    freeze = read_json(paths["freeze"])
    selection_hash = _validate_self_hash(freeze, "selection_hash", "ImageNet VR3 freeze")
    if selection_hash != EXPECTED_VR3_SELECTION_HASH:
        raise AssertionError("ImageNet VR3 selection hash differs from the audited result")
    if freeze.get("protocol_hash") != protocol_hash or freeze.get("status") != "PASS":
        raise AssertionError("ImageNet VR3 freeze/protocol identity is invalid")
    if freeze.get("authorized_next_stage") != (
        "ImageNet_ViT_reference_crossarch_VR4_disjoint_1000_image_"
        "confirmatory_attack"
    ):
        raise PermissionError("ImageNet VR3 did not authorize VR4")
    if freeze.get("paper_claim_authorized_by_pilot") is not False:
        raise AssertionError("ImageNet VR3 pilot incorrectly authorizes a paper claim")
    if freeze.get("confirmatory_partition") != {
        "examples": CONFIRMATORY_EXAMPLES,
        "rows_sha256": EXPECTED_CONFIRMATORY_ROWS_SHA256,
        "outcomes_read": False,
    }:
        raise AssertionError("ImageNet VR3 confirmatory declaration changed")
    if freeze.get("confirmatory_search_seeds") != list(SEARCH_SEEDS):
        raise AssertionError("ImageNet VR3 confirmatory seeds changed")
    if freeze.get("frozen_defense") != FROZEN_DEFENSE:
        raise AssertionError("ImageNet VR3 defense freeze changed")
    if freeze.get("attack_configs") != {
        "SquareAdaptive": ADAPTIVE_CONFIG,
        "SquareRacing": RACING_CONFIG,
    }:
        raise AssertionError("ImageNet VR3 attack configuration changed")
    frozen_scalars = {
        "primary_k": PRIMARY_K,
        "secondary_k": SECONDARY_K,
        "epsilon_px": EPSILON_PX,
        "api_budget_per_example": API_BUDGET,
        "square_p_init": SQUARE_P_INIT,
        "candidate_reevaluation_draws": vr3.REEVAL_DRAWS,
        "candidate_reevaluation_seed": vr3.REEVAL_SEED,
    }
    for key, expected in frozen_scalars.items():
        if freeze.get(key) != expected:
            raise AssertionError(f"ImageNet VR3 freeze changed: {key}")
    observed_map = {
        method: str(freeze.get("method_attack_selection", {}).get(method, {}).get("attack", ""))
        for method in METHODS
    }
    if observed_map != METHOD_ATTACK:
        raise AssertionError(f"Unexpected ImageNet VR3 method/attack freeze: {observed_map}")
    if freeze.get("no_per_example_candidate_selection") is not True:
        raise AssertionError("ImageNet VR3 freeze permits per-example selection")

    selected = read_csv(paths["selected"])
    if len(selected) != len(METHODS):
        raise AssertionError("ImageNet VR3 selected-attack table must contain four rows")
    if {row["method"]: row["selected_attack"] for row in selected} != METHOD_ATTACK:
        raise AssertionError("ImageNet VR3 CSV and JSON attack freezes differ")
    if any(str(row.get("paper_claim_authorized", "")).lower() not in {"false", "0"} for row in selected):
        raise AssertionError("ImageNet VR3 pilot CSV incorrectly authorizes a claim")

    final = read_json(paths["final"])
    if final.get("status") != "PASS":
        raise PermissionError("ImageNet VR3 pilot did not pass")
    if final.get("confirmatory_attack_freeze_hash") != selection_hash:
        raise AssertionError("ImageNet VR3 final/freeze hash mismatch")
    if final.get("confirmatory_partition_outcomes_read") is not False:
        raise AssertionError("ImageNet VR3 final reports confirmatory-outcome access")
    if final.get("paper_claim_status") != "PILOT_ONLY_NO_ADVERSARIAL_CLAIM":
        raise AssertionError("ImageNet VR3 final summary overstates the pilot")
    return {
        "vr3_root": vr3_root,
        "paths": paths,
        "protocol": protocol,
        "protocol_hash": protocol_hash,
        "freeze": freeze,
        "selection_hash": selection_hash,
        "final": final,
    }


def build_protocol_payload(
    *,
    parent: Mapping[str, Any],
    vr3_info: Mapping[str, Any],
    data_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    rows = canonical_confirmatory_rows(parent)
    scripts = {name: file_sha256(PROJECT_DIR / name) for name in SCRIPT_NAMES}
    dependencies = {name: file_sha256(PROJECT_DIR / name) for name in DEPENDENCY_NAMES}
    payload: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "parent_protocol_id": PARENT_PROTOCOL_ID,
        "vr3_protocol_id": VR3_PROTOCOL_ID,
        "parent_vr2_config_hash": vr3.CLEAN_CONFIG_HASH,
        "parent_vr2_protocol_hash": vr3.CLEAN_PROTOCOL_HASH,
        "vr1_selection_hash": vr3.CALIBRATION_SELECTION_HASH,
        "vr3_protocol_hash": vr3_info["protocol_hash"],
        "vr3_selection_hash": vr3_info["selection_hash"],
        "dataset": "ImageNet-1k validation",
        "architecture_roles": {
            "standard_reference": "torchvision ViT-B/16",
            "defended_backbone": "torchvision ResNet-50",
            "cross_architecture_reference": True,
        },
        "experiment_scope": (
            "cross-architecture reference transfer with ViT-B/16 reference and "
            "robust ResNet-50 defended backbone; not a robust-ViT evaluation"
        ),
        "paths": {
            "data_root": str(Path(data_root).expanduser().resolve()),
            "r0_dir": str(parent["r0_dir"]),
            "calibration_dir": str(parent["calibration_dir"]),
            "clean_dir": str(parent["clean_dir"]),
            "vr3_root": str(Path(vr3_info["vr3_root"])),
            "output_root": str(Path(output_root).expanduser().resolve()),
        },
        "checkpoints": parent["checkpoints"],
        "split": "predeclared_disjoint_attack_confirmatory",
        "examples": CONFIRMATORY_EXAMPLES,
        "examples_per_class": 1,
        "confirmatory_rows_sha256": path_rows_sha256(rows),
        "pilot_rows_sha256": vr3_info["protocol"]["pilot_rows_sha256"],
        "pilot_and_confirmatory_disjoint": True,
        "confirmatory_outcomes_read_before_protocol_freeze": False,
        "methods": list(METHODS),
        "method_attack": dict(METHOD_ATTACK),
        "search_seeds": list(SEARCH_SEEDS),
        "pilot_search_seeds": list(PILOT_SEARCH_SEEDS),
        "primary_k": PRIMARY_K,
        "secondary_k": SECONDARY_K,
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
            "batch_size": REEVAL_BATCH_SIZE,
            "mc_chunk": REEVAL_MC_CHUNK,
            "common_random_numbers": True,
            "not_charged_to_attack_budget": True,
        },
        "search_batch_size_by_method": dict(SEARCH_BATCH_SIZE_BY_METHOD),
        "reference_batch_size": REFERENCE_BATCH_SIZE,
        "frozen_defense": dict(FROZEN_DEFENSE),
        "analysis_plan": {
            "unit": "confirmatory image (one image per ImageNet class)",
            "seed_aggregation": "arithmetic_mean_over_three_fixed_search_seeds",
            "primary_comparison": PRIMARY_COMPARISON,
            "primary_decision_rule": "paired_bootstrap_95pct_lower_bound_gt_0",
            "secondary_comparisons": list(SECONDARY_COMPARISONS),
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "no_attack_family_selection": True,
            "no_seed_selection": True,
            "no_per_example_candidate_selection": True,
            "primary_claim_scope": (
                "Frozen full defense versus the frozen robust ResNet-50 anchor on "
                "the disjoint 1,000-class ImageNet-1k validation holdout for Exact "
                "Top-5 set preservation relative to a frozen ViT-B/16 reference, "
                "at epsilon=4/255 and Q=5000"
            ),
        },
        "candidate_schema": "shared_clean_manifest_plus_adversarial_tensor_v1",
        "script_sha256": scripts,
        "dependency_sha256": dependencies,
        "no_training": True,
        "no_defense_retuning": True,
        "no_attack_retuning": True,
        "no_attack_family_selection": True,
        "no_seed_selection": True,
        "no_per_example_candidate_selection": True,
        "budgets_run_independently": True,
        "every_eot_refresh_race_confirmation_draw_charged": True,
        "attack_reference_source": (
            "clean standard ViT-B/16 ranked Top-5, frozen once in VR4 preparation"
        ),
        "defense_reference_behavior": (
            "full methods recompute the standard ViT-B/16 Top-5 on each queried "
            "candidate before the frozen strengthening transform"
        ),
        "standard_reference_adversarially_robust": False,
        "robust_vit_claim_authorized": False,
    }
    payload["protocol_hash"] = object_sha256(payload)
    return payload


def validate_protocol(payload: Mapping[str, Any]) -> str:
    claimed = _validate_self_hash(payload, "protocol_hash", "ImageNet ViT-reference VR4 protocol")
    expected = {
        "protocol_id": PROTOCOL_ID,
        "parent_protocol_id": PARENT_PROTOCOL_ID,
        "vr3_protocol_id": VR3_PROTOCOL_ID,
        "parent_vr2_config_hash": vr3.CLEAN_CONFIG_HASH,
        "parent_vr2_protocol_hash": vr3.CLEAN_PROTOCOL_HASH,
        "vr1_selection_hash": vr3.CALIBRATION_SELECTION_HASH,
        "vr3_protocol_hash": EXPECTED_VR3_PROTOCOL_HASH,
        "vr3_selection_hash": EXPECTED_VR3_SELECTION_HASH,
        "dataset": "ImageNet-1k validation",
        "architecture_roles": {
            "standard_reference": "torchvision ViT-B/16",
            "defended_backbone": "torchvision ResNet-50",
            "cross_architecture_reference": True,
        },
        "experiment_scope": (
            "cross-architecture reference transfer with ViT-B/16 reference and "
            "robust ResNet-50 defended backbone; not a robust-ViT evaluation"
        ),
        "split": "predeclared_disjoint_attack_confirmatory",
        "examples": CONFIRMATORY_EXAMPLES,
        "examples_per_class": 1,
        "confirmatory_rows_sha256": EXPECTED_CONFIRMATORY_ROWS_SHA256,
        "pilot_and_confirmatory_disjoint": True,
        "confirmatory_outcomes_read_before_protocol_freeze": False,
        "methods": list(METHODS),
        "method_attack": dict(METHOD_ATTACK),
        "search_seeds": list(SEARCH_SEEDS),
        "pilot_search_seeds": list(PILOT_SEARCH_SEEDS),
        "primary_k": PRIMARY_K,
        "secondary_k": SECONDARY_K,
        "epsilon_px": EPSILON_PX,
        "api_budget_per_example": API_BUDGET,
        "square_p_init": SQUARE_P_INIT,
        "frozen_defense": FROZEN_DEFENSE,
        "no_training": True,
        "no_defense_retuning": True,
        "no_attack_retuning": True,
        "no_attack_family_selection": True,
        "no_seed_selection": True,
        "no_per_example_candidate_selection": True,
        "candidate_schema": "shared_clean_manifest_plus_adversarial_tensor_v1",
        "attack_reference_source": (
            "clean standard ViT-B/16 ranked Top-5, frozen once in VR4 preparation"
        ),
        "defense_reference_behavior": (
            "full methods recompute the standard ViT-B/16 Top-5 on each queried "
            "candidate before the frozen strengthening transform"
        ),
        "standard_reference_adversarially_robust": False,
        "robust_vit_claim_authorized": False,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise AssertionError(f"Frozen ImageNet ViT-reference VR4 field changed: {key}")
    if payload.get("attack_configs") != {
        "SquareAdaptive": ADAPTIVE_CONFIG,
        "SquareRacing": RACING_CONFIG,
    }:
        raise AssertionError("ImageNet ViT-reference VR4 attack configurations changed")
    if payload.get("search_batch_size_by_method") != SEARCH_BATCH_SIZE_BY_METHOD:
        raise AssertionError("ImageNet ViT-reference VR4 search batch sizes changed")
    expected_reeval = {
        "draws_stochastic": REEVAL_DRAWS,
        "draws_deterministic": 1,
        "seed": REEVAL_SEED,
        "batch_size": REEVAL_BATCH_SIZE,
        "mc_chunk": REEVAL_MC_CHUNK,
        "common_random_numbers": True,
        "not_charged_to_attack_budget": True,
    }
    if payload.get("candidate_reevaluation") != expected_reeval:
        raise AssertionError("ImageNet ViT-reference VR4 reevaluation configuration changed")
    plan = payload.get("analysis_plan", {})
    if plan.get("primary_comparison") != PRIMARY_COMPARISON:
        raise AssertionError("ImageNet ViT-reference VR4 primary comparison changed")
    if plan.get("secondary_comparisons") != list(SECONDARY_COMPARISONS):
        raise AssertionError("ImageNet ViT-reference VR4 secondary comparisons changed")
    if plan.get("bootstrap_resamples") != BOOTSTRAP_RESAMPLES:
        raise AssertionError("ImageNet ViT-reference VR4 bootstrap count changed")
    if plan.get("bootstrap_seed") != BOOTSTRAP_SEED:
        raise AssertionError("ImageNet ViT-reference VR4 bootstrap seed changed")
    recorded_scripts = payload.get("script_sha256", {})
    recorded_dependencies = payload.get("dependency_sha256", {})
    if set(recorded_scripts) != set(SCRIPT_NAMES):
        raise AssertionError("ImageNet ViT-reference VR4 source-hash manifest is incomplete")
    if set(recorded_dependencies) != set(DEPENDENCY_NAMES):
        raise AssertionError("ImageNet ViT-reference VR4 dependency-hash manifest is incomplete")
    for name in SCRIPT_NAMES:
        path = PROJECT_DIR / name
        if not path.is_file() or file_sha256(path) != recorded_scripts[name]:
            raise AssertionError(f"ImageNet ViT-reference VR4 source changed after protocol freeze: {name}")
    for name in DEPENDENCY_NAMES:
        path = PROJECT_DIR / name
        if not path.is_file() or file_sha256(path) != recorded_dependencies[name]:
            raise AssertionError(f"ImageNet ViT-reference VR4 dependency changed after protocol freeze: {name}")
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
        if str(row["attack"]) != METHOD_ATTACK[method]:
            raise ValueError(f"Unexpected attack row for {method}: {row['attack']}")
        seed = int(row["search_seed"])
        position = int(row["confirmatory_position"])
        if position in by_seed.setdefault(seed, {}):
            raise ValueError(f"Duplicate ImageNet ViT-reference VR4 row: {method}/{seed}/{position}")
        by_seed[seed][position] = float(row[metric])
    if set(by_seed) != set(SEARCH_SEEDS):
        raise ValueError(f"Missing ImageNet ViT-reference VR4 seeds for {method}")
    positions = sorted(next(iter(by_seed.values())))
    if positions != list(range(CONFIRMATORY_EXAMPLES)):
        raise ValueError(f"Incomplete ImageNet ViT-reference VR4 positions for {method}")
    if any(sorted(values) != positions for values in by_seed.values()):
        raise ValueError(f"Unaligned ImageNet ViT-reference VR4 seed rows for {method}")
    matrix = np.asarray(
        [[by_seed[seed][position] for position in positions] for seed in SEARCH_SEEDS],
        dtype=np.float64,
    )
    return positions, matrix.mean(axis=0)


def clean_values(
    rows: Sequence[Mapping[str, Any]], *, method: str, metric: str
) -> tuple[list[int], np.ndarray]:
    first_seed = SEARCH_SEEDS[0]
    selected = [
        row for row in rows
        if str(row["method"]) == method and int(row["search_seed"]) == first_seed
    ]
    by_position = {
        int(row["confirmatory_position"]): float(row[metric]) for row in selected
    }
    if sorted(by_position) != list(range(CONFIRMATORY_EXAMPLES)):
        raise ValueError(f"Incomplete clean ImageNet ViT-reference VR4 values for {method}")
    positions = sorted(by_position)
    return positions, np.asarray([by_position[position] for position in positions], dtype=np.float64)


def paired_bootstrap(
    delta: np.ndarray,
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    chunk_size: int = 100,
) -> dict[str, float | int | str]:
    delta = np.asarray(delta, dtype=np.float64)
    if delta.ndim != 1 or len(delta) != CONFIRMATORY_EXAMPLES or not np.isfinite(delta).all():
        raise ValueError("Expected one finite paired value per ImageNet ViT-reference VR4 image")
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
    if len(labels) != CONFIRMATORY_EXAMPLES:
        raise AssertionError("ImageNet ViT-reference VR4 must contain exactly 1,000 labels")
    if Counter(int(value) for value in labels) != Counter({class_id: 1 for class_id in range(1000)}):
        raise AssertionError("ImageNet ViT-reference VR4 must contain exactly one image per class")
