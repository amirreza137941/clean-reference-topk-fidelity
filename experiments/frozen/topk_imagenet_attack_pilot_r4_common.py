#!/usr/bin/env python3
"""Frozen contracts and dependency-light helpers for ImageNet attack pilot R4.

R4 is an attack-family-selection pilot.  It may read only the R0
``attack_pilot`` partition and may never authorize a paper adversarial claim.
The disjoint 1,000-image ``attack_confirmatory`` partition remains outcome
blind until R5.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PROTOCOL_ID = "imagenet_topk_attack_family_pilot_r4_20260911"
PARENT_PROTOCOL_ID = "imagenet_topk_clean_confirmation_r3_20260911"
R0_PROTOCOL_ID = "imagenet_topk_preflight_r0_20260910"
R2_PROTOCOL_ID = "imagenet_topk_defense_calibration_dev_r2_20260911"

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = Path("./datasets/imagenet")
DEFAULT_R0_DIR = Path(
    "./models/imagenet_topk/"
    "paper_topk_imagenet_preflight_r0"
)
DEFAULT_R2_DIR = Path(
    "./models/imagenet_topk/"
    "paper_topk_imagenet_defense_calibration_dev_r2"
)
DEFAULT_R3_DIR = Path(
    "./models/imagenet_topk/"
    "paper_topk_imagenet_clean_confirmation_k5_mc50_r3"
)
DEFAULT_OUTPUT_ROOT = Path(
    "./models/imagenet_topk/"
    "paper_topk_imagenet_attack_family_pilot_k5_q5k_r4"
)
DEFAULT_CIFAR_CORE = Path(
    "./projects/CIFAR100/"
    "topk_cifar100_square_attack_core_r4.py"
)

FROZEN_SPLIT_SHA256 = "829b84ffac9f2fe4d97f812b42bf91d8d6254662a8dcfdee99ec85345ca88619"
FROZEN_SPLIT_FILE_SHA256 = "1d31a794a7738d7a0876f521f394b63013297c85bf7a8a7d80c7ce3957d4cae2"
STANDARD_CHECKPOINT_SHA256 = "c377ee6a665d9cb868b071b2623dff729084f6b3fe0faad6b79c0ab51b99a33f"
ROBUST_CHECKPOINT_SHA256 = "29d4db62dfc2635ca364c1b98c1437af84b10620c99407a3c6ca6967c9e6bfc6"
ROBUST_STATE_FINGERPRINT = "ab15ff6b4fe3d75542686d7138ce127bca3467fcff86820f5049b1540e50251d"
R2_CONFIG_HASH = "e09f3e09e5279957e6da1c7a4f05ac5a5812e3e14c4134cbaea99fd90d9503c4"
R2_SELECTION_HASH = "e078d36935326bac17b810e5c888a201d801957a4552ea788442eeff899d0386"
R3_CONFIG_HASH = "d66850e9684c125ffbf78b4ef30e7e4763b3dc75bc143c3a6551b07bf71e4b9e"
R3_PROTOCOL_HASH = "77d5dfabf8002f3bb6de2b796be00d63e30603314780348bde6ceccf7de21a8d"

EXPECTED_R3_ARTIFACT_SHA256 = {
    "run_config.json": "0bfbeb06fc77b5a6f9a3ae687bc9c042a70345062ccda2198d7444129792a207",
    "frozen_protocol_r3.json": "ea103320a84830e9f5cd67e791cdeb29609d1fb5957aee2e9d36fe2c3fd243a0",
    "clean_confirmation_manifest.json": "5ff1221ec363d1c49589e60c856a5c917dee28481b5986769695d81e70b8be06",
    "clean_confirmation_summary.csv": "6e5cfea799b29893a9f21c53d8d5323b44b8571e2df4a40fe782aef4fa7e6207",
    "preregistered_clean_gates.csv": "d45465e8bc01026f4f1fefc225387d431cc51580dc9f544def701ccd94c5c3ed",
    "final_summary.json": "175b7d475a7b3b55d3cf6ac7d30d4bf80b0d30c7fd1c46a28835da70bfb4fd74",
}

# The attack core is the already-audited, dataset-agnostic implementation used
# in CIFAR-100 R4.  A bridge checks these exact bytes before importing it.
EXPECTED_ATTACK_CORE_SHA256 = "0dc459984ce69173ab83bfceb0b8fb824a5d38b565550c4a6b4ea515c30dded7"

PRIMARY_K = 5
SECONDARY_K = 2
PILOT_POOL_EXAMPLES = 1000
PILOT_EXAMPLES = 100
CONFIRMATORY_EXAMPLES = 1000
METHODS = ("robust", "full_sigma0", "noise_only", "full")
LIGHT_METHODS = ("robust", "noise_only")
FULL_METHODS = ("full_sigma0", "full")
ATTACKS = ("SquareAdaptive", "SquareRacing")
SEARCH_SEEDS = (141071, 252073, 363083)
CONFIRMATORY_SEARCH_SEEDS = (474091, 585101, 696113)

EPSILON_PX = 4.0
API_BUDGET = 5000
SQUARE_P_INIT = 0.05
REEVAL_DRAWS = 1000
REEVAL_SEED = 2026091111
REEVAL_BATCH_SIZE = 4
REEVAL_MC_CHUNK = 10
BOOTSTRAP_RESAMPLES = 50_000
BOOTSTRAP_SEED = 2026091112
REFERENCE_BATCH_SIZE = 16
SEARCH_BATCH_SIZE_BY_METHOD = {
    "robust": 16,
    "noise_only": 16,
    "full_sigma0": 8,
    "full": 8,
}

FROZEN_DEFENSE = {
    "candidate_id": "global_a1_s4_w4",
    "gate_mode": "global",
    "alpha_px": 1.0,
    "epsilon_px": 4.0,
    "steps": 4,
    "top1_weight": 1.0,
    "set_weight": 4.0,
    "sigma": 0.025,
}

ADAPTIVE_CONFIG = {
    "search_eot": 1,
    "refresh_eot": 1,
    "confirm_candidates": 5,
    "confirm_fraction": 0.20,
    "objective": "exact_top5_set_margin_of_mean_logits",
}
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

SCRIPT_NAMES = (
    "topk_imagenet_attack_pilot_r4_common.py",
    "topk_imagenet_square_attack_core_r4.py",
    "prepare_topk_imagenet_attack_pilot_r4.py",
    "search_topk_imagenet_attack_pilot_r4.py",
    "reevaluate_topk_imagenet_attack_pilot_r4_mc1000.py",
    "audit_topk_imagenet_attack_pilot_r4.py",
    "test_topk_imagenet_attack_pilot_r4.py",
    "topk_imagenet_attack_pilot_r4_launcher.py",
    "run_topk_imagenet_attack_pilot_r4_light.py",
    "run_topk_imagenet_attack_pilot_r4_full.py",
    "run_topk_imagenet_attack_pilot_r4_reeval.py",
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
        sigma=FROZEN_DEFENSE["sigma"] if noise else 0.0,
        strength_steps=FROZEN_DEFENSE["steps"] if strength else 0,
        strength_alpha_px=FROZEN_DEFENSE["alpha_px"] if strength else 0.0,
        strength_epsilon_px=FROZEN_DEFENSE["epsilon_px"] if strength else 0.0,
        top1_weight=FROZEN_DEFENSE["top1_weight"],
        set_weight=FROZEN_DEFENSE["set_weight"],
    )


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def object_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def atomic_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = [dict(row) for row in rows]
    if not materialized:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fields:
                fields.append(key)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(materialized)
    os.replace(temporary, path)


def write_or_validate_json(path: Path, payload: Mapping[str, Any]) -> None:
    if Path(path).is_file():
        if read_json(path) != dict(payload):
            raise AssertionError(f"Frozen JSON differs from this invocation: {path}")
    else:
        atomic_json(path, payload)


def stable_seed(base_seed: int, label: str) -> int:
    value = hashlib.sha256(f"{int(base_seed)}|{label}".encode()).hexdigest()
    return int(value[:15], 16) % (2**31 - 1)


def path_rows_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    canonical = [
        {
            "target": int(row["target"]),
            "class_name": str(row["class_name"]),
            "relative_path": str(row["relative_path"]),
        }
        for row in rows
    ]
    return object_sha256(canonical)


def pilot_rank(relative_path: str) -> str:
    return hashlib.sha256(
        f"{PROTOCOL_ID}|outcome-blind-pilot-subset|{relative_path}".encode()
    ).hexdigest()


def select_pilot_rows(split_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    pool = [dict(row) for row in split_rows if row.get("partition") == "attack_pilot"]
    if len(pool) != PILOT_POOL_EXAMPLES:
        raise AssertionError(f"Expected {PILOT_POOL_EXAMPLES} attack-pilot rows")
    if len({str(row["relative_path"]) for row in pool}) != len(pool):
        raise AssertionError("Duplicate attack-pilot path")
    ordered = sorted(pool, key=lambda row: pilot_rank(str(row["relative_path"])))
    selected = ordered[:PILOT_EXAMPLES]
    if len({int(row["target"]) for row in selected}) != PILOT_EXAMPLES:
        raise AssertionError("ImageNet pilot subset should contain distinct classes")
    return [
        {
            "pilot_position": position,
            "partition": "attack_pilot",
            "target": int(row["target"]),
            "class_name": str(row["class_name"]),
            "relative_path": str(row["relative_path"]),
            "selection_rank_sha256": pilot_rank(str(row["relative_path"])),
        }
        for position, row in enumerate(selected)
    ]


def validate_parent(r0_dir: Path, r2_dir: Path, r3_dir: Path) -> dict[str, Any]:
    r0_dir, r2_dir, r3_dir = map(lambda p: Path(p).expanduser().resolve(), (r0_dir, r2_dir, r3_dir))
    for name, expected in EXPECTED_R3_ARTIFACT_SHA256.items():
        path = r3_dir / name
        if not path.is_file() or file_sha256(path) != expected:
            raise AssertionError(f"R3 artifact missing or changed: {path}")
    config = read_json(r3_dir / "run_config.json")
    protocol = read_json(r3_dir / "frozen_protocol_r3.json")
    final = read_json(r3_dir / "final_summary.json")
    if config.get("protocol_id") != PARENT_PROTOCOL_ID or config.get("config_hash") != R3_CONFIG_HASH:
        raise AssertionError("Unexpected R3 configuration identity")
    if protocol.get("protocol_id") != PARENT_PROTOCOL_ID or protocol.get("protocol_hash") != R3_PROTOCOL_HASH:
        raise AssertionError("Unexpected R3 protocol identity")
    if final.get("status") != "PASS" or final.get("attack_stage_ready") is not True:
        raise PermissionError("R3 did not authorize the attack-pilot stage")
    if final.get("official_clean_confirmation") != "PASS":
        raise PermissionError("R3 official clean confirmation did not pass")
    if final.get("r2_selection_hash") != R2_SELECTION_HASH:
        raise AssertionError("R3 points to a different R2 selection")
    selection_path = r2_dir / "defense_selection_r2.json"
    selection = read_json(selection_path)
    if selection.get("protocol_id") != R2_PROTOCOL_ID:
        raise AssertionError("Unexpected R2 protocol")
    if selection.get("config_hash") != R2_CONFIG_HASH or selection.get("selection_hash") != R2_SELECTION_HASH:
        raise AssertionError("R2 selection identity changed")
    frozen = dict(selection["strength"])
    frozen.pop("positive_strength_selected", None)
    frozen.pop("selected_dev_summary", None)
    frozen["sigma"] = float(selection["randomization"]["sigma"])
    if frozen != FROZEN_DEFENSE:
        raise AssertionError(f"R2/R4 frozen-defense mismatch: {frozen}")
    split_path = r0_dir / "frozen_imagenet_val_split.csv"
    if not split_path.is_file() or file_sha256(split_path) != FROZEN_SPLIT_FILE_SHA256:
        raise AssertionError("Frozen R0 split CSV is missing or changed")
    split_rows = read_csv(split_path)
    if len(split_rows) != 50_000:
        raise AssertionError("R0 split must contain exactly 50,000 rows")
    pilot_pool = [row for row in split_rows if row["partition"] == "attack_pilot"]
    confirmatory = [row for row in split_rows if row["partition"] == "attack_confirmatory"]
    if len(pilot_pool) != PILOT_POOL_EXAMPLES or len(confirmatory) != CONFIRMATORY_EXAMPLES:
        raise AssertionError("R0 attack partitions changed")
    if set(row["relative_path"] for row in pilot_pool) & set(row["relative_path"] for row in confirmatory):
        raise AssertionError("Pilot and confirmatory partitions overlap")
    checkpoints = config.get("checkpoints", {})
    for name, expected in (("standard", STANDARD_CHECKPOINT_SHA256), ("robust", ROBUST_CHECKPOINT_SHA256)):
        item = checkpoints.get(name, {})
        path = Path(str(item.get("path", ""))).expanduser().resolve()
        if item.get("sha256") != expected or not path.is_file() or file_sha256(path) != expected:
            raise AssertionError(f"Frozen {name} checkpoint is missing or changed")
    if checkpoints["robust"].get("state_dict_fingerprint_sha256") != ROBUST_STATE_FINGERPRINT:
        raise AssertionError("Robust tensor-state fingerprint changed")
    return {
        "r0_dir": r0_dir,
        "r2_dir": r2_dir,
        "r3_dir": r3_dir,
        "config": config,
        "protocol": protocol,
        "final": final,
        "selection": selection,
        "split_rows": split_rows,
        "pilot_pool": pilot_pool,
        "confirmatory": confirmatory,
        "checkpoints": checkpoints,
    }


def build_protocol_payload(
    *, parent: Mapping[str, Any], data_root: Path, output_root: Path, core_path: Path
) -> dict[str, Any]:
    selected = select_pilot_rows(parent["split_rows"])
    scripts = {name: file_sha256(PROJECT_DIR / name) for name in SCRIPT_NAMES}
    core_path = Path(core_path).expanduser().resolve()
    if not core_path.is_file() or file_sha256(core_path) != EXPECTED_ATTACK_CORE_SHA256:
        raise AssertionError(f"Audited Square core missing or changed: {core_path}")
    confirmatory_rows = [
        {
            "target": int(row["target"]),
            "class_name": str(row["class_name"]),
            "relative_path": str(row["relative_path"]),
        }
        for row in parent["confirmatory"]
    ]
    payload: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "parent_protocol_id": PARENT_PROTOCOL_ID,
        "parent_r3_config_hash": R3_CONFIG_HASH,
        "parent_r3_protocol_hash": R3_PROTOCOL_HASH,
        "r2_selection_hash": R2_SELECTION_HASH,
        "frozen_split_sha256": FROZEN_SPLIT_SHA256,
        "dataset": "ImageNet-1k validation",
        "architecture": "torchvision ResNet-50",
        "paths": {
            "data_root": str(Path(data_root).expanduser().resolve()),
            "r0_dir": str(parent["r0_dir"]),
            "r2_dir": str(parent["r2_dir"]),
            "r3_dir": str(parent["r3_dir"]),
            "output_root": str(Path(output_root).expanduser().resolve()),
            "attack_core": str(core_path),
        },
        "checkpoints": parent["checkpoints"],
        "partition": "attack_pilot",
        "pilot_pool_examples": PILOT_POOL_EXAMPLES,
        "pilot_examples": PILOT_EXAMPLES,
        "pilot_subset_rule": "lowest SHA256(PROTOCOL_ID|outcome-blind-pilot-subset|relative_path)",
        "pilot_rows_sha256": path_rows_sha256(selected),
        "confirmatory_examples": CONFIRMATORY_EXAMPLES,
        "confirmatory_rows_sha256": path_rows_sha256(confirmatory_rows),
        "confirmatory_outcomes_read": False,
        "methods": list(METHODS),
        "attacks": list(ATTACKS),
        "search_seeds": list(SEARCH_SEEDS),
        "confirmatory_search_seeds_predeclared": list(CONFIRMATORY_SEARCH_SEEDS),
        "primary_k": PRIMARY_K,
        "secondary_k": SECONDARY_K,
        "epsilon_px": EPSILON_PX,
        "api_budget_per_example": API_BUDGET,
        "square_p_init": SQUARE_P_INIT,
        "adaptive": dict(ADAPTIVE_CONFIG),
        "racing": dict(RACING_CONFIG),
        "frozen_defense": dict(FROZEN_DEFENSE),
        "reference_batch_size": REFERENCE_BATCH_SIZE,
        "search_batch_size_by_method": dict(SEARCH_BATCH_SIZE_BY_METHOD),
        "candidate_reevaluation": {
            "draws_stochastic": REEVAL_DRAWS,
            "draws_deterministic": 1,
            "seed": REEVAL_SEED,
            "batch_size": REEVAL_BATCH_SIZE,
            "mc_chunk": REEVAL_MC_CHUNK,
            "common_random_numbers": True,
            "not_charged_to_attack_budget": True,
        },
        "selection_rule": {
            "unit": "method",
            "criterion": "lowest_three_seed_mean_common_mc1000_exact_set_k5_probability",
            "tie_break": "SquareAdaptive",
            "no_per_example_selection": True,
            "pilot_only": True,
        },
        "bootstrap": {
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "unit": "pilot image after seed averaging",
        },
        "script_sha256": scripts,
        "external_attack_core_sha256": EXPECTED_ATTACK_CORE_SHA256,
        "no_training": True,
        "no_defense_retuning": True,
        "paper_claim_authorized": False,
        "budgets_run_independently": True,
        "all_eot_refresh_race_confirmation_draws_charged": True,
        "attack_reference": "ranked Top-5 of the clean standard model, frozen once before search",
    }
    payload["protocol_hash"] = object_sha256(payload)
    return payload


def validate_protocol(payload: Mapping[str, Any]) -> str:
    claimed = str(payload.get("protocol_hash", ""))
    base = dict(payload)
    base.pop("protocol_hash", None)
    if not claimed or object_sha256(base) != claimed:
        raise AssertionError("R4 protocol self-hash mismatch")
    expected = {
        "protocol_id": PROTOCOL_ID,
        "parent_protocol_id": PARENT_PROTOCOL_ID,
        "parent_r3_config_hash": R3_CONFIG_HASH,
        "parent_r3_protocol_hash": R3_PROTOCOL_HASH,
        "r2_selection_hash": R2_SELECTION_HASH,
        "frozen_split_sha256": FROZEN_SPLIT_SHA256,
        "partition": "attack_pilot",
        "pilot_pool_examples": PILOT_POOL_EXAMPLES,
        "pilot_examples": PILOT_EXAMPLES,
        "confirmatory_examples": CONFIRMATORY_EXAMPLES,
        "confirmatory_outcomes_read": False,
        "primary_k": PRIMARY_K,
        "secondary_k": SECONDARY_K,
        "epsilon_px": EPSILON_PX,
        "api_budget_per_example": API_BUDGET,
        "square_p_init": SQUARE_P_INIT,
        "no_training": True,
        "no_defense_retuning": True,
        "paper_claim_authorized": False,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise AssertionError(f"Frozen R4 field changed: {key}")
    for key, value in (
        ("methods", list(METHODS)),
        ("attacks", list(ATTACKS)),
        ("search_seeds", list(SEARCH_SEEDS)),
        ("confirmatory_search_seeds_predeclared", list(CONFIRMATORY_SEARCH_SEEDS)),
        ("adaptive", ADAPTIVE_CONFIG),
        ("racing", RACING_CONFIG),
        ("frozen_defense", FROZEN_DEFENSE),
        ("search_batch_size_by_method", SEARCH_BATCH_SIZE_BY_METHOD),
    ):
        if payload.get(key) != value:
            raise AssertionError(f"Frozen R4 structure changed: {key}")
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
        raise AssertionError("Frozen R4 reevaluation configuration changed")
    expected_selection = {
        "unit": "method",
        "criterion": "lowest_three_seed_mean_common_mc1000_exact_set_k5_probability",
        "tie_break": "SquareAdaptive",
        "no_per_example_selection": True,
        "pilot_only": True,
    }
    if payload.get("selection_rule") != expected_selection:
        raise AssertionError("Frozen R4 selection rule changed")
    core = Path(payload["paths"]["attack_core"])
    if not core.is_file() or file_sha256(core) != EXPECTED_ATTACK_CORE_SHA256:
        raise AssertionError("Audited external Square core missing or changed")
    recorded_scripts = payload.get("script_sha256", {})
    if set(recorded_scripts) != set(SCRIPT_NAMES):
        raise AssertionError("R4 source-hash manifest is incomplete")
    for name in SCRIPT_NAMES:
        path = PROJECT_DIR / name
        if not path.is_file() or file_sha256(path) != recorded_scripts[name]:
            raise AssertionError(f"R4 source changed after protocol freeze: {name}")
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


def paired_bootstrap(delta: np.ndarray) -> dict[str, Any]:
    delta = np.asarray(delta, dtype=np.float64)
    if delta.ndim != 1 or len(delta) != PILOT_EXAMPLES:
        raise ValueError("Expected one aligned pilot value per image")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    cursor = 0
    while cursor < BOOTSTRAP_RESAMPLES:
        count = min(250, BOOTSTRAP_RESAMPLES - cursor)
        sampled = rng.integers(0, len(delta), size=(count, len(delta)))
        boot[cursor : cursor + count] = delta[sampled].mean(1)
        cursor += count
    low, high = np.quantile(boot, [0.025, 0.975])
    return {
        "difference_pp": 100.0 * float(delta.mean()),
        "ci95_low_pp": 100.0 * float(low),
        "ci95_high_pp": 100.0 * float(high),
        "examples": len(delta),
        "resamples": BOOTSTRAP_RESAMPLES,
        "seed": BOOTSTRAP_SEED,
        "status": "positive" if low > 0 else "negative" if high < 0 else "statistically_indistinguishable",
    }
