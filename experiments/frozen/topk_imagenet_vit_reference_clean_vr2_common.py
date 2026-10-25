#!/usr/bin/env python3
"""Pure contracts for ImageNet ViT-reference official clean confirmation VR2.

The operating point was selected once on ``calibration_dev`` by VR1.  VR2
only evaluates that frozen point on the untouched, class-balanced
``clean_confirmation`` partition.  No parameter selection and no attack are
permitted in this stage.
"""

from __future__ import annotations

import csv
import gzip
import io
import os
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

import topk_imagenet_vit_reference_calibration_vr1_common as calibration_common


PROTOCOL_ID = "imagenet_topk_vit_reference_crossarch_clean_confirmation_vr2_20260917"
PARENT_PROTOCOL_ID = calibration_common.PROTOCOL_ID

FROZEN_SPLIT_SHA256 = calibration_common.FROZEN_SPLIT_SHA256
STANDARD_CHECKPOINT_SHA256 = calibration_common.STANDARD_CHECKPOINT_SHA256
ROBUST_CHECKPOINT_SHA256 = calibration_common.ROBUST_CHECKPOINT_SHA256
ROBUST_STATE_FINGERPRINT = calibration_common.ROBUST_STATE_FINGERPRINT

# Exact artifacts from the completed and audited VR1 calibration supplied by
# the user.  These locks prevent clean-confirmation results from being mixed
# with a different or subsequently retuned operating point.
EXPECTED_CALIBRATION_CONFIG_HASH = (
    "4899590956da9c01862538facaedefdf68a79538a864ce06e5f8840fc318d37f"
)
EXPECTED_SELECTION_HASH = (
    "3b754fffc763821d800bf40be1c47d07cd8ac32b3b07c806d6e999523ee37a14"
)
EXPECTED_SELECTION_FILE_SHA256 = (
    "1e4ebadb1af1384be01c2feba40dc099336a60c66e5f66fac758f91f99a26083"
)
EXPECTED_CALIBRATION_FINAL_SHA256 = (
    "0c9dedff23b35601fa0b7f6f5a941786e3a35643448f23e69bb33f0d978f921b"
)
EXPECTED_CALIBRATION_RUN_CONFIG_SHA256 = (
    "42a9bf357b3d95023f8af9d608622dcd0e1d3bdc92352969a53ca341573490ad"
)

EXPECTED_CALIBRATION_SOURCE_HASHES = {
    "topk_imagenet_vit_reference_calibration_vr1_common.py": (
        "429b0f09fdc9f9dd0737ab37ceb38d2eca680098b74f69ea4d480c8a3f935c0d"
    ),
    "calibrate_topk_imagenet_vit_reference_vr1.py": (
        "83a5fd31bac13053fc00fbf060c247a330e93fcae6a6a36e39a5f0f22b3ad53e"
    ),
    "audit_topk_imagenet_vit_reference_calibration_vr1.py": (
        "54433dea0f5be28e0cf0dc408f1d54c9c4f906ed8a02f55df89b7fb14522dd99"
    ),
    "test_topk_imagenet_vit_reference_calibration_vr1.py": (
        "84f433f57304b5bcb72ac6801ac47a50d8c80fa49c5eef2a9c8b33c50d377b6d"
    ),
    "run_topk_imagenet_vit_reference_calibration_vr1.py": (
        "dbe82f4f4278c9ade9dc2351e5e9176ef3669a63378fe45a0719033c014420b4"
    ),
}

CLEAN_PARTITION = "clean_confirmation"
CLEAN_EXAMPLES = 45_000
CLEAN_CLASSES = 1_000
CLEAN_EXAMPLES_PER_CLASS = 45
EXAMPLES_PER_SHARD = 500
CLEAN_SHARDS = CLEAN_EXAMPLES // EXAMPLES_PER_SHARD
MC_DRAWS = 50
NONINFERIORITY_MARGIN_PP = 2.0
BOOTSTRAP_RESAMPLES = 50_000
EVALUATION_SEED = 2026091705
BOOTSTRAP_SEED = 2026091706

PRIMARY_K = calibration_common.PRIMARY_K
SECONDARY_K = calibration_common.SECONDARY_K
METHODS = ("standard", "robust", "full_sigma0", "noise_only", "full")
STOCHASTIC_METHODS = ("noise_only", "full")

PROBABILITY_METRICS = dict(calibration_common.PROBABILITY_METRICS)
REPORT_METRICS = ("top1", "top5", "exact_k5", "exact_k2")

# Every comparison below is reported.  The scientific gate uses only the
# preregistered subset in GATE_RULES.
COMPARISON_PAIRS = (
    ("noise_only", "robust"),
    ("full_sigma0", "robust"),
    ("full", "robust"),
    ("full", "noise_only"),
    ("full", "full_sigma0"),
)

GATE_RULES = {
    # Noise must preserve the robust backbone's clean utility/fidelity.
    "noise_only_minus_robust_top1": ("noninferiority", -NONINFERIORITY_MARGIN_PP),
    "noise_only_minus_robust_top5": ("noninferiority", -NONINFERIORITY_MARGIN_PP),
    "noise_only_minus_robust_exact_k5": (
        "noninferiority",
        -NONINFERIORITY_MARGIN_PP,
    ),
    # Full must not sacrifice ordinary clean utility relative to robust.
    "full_minus_robust_top1": ("noninferiority", -NONINFERIORITY_MARGIN_PP),
    "full_minus_robust_top5": ("noninferiority", -NONINFERIORITY_MARGIN_PP),
    # The frozen positive sigma must preserve the deterministic strengthened
    # operating point within the same 2 pp clean tolerance used at selection.
    "full_minus_full_sigma0_top1": (
        "noninferiority",
        -NONINFERIORITY_MARGIN_PP,
    ),
    "full_minus_full_sigma0_top5": (
        "noninferiority",
        -NONINFERIORITY_MARGIN_PP,
    ),
    "full_minus_full_sigma0_exact_k5": (
        "noninferiority",
        -NONINFERIORITY_MARGIN_PP,
    ),
    # The cross-architecture alignment module must provide a replicated
    # Exact-Top5 gain over both the robust anchor and randomization alone
    # before any attack outcome is opened.
    "full_minus_robust_exact_k5": ("superiority", 0.0),
    "full_minus_noise_only_exact_k5": ("superiority", 0.0),
}

NEW_SCRIPT_NAMES = (
    "topk_imagenet_vit_reference_clean_vr2_common.py",
    "confirm_topk_imagenet_vit_reference_clean_vr2.py",
    "audit_topk_imagenet_vit_reference_clean_vr2.py",
    "test_topk_imagenet_vit_reference_clean_vr2.py",
    "pack_topk_imagenet_vit_reference_clean_vr2.py",
    "run_topk_imagenet_vit_reference_clean_vr2.py",
)


canonical_json_bytes = calibration_common.canonical_json_bytes
object_sha256 = calibration_common.object_sha256
file_sha256 = calibration_common.file_sha256
read_json = calibration_common.read_json
read_csv = calibration_common.read_csv
atomic_json = calibration_common.atomic_json
atomic_csv = calibration_common.atomic_csv
aligned_delta = calibration_common.aligned_delta
rows_by_sample = calibration_common.rows_by_sample


def atomic_gzip_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Atomically write a deterministic, compressed per-example table."""

    path = Path(path)
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fieldnames = list(rows[0].keys())
    expected = set(fieldnames)
    if any(set(row) != expected for row in rows):
        raise ValueError(f"Non-uniform row schema while writing {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=raw, compresslevel=3, mtime=0
        ) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
    os.replace(temporary, path)


def iter_csv(path: Path) -> Iterator[dict[str, str]]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle)


def shard_id(index: int) -> str:
    if not 0 <= int(index) < CLEAN_SHARDS:
        raise ValueError(f"Shard index outside [0,{CLEAN_SHARDS - 1}]: {index}")
    return f"shard_{int(index):03d}"


def comparison_name(left: str, right: str, metric: str) -> str:
    return f"{left}_minus_{right}_{metric}"


def frozen_protocol_payload() -> dict[str, Any]:
    payload = {
        "protocol_id": PROTOCOL_ID,
        "parent_protocol_id": PARENT_PROTOCOL_ID,
        "partition": CLEAN_PARTITION,
        "examples": CLEAN_EXAMPLES,
        "classes": CLEAN_CLASSES,
        "examples_per_class": CLEAN_EXAMPLES_PER_CLASS,
        "shards": CLEAN_SHARDS,
        "examples_per_shard": EXAMPLES_PER_SHARD,
        "primary_k": PRIMARY_K,
        "secondary_k": SECONDARY_K,
        "methods": list(METHODS),
        "selection_hash": EXPECTED_SELECTION_HASH,
        "strength": {
            "candidate_id": "global_a1_s4_w4",
            "gate_mode": "global",
            "alpha_px": 1.0,
            "epsilon_px": 4.0,
            "steps": 4,
            "top1_weight": 1.0,
            "set_weight": 4.0,
        },
        "sigma": 0.025,
        "mc_draws": MC_DRAWS,
        "common_random_numbers": True,
        "numerics": {
            "batch_size": 16,
            "mc_chunk": 4,
            "amp": False,
            "model_logits": "FP32",
            "mc_accumulators": "FP64",
        },
        "bootstrap": {
            "unit": "ImageNet class cluster",
            "clusters": CLEAN_CLASSES,
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "interval": "paired percentile 95%",
        },
        "evaluation_seed": EVALUATION_SEED,
        "noninferiority_margin_pp": NONINFERIORITY_MARGIN_PP,
        "gate_rules": {
            name: {"rule": rule, "threshold_pp": threshold}
            for name, (rule, threshold) in GATE_RULES.items()
        },
        "selection_or_retuning_allowed": False,
        "attack_allowed": False,
        "training_allowed": False,
        "attack_partitions_read": False,
        "next_stage_if_all_gates_pass": (
            "vit_reference_crossarch_adaptive_attack_pilot_on_attack_pilot_partition"
        ),
        "next_stage_if_any_gate_fails": (
            "stop_and_report_failed_official_clean_confirmation"
        ),
    }
    payload["protocol_hash"] = object_sha256(payload)
    return payload


def class_cluster_delta(
    left_rows: Sequence[Mapping[str, Any]],
    right_rows: Sequence[Mapping[str, Any]],
    metric_key: str,
) -> tuple[np.ndarray, int]:
    left = rows_by_sample(left_rows)
    right = rows_by_sample(right_rows)
    paths = sorted(left)
    if paths != sorted(right):
        raise ValueError("Paired methods do not share identical samples")
    by_class: dict[int, list[float]] = {}
    for path in paths:
        left_row = left[path]
        right_row = right[path]
        if int(left_row["label"]) != int(right_row["label"]):
            raise ValueError(f"Paired labels differ for {path}")
        target = int(left_row["label"])
        by_class.setdefault(target, []).append(
            float(left_row[metric_key]) - float(right_row[metric_key])
        )
    if sorted(by_class) != list(range(CLEAN_CLASSES)):
        raise ValueError("Class-cluster bootstrap requires all 1,000 ImageNet classes")
    if {len(values) for values in by_class.values()} != {CLEAN_EXAMPLES_PER_CLASS}:
        raise ValueError("Every class cluster must contain exactly 45 paired examples")
    return (
        np.asarray(
            [np.mean(by_class[index]) for index in range(CLEAN_CLASSES)],
            dtype=np.float64,
        ),
        len(paths),
    )


def paired_class_bootstrap_many(
    comparisons: Sequence[Mapping[str, Any]],
    resamples: int,
    seed: int,
    chunk_size: int = 100,
) -> list[dict[str, Any]]:
    """Paired percentile bootstrap with ImageNet class as the sampling unit."""

    if not comparisons or int(resamples) <= 0 or int(chunk_size) <= 0:
        raise ValueError("Bootstrap requires comparisons and positive sizes")
    vectors = [np.asarray(item["class_delta"], dtype=np.float64) for item in comparisons]
    cluster_count = len(vectors[0])
    if cluster_count <= 0 or any(len(vector) != cluster_count for vector in vectors):
        raise ValueError("Class-delta vectors must be non-empty and aligned")
    matrix = np.column_stack(vectors)
    rng = np.random.default_rng(int(seed))
    boot = np.empty((int(resamples), matrix.shape[1]), dtype=np.float64)
    cursor = 0
    while cursor < int(resamples):
        count = min(int(chunk_size), int(resamples) - cursor)
        sampled = rng.integers(0, cluster_count, size=(count, cluster_count))
        boot[cursor : cursor + count] = matrix[sampled].mean(axis=1)
        cursor += count
    output: list[dict[str, Any]] = []
    for position, (source, delta) in enumerate(zip(comparisons, vectors)):
        low, high = np.quantile(boot[:, position], [0.025, 0.975])
        row = {key: value for key, value in source.items() if key != "class_delta"}
        row.update(
            {
                "difference_pp": 100.0 * float(delta.mean()),
                "ci95_low_pp": 100.0 * float(low),
                "ci95_high_pp": 100.0 * float(high),
                "examples": int(source["examples"]),
                "clusters": cluster_count,
                "cluster_unit": "ImageNet class",
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


def evaluate_clean_gate(
    bootstrap_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate the frozen clean-confirmation gate from paired CI rows."""

    by_comparison: dict[str, Mapping[str, Any]] = {}
    for row in bootstrap_rows:
        name = str(row["comparison"])
        if name in by_comparison:
            raise ValueError(f"Duplicate bootstrap comparison: {name}")
        by_comparison[name] = row

    decisions: list[dict[str, Any]] = []
    blockers: list[str] = []
    for name, (rule, threshold) in GATE_RULES.items():
        if name not in by_comparison:
            raise ValueError(f"Missing preregistered gate comparison: {name}")
        row = by_comparison[name]
        lower = float(row["ci95_low_pp"])
        passed = lower >= float(threshold) - 1e-12 if rule == "noninferiority" else lower > 0.0
        decision = {
            "comparison": name,
            "rule": rule,
            "threshold_pp": float(threshold),
            "difference_pp": float(row["difference_pp"]),
            "ci95_low_pp": lower,
            "ci95_high_pp": float(row["ci95_high_pp"]),
            "passed": bool(passed),
        }
        decisions.append(decision)
        if not passed:
            relation = ">= -2 pp" if rule == "noninferiority" else "> 0 pp"
            blockers.append(
                f"{name}: lower 95% CI {lower:.6f} pp did not satisfy {relation}"
            )
    return {
        "gate_policy": (
            "all eight preregistered -2 pp non-inferiority lower-CI constraints "
            "plus replicated Full Exact-Top5 superiority over Robust and Noise-only"
        ),
        "pointwise_ci_note": (
            "The gate intersects preregistered pointwise paired 95% bootstrap CIs; "
            "these intervals are not advertised as familywise-adjusted confidence bounds."
        ),
        "rules": decisions,
        "passed": not blockers,
        "blockers": blockers,
    }


def validate_frozen_operating_point(selection: Mapping[str, Any]) -> None:
    """Reject any selection other than the audited VR1 operating point."""

    if str(selection.get("selection_hash")) != EXPECTED_SELECTION_HASH:
        raise AssertionError("VR1 selection hash differs from the frozen clean protocol")
    if str(selection.get("config_hash")) != EXPECTED_CALIBRATION_CONFIG_HASH:
        raise AssertionError("VR1 calibration config hash differs from the frozen protocol")
    if str(selection.get("selection_split")) != "calibration_dev":
        raise AssertionError("Operating point was not selected on calibration_dev")
    strength = selection.get("strength", {})
    expected_strength = {
        "candidate_id": "global_a1_s4_w4",
        "gate_mode": "global",
        "alpha_px": 1.0,
        "epsilon_px": 4.0,
        "steps": 4,
        "top1_weight": 1.0,
        "set_weight": 4.0,
    }
    for key, expected in expected_strength.items():
        observed = strength.get(key)
        if isinstance(expected, float):
            if abs(float(observed) - expected) > 1e-12:
                raise AssertionError(f"Frozen strength mismatch for {key}: {observed}")
        elif observed != expected:
            raise AssertionError(f"Frozen strength mismatch for {key}: {observed}")
    if strength.get("positive_strength_selected") is not True:
        raise AssertionError("VR1 did not select positive strengthening")
    if strength.get("selected_dev_summary", {}).get("jointly_ci_feasible") is not True:
        raise AssertionError("VR1 strengthening did not pass its paired-CI constraints")
    randomization = selection.get("randomization", {})
    if abs(float(randomization.get("sigma", -1.0)) - 0.025) > 1e-12:
        raise AssertionError("Frozen sigma must be exactly 0.025")
    if int(randomization.get("draws_during_calibration", -1)) != 50:
        raise AssertionError("Unexpected VR1 calibration draw count")
    if randomization.get("positive_sigma_selected") is not True:
        raise AssertionError("VR1 did not select a positive sigma")
    if randomization.get("selected_dev_constraints", {}).get("jointly_ci_feasible") is not True:
        raise AssertionError("VR1 sigma did not pass its paired-CI constraints")
    governance = selection.get("data_governance", {})
    if governance.get("partitions_read") != ["calibration_dev"]:
        raise AssertionError("VR1 selection has unexpected data provenance")
    if governance.get("attack_outcomes_read") is not False:
        raise AssertionError("VR1 selection is not outcome-blind")
    if governance.get("training_performed") is not False:
        raise AssertionError("VR1 selection unexpectedly performed training")
    if governance.get("cifar_selected_hyperparameters_transferred_as_fixed_values") is not False:
        raise AssertionError("VR1 selection transferred CIFAR hyperparameters")
    if governance.get("claim_scope") != "cross_architecture_reference_only_not_robust_vit":
        raise AssertionError("VR1 claim scope differs from the frozen protocol")
