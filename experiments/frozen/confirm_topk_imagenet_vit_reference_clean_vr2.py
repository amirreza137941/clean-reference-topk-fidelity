#!/usr/bin/env python3
"""Official ImageNet clean confirmation for the frozen ViT-reference VR1 point.

This stage reads all 45,000 images in the untouched ``clean_confirmation``
partition.  It evaluates exactly one operating point selected by VR1 and may
authorize the later adaptive attack pilot.  It performs no tuning, training,
download, or attack.
"""

from __future__ import annotations

import argparse
import hashlib
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

import calibrate_topk_imagenet_vit_reference_vr1 as calibration
import topk_imagenet_vit_reference_clean_vr2_common as common
from topk_imagenet_preflight_common import (
    partition_rows,
    read_csv as read_r0_csv,
    validate_frozen_split,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = Path(".")
DEFAULT_DATA_ROOT = DEFAULT_ROOT / "datasets/imagenet"
DEFAULT_R0_DIR = DEFAULT_ROOT / "models/imagenet_topk/paper_topk_imagenet_preflight_r0"
DEFAULT_R1_DIR = DEFAULT_ROOT / "models/imagenet_topk/paper_topk_imagenet_robust_anchor_r1"
DEFAULT_CALIBRATION_DIR = (
    DEFAULT_ROOT
    / "models/imagenet_topk/paper_topk_imagenet_vit_reference_crossarch_calibration_dev_vr1"
)
DEFAULT_STANDARD_CHECKPOINT = DEFAULT_ROOT / "models/vit/imagenet/vit_b16_imagenet.pth"
DEFAULT_ROBUST_CHECKPOINT = (
    DEFAULT_ROOT / "models/resnet50/imagenet/salman2020do_r50_linf4.pth"
)
DEFAULT_OUTPUT_DIR = (
    DEFAULT_ROOT
    / "models/imagenet_topk/paper_topk_imagenet_vit_reference_crossarch_clean_confirmation_vr2"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Official ImageNet ViT-reference clean confirmation VR2"
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--r0-dir", type=Path, default=DEFAULT_R0_DIR)
    parser.add_argument("--r1-dir", type=Path, default=DEFAULT_R1_DIR)
    parser.add_argument("--calibration-dir", type=Path, default=DEFAULT_CALIBRATION_DIR)
    parser.add_argument("--standard-checkpoint", type=Path, default=DEFAULT_STANDARD_CHECKPOINT)
    parser.add_argument("--robust-checkpoint", type=Path, default=DEFAULT_ROBUST_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--mc-chunk", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.45)
    parser.add_argument("--mc-draws", type=int, default=common.MC_DRAWS)
    parser.add_argument(
        "--bootstrap-resamples", type=int, default=common.BOOTSTRAP_RESAMPLES
    )
    parser.add_argument("--seed", type=int, default=common.EVALUATION_SEED)
    parser.add_argument("--bootstrap-seed", type=int, default=common.BOOTSTRAP_SEED)
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if int(args.batch_size) != 16:
        raise ValueError("The frozen VR2 protocol requires --batch-size 16")
    if int(args.mc_chunk) != 4:
        raise ValueError("The frozen VR2 protocol requires --mc-chunk 4")
    if int(args.num_workers) < 0:
        raise ValueError("--num-workers must be non-negative")
    if not 0.0 < float(args.gpu_memory_fraction) <= 1.0:
        raise ValueError("--gpu-memory-fraction must be in (0,1]")
    if int(args.mc_draws) != common.MC_DRAWS:
        raise ValueError(f"The frozen VR2 protocol requires exactly {common.MC_DRAWS} draws")
    if int(args.bootstrap_resamples) != common.BOOTSTRAP_RESAMPLES:
        raise ValueError(
            f"The frozen VR2 protocol requires {common.BOOTSTRAP_RESAMPLES:,} resamples"
        )
    if int(args.seed) != common.EVALUATION_SEED:
        raise ValueError("The VR2 evaluation seed is frozen")
    if int(args.bootstrap_seed) != common.BOOTSTRAP_SEED:
        raise ValueError("The VR2 bootstrap seed is frozen")
    if bool(args.amp):
        raise ValueError("AMP is forbidden because the selected VR1 point used FP32")


def resolved(path: Path) -> Path:
    return path.expanduser().resolve()


def _write_or_validate_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.is_file():
        if common.read_json(path) != payload:
            raise AssertionError(f"Frozen artifact differs from this invocation: {path}")
    else:
        common.atomic_json(path, payload)


def validate_calibration_parent(
    paths: Mapping[str, Path],
) -> tuple[list[dict[str, object]], dict[str, Any]]:
    """Validate R0/R1 and the exact audited VR1 selection before opening clean data."""

    # Reuse every checkpoint, loader, R0/R1, and source-integrity contract from
    # calibration.  This reads split metadata but opens no calibration image.
    calibration.validate_parents(paths)

    calibration_dir = paths["calibration_dir"]
    required = {
        "run_config": calibration_dir / "run_config.json",
        "selection": calibration_dir / "defense_selection_vr1.json",
        "final": calibration_dir / "final_summary.json",
        "model_audit": calibration_dir / "model_load_audit.json",
    }
    for path in required.values():
        if not path.is_file():
            raise FileNotFoundError(f"Required audited VR1 artifact is missing: {path}")
    if common.file_sha256(required["run_config"]) != common.EXPECTED_CALIBRATION_RUN_CONFIG_SHA256:
        raise AssertionError("VR1 run_config.json differs from the audited calibration")
    if common.file_sha256(required["selection"]) != common.EXPECTED_SELECTION_FILE_SHA256:
        raise AssertionError("VR1 defense_selection_vr1.json differs from the audited selection")
    if common.file_sha256(required["final"]) != common.EXPECTED_CALIBRATION_FINAL_SHA256:
        raise AssertionError("VR1 final_summary.json differs from the audited calibration")

    run_config = common.read_json(required["run_config"])
    selection = common.read_json(required["selection"])
    final = common.read_json(required["final"])
    hash_payload = dict(selection)
    claimed_hash = str(hash_payload.pop("selection_hash"))
    if common.object_sha256(hash_payload) != claimed_hash:
        raise AssertionError("VR1 selection object hash does not reproduce")
    common.validate_frozen_operating_point(selection)
    if run_config.get("protocol_id") != common.PARENT_PROTOCOL_ID:
        raise AssertionError("Unexpected VR1 parent protocol")
    if run_config.get("config_hash") != common.EXPECTED_CALIBRATION_CONFIG_HASH:
        raise AssertionError("Unexpected VR1 parent config hash")
    if run_config.get("partitions_read") != ["calibration_dev"]:
        raise AssertionError("VR1 parent read an unexpected partition")
    if run_config.get("attack_outcomes_read") is not False:
        raise AssertionError("VR1 parent is not attack-outcome blind")
    recorded_sources = run_config.get("script_sha256", {})
    for filename, expected in common.EXPECTED_CALIBRATION_SOURCE_HASHES.items():
        source = PROJECT_DIR / filename
        if not source.is_file() or common.file_sha256(source) != expected:
            raise AssertionError(f"Frozen VR1 source changed or is missing: {filename}")
        if recorded_sources.get(filename) != expected:
            raise AssertionError(f"VR1 run did not record expected source: {filename}")
    if final.get("status") != "PASS":
        raise AssertionError("VR1 calibration did not pass")
    if final.get("official_clean_stage_ready") is not True:
        raise AssertionError("VR1 did not authorize official clean confirmation")
    if final.get("official_clean_stage_blockers") != []:
        raise AssertionError("VR1 clean-stage authorization has blockers")
    if final.get("attack_stage_ready") is not False:
        raise AssertionError("VR1 unexpectedly opened the attack stage")
    if final.get("paper_claim_authorized") is not False:
        raise AssertionError("VR1 unexpectedly authorized a paper claim")
    if final.get("selection_hash") != common.EXPECTED_SELECTION_HASH:
        raise AssertionError("VR1 final summary selection mismatch")
    if final.get("selection") != selection:
        raise AssertionError("VR1 final summary embeds a different selection")

    split_rows = read_r0_csv(paths["r0_dir"] / "frozen_imagenet_val_split.csv")
    split_check = validate_frozen_split(split_rows)
    if split_check.get("split_sha256") != common.FROZEN_SPLIT_SHA256:
        raise AssertionError("Frozen ImageNet split changed")
    clean_rows = partition_rows(split_rows, common.CLEAN_PARTITION)
    if len(clean_rows) != common.CLEAN_EXAMPLES:
        raise AssertionError(
            f"clean_confirmation must contain {common.CLEAN_EXAMPLES:,} examples"
        )
    targets = np.asarray([int(row["target"]) for row in clean_rows])
    expected = np.full(1000, common.CLEAN_EXAMPLES_PER_CLASS)
    if not np.array_equal(np.bincount(targets, minlength=1000), expected):
        raise AssertionError("clean_confirmation must contain 45 images per class")
    return clean_rows, {
        "run_config": run_config,
        "selection": selection,
        "final": final,
        "split_check": split_check,
    }


def outcome_blind_shards(
    rows: Sequence[Mapping[str, object]],
) -> list[list[dict[str, object]]]:
    def rank(row: Mapping[str, object]) -> str:
        label = f"{common.PROTOCOL_ID}|clean-shard|{row['relative_path']}"
        return hashlib.sha256(label.encode("utf-8")).hexdigest()

    ordered = [dict(row) for row in sorted(rows, key=rank)]
    shards = [
        ordered[start : start + common.EXAMPLES_PER_SHARD]
        for start in range(0, len(ordered), common.EXAMPLES_PER_SHARD)
    ]
    if len(shards) != common.CLEAN_SHARDS:
        raise AssertionError("Unexpected number of clean-confirmation shards")
    if any(len(shard) != common.EXAMPLES_PER_SHARD for shard in shards):
        raise AssertionError("Clean-confirmation shards are not all size 500")
    return shards


def _strength_spec(selection: Mapping[str, Any]) -> calibration.MethodSpec:
    strength = selection["strength"]
    return calibration.MethodSpec(
        method="full_sigma0",
        candidate_id="full_sigma0",
        gate_mode=str(strength["gate_mode"]),
        strength_alpha_px=float(strength["alpha_px"]),
        strength_epsilon_px=float(strength["epsilon_px"]),
        strength_steps=int(strength["steps"]),
        top1_weight=float(strength["top1_weight"]),
        set_weight=float(strength["set_weight"]),
        sigma=0.0,
        draws=1,
    )


def _with_protocol(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        row["protocol_id"] = common.PROTOCOL_ID
        row["partition"] = common.CLEAN_PARTITION
        output.append(row)
    return output


def evaluate_clean_shard(
    shard: str,
    loader: DataLoader,
    standard: torch.nn.Module,
    robust: torch.nn.Module,
    selection: Mapping[str, Any],
    device: torch.device,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    strength_spec = _strength_spec(selection)
    sigma = float(selection["randomization"]["sigma"])
    standard_spec = calibration.MethodSpec("standard", "standard")
    robust_spec = calibration.MethodSpec("robust", "robust")
    noise_spec = calibration.MethodSpec(
        method="noise_only",
        candidate_id="noise_only",
        sigma=sigma,
        draws=int(args.mc_draws),
    )
    full_values = asdict(strength_spec)
    full_values.update(
        {"method": "full", "candidate_id": "full", "sigma": sigma, "draws": int(args.mc_draws)}
    )
    full_spec = calibration.MethodSpec(**full_values)

    for batch_number, (images, labels, relative_paths) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        paths = [str(value) for value in relative_paths]
        with torch.no_grad(), calibration.autocast_context(device, False):
            standard_logits = standard(images).float()
            robust_logits = robust(images).float()
            reference_k5 = calibration.ranked_reference(
                standard_logits, common.PRIMARY_K
            ).clone()
            disagreement = standard_logits.argmax(1).ne(robust_logits.argmax(1))
        full_base, full_gate = calibration.selected_strength_base(
            images, reference_k5, disagreement, robust, strength_spec, False
        )
        inactive = torch.zeros(len(images), dtype=torch.bool, device=device)
        with torch.inference_mode(), calibration.autocast_context(device, False):
            full0_logits = robust(full_base).float()

        for spec, logits, active in (
            (standard_spec, standard_logits, inactive),
            (robust_spec, robust_logits, inactive),
            (strength_spec, full0_logits, full_gate),
        ):
            output.extend(
                _with_protocol(
                    calibration.deterministic_rows(
                        "official_clean",
                        shard,
                        spec,
                        logits,
                        labels,
                        paths,
                        reference_k5,
                        active,
                    )
                )
            )

        counts, logit_sums = calibration._empty_mc_store([sigma], len(images), device)
        generator = torch.Generator(device=device)
        generator.manual_seed(
            calibration.stable_seed(
                args.seed, f"official-clean-common|{shard}|batch-{batch_number:04d}"
            )
        )
        completed = 0
        while completed < int(args.mc_draws):
            chunk = min(int(args.mc_chunk), int(args.mc_draws) - completed)
            noise = torch.randn(
                (chunk,) + tuple(images.shape),
                dtype=images.dtype,
                device=device,
                generator=generator,
            )
            tensors = (
                (images.unsqueeze(0) + sigma * noise).clamp(0.0, 1.0),
                (full_base.unsqueeze(0) + sigma * noise).clamp(0.0, 1.0),
            )
            flattened = torch.cat([tensor.flatten(0, 1) for tensor in tensors], dim=0)
            with torch.inference_mode(), calibration.autocast_context(device, False):
                combined_logits = robust(flattened).float()
            block = chunk * len(images)
            for position, method in enumerate(("noise_only", "full")):
                logits = combined_logits[position * block : (position + 1) * block].view(
                    chunk, len(images), 1000
                )
                flat_metrics = calibration.metric_tensors(
                    logits.flatten(0, 1),
                    labels.repeat(chunk),
                    reference_k5.repeat(chunk, 1),
                )
                key = (sigma, method)
                for metric, values in flat_metrics.items():
                    counts[key][metric] += values.view(chunk, len(images)).sum(0).double()
                logit_sums[key] += logits.double().sum(0)
            completed += chunk

        for method, spec, active in (
            ("noise_only", noise_spec, inactive),
            ("full", full_spec, full_gate),
        ):
            key = (sigma, method)
            mean_logits = (logit_sums[key] / int(args.mc_draws)).float()
            output.extend(
                _with_protocol(
                    calibration.rows_from_statistics(
                        "official_clean",
                        shard,
                        spec,
                        labels,
                        paths,
                        reference_k5,
                        active,
                        counts[key],
                        mean_logits,
                    )
                )
            )
        print(
            f"    official_clean/{shard}: batch {batch_number}/{len(loader)}",
            flush=True,
        )
    return output


def _partial_path(output_dir: Path, shard: str) -> Path:
    return output_dir / "partials" / "official_clean" / f"{shard}.json"


def run_or_resume_shard(
    output_dir: Path,
    shard: str,
    config_hash: str,
    protocol_hash: str,
    rows: Sequence[Mapping[str, object]],
    evaluator,
) -> dict[str, Any]:
    path = _partial_path(output_dir, shard)
    expected_paths = sorted(str(row["relative_path"]) for row in rows)
    if path.is_file():
        payload = common.read_json(path)
        if payload.get("protocol_id") != common.PROTOCOL_ID:
            raise AssertionError(f"Partial protocol mismatch: {path}")
        if payload.get("config_hash") != config_hash:
            raise AssertionError(f"Partial invocation mismatch: {path}")
        if payload.get("protocol_hash") != protocol_hash:
            raise AssertionError(f"Partial preregistration mismatch: {path}")
        if payload.get("phase") != "official_clean" or payload.get("shard_id") != shard:
            raise AssertionError(f"Partial phase/shard mismatch: {path}")
        if payload.get("selection_hash") != common.EXPECTED_SELECTION_HASH:
            raise AssertionError(f"Partial selection mismatch: {path}")
        observed = sorted({str(row["relative_path"]) for row in payload["examples"]})
        if observed != expected_paths:
            raise AssertionError(f"Partial sample membership mismatch: {path}")
        print(f"RESUME official_clean/{shard}", flush=True)
        return payload
    print(f"RUN official_clean/{shard}", flush=True)
    started = time.monotonic()
    examples = evaluator()
    payload = {
        "protocol_id": common.PROTOCOL_ID,
        "protocol_hash": protocol_hash,
        "config_hash": config_hash,
        "partition": common.CLEAN_PARTITION,
        "phase": "official_clean",
        "shard_id": shard,
        "selection_hash": common.EXPECTED_SELECTION_HASH,
        "input_examples": len(rows),
        "wall_seconds": time.monotonic() - started,
        "examples": examples,
    }
    common.atomic_json(path, payload)
    print(f"PASS official_clean/{shard}: {len(examples):,} rows", flush=True)
    return payload


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["method"]), []).append(row)
    if set(grouped) != set(common.METHODS):
        raise AssertionError(f"Unexpected method set: {sorted(grouped)}")
    output: list[dict[str, Any]] = []
    for method in common.METHODS:
        group = grouped[method]
        first = group[0]
        paths = {str(row["relative_path"]) for row in group}
        if len(group) != common.CLEAN_EXAMPLES or len(paths) != common.CLEAN_EXAMPLES:
            raise AssertionError(f"{method} does not have 45,000 unique examples")

        def percent(key: str) -> float:
            return 100.0 * float(np.mean([float(row[key]) for row in group]))

        output.append(
            {
                "protocol_id": common.PROTOCOL_ID,
                "partition": common.CLEAN_PARTITION,
                "phase": "official_clean",
                "method": method,
                "examples": len(group),
                "draws": int(first["draws"]),
                "sigma": float(first["sigma"]),
                "gate_mode": str(first["gate_mode"]),
                "gate_activation_percent": percent("gate_active"),
                "strength_alpha_px": float(first["strength_alpha_px"]),
                "strength_epsilon_px": float(first["strength_epsilon_px"]),
                "strength_steps": int(first["strength_steps"]),
                "top1_weight": float(first["top1_weight"]),
                "set_weight": float(first["set_weight"]),
                "top1_accuracy_percent": percent("top1_accuracy_probability"),
                "reference_top1_match_probability_percent": percent(
                    "reference_top1_match_probability"
                ),
                "label_coverage_k5_probability_percent": percent(
                    "label_coverage_k5_probability"
                ),
                "exact_set_k5_probability_percent": percent(
                    "exact_set_k5_probability"
                ),
                "label_coverage_k2_probability_percent": percent(
                    "label_coverage_k2_probability"
                ),
                "exact_set_k2_probability_percent": percent(
                    "exact_set_k2_probability"
                ),
                "mean_logits_top1_accuracy_percent": percent(
                    "mean_logits_top1_accuracy"
                ),
                "mean_logits_label_coverage_k5_percent": percent(
                    "mean_logits_label_coverage_k5"
                ),
                "mean_logits_exact_set_k5_percent": percent(
                    "mean_logits_exact_set_k5"
                ),
                "mean_logits_exact_set_k2_percent": percent(
                    "mean_logits_exact_set_k2"
                ),
                "mean_boundary_margin_k5": float(
                    np.mean([float(row["mean_boundary_margin_k5"]) for row in group])
                ),
                "mean_boundary_margin_k2": float(
                    np.mean([float(row["mean_boundary_margin_k2"]) for row in group])
                ),
            }
        )
    return output


def bootstrap_specs(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups = {
        method: [row for row in rows if str(row["method"]) == method]
        for method in common.METHODS
    }
    specs: list[dict[str, Any]] = []
    for left, right in common.COMPARISON_PAIRS:
        for metric in common.REPORT_METRICS:
            key = common.PROBABILITY_METRICS[metric]
            class_delta, examples = common.class_cluster_delta(
                groups[left], groups[right], key
            )
            specs.append(
                {
                    "protocol_id": common.PROTOCOL_ID,
                    "group_id": "official_clean",
                    "comparison": common.comparison_name(left, right, metric),
                    "comparison_label": f"{left} - {right}",
                    "left": left,
                    "right": right,
                    "metric": metric,
                    "base_metric_key": key,
                    "examples": examples,
                    "class_delta": class_delta,
                }
            )
    return specs


def classwise_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for method in common.METHODS:
        method_rows = [row for row in rows if str(row["method"]) == method]
        by_class: dict[int, list[Mapping[str, Any]]] = {}
        for row in method_rows:
            by_class.setdefault(int(row["label"]), []).append(row)
        if sorted(by_class) != list(range(common.CLEAN_CLASSES)):
            raise AssertionError(f"Missing ImageNet classes for {method}")
        for target in range(common.CLEAN_CLASSES):
            group = by_class[target]
            if len(group) != common.CLEAN_EXAMPLES_PER_CLASS:
                raise AssertionError(f"Class {target} row count mismatch for {method}")

            def percent(key: str) -> float:
                return 100.0 * float(np.mean([float(row[key]) for row in group]))

            output.append(
                {
                    "protocol_id": common.PROTOCOL_ID,
                    "partition": common.CLEAN_PARTITION,
                    "method": method,
                    "target": target,
                    "examples": len(group),
                    "top1_accuracy_percent": percent("top1_accuracy_probability"),
                    "label_coverage_k5_probability_percent": percent(
                        "label_coverage_k5_probability"
                    ),
                    "exact_set_k5_probability_percent": percent(
                        "exact_set_k5_probability"
                    ),
                    "exact_set_k2_probability_percent": percent(
                        "exact_set_k2_probability"
                    ),
                }
            )
    return output


def reference_quality_decomposition(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_method = {
        method: common.rows_by_sample(
            [row for row in rows if str(row["method"]) == method]
        )
        for method in common.METHODS
    }
    paths = sorted(by_method["standard"])
    if any(sorted(table) != paths for table in by_method.values()):
        raise AssertionError("Methods do not share identical clean-confirmation samples")
    groups = {
        "all": paths,
        "standard_correct": [
            path
            for path in paths
            if float(by_method["standard"][path]["top1_accuracy_probability"]) == 1.0
        ],
        "standard_wrong": [
            path
            for path in paths
            if float(by_method["standard"][path]["top1_accuracy_probability"]) != 1.0
        ],
    }
    output: list[dict[str, Any]] = []
    for group_name, group_paths in groups.items():
        if not group_paths:
            raise AssertionError(f"Empty reference-quality group: {group_name}")
        for method, table in by_method.items():
            def percent(key: str) -> float:
                return 100.0 * float(
                    np.mean([float(table[path][key]) for path in group_paths])
                )

            output.append(
                {
                    "protocol_id": common.PROTOCOL_ID,
                    "group": group_name,
                    "method": method,
                    "examples": len(group_paths),
                    "top1_accuracy_percent": percent("top1_accuracy_probability"),
                    "label_coverage_k5_probability_percent": percent(
                        "label_coverage_k5_probability"
                    ),
                    "exact_set_k5_probability_percent": percent(
                        "exact_set_k5_probability"
                    ),
                    "exact_set_k2_probability_percent": percent(
                        "exact_set_k2_probability"
                    ),
                }
            )
    return output


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = calibration.configure_device(args)
    calibration.seed_everything(args.seed)
    paths = {
        "data_root": resolved(args.data_root),
        "r0_dir": resolved(args.r0_dir),
        "r1_dir": resolved(args.r1_dir),
        "calibration_dir": resolved(args.calibration_dir),
        "standard_checkpoint": resolved(args.standard_checkpoint),
        "robust_checkpoint": resolved(args.robust_checkpoint),
        "output_dir": resolved(args.output_dir),
    }
    paths["output_dir"].mkdir(parents=True, exist_ok=True)
    print(f"PROTOCOL_ID={common.PROTOCOL_ID}")
    print(f"sys.executable={sys.executable}")
    print(
        "scope: frozen selection on clean_confirmation only; no selection, attack, "
        "training, download, or calibration-image read",
        flush=True,
    )

    protocol = common.frozen_protocol_payload()
    protocol_hash = str(protocol["protocol_hash"])
    _write_or_validate_json(
        paths["output_dir"] / "frozen_protocol_vr2.json", protocol
    )

    clean_rows, parents = validate_calibration_parent(paths)
    selection = parents["selection"]
    shards = outcome_blind_shards(clean_rows)
    _transform, preprocessing = calibration.raw_cross_architecture_preprocessing()
    del _transform
    manifest_rows: list[dict[str, Any]] = []
    for shard_number, shard_rows in enumerate(shards):
        for row in shard_rows:
            manifest_rows.append({**dict(row), "shard_id": common.shard_id(shard_number)})
    manifest = {
        "protocol_id": common.PROTOCOL_ID,
        "protocol_hash": protocol_hash,
        "partition": common.CLEAN_PARTITION,
        "examples": common.CLEAN_EXAMPLES,
        "classes": common.CLEAN_CLASSES,
        "examples_per_class": common.CLEAN_EXAMPLES_PER_CLASS,
        "shards": common.CLEAN_SHARDS,
        "examples_per_shard": common.EXAMPLES_PER_SHARD,
        "assignment": "SHA256 rank of relative_path before inference, then contiguous chunks",
        "rows_sha256": common.object_sha256(manifest_rows),
        "parent_split_sha256": common.FROZEN_SPLIT_SHA256,
        "selection_hash": common.EXPECTED_SELECTION_HASH,
        "image_partitions_read": [common.CLEAN_PARTITION],
        "image_partitions_not_read": [
            "preflight_sanity",
            "calibration_dev",
            "attack_pilot",
            "attack_confirmatory",
        ],
        "attack_outcomes_read": False,
        "preprocessing": preprocessing,
    }
    _write_or_validate_json(
        paths["output_dir"] / "clean_confirmation_manifest.json", manifest
    )
    common.atomic_csv(
        paths["output_dir"] / "clean_confirmation_manifest.csv", manifest_rows
    )

    source_paths = {filename: PROJECT_DIR / filename for filename in common.NEW_SCRIPT_NAMES}
    source_paths.update(
        {
            filename: PROJECT_DIR / filename
            for filename in common.EXPECTED_CALIBRATION_SOURCE_HASHES
        }
    )
    source_paths.update(
        {
            filename: PROJECT_DIR / filename
            for filename in calibration.common.EXPECTED_PARENT_SOURCE_HASHES
        }
    )
    for path in source_paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Required source is missing: {path}")
    config = {
        "protocol_id": common.PROTOCOL_ID,
        "protocol_hash": protocol_hash,
        "parent_protocol_id": common.PARENT_PROTOCOL_ID,
        "dataset": "ImageNet-1k validation",
        "stage": "official_clean_confirmation",
        "experiment_scope": (
            "cross-architecture reference transfer with ViT-B/16 reference and "
            "robust ResNet-50 defended backbone; not a robust-ViT evaluation"
        ),
        "architecture_roles": {
            "standard_reference": "torchvision ViT-B/16",
            "defended_backbone": "torchvision ResNet-50",
            "cross_architecture_reference": True,
        },
        "checkpoints": {
            "standard": {
                "path": str(paths["standard_checkpoint"]),
                "sha256": common.STANDARD_CHECKPOINT_SHA256,
                "model_id": "torchvision_vit_b_16",
                "role": "standard_reference_only",
            },
            "robust": {
                "path": str(paths["robust_checkpoint"]),
                "sha256": common.ROBUST_CHECKPOINT_SHA256,
                "state_dict_fingerprint_sha256": common.ROBUST_STATE_FINGERPRINT,
                "model_id": "Salman2020Do_R50",
                "role": "defended_backbone",
                "training_epsilon_linf_px": 4.0,
            },
        },
        "paths": {key: str(value) for key, value in paths.items()},
        "parent_artifacts": {
            "frozen_split_sha256": common.FROZEN_SPLIT_SHA256,
            "calibration_config_hash": common.EXPECTED_CALIBRATION_CONFIG_HASH,
            "selection_hash": common.EXPECTED_SELECTION_HASH,
            "selection_file_sha256": common.EXPECTED_SELECTION_FILE_SHA256,
            "calibration_final_sha256": common.EXPECTED_CALIBRATION_FINAL_SHA256,
        },
        "frozen_operating_point": {
            "strength": selection["strength"],
            "randomization": {
                "sigma": float(selection["randomization"]["sigma"]),
                "official_clean_draws": int(args.mc_draws),
            },
        },
        "primary_k": common.PRIMARY_K,
        "secondary_k": common.SECONDARY_K,
        "partition": common.CLEAN_PARTITION,
        "clean_examples": common.CLEAN_EXAMPLES,
        "clean_shards": common.CLEAN_SHARDS,
        "batch_size": int(args.batch_size),
        "mc_chunk": int(args.mc_chunk),
        "num_workers": int(args.num_workers),
        "gpu_memory_fraction": float(args.gpu_memory_fraction),
        "amp": False,
        "mc_draws": int(args.mc_draws),
        "bootstrap": {
            "unit": "ImageNet class cluster",
            "clusters": common.CLEAN_CLASSES,
            "resamples": int(args.bootstrap_resamples),
            "seed": int(args.bootstrap_seed),
            "interval": "paired percentile 95%",
        },
        "seed": int(args.seed),
        "bootstrap_seed": int(args.bootstrap_seed),
        "noninferiority_margin_pp": common.NONINFERIORITY_MARGIN_PP,
        "gate_rules": {
            name: {"rule": rule, "threshold_pp": threshold}
            for name, (rule, threshold) in common.GATE_RULES.items()
        },
        "common_random_numbers": True,
        "offline": True,
        "downloads_allowed": False,
        "training_allowed": False,
        "selection_allowed": False,
        "attack_allowed": False,
        "attack_outcomes_read": False,
        "image_partitions_read": [common.CLEAN_PARTITION],
        "image_partitions_not_read": [
            "preflight_sanity",
            "calibration_dev",
            "attack_pilot",
            "attack_confirmatory",
        ],
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
        },
        "script_sha256": {
            name: common.file_sha256(path) for name, path in source_paths.items()
        },
    }
    config_hash = common.object_sha256(config)
    config["config_hash"] = config_hash
    _write_or_validate_json(paths["output_dir"] / "run_config.json", config)

    standard, robust, model_audit = calibration.load_models(
        paths["standard_checkpoint"], paths["robust_checkpoint"], device
    )
    _write_or_validate_json(paths["output_dir"] / "model_load_audit.json", model_audit)

    partials: list[dict[str, Any]] = []
    for shard_number, shard_rows in enumerate(shards):
        shard = common.shard_id(shard_number)
        loader = calibration.make_loader(
            paths["data_root"], shard_rows, args.batch_size, args.num_workers, device
        )
        partials.append(
            run_or_resume_shard(
                paths["output_dir"],
                shard,
                config_hash,
                protocol_hash,
                shard_rows,
                lambda loader=loader, shard=shard: evaluate_clean_shard(
                    shard, loader, standard, robust, selection, device, args
                ),
            )
        )

    rows = [row for payload in partials for row in payload["examples"]]
    expected_rows = common.CLEAN_EXAMPLES * len(common.METHODS)
    if len(rows) != expected_rows:
        raise AssertionError(f"Expected {expected_rows:,} per-example rows; got {len(rows):,}")
    summaries = summarize_rows(rows)
    print(
        "Computing 50,000-resample paired class-cluster bootstrap on 45,000 images",
        flush=True,
    )
    bootstrap = common.paired_class_bootstrap_many(
        bootstrap_specs(rows), args.bootstrap_resamples, args.bootstrap_seed
    )
    gate = common.evaluate_clean_gate(bootstrap)
    decomposition = reference_quality_decomposition(rows)
    classwise = classwise_summary(rows)
    runtime_rows = [
        {
            "protocol_id": common.PROTOCOL_ID,
            "shard_id": payload["shard_id"],
            "input_examples": int(payload["input_examples"]),
            "output_rows": len(payload["examples"]),
            "wall_seconds": float(payload["wall_seconds"]),
        }
        for payload in partials
    ]
    del partials

    common.atomic_csv(paths["output_dir"] / "official_clean_summary.csv", summaries)
    common.atomic_csv(
        paths["output_dir"] / "paired_class_bootstrap.csv", bootstrap
    )
    common.atomic_csv(
        paths["output_dir"] / "preregistered_clean_gates.csv", gate["rules"]
    )
    common.atomic_csv(
        paths["output_dir"] / "reference_quality_decomposition.csv", decomposition
    )
    common.atomic_csv(paths["output_dir"] / "classwise_summary.csv", classwise)
    common.atomic_csv(
        paths["output_dir"] / "partial_runtime_summary.csv", runtime_rows
    )
    common.atomic_json(paths["output_dir"] / "official_clean_gate.json", gate)
    per_example_path = paths["output_dir"] / "official_clean_per_example.csv.gz"
    common.atomic_gzip_csv(per_example_path, rows)

    attack_ready = bool(gate["passed"])
    failed_gates = [
        str(rule["comparison"]) for rule in gate["rules"] if not bool(rule["passed"])
    ]
    final_summary = {
        "protocol_id": common.PROTOCOL_ID,
        "protocol_hash": protocol_hash,
        "config_hash": config_hash,
        "selection_hash": common.EXPECTED_SELECTION_HASH,
        "status": "PASS",
        "official_clean_confirmation_completed": True,
        "official_clean_confirmation_passed": attack_ready,
        "failed_gates": failed_gates,
        "frozen_operating_point": {
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
            "mc_draws": common.MC_DRAWS,
        },
        "official_clean_gate": gate,
        "official_clean_summaries": summaries,
        "official_clean_paired_bootstrap": bootstrap,
        "attack_stage_ready": attack_ready,
        "clean_paper_claim_authorized": attack_ready,
        "adversarial_paper_claim_authorized": False,
        "paper_claim_authorized": False,
        "next_stage": (
            "vit_reference_crossarch_adaptive_attack_pilot_on_attack_pilot_partition"
            if attack_ready
            else "stop_and_report_failed_official_clean_confirmation"
        ),
        "monte_carlo_precision": {
            "aggregate_probability_standard_error_upper_bound_pp": (
                100.0
                * np.sqrt(0.25 / (common.CLEAN_EXAMPLES * common.MC_DRAWS))
            ),
            "note": (
                "upper bound only; paired class-cluster CIs are the "
                "inferential intervals"
            ),
        },
        "data_governance": {
            "image_partitions_read": [common.CLEAN_PARTITION],
            "image_partitions_not_read": [
                "preflight_sanity",
                "calibration_dev",
                "attack_pilot",
                "attack_confirmatory",
            ],
            "calibration_artifacts_read": True,
            "calibration_images_read": False,
            "attack_outcomes_read": False,
            "no_training": True,
            "no_download": True,
            "no_attack": True,
            "no_selection": True,
        },
        "artifacts": {
            "per_example_file": per_example_path.name,
            "per_example_file_sha256": common.file_sha256(per_example_path),
            "per_example_rows": len(rows),
            "partial_files": common.CLEAN_SHARDS,
            "partial_wall_seconds_sum": float(
                sum(float(row["wall_seconds"]) for row in runtime_rows)
            ),
            "classwise_rows": len(classwise),
        },
    }
    common.atomic_json(paths["output_dir"] / "final_summary.json", final_summary)

    table = {str(row["method"]): row for row in summaries}
    print("\nOfficial ImageNet ViT-reference clean confirmation", flush=True)
    print(
        "  frozen strength=global_a1_s4_w4, alpha=1/255, epsilon=4/255, "
        "steps=4, weights=1:4; sigma=0.025; MC=50",
        flush=True,
    )
    for method in common.METHODS:
        row = table[method]
        print(
            f"  {method:>11}: Top1={float(row['top1_accuracy_percent']):.3f}% "
            f"Top5={float(row['label_coverage_k5_probability_percent']):.3f}% "
            f"Exact5={float(row['exact_set_k5_probability_percent']):.3f}% "
            f"Exact2={float(row['exact_set_k2_probability_percent']):.3f}%",
            flush=True,
        )
    for rule in gate["rules"]:
        print(
            f"  gate {rule['comparison']}: {float(rule['difference_pp']):+.3f} pp "
            f"[{float(rule['ci95_low_pp']):+.3f}, "
            f"{float(rule['ci95_high_pp']):+.3f}] -> "
            f"{'PASS' if rule['passed'] else 'FAIL'}",
            flush=True,
        )
    print(
        "OFFICIAL_CLEAN_CONFIRMATION=" + ("PASS" if attack_ready else "FAIL"),
        flush=True,
    )
    print("ATTACK_STAGE_READY=" + ("YES" if attack_ready else "NO"), flush=True)
    print("PAPER_CLAIM_AUTHORIZED=NO (no attack result yet)", flush=True)
    print(f"Complete: {paths['output_dir']}", flush=True)


if __name__ == "__main__":
    main()
