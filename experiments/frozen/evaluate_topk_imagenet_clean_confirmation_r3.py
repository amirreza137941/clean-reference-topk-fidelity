#!/usr/bin/env python3
"""Frozen ImageNet clean confirmation for the R2 Top-k defense.

This stage reads only the 45,000-image ``clean_confirmation`` partition from
the split frozen in R0.  It evaluates one R2-selected operating point, uses
common random numbers for stochastic methods, and performs no selection,
training, attack, or hyperparameter tuning.
"""

from __future__ import annotations

import argparse
import contextlib
import platform
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import calibrate_topk_imagenet_defense_r2 as r2
import topk_imagenet_clean_confirmation_r3_common as common
from topk_imagenet_preflight_common import (
    partition_rows,
    read_csv as read_r0_csv,
    validate_frozen_split,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = Path("./datasets/imagenet")
DEFAULT_R0_DIR = Path(
    "./models/imagenet_topk/"
    "paper_topk_imagenet_preflight_r0"
)
DEFAULT_R1_DIR = Path(
    "./models/imagenet_topk/"
    "paper_topk_imagenet_robust_anchor_r1"
)
DEFAULT_R2_DIR = Path(
    "./models/imagenet_topk/"
    "paper_topk_imagenet_defense_calibration_dev_r2"
)
DEFAULT_STANDARD_CHECKPOINT = Path(
    "./models/resnet50/imagenet/resnet50_imagenet.pth"
)
DEFAULT_ROBUST_CHECKPOINT = Path(
    "./models/resnet50/imagenet/"
    "salman2020do_r50_linf4.pth"
)
DEFAULT_OUTPUT_DIR = Path(
    "./models/imagenet_topk/"
    "paper_topk_imagenet_clean_confirmation_k5_mc50_r3"
)


@dataclass(frozen=True)
class DefenseSpec:
    method: str
    gate_mode: str = "none"
    gate_active: bool = False
    sigma: float = 0.0
    draws: int = 1
    strength_alpha_px: float = 0.0
    strength_epsilon_px: float = 0.0
    strength_steps: int = 0
    top1_weight: float = 0.0
    set_weight: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ImageNet Top-k frozen clean confirmation R3")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--r0-dir", type=Path, default=DEFAULT_R0_DIR)
    parser.add_argument("--r1-dir", type=Path, default=DEFAULT_R1_DIR)
    parser.add_argument("--r2-dir", type=Path, default=DEFAULT_R2_DIR)
    parser.add_argument("--standard-checkpoint", type=Path, default=DEFAULT_STANDARD_CHECKPOINT)
    parser.add_argument("--robust-checkpoint", type=Path, default=DEFAULT_ROBUST_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--mc-chunk", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.45)
    parser.add_argument("--mc-draws", type=int, default=common.MC_DRAWS)
    parser.add_argument("--bootstrap-resamples", type=int, default=common.BOOTSTRAP_RESAMPLES)
    parser.add_argument("--seed", type=int, default=common.EVALUATION_SEED)
    parser.add_argument("--bootstrap-seed", type=int, default=common.BOOTSTRAP_SEED)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size != 16:
        raise ValueError("The frozen R3 protocol requires --batch-size 16")
    if args.mc_chunk != 4:
        raise ValueError("The frozen R3 protocol requires --mc-chunk 4")
    if args.mc_draws != common.MC_DRAWS:
        raise ValueError(f"The frozen R3 protocol requires {common.MC_DRAWS} MC draws")
    if args.bootstrap_resamples != common.BOOTSTRAP_RESAMPLES:
        raise ValueError(
            f"The frozen R3 protocol requires {common.BOOTSTRAP_RESAMPLES:,} resamples"
        )
    if args.seed != common.EVALUATION_SEED or args.bootstrap_seed != common.BOOTSTRAP_SEED:
        raise ValueError("R3 evaluation and bootstrap seeds are frozen")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if not 0.0 < args.gpu_memory_fraction <= 1.0:
        raise ValueError("--gpu-memory-fraction must be in (0,1]")


def resolved(path: Path) -> Path:
    return path.expanduser().resolve()


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False


def configure_device(args: argparse.Namespace) -> torch.device:
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(
            float(args.gpu_memory_fraction), device.index or 0
        )
    return device


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", enabled=False)
    return contextlib.nullcontext()


def _write_or_validate_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.is_file():
        if common.read_json(path) != payload:
            raise AssertionError(f"Frozen artifact differs from this invocation: {path}")
    else:
        common.atomic_json(path, payload)


def _validate_object_hash(payload: Mapping[str, Any], hash_key: str, expected: str) -> None:
    working = dict(payload)
    claimed = str(working.pop(hash_key))
    if claimed != expected or common.object_sha256(working) != expected:
        raise AssertionError(f"Invalid {hash_key}: expected {expected}, got {claimed}")


def validate_parents(paths: Mapping[str, Path]) -> tuple[list[dict[str, object]], dict[str, Any]]:
    required = [
        paths["r0_dir"] / "frozen_imagenet_val_split.csv",
        paths["r0_dir"] / "run_manifest.json",
        paths["r1_dir"] / "run_manifest.json",
        paths["standard_checkpoint"],
        paths["robust_checkpoint"],
    ]
    required.extend(paths["r2_dir"] / name for name in common.EXPECTED_R2_ARTIFACT_HASHES)
    required.extend(PROJECT_DIR / name for name in common.EXPECTED_R2_SOURCE_HASHES)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Required frozen parent artifact is missing: {path}")

    if common.file_sha256(paths["standard_checkpoint"]) != common.STANDARD_CHECKPOINT_SHA256:
        raise AssertionError("Standard checkpoint SHA256 mismatch")
    if common.file_sha256(paths["robust_checkpoint"]) != common.ROBUST_CHECKPOINT_SHA256:
        raise AssertionError("Robust checkpoint SHA256 mismatch")
    for name, expected in common.EXPECTED_R2_ARTIFACT_HASHES.items():
        if common.file_sha256(paths["r2_dir"] / name) != expected:
            raise AssertionError(f"Frozen R2 artifact changed: {name}")
    for name, expected in common.EXPECTED_R2_SOURCE_HASHES.items():
        if common.file_sha256(PROJECT_DIR / name) != expected:
            raise AssertionError(f"Frozen R2 source changed: {name}")

    r2_config = common.read_json(paths["r2_dir"] / "run_config.json")
    r2_selection = common.read_json(paths["r2_dir"] / "defense_selection_r2.json")
    r2_summary = common.read_json(paths["r2_dir"] / "final_summary.json")
    _validate_object_hash(r2_config, "config_hash", common.R2_CONFIG_HASH)
    _validate_object_hash(r2_selection, "selection_hash", common.R2_SELECTION_HASH)
    if r2_config.get("protocol_id") != common.PARENT_R2_PROTOCOL_ID:
        raise AssertionError("Unexpected R2 protocol")
    if r2_summary.get("protocol_id") != common.PARENT_R2_PROTOCOL_ID:
        raise AssertionError("R2 final-summary protocol mismatch")
    if r2_summary.get("status") != "PASS":
        raise AssertionError("R2 did not finish with PASS")
    if r2_summary.get("official_clean_stage_ready") is not True:
        raise AssertionError("R2 did not authorize the frozen clean stage")
    if r2_summary.get("official_clean_stage_blockers") != []:
        raise AssertionError("R2 has clean-stage blockers")
    if r2_summary.get("attack_stage_ready") is not False:
        raise AssertionError("R2 unexpectedly authorized attacks before clean confirmation")
    if r2_summary.get("selection_hash") != common.R2_SELECTION_HASH:
        raise AssertionError("R2 final summary does not lock the selected defense")
    for name, expected in common.EXPECTED_R2_SOURCE_HASHES.items():
        if r2_config.get("script_sha256", {}).get(name) != expected:
            raise AssertionError(f"R2 config source lock mismatch: {name}")

    strength = r2_selection.get("strength", {})
    randomization = r2_selection.get("randomization", {})
    expected_strength = common.FROZEN_STRENGTH
    observed = {
        "candidate_id": strength.get("candidate_id"),
        "gate_mode": strength.get("gate_mode"),
        "alpha_px": float(strength.get("alpha_px")),
        "epsilon_px": float(strength.get("epsilon_px")),
        "steps": int(strength.get("steps")),
        "top1_weight": float(strength.get("top1_weight")),
        "set_weight": float(strength.get("set_weight")),
    }
    if observed != expected_strength:
        raise AssertionError(f"R2 selected strength changed: {observed}")
    if abs(float(randomization.get("sigma")) - common.SIGMA) > 1e-12:
        raise AssertionError("R2 selected sigma changed")
    if r2_selection.get("selection_split") != "calibration_dev":
        raise AssertionError("R2 selection did not use calibration_dev")
    if r2_selection.get("data_governance", {}).get("attack_outcomes_read") is not False:
        raise AssertionError("R2 selection was not attack-outcome blind")

    split_rows = read_r0_csv(paths["r0_dir"] / "frozen_imagenet_val_split.csv")
    split_check = validate_frozen_split(split_rows)
    if split_check.get("split_sha256") != common.FROZEN_SPLIT_SHA256:
        raise AssertionError("Frozen ImageNet split changed")
    holdout_rows = partition_rows(split_rows, common.PARTITION)
    if len(holdout_rows) != common.EXAMPLES:
        raise AssertionError("clean_confirmation must contain exactly 45,000 examples")
    targets = np.asarray([int(row["target"]) for row in holdout_rows])
    if not np.array_equal(
        np.bincount(targets, minlength=common.CLASSES),
        np.full(common.CLASSES, common.EXAMPLES_PER_CLASS),
    ):
        raise AssertionError("clean_confirmation must contain 45 images per class")
    return holdout_rows, {
        "r2_config": r2_config,
        "r2_selection": r2_selection,
        "r2_final_summary": r2_summary,
        "split_check": split_check,
    }


def outcome_blind_shards(rows: Sequence[Mapping[str, object]]) -> list[list[dict[str, object]]]:
    def rank(row: Mapping[str, object]) -> str:
        label = f"{common.PROTOCOL_ID}|clean-confirmation-shard|{row['relative_path']}"
        import hashlib

        return hashlib.sha256(label.encode("utf-8")).hexdigest()

    ordered = [dict(row) for row in sorted(rows, key=rank)]
    shards = [
        ordered[start : start + common.EXAMPLES_PER_SHARD]
        for start in range(0, len(ordered), common.EXAMPLES_PER_SHARD)
    ]
    if len(shards) != common.SHARDS:
        raise AssertionError("Unexpected number of clean-confirmation shards")
    if any(len(shard) != common.EXAMPLES_PER_SHARD for shard in shards):
        raise AssertionError("Clean-confirmation shards are not all size 500")
    return shards


def load_models(
    standard_path: Path, robust_path: Path, device: torch.device
) -> tuple[nn.Module, nn.Module, dict[str, Any]]:
    standard, robust, audit = r2.load_models(standard_path, robust_path, device)
    if audit["robust"]["state_dict_fingerprint_sha256"] != common.ROBUST_STATE_FINGERPRINT:
        raise AssertionError("Robust state fingerprint changed")
    return standard, robust, audit


def make_loader(
    data_root: Path,
    rows: Sequence[Mapping[str, object]],
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    return r2.make_loader(data_root, rows, batch_size, num_workers, device)


def _reference_string(reference: np.ndarray) -> str:
    return "|".join(str(int(value)) for value in sorted(reference.tolist()))


def rows_from_statistics(
    shard: str,
    spec: DefenseSpec,
    labels: torch.Tensor,
    paths: Sequence[str],
    reference_k5: torch.Tensor,
    counts: Mapping[str, torch.Tensor],
    mean_logits: torch.Tensor,
) -> list[dict[str, Any]]:
    mean_metrics = r2.metric_tensors(mean_logits, labels, reference_k5)
    margin5 = r2.boundary_margin(mean_logits, reference_k5)
    margin2 = r2.boundary_margin(mean_logits, reference_k5[:, : common.SECONDARY_K])
    labels_np = labels.detach().cpu().numpy()
    reference_np = reference_k5.detach().cpu().numpy()
    count_np = {key: value.detach().cpu().numpy() for key, value in counts.items()}
    mean_np = {key: value.detach().cpu().numpy() for key, value in mean_metrics.items()}
    margin5_np = margin5.detach().cpu().numpy()
    margin2_np = margin2.detach().cpu().numpy()
    rows: list[dict[str, Any]] = []
    for position, relative_path in enumerate(paths):
        reference = reference_np[position]
        row = {
            "protocol_id": common.PROTOCOL_ID,
            "partition": common.PARTITION,
            "shard_id": shard,
            "method": spec.method,
            "relative_path": str(relative_path),
            "label": int(labels_np[position]),
            "reference_top1": int(reference[0]),
            "reference_set_k5": _reference_string(reference),
            "reference_set_k2": _reference_string(reference[: common.SECONDARY_K]),
            "gate_mode": spec.gate_mode,
            "gate_active": int(spec.gate_active),
            "draws": int(spec.draws),
            "sigma": float(spec.sigma),
            "strength_alpha_px": float(spec.strength_alpha_px),
            "strength_epsilon_px": float(spec.strength_epsilon_px),
            "strength_steps": int(spec.strength_steps),
            "top1_weight": float(spec.top1_weight),
            "set_weight": float(spec.set_weight),
        }
        for output_key, metric_key in (
            ("top1_accuracy_probability", "top1_accuracy"),
            ("reference_top1_match_probability", "reference_top1_match"),
            ("label_coverage_k5_probability", "label_coverage_k5"),
            ("exact_set_k5_probability", "exact_set_k5"),
            ("label_coverage_k2_probability", "label_coverage_k2"),
            ("exact_set_k2_probability", "exact_set_k2"),
        ):
            row[output_key] = float(count_np[metric_key][position] / int(spec.draws))
        for output_key, metric_key in (
            ("mean_logits_top1_accuracy", "top1_accuracy"),
            ("mean_logits_label_coverage_k5", "label_coverage_k5"),
            ("mean_logits_exact_set_k5", "exact_set_k5"),
            ("mean_logits_exact_set_k2", "exact_set_k2"),
        ):
            row[output_key] = float(mean_np[metric_key][position])
        row["mean_boundary_margin_k5"] = float(margin5_np[position])
        row["mean_boundary_margin_k2"] = float(margin2_np[position])
        rows.append(row)
    return rows


def deterministic_rows(
    shard: str,
    spec: DefenseSpec,
    logits: torch.Tensor,
    labels: torch.Tensor,
    paths: Sequence[str],
    reference_k5: torch.Tensor,
) -> list[dict[str, Any]]:
    return rows_from_statistics(
        shard,
        spec,
        labels,
        paths,
        reference_k5,
        r2.metric_tensors(logits, labels, reference_k5),
        logits,
    )


def frozen_strength_spec() -> r2.MethodSpec:
    strength = common.FROZEN_STRENGTH
    return r2.MethodSpec(
        method="full_sigma0",
        candidate_id=str(strength["candidate_id"]),
        gate_mode=str(strength["gate_mode"]),
        strength_alpha_px=float(strength["alpha_px"]),
        strength_epsilon_px=float(strength["epsilon_px"]),
        strength_steps=int(strength["steps"]),
        top1_weight=float(strength["top1_weight"]),
        set_weight=float(strength["set_weight"]),
    )


def defense_specs() -> dict[str, DefenseSpec]:
    strength = common.FROZEN_STRENGTH
    strength_fields = {
        "gate_mode": str(strength["gate_mode"]),
        "gate_active": True,
        "strength_alpha_px": float(strength["alpha_px"]),
        "strength_epsilon_px": float(strength["epsilon_px"]),
        "strength_steps": int(strength["steps"]),
        "top1_weight": float(strength["top1_weight"]),
        "set_weight": float(strength["set_weight"]),
    }
    return {
        "standard": DefenseSpec("standard"),
        "robust": DefenseSpec("robust"),
        "full_sigma0": DefenseSpec("full_sigma0", **strength_fields),
        "noise_only": DefenseSpec(
            "noise_only", sigma=common.SIGMA, draws=common.MC_DRAWS
        ),
        "full": DefenseSpec(
            "full", sigma=common.SIGMA, draws=common.MC_DRAWS, **strength_fields
        ),
    }


def _empty_counts(batch: int, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        metric: torch.zeros(batch, dtype=torch.float64, device=device)
        for metric in (
            "top1_accuracy",
            "reference_top1_match",
            "label_coverage_k5",
            "exact_set_k5",
            "label_coverage_k2",
            "exact_set_k2",
        )
    }


def evaluate_shard(
    shard: str,
    loader: DataLoader,
    standard: nn.Module,
    robust: nn.Module,
    device: torch.device,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    specs = defense_specs()
    strength_spec = frozen_strength_spec()
    for batch_number, (images, labels, relative_paths) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        paths = [str(value) for value in relative_paths]
        with torch.no_grad(), autocast_context(device):
            standard_logits = standard(images).float()
            robust_logits = robust(images).float()
            reference_k5 = r2.ranked_reference(standard_logits, common.PRIMARY_K).clone()
            disagreement = standard_logits.argmax(1).ne(robust_logits.argmax(1))

        full_base, full_gate = r2.selected_strength_base(
            images, reference_k5, disagreement, robust, strength_spec, False
        )
        if not bool(full_gate.all().item()):
            raise AssertionError("The frozen global gate must activate for every image")
        with torch.no_grad(), autocast_context(device):
            full0_logits = robust(full_base).float()
        output.extend(
            deterministic_rows(
                shard, specs["standard"], standard_logits, labels, paths, reference_k5
            )
        )
        output.extend(
            deterministic_rows(
                shard, specs["robust"], robust_logits, labels, paths, reference_k5
            )
        )
        output.extend(
            deterministic_rows(
                shard, specs["full_sigma0"], full0_logits, labels, paths, reference_k5
            )
        )

        counts = {
            "noise_only": _empty_counts(len(images), device),
            "full": _empty_counts(len(images), device),
        }
        logit_sums = {
            method: torch.zeros(len(images), 1000, dtype=torch.float64, device=device)
            for method in ("noise_only", "full")
        }
        generator = torch.Generator(device=device)
        generator.manual_seed(
            common.stable_seed(
                args.seed, f"clean-common-noise|{shard}|batch-{batch_number:04d}"
            )
        )
        completed = 0
        while completed < args.mc_draws:
            chunk = min(args.mc_chunk, args.mc_draws - completed)
            noise = torch.randn(
                (chunk,) + tuple(images.shape),
                dtype=images.dtype,
                device=device,
                generator=generator,
            )
            noisy = (images.unsqueeze(0) + common.SIGMA * noise).clamp(0.0, 1.0)
            full_noisy = (full_base.unsqueeze(0) + common.SIGMA * noise).clamp(0.0, 1.0)
            flattened = torch.cat((noisy.flatten(0, 1), full_noisy.flatten(0, 1)), dim=0)
            with torch.inference_mode(), autocast_context(device):
                combined_logits = robust(flattened).float()
            block = chunk * len(images)
            for position, method in enumerate(("noise_only", "full")):
                logits = combined_logits[position * block : (position + 1) * block].view(
                    chunk, len(images), 1000
                )
                metrics = r2.metric_tensors(
                    logits.flatten(0, 1),
                    labels.repeat(chunk),
                    reference_k5.repeat(chunk, 1),
                )
                for metric, values in metrics.items():
                    counts[method][metric] += values.view(chunk, len(images)).sum(0).double()
                logit_sums[method] += logits.double().sum(0)
            completed += chunk
        for method in ("noise_only", "full"):
            output.extend(
                rows_from_statistics(
                    shard,
                    specs[method],
                    labels,
                    paths,
                    reference_k5,
                    counts[method],
                    (logit_sums[method] / args.mc_draws).float(),
                )
            )
        if batch_number == 1 or batch_number % 8 == 0 or batch_number == len(loader):
            print(
                f"    {shard}: batch {batch_number}/{len(loader)}, draws={args.mc_draws}",
                flush=True,
            )
    return output


def partial_path(output_dir: Path, shard: str) -> Path:
    return output_dir / "partials" / f"{shard}.json"


def run_or_resume_shard(
    output_dir: Path,
    shard: str,
    config_hash: str,
    protocol_hash: str,
    rows: Sequence[Mapping[str, object]],
    evaluator,
) -> dict[str, Any]:
    path = partial_path(output_dir, shard)
    expected_paths = sorted(str(row["relative_path"]) for row in rows)
    if path.is_file():
        payload = common.read_json(path)
        if payload.get("protocol_id") != common.PROTOCOL_ID:
            raise AssertionError(f"Partial protocol mismatch: {path}")
        if payload.get("config_hash") != config_hash:
            raise AssertionError(f"Partial invocation mismatch: {path}")
        if payload.get("protocol_hash") != protocol_hash:
            raise AssertionError(f"Partial preregistration mismatch: {path}")
        if payload.get("r2_selection_hash") != common.R2_SELECTION_HASH:
            raise AssertionError(f"Partial R2 selection mismatch: {path}")
        if payload.get("shard_id") != shard:
            raise AssertionError(f"Partial shard mismatch: {path}")
        examples = payload.get("examples", [])
        observed_paths = sorted({str(row["relative_path"]) for row in examples})
        if observed_paths != expected_paths or len(examples) != len(expected_paths) * len(common.METHODS):
            raise AssertionError(f"Partial sample membership mismatch: {path}")
        print(f"RESUME {shard}", flush=True)
        return payload
    print(f"RUN {shard}", flush=True)
    started = time.monotonic()
    examples = evaluator()
    payload = {
        "protocol_id": common.PROTOCOL_ID,
        "protocol_hash": protocol_hash,
        "config_hash": config_hash,
        "r2_selection_hash": common.R2_SELECTION_HASH,
        "partition": common.PARTITION,
        "shard_id": shard,
        "input_examples": len(rows),
        "wall_seconds": time.monotonic() - started,
        "examples": examples,
    }
    common.atomic_json(path, payload)
    print(f"PASS {shard}: {len(examples):,} method rows", flush=True)
    return payload


def method_groups(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    groups = {
        method: [row for row in rows if str(row["method"]) == method]
        for method in common.METHODS
    }
    for method, group in groups.items():
        if len(group) != common.EXAMPLES:
            raise AssertionError(f"Expected {common.EXAMPLES:,} rows for {method}; got {len(group):,}")
    return groups


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups = method_groups(rows)
    output: list[dict[str, Any]] = []
    for method in common.METHODS:
        group = groups[method]
        first = group[0]

        def percent(key: str) -> float:
            return 100.0 * float(np.mean([float(row[key]) for row in group]))

        output.append(
            {
                "protocol_id": common.PROTOCOL_ID,
                "partition": common.PARTITION,
                "method": method,
                "examples": len(group),
                "draws": int(first["draws"]),
                "sigma": float(first["sigma"]),
                "gate_mode": first["gate_mode"],
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
                "exact_set_k5_probability_percent": percent("exact_set_k5_probability"),
                "label_coverage_k2_probability_percent": percent(
                    "label_coverage_k2_probability"
                ),
                "exact_set_k2_probability_percent": percent("exact_set_k2_probability"),
                "mean_logits_top1_accuracy_percent": percent("mean_logits_top1_accuracy"),
                "mean_logits_label_coverage_k5_percent": percent(
                    "mean_logits_label_coverage_k5"
                ),
                "mean_logits_exact_set_k5_percent": percent("mean_logits_exact_set_k5"),
                "mean_logits_exact_set_k2_percent": percent("mean_logits_exact_set_k2"),
                "mean_boundary_margin_k5": float(
                    np.mean([float(row["mean_boundary_margin_k5"]) for row in group])
                ),
                "mean_boundary_margin_k2": float(
                    np.mean([float(row["mean_boundary_margin_k2"]) for row in group])
                ),
            }
        )
    return output


def build_bootstrap_specs(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups = method_groups(rows)
    specs: list[dict[str, Any]] = []
    for left, right in common.PAIRWISE_METHODS:
        for metric, metric_key in common.METRIC_KEYS.items():
            class_delta, examples = common.class_cluster_delta(
                groups[left], groups[right], metric_key
            )
            specs.append(
                {
                    "protocol_id": common.PROTOCOL_ID,
                    "comparison_id": f"{left}_minus_{right}_{metric}",
                    "comparison": f"{left} - {right}",
                    "left": left,
                    "right": right,
                    "metric": metric,
                    "metric_key": metric_key,
                    "examples": examples,
                    "class_delta": class_delta,
                }
            )
    return specs


def reference_quality_decomposition(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    groups = {method: common.rows_by_sample(group) for method, group in method_groups(rows).items()}
    paths = sorted(groups["standard"])
    subsets = {
        "all": paths,
        "standard_correct": [
            path
            for path in paths
            if float(groups["standard"][path]["top1_accuracy_probability"]) == 1.0
        ],
        "standard_wrong": [
            path
            for path in paths
            if float(groups["standard"][path]["top1_accuracy_probability"]) != 1.0
        ],
    }
    output: list[dict[str, Any]] = []
    for subset, selected_paths in subsets.items():
        if not selected_paths:
            raise AssertionError(f"Empty reference-quality subset: {subset}")
        for method in common.METHODS:
            table = groups[method]

            def percent(key: str) -> float:
                return 100.0 * float(
                    np.mean([float(table[path][key]) for path in selected_paths])
                )

            output.append(
                {
                    "protocol_id": common.PROTOCOL_ID,
                    "partition": common.PARTITION,
                    "subset": subset,
                    "method": method,
                    "examples": len(selected_paths),
                    "top1_accuracy_percent": percent("top1_accuracy_probability"),
                    "label_coverage_k5_probability_percent": percent(
                        "label_coverage_k5_probability"
                    ),
                    "exact_set_k5_probability_percent": percent("exact_set_k5_probability"),
                    "exact_set_k2_probability_percent": percent("exact_set_k2_probability"),
                }
            )
    return output


def classwise_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for method, method_rows in method_groups(rows).items():
        by_class: dict[int, list[Mapping[str, Any]]] = {}
        for row in method_rows:
            by_class.setdefault(int(row["label"]), []).append(row)
        if sorted(by_class) != list(range(common.CLASSES)):
            raise AssertionError(f"Missing classes for {method}")
        for target in range(common.CLASSES):
            group = by_class[target]
            if len(group) != common.EXAMPLES_PER_CLASS:
                raise AssertionError(f"Class {target} row count mismatch for {method}")

            def percent(key: str) -> float:
                return 100.0 * float(np.mean([float(row[key]) for row in group]))

            output.append(
                {
                    "protocol_id": common.PROTOCOL_ID,
                    "partition": common.PARTITION,
                    "method": method,
                    "target": target,
                    "examples": len(group),
                    "top1_accuracy_percent": percent("top1_accuracy_probability"),
                    "label_coverage_k5_probability_percent": percent(
                        "label_coverage_k5_probability"
                    ),
                    "exact_set_k5_probability_percent": percent("exact_set_k5_probability"),
                    "exact_set_k2_probability_percent": percent("exact_set_k2_probability"),
                }
            )
    return output


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = configure_device(args)
    seed_everything(args.seed)
    paths = {
        "data_root": resolved(args.data_root),
        "r0_dir": resolved(args.r0_dir),
        "r1_dir": resolved(args.r1_dir),
        "r2_dir": resolved(args.r2_dir),
        "standard_checkpoint": resolved(args.standard_checkpoint),
        "robust_checkpoint": resolved(args.robust_checkpoint),
        "output_dir": resolved(args.output_dir),
    }
    paths["output_dir"].mkdir(parents=True, exist_ok=True)

    protocol = common.frozen_protocol_payload()
    _write_or_validate_json(paths["output_dir"] / "frozen_protocol_r3.json", protocol)
    protocol_hash = str(protocol["protocol_hash"])
    print(f"PROTOCOL_ID={common.PROTOCOL_ID}")
    print(f"PROTOCOL_HASH={protocol_hash}")
    print(f"sys.executable={sys.executable}")
    print(
        "scope: clean_confirmation only; frozen R2 defense; no selection, training, "
        "attack, download, or attack-split read",
        flush=True,
    )

    holdout_rows, parents = validate_parents(paths)
    shards = outcome_blind_shards(holdout_rows)
    manifest_rows: list[dict[str, Any]] = []
    for shard_number, shard_rows in enumerate(shards):
        for row in shard_rows:
            manifest_rows.append({**dict(row), "shard_id": common.shard_id(shard_number)})
    _, preprocessing = r2.r1.raw_preprocessing()
    manifest = {
        "protocol_id": common.PROTOCOL_ID,
        "protocol_hash": protocol_hash,
        "partition": common.PARTITION,
        "examples": common.EXAMPLES,
        "classes": common.CLASSES,
        "examples_per_class": common.EXAMPLES_PER_CLASS,
        "shards": common.SHARDS,
        "examples_per_shard": common.EXAMPLES_PER_SHARD,
        "assignment": "SHA256 rank of relative_path before inference, then contiguous chunks",
        "rows_sha256": common.object_sha256(manifest_rows),
        "parent_split_sha256": common.FROZEN_SPLIT_SHA256,
        "partitions_read": [common.PARTITION],
        "partitions_not_read": [
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

    source_paths = {name: PROJECT_DIR / name for name in common.NEW_SCRIPT_NAMES}
    source_paths.update(
        {name: PROJECT_DIR / name for name in common.EXPECTED_R2_SOURCE_HASHES}
    )
    for path in source_paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Required source is missing: {path}")
    config = {
        "protocol_id": common.PROTOCOL_ID,
        "protocol_hash": protocol_hash,
        "parent_protocols": {
            "r0": common.PARENT_R0_PROTOCOL_ID,
            "r1": common.PARENT_R1_PROTOCOL_ID,
            "r2": common.PARENT_R2_PROTOCOL_ID,
        },
        "dataset": "ImageNet-1k validation clean_confirmation",
        "architecture": "torchvision ResNet-50",
        "primary_k": common.PRIMARY_K,
        "secondary_k": common.SECONDARY_K,
        "paths": {key: str(value) for key, value in paths.items()},
        "checkpoints": {
            "standard": {
                "path": str(paths["standard_checkpoint"]),
                "sha256": common.STANDARD_CHECKPOINT_SHA256,
            },
            "robust": {
                "path": str(paths["robust_checkpoint"]),
                "sha256": common.ROBUST_CHECKPOINT_SHA256,
                "state_dict_fingerprint_sha256": common.ROBUST_STATE_FINGERPRINT,
                "model_id": "Salman2020Do_R50",
                "training_epsilon_linf_px": 4.0,
            },
        },
        "parent_artifacts": {
            "frozen_split_sha256": common.FROZEN_SPLIT_SHA256,
            "r2_config_hash": common.R2_CONFIG_HASH,
            "r2_selection_hash": common.R2_SELECTION_HASH,
            "r2_artifact_sha256": dict(common.EXPECTED_R2_ARTIFACT_HASHES),
        },
        "partition": common.PARTITION,
        "examples": common.EXAMPLES,
        "classes": common.CLASSES,
        "examples_per_class": common.EXAMPLES_PER_CLASS,
        "shards": common.SHARDS,
        "batch_size": args.batch_size,
        "mc_chunk": args.mc_chunk,
        "num_workers": args.num_workers,
        "gpu_memory_fraction": args.gpu_memory_fraction,
        "amp": False,
        "mc_draws": args.mc_draws,
        "evaluation_seed": args.seed,
        "bootstrap": {
            "unit": "ImageNet class cluster",
            "resamples": args.bootstrap_resamples,
            "seed": args.bootstrap_seed,
            "interval": "paired percentile 95%",
        },
        "frozen_strength": dict(common.FROZEN_STRENGTH),
        "sigma": common.SIGMA,
        "noninferiority_margin_pp": common.NONINFERIORITY_MARGIN_PP,
        "gate_definitions": [
            {"gate_id": gate, "comparison_id": comparison, "kind": kind}
            for gate, comparison, kind in common.GATE_DEFINITIONS
        ],
        "common_random_numbers": True,
        "selection_or_retuning_allowed": False,
        "offline": True,
        "downloads_allowed": False,
        "training_allowed": False,
        "attack_allowed": False,
        "attack_outcomes_read": False,
        "partitions_read": [common.PARTITION],
        "forbidden_partitions_not_read": ["attack_pilot", "attack_confirmatory"],
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
        },
        "script_sha256": {name: common.file_sha256(path) for name, path in source_paths.items()},
    }
    config_hash = common.object_sha256(config)
    config["config_hash"] = config_hash
    _write_or_validate_json(paths["output_dir"] / "run_config.json", config)

    standard, robust, model_audit = load_models(
        paths["standard_checkpoint"], paths["robust_checkpoint"], device
    )
    _write_or_validate_json(paths["output_dir"] / "model_load_audit.json", model_audit)

    payloads: list[dict[str, Any]] = []
    for shard_number, shard_rows in enumerate(shards):
        shard = common.shard_id(shard_number)
        loader = make_loader(
            paths["data_root"], shard_rows, args.batch_size, args.num_workers, device
        )
        payloads.append(
            run_or_resume_shard(
                paths["output_dir"],
                shard,
                config_hash,
                protocol_hash,
                shard_rows,
                lambda loader=loader, shard=shard: evaluate_shard(
                    shard, loader, standard, robust, device, args
                ),
            )
        )

    all_rows = [dict(row) for payload in payloads for row in payload["examples"]]
    if len(all_rows) != common.EXAMPLES * len(common.METHODS):
        raise AssertionError("Combined clean-confirmation row count is incomplete")
    summaries = summarize_rows(all_rows)
    print("Computing preregistered paired class-cluster bootstrap", flush=True)
    bootstrap = common.paired_class_bootstrap_many(
        build_bootstrap_specs(all_rows), args.bootstrap_resamples, args.bootstrap_seed
    )
    gates = common.apply_preregistered_gates(bootstrap)
    decomposition = reference_quality_decomposition(all_rows)
    classwise = classwise_summary(all_rows)

    common.atomic_csv(paths["output_dir"] / "clean_confirmation_summary.csv", summaries)
    common.atomic_csv(paths["output_dir"] / "paired_class_bootstrap.csv", bootstrap)
    common.atomic_csv(paths["output_dir"] / "preregistered_clean_gates.csv", gates)
    common.atomic_csv(
        paths["output_dir"] / "reference_quality_decomposition.csv", decomposition
    )
    common.atomic_csv(paths["output_dir"] / "classwise_summary.csv", classwise)
    per_example_path = paths["output_dir"] / "clean_confirmation_per_example.csv.gz"
    common.atomic_gzip_csv(per_example_path, all_rows)

    clean_pass = all(bool(row["passed"]) for row in gates)
    failed_gates = [str(row["gate_id"]) for row in gates if not bool(row["passed"])]
    table = {str(row["method"]): row for row in summaries}
    final_summary = {
        "protocol_id": common.PROTOCOL_ID,
        "protocol_hash": protocol_hash,
        "config_hash": config_hash,
        "r2_selection_hash": common.R2_SELECTION_HASH,
        "status": "PASS",
        "frozen_operating_point": {
            "strength": dict(common.FROZEN_STRENGTH),
            "sigma": common.SIGMA,
            "mc_draws": common.MC_DRAWS,
        },
        "clean_confirmation_summaries": summaries,
        "preregistered_clean_gates": gates,
        "official_clean_confirmation": "PASS" if clean_pass else "FAIL",
        "official_clean_stage_passed": clean_pass,
        "failed_gates": failed_gates,
        "attack_stage_ready": clean_pass,
        "clean_paper_claim_authorized": clean_pass,
        "adversarial_paper_claim_authorized": False,
        "next_stage": (
            "attack_family_pilot_on_attack_pilot_partition"
            if clean_pass
            else "stop_and_report_without_holdout_retuning"
        ),
        "monte_carlo_precision": {
            "aggregate_probability_standard_error_upper_bound_pp": (
                100.0 * np.sqrt(0.25 / (common.EXAMPLES * common.MC_DRAWS))
            ),
            "note": "upper bound only; paired class-cluster CIs are the inferential intervals",
        },
        "artifacts": {
            "per_example_file": per_example_path.name,
            "per_example_rows": len(all_rows),
            "per_example_file_sha256": common.file_sha256(per_example_path),
            "partial_files": len(payloads),
            "partial_wall_seconds_sum": float(
                sum(float(payload["wall_seconds"]) for payload in payloads)
            ),
        },
        "data_governance": {
            "partitions_read": [common.PARTITION],
            "partitions_not_read": [
                "preflight_sanity",
                "calibration_dev",
                "attack_pilot",
                "attack_confirmatory",
            ],
            "no_selection_or_retuning": True,
            "no_training": True,
            "no_download": True,
            "no_attack": True,
        },
    }
    common.atomic_json(paths["output_dir"] / "final_summary.json", final_summary)

    print("\nImageNet R3 frozen clean-confirmation results", flush=True)
    for method in common.METHODS:
        row = table[method]
        print(
            f"  {method:>11}: Top1={float(row['top1_accuracy_percent']):.3f}% "
            f"Top5={float(row['label_coverage_k5_probability_percent']):.3f}% "
            f"Exact5={float(row['exact_set_k5_probability_percent']):.3f}% "
            f"Exact2={float(row['exact_set_k2_probability_percent']):.3f}%",
            flush=True,
        )
    print("Preregistered clean gates", flush=True)
    for row in gates:
        print(
            f"  {row['gate_id']}: {float(row['difference_pp']):+.3f} pp "
            f"[{float(row['ci95_low_pp']):+.3f}, {float(row['ci95_high_pp']):+.3f}] "
            f"-> {row['status']}",
            flush=True,
        )
    print("OFFICIAL_CLEAN_CONFIRMATION=" + ("PASS" if clean_pass else "FAIL"))
    print("ATTACK_STAGE_READY=" + ("YES" if clean_pass else "NO"))
    print("CLEAN_PAPER_CLAIM_AUTHORIZED=" + ("YES" if clean_pass else "NO"))
    print("ADVERSARIAL_PAPER_CLAIM_AUTHORIZED=NO (no ImageNet attack yet)")
    print(f"Complete: {paths['output_dir']}")


if __name__ == "__main__":
    main()
