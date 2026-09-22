#!/usr/bin/env python3
"""Outcome-blind ImageNet calibration with a ViT-B/16 reference.

VR1 reads exactly the 2,000 images assigned to ``calibration_dev`` by the
pre-model R0 split.  The local standard ViT-B/16 defines the reference Top-5,
while the published robust ResNet-50 remains the defended backbone.  The stage
selects deterministic input strengthening and Gaussian input noise under paired
Top-1/Top-5/Exact-Top5 non-inferiority constraints.  It never reads an attack
result or any pilot, confirmatory, or clean-confirmation image.

This protocol tests cross-architecture reference transfer.  It must not be
described as a robust-ViT evaluation because the ViT is the standard reference,
not the robust backbone.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

import preflight_topk_imagenet_r0 as r0
import prepare_topk_imagenet_robust_anchor_r1 as r1
import topk_imagenet_vit_reference_calibration_vr1_common as common
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
DEFAULT_STANDARD_CHECKPOINT = Path(
    "./models/vit/imagenet/vit_b16_imagenet.pth"
)
DEFAULT_ROBUST_CHECKPOINT = Path(
    "./models/resnet50/imagenet/"
    "salman2020do_r50_linf4.pth"
)
DEFAULT_OUTPUT_DIR = Path(
    "./models/imagenet_topk/"
    "paper_topk_imagenet_vit_reference_crossarch_calibration_dev_vr1"
)


@dataclass(frozen=True)
class MethodSpec:
    method: str
    candidate_id: str
    gate_mode: str = "none"
    strength_alpha_px: float = 0.0
    strength_epsilon_px: float = 0.0
    strength_steps: int = 0
    top1_weight: float = 0.0
    set_weight: float = 0.0
    sigma: float = 0.0
    draws: int = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ImageNet Top-k cross-architecture ViT-reference calibration VR1"
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--r0-dir", type=Path, default=DEFAULT_R0_DIR)
    parser.add_argument("--r1-dir", type=Path, default=DEFAULT_R1_DIR)
    parser.add_argument("--standard-checkpoint", type=Path, default=DEFAULT_STANDARD_CHECKPOINT)
    parser.add_argument("--robust-checkpoint", type=Path, default=DEFAULT_ROBUST_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--mc-chunk", type=int, default=4)
    parser.add_argument("--sigma-forward-group", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.85)
    parser.add_argument("--sigma-draws", type=int, default=50)
    parser.add_argument("--max-clean-drop-pp", type=float, default=2.0)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026091601)
    parser.add_argument("--bootstrap-seed-strength", type=int, default=2026091602)
    parser.add_argument("--bootstrap-seed-sigma", type=int, default=2026091603)
    parser.add_argument("--bootstrap-seed-selected", type=int, default=2026091604)
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if min(args.batch_size, args.mc_chunk, args.sigma_forward_group, args.sigma_draws) <= 0:
        raise ValueError("Batch, MC, sigma grouping, and draw counts must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if not 0.0 < args.gpu_memory_fraction <= 1.0:
        raise ValueError("--gpu-memory-fraction must be in (0,1]")
    if args.sigma_draws != 50:
        raise ValueError("The frozen VR1 calibration protocol requires exactly 50 MC draws")
    if abs(args.max_clean_drop_pp - 2.0) > 1e-12:
        raise ValueError("The frozen VR1 non-inferiority margin is exactly 2 pp")
    if args.bootstrap_resamples < 10000:
        raise ValueError("VR1 requires at least 10,000 paired bootstrap resamples")


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


def autocast_context(device: torch.device, enabled: bool):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", enabled=bool(enabled))
    return contextlib.nullcontext()


def stable_seed(base_seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{int(base_seed)}|{label}".encode("utf-8")).hexdigest()
    return int(digest[:15], 16) % (2**63 - 1)


def _write_or_validate_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.is_file():
        if common.read_json(path) != payload:
            raise AssertionError(f"Frozen artifact differs from this invocation: {path}")
    else:
        common.atomic_json(path, payload)


def validate_parents(paths: Mapping[str, Path]) -> tuple[list[dict[str, object]], dict[str, Any]]:
    required = (
        paths["r0_dir"] / "frozen_imagenet_val_split.csv",
        paths["r0_dir"] / "frozen_imagenet_val_split_manifest.json",
        paths["r0_dir"] / "run_manifest.json",
        paths["r0_dir"] / "preflight_report.json",
        paths["r1_dir"] / "run_manifest.json",
        paths["r1_dir"] / "robust_anchor_report.json",
        paths["r1_dir"] / "calibration_handoff_r1.json",
        paths["standard_checkpoint"],
        paths["robust_checkpoint"],
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Required frozen parent artifact is missing: {path}")

    if common.file_sha256(paths["standard_checkpoint"]) != common.STANDARD_CHECKPOINT_SHA256:
        raise AssertionError("Standard ImageNet checkpoint SHA256 mismatch")
    if common.file_sha256(paths["robust_checkpoint"]) != common.ROBUST_CHECKPOINT_SHA256:
        raise AssertionError("Robust ImageNet checkpoint SHA256 mismatch")
    handoff_path = paths["r1_dir"] / "calibration_handoff_r1.json"
    report_path = paths["r1_dir"] / "robust_anchor_report.json"
    if common.file_sha256(handoff_path) != common.R1_HANDOFF_SHA256:
        raise AssertionError("R1 calibration handoff changed")
    if common.file_sha256(report_path) != common.R1_REPORT_SHA256:
        raise AssertionError("R1 robust-anchor report changed")

    r0_manifest = common.read_json(paths["r0_dir"] / "run_manifest.json")
    r0_report = common.read_json(paths["r0_dir"] / "preflight_report.json")
    r1_manifest = common.read_json(paths["r1_dir"] / "run_manifest.json")
    report = common.read_json(report_path)
    handoff = common.read_json(handoff_path)
    if r0_manifest.get("protocol_id") != common.PARENT_R0_PROTOCOL_ID:
        raise AssertionError("Unexpected R0 protocol")
    if r1_manifest.get("protocol_id") != common.PARENT_R1_PROTOCOL_ID:
        raise AssertionError("Unexpected R1 protocol")
    if report.get("protocol_id") != common.PARENT_R1_PROTOCOL_ID:
        raise AssertionError("R1 report protocol mismatch")
    if report.get("status") != "PASS" or report.get("defense_calibration_ready") is not True:
        raise AssertionError("R1 did not authorize defense calibration")
    if handoff.get("status") != "READY" or handoff.get("blockers") != []:
        raise AssertionError("R1 handoff is not unconditionally ready")
    if handoff.get("calibration_partition") != "calibration_dev":
        raise AssertionError("R1 handoff does not authorize calibration_dev")
    if handoff.get("frozen_split_sha256") != common.FROZEN_SPLIT_SHA256:
        raise AssertionError("R1 handoff split hash mismatch")
    if handoff.get("do_not_transfer_cifar_hyperparameters") is not True:
        raise AssertionError("R1 hyperparameter-transfer guard is missing")
    if r1_manifest.get("outcomes_read_from_partitions") != ["preflight_sanity"]:
        raise AssertionError("R1 read an unexpected ImageNet partition")

    vit_entries = [
        item
        for item in r0_report.get("checkpoints", [])
        if item.get("checkpoint_id") == "vit_b16_local"
    ]
    if len(vit_entries) != 1:
        raise AssertionError("R0 must contain exactly one vit_b16_local preflight entry")
    vit_entry = vit_entries[0]
    if vit_entry.get("sha256") != common.STANDARD_CHECKPOINT_SHA256:
        raise AssertionError("R0 ViT checkpoint hash differs from the frozen VR1 reference")
    architecture = vit_entry.get("architecture", {})
    if architecture.get("architecture_id") != "torchvision_vit_b_16":
        raise AssertionError("R0 did not detect the reference as torchvision ViT-B/16")
    if architecture.get("family") != "vit_b16":
        raise AssertionError("R0 ViT family declaration changed")
    match = vit_entry.get("state_dict_match", {})
    if match.get("load_ready") is not True:
        raise AssertionError("R0 did not authorize loading the ViT checkpoint")
    if float(match.get("parameter_numel_coverage", 0.0)) != 1.0:
        raise AssertionError("R0 ViT parameter coverage is not exactly 100%")
    if match.get("missing_keys") or match.get("shape_mismatches"):
        raise AssertionError("R0 ViT load audit contains missing or mismatched tensors")
    if vit_entry.get("sanity", {}).get("sanity_gate") != "PASS":
        raise AssertionError("R0 ViT preflight sanity did not pass")

    for filename, expected in common.EXPECTED_PARENT_SOURCE_HASHES.items():
        source = PROJECT_DIR / filename
        if not source.is_file() or common.file_sha256(source) != expected:
            raise AssertionError(f"Frozen parent source changed or is missing: {filename}")
        if r1_manifest.get("script_sha256", {}).get(filename) != expected:
            raise AssertionError(f"R1 did not record the expected source hash: {filename}")

    split_rows = read_r0_csv(paths["r0_dir"] / "frozen_imagenet_val_split.csv")
    split_check = validate_frozen_split(split_rows)
    if split_check.get("split_sha256") != common.FROZEN_SPLIT_SHA256:
        raise AssertionError("Frozen ImageNet row split changed")
    calibration_rows = partition_rows(split_rows, "calibration_dev")
    if len(calibration_rows) != common.CALIBRATION_EXAMPLES:
        raise AssertionError("calibration_dev must contain exactly 2,000 examples")
    targets = np.asarray([int(row["target"]) for row in calibration_rows])
    if not np.array_equal(np.bincount(targets, minlength=1000), np.full(1000, 2)):
        raise AssertionError("calibration_dev must contain exactly two images per class")
    forbidden = {"attack_pilot", "attack_confirmatory", "clean_confirmation"}
    if set(handoff.get("forbidden_until_later", [])) != forbidden:
        raise AssertionError("R1 forbidden-partition contract changed")
    return calibration_rows, {
        "r0_manifest": r0_manifest,
        "r0_vit_preflight": vit_entry,
        "r1_manifest": r1_manifest,
        "r1_report": report,
        "r1_handoff": handoff,
        "split_check": split_check,
    }


def outcome_blind_shards(rows: Sequence[Mapping[str, object]]) -> list[list[dict[str, object]]]:
    def rank(row: Mapping[str, object]) -> str:
        label = f"{common.PROTOCOL_ID}|calibration-shard|{row['relative_path']}"
        return hashlib.sha256(label.encode("utf-8")).hexdigest()

    ordered = [dict(row) for row in sorted(rows, key=rank)]
    shards = [
        ordered[start : start + common.EXAMPLES_PER_SHARD]
        for start in range(0, len(ordered), common.EXAMPLES_PER_SHARD)
    ]
    if len(shards) != common.CALIBRATION_SHARDS:
        raise AssertionError("Unexpected number of calibration shards")
    if any(len(shard) != common.EXAMPLES_PER_SHARD for shard in shards):
        raise AssertionError("Calibration shards are not all size 100")
    return shards


def load_models(
    standard_path: Path, robust_path: Path, device: torch.device
) -> tuple[nn.Module, nn.Module, dict[str, Any]]:
    payload, load_mode = r0.torch_load_local(standard_path, allow_unsafe=False)
    source_state, state_route = r0.extract_state_dict(payload)
    standard_bare, standard_architecture, standard_match = r0.select_architecture(
        source_state
    )
    if standard_architecture.get("architecture_id") != "torchvision_vit_b_16":
        raise AssertionError(
            "Frozen standard reference is not detected as torchvision ViT-B/16"
        )
    if standard_architecture.get("family") != "vit_b16":
        raise AssertionError("Frozen standard reference family is not ViT-B/16")
    if float(standard_match.get("parameter_numel_coverage", 0.0)) != 1.0:
        raise AssertionError("ViT reference does not have 100% parameter coverage")
    if float(standard_match.get("numel_coverage", 0.0)) != 1.0:
        raise AssertionError("ViT reference does not have 100% state coverage")
    if standard_match.get("missing_keys") or standard_match.get("shape_mismatches"):
        raise AssertionError("ViT reference contains missing or shape-mismatched tensors")
    standard_audit = {
        "role": "standard_reference_top5_only",
        "adversarially_robust_checkpoint": False,
        "load_mode": load_mode,
        "state_dict_route": state_route,
        "source_tensor_count": len(source_state),
        "architecture": standard_architecture,
        "match": standard_match,
        "normalized_tensor_count": len(standard_bare.state_dict()),
        "state_dict_fingerprint_sha256": r1.state_dict_fingerprint(
            standard_bare.state_dict()
        ),
    }
    robust_bare, robust_audit = r1.load_resnet50_checkpoint(robust_path)
    robust_audit.pop("normalized_state", None)
    robust_audit["role"] = "robust_defended_backbone"
    robust_audit["adversarially_robust_checkpoint"] = True
    if robust_audit["state_dict_fingerprint_sha256"] != common.ROBUST_STATE_FINGERPRINT:
        raise AssertionError("Robust checkpoint tensor fingerprint mismatch")
    standard = r1.NormalizeThenModel(standard_bare).to(device).eval()
    robust = r1.NormalizeThenModel(robust_bare).to(device).eval()
    for model in (standard, robust):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    return standard, robust, {"standard": standard_audit, "robust": robust_audit}


def raw_cross_architecture_preprocessing() -> tuple[Any, dict[str, Any]]:
    """Return a shared raw-pixel view using the ViT architecture default."""

    transform = transforms.Compose(
        [
            transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )
    return transform, {
        "selection_rule": "ViT-B/16 architecture default; fixed before calibration",
        "resize_shorter_side": 256,
        "crop_size": 224,
        "interpolation": "bicubic",
        "range": [0.0, 1.0],
        "normalization_inside_model": True,
        "shared_raw_tensor_for_reference_and_backbone": True,
    }


def make_loader(
    data_root: Path,
    rows: Sequence[Mapping[str, object]],
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    transform, _ = raw_cross_architecture_preprocessing()
    dataset = r1.RawManifestDataset(data_root, rows, transform)
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=device.type == "cuda",
        persistent_workers=int(num_workers) > 0,
    )


def ranked_reference(logits: torch.Tensor, k: int) -> torch.Tensor:
    return logits.topk(int(k), dim=1, largest=True, sorted=True).indices


def _sorted_set(logits: torch.Tensor, k: int) -> torch.Tensor:
    return logits.topk(int(k), dim=1, largest=True, sorted=False).indices.sort(1).values


def metric_tensors(
    logits: torch.Tensor,
    labels: torch.Tensor,
    reference_k5: torch.Tensor,
) -> dict[str, torch.Tensor]:
    reference_k2 = reference_k5[:, : common.SECONDARY_K]
    predicted_k5 = _sorted_set(logits, common.PRIMARY_K)
    predicted_k2 = _sorted_set(logits, common.SECONDARY_K)
    reference_set_k5 = reference_k5.sort(1).values
    reference_set_k2 = reference_k2.sort(1).values
    prediction = logits.argmax(1)
    return {
        "top1_accuracy": prediction.eq(labels).float(),
        "reference_top1_match": prediction.eq(reference_k5[:, 0]).float(),
        "label_coverage_k5": predicted_k5.eq(labels[:, None]).any(1).float(),
        "exact_set_k5": predicted_k5.eq(reference_set_k5).all(1).float(),
        "label_coverage_k2": predicted_k2.eq(labels[:, None]).any(1).float(),
        "exact_set_k2": predicted_k2.eq(reference_set_k2).all(1).float(),
    }


def boundary_margin(logits: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    selected_min = logits.gather(1, reference).min(1).values
    outside_mask = torch.ones_like(logits, dtype=torch.bool)
    outside_mask.scatter_(1, reference, False)
    outside_max = logits.masked_fill(~outside_mask, -torch.inf).max(1).values
    return selected_min - outside_max


def strengthening_objective(
    logits: torch.Tensor,
    reference_k5: torch.Tensor,
    top1_weight: float,
    set_weight: float,
) -> torch.Tensor:
    reference_top1 = reference_k5[:, :1]
    top1_selected = logits.gather(1, reference_top1).squeeze(1)
    top1_outside = torch.ones_like(logits, dtype=torch.bool)
    top1_outside.scatter_(1, reference_top1, False)
    top1_margin = top1_selected - logits.masked_fill(~top1_outside, -torch.inf).max(1).values
    set_margin = boundary_margin(logits, reference_k5)
    return float(top1_weight) * top1_margin + float(set_weight) * set_margin


def strengthen_trajectory(
    clean: torch.Tensor,
    reference_k5: torch.Tensor,
    robust: nn.Module,
    alpha_px: float,
    epsilon_px: float,
    capture_steps: Sequence[int],
    top1_weight: float,
    set_weight: float,
    amp: bool,
) -> dict[int, torch.Tensor]:
    wanted = {int(step) for step in capture_steps}
    if not wanted or min(wanted) <= 0:
        raise ValueError("capture_steps must contain positive integers")
    lower = (clean - float(epsilon_px) / 255.0).clamp(0.0, 1.0)
    upper = (clean + float(epsilon_px) / 255.0).clamp(0.0, 1.0)
    current = clean.detach().clone()
    captured: dict[int, torch.Tensor] = {}
    for step in range(1, max(wanted) + 1):
        current = current.detach().requires_grad_(True)
        with torch.enable_grad(), autocast_context(clean.device, amp):
            logits = robust(current).float()
            objective = strengthening_objective(
                logits, reference_k5, top1_weight, set_weight
            ).sum()
        gradient = torch.autograd.grad(objective, current, only_inputs=True)[0]
        updated = current.detach() + float(alpha_px) / 255.0 * gradient.sign()
        current = torch.maximum(torch.minimum(updated, upper), lower).clamp(0.0, 1.0)
        if step in wanted:
            captured[step] = current.detach().clone()
    return captured


def gated_base(
    clean: torch.Tensor,
    strengthened: torch.Tensor,
    disagreement: torch.Tensor,
    gate_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if gate_mode == "global":
        active = torch.ones(len(clean), dtype=torch.bool, device=clean.device)
    elif gate_mode == "disagreement":
        active = disagreement
    elif gate_mode == "none":
        active = torch.zeros(len(clean), dtype=torch.bool, device=clean.device)
    else:
        raise ValueError(f"Unknown gate mode: {gate_mode}")
    return torch.where(active[:, None, None, None], strengthened, clean), active


def _reference_string(reference: np.ndarray) -> str:
    return "|".join(str(int(value)) for value in sorted(reference.tolist()))


def rows_from_statistics(
    phase: str,
    shard: str,
    spec: MethodSpec,
    labels: torch.Tensor,
    paths: Sequence[str],
    reference_k5: torch.Tensor,
    gate_active: torch.Tensor,
    counts: Mapping[str, torch.Tensor],
    mean_logits: torch.Tensor,
) -> list[dict[str, Any]]:
    mean_metrics = metric_tensors(mean_logits, labels, reference_k5)
    margin5 = boundary_margin(mean_logits, reference_k5)
    margin2 = boundary_margin(mean_logits, reference_k5[:, : common.SECONDARY_K])
    labels_np = labels.detach().cpu().numpy()
    reference_np = reference_k5.detach().cpu().numpy()
    gate_np = gate_active.detach().cpu().numpy()
    count_np = {key: value.detach().cpu().numpy() for key, value in counts.items()}
    mean_np = {key: value.detach().cpu().numpy() for key, value in mean_metrics.items()}
    margin5_np = margin5.detach().cpu().numpy()
    margin2_np = margin2.detach().cpu().numpy()
    rows: list[dict[str, Any]] = []
    for position, relative_path in enumerate(paths):
        reference = reference_np[position]
        row = {
            "protocol_id": common.PROTOCOL_ID,
            "phase": phase,
            "shard_id": shard,
            "method": spec.method,
            "candidate_id": spec.candidate_id,
            "relative_path": str(relative_path),
            "label": int(labels_np[position]),
            "reference_top1": int(reference[0]),
            "reference_set_k5": _reference_string(reference),
            "reference_set_k2": _reference_string(reference[: common.SECONDARY_K]),
            "gate_mode": spec.gate_mode,
            "gate_active": int(bool(gate_np[position])),
            "draws": int(spec.draws),
            "sigma": float(spec.sigma),
            "strength_alpha_px": float(spec.strength_alpha_px),
            "strength_epsilon_px": float(spec.strength_epsilon_px),
            "strength_steps": int(spec.strength_steps),
            "top1_weight": float(spec.top1_weight),
            "set_weight": float(spec.set_weight),
        }
        for short, key in (
            ("top1_accuracy_probability", "top1_accuracy"),
            ("reference_top1_match_probability", "reference_top1_match"),
            ("label_coverage_k5_probability", "label_coverage_k5"),
            ("exact_set_k5_probability", "exact_set_k5"),
            ("label_coverage_k2_probability", "label_coverage_k2"),
            ("exact_set_k2_probability", "exact_set_k2"),
        ):
            row[short] = float(count_np[key][position] / int(spec.draws))
        for short, key in (
            ("mean_logits_top1_accuracy", "top1_accuracy"),
            ("mean_logits_label_coverage_k5", "label_coverage_k5"),
            ("mean_logits_exact_set_k5", "exact_set_k5"),
            ("mean_logits_exact_set_k2", "exact_set_k2"),
        ):
            row[short] = float(mean_np[key][position])
        row["mean_boundary_margin_k5"] = float(margin5_np[position])
        row["mean_boundary_margin_k2"] = float(margin2_np[position])
        rows.append(row)
    return rows


def deterministic_rows(
    phase: str,
    shard: str,
    spec: MethodSpec,
    logits: torch.Tensor,
    labels: torch.Tensor,
    paths: Sequence[str],
    reference_k5: torch.Tensor,
    gate_active: torch.Tensor,
) -> list[dict[str, Any]]:
    metrics = metric_tensors(logits, labels, reference_k5)
    return rows_from_statistics(
        phase, shard, spec, labels, paths, reference_k5, gate_active, metrics, logits
    )


def _candidate_spec(gate_mode: str, candidate: Mapping[str, Any]) -> MethodSpec:
    return MethodSpec(
        method="full_sigma0",
        candidate_id=common.strength_candidate_id(gate_mode, candidate),
        gate_mode=gate_mode,
        strength_alpha_px=float(candidate["alpha_px"]),
        strength_epsilon_px=float(candidate["epsilon_px"]),
        strength_steps=int(candidate["steps"]),
        top1_weight=float(candidate["top1_weight"]),
        set_weight=float(candidate["set_weight"]),
    )


def evaluate_strength_shard(
    shard: str,
    loader: DataLoader,
    standard: nn.Module,
    robust: nn.Module,
    device: torch.device,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    capture_steps = sorted({int(row["steps"]) for row in common.STRENGTH_GRID})
    trajectory_groups = sorted(
        {
            (
                float(row["alpha_px"]),
                float(row["epsilon_px"]),
                float(row["top1_weight"]),
                float(row["set_weight"]),
            )
            for row in common.STRENGTH_GRID
        }
    )
    for batch_number, (images, labels, relative_paths) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        paths = [str(value) for value in relative_paths]
        with torch.no_grad(), autocast_context(device, args.amp):
            standard_logits = standard(images).float()
            robust_logits = robust(images).float()
            reference_k5 = ranked_reference(standard_logits, common.PRIMARY_K).clone()
            disagreement = standard_logits.argmax(1).ne(robust_logits.argmax(1))
        inactive = torch.zeros(len(images), dtype=torch.bool, device=device)
        output.extend(
            deterministic_rows(
                "strength", shard, MethodSpec("standard", "standard"), standard_logits,
                labels, paths, reference_k5, inactive
            )
        )
        output.extend(
            deterministic_rows(
                "strength", shard, MethodSpec("robust", "identity"), robust_logits,
                labels, paths, reference_k5, inactive
            )
        )

        for alpha_px, epsilon_px, top1_weight, set_weight in trajectory_groups:
            trajectories = strengthen_trajectory(
                images,
                reference_k5,
                robust,
                alpha_px,
                epsilon_px,
                capture_steps,
                top1_weight,
                set_weight,
                args.amp,
            )
            specs: list[MethodSpec] = []
            bases: list[torch.Tensor] = []
            gates: list[torch.Tensor] = []
            for steps in capture_steps:
                candidate = {
                    "alpha_px": alpha_px,
                    "epsilon_px": epsilon_px,
                    "steps": steps,
                    "top1_weight": top1_weight,
                    "set_weight": set_weight,
                }
                for gate_mode in common.GATE_MODES:
                    spec = _candidate_spec(gate_mode, candidate)
                    base, active = gated_base(
                        images, trajectories[steps], disagreement, gate_mode
                    )
                    specs.append(spec)
                    bases.append(base)
                    gates.append(active)
            with torch.inference_mode(), autocast_context(device, args.amp):
                combined_logits = robust(torch.cat(bases, dim=0)).float()
            for position, spec in enumerate(specs):
                start = position * len(images)
                logits = combined_logits[start : start + len(images)]
                output.extend(
                    deterministic_rows(
                        "strength", shard, spec, logits, labels, paths,
                        reference_k5, gates[position]
                    )
                )
        print(f"    strength/{shard}: batch {batch_number}/{len(loader)}", flush=True)
    return output


def _selected_strength_spec(selection: Mapping[str, Any]) -> MethodSpec:
    return MethodSpec(
        method="full_sigma0",
        candidate_id=str(selection["candidate_id"]),
        gate_mode=str(selection["gate_mode"]),
        strength_alpha_px=float(selection["strength_alpha_px"]),
        strength_epsilon_px=float(selection["strength_epsilon_px"]),
        strength_steps=int(selection["strength_steps"]),
        top1_weight=float(selection["top1_weight"]),
        set_weight=float(selection["set_weight"]),
    )


def selected_strength_base(
    clean: torch.Tensor,
    reference_k5: torch.Tensor,
    disagreement: torch.Tensor,
    robust: nn.Module,
    spec: MethodSpec,
    amp: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if spec.strength_steps == 0:
        inactive = torch.zeros(len(clean), dtype=torch.bool, device=clean.device)
        return clean, inactive
    trajectory = strengthen_trajectory(
        clean,
        reference_k5,
        robust,
        spec.strength_alpha_px,
        spec.strength_epsilon_px,
        (spec.strength_steps,),
        spec.top1_weight,
        spec.set_weight,
        amp,
    )
    return gated_base(clean, trajectory[spec.strength_steps], disagreement, spec.gate_mode)


def _empty_mc_store(
    sigmas: Sequence[float], batch: int, device: torch.device
) -> tuple[dict[tuple[float, str], dict[str, torch.Tensor]], dict[tuple[float, str], torch.Tensor]]:
    counts: dict[tuple[float, str], dict[str, torch.Tensor]] = {}
    sums: dict[tuple[float, str], torch.Tensor] = {}
    for sigma in sigmas:
        for method in ("noise_only", "full"):
            key = (float(sigma), method)
            counts[key] = {
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
            sums[key] = torch.zeros(batch, 1000, dtype=torch.float64, device=device)
    return counts, sums


def evaluate_sigma_shard(
    shard: str,
    loader: DataLoader,
    standard: nn.Module,
    robust: nn.Module,
    selected_strength: Mapping[str, Any],
    device: torch.device,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    strength_spec = _selected_strength_spec(selected_strength)
    positive_sigmas = [float(value) for value in common.SIGMA_GRID if value > 0.0]
    sigma_groups = [
        positive_sigmas[start : start + int(args.sigma_forward_group)]
        for start in range(0, len(positive_sigmas), int(args.sigma_forward_group))
    ]
    for batch_number, (images, labels, relative_paths) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        paths = [str(value) for value in relative_paths]
        with torch.no_grad(), autocast_context(device, args.amp):
            standard_logits = standard(images).float()
            robust_logits = robust(images).float()
            reference_k5 = ranked_reference(standard_logits, common.PRIMARY_K).clone()
            disagreement = standard_logits.argmax(1).ne(robust_logits.argmax(1))
        full_base, full_gate = selected_strength_base(
            images, reference_k5, disagreement, robust, strength_spec, args.amp
        )
        inactive = torch.zeros(len(images), dtype=torch.bool, device=device)
        noise0 = MethodSpec("noise_only", "sigma_0_noise_only")
        full0 = MethodSpec(
            method="full",
            candidate_id="sigma_0_full",
            gate_mode=strength_spec.gate_mode,
            strength_alpha_px=strength_spec.strength_alpha_px,
            strength_epsilon_px=strength_spec.strength_epsilon_px,
            strength_steps=strength_spec.strength_steps,
            top1_weight=strength_spec.top1_weight,
            set_weight=strength_spec.set_weight,
        )
        output.extend(
            deterministic_rows(
                "sigma", shard, noise0, robust_logits, labels, paths, reference_k5, inactive
            )
        )
        with torch.inference_mode(), autocast_context(device, args.amp):
            full0_logits = robust(full_base).float()
        output.extend(
            deterministic_rows(
                "sigma", shard, full0, full0_logits, labels, paths, reference_k5, full_gate
            )
        )

        counts, logit_sums = _empty_mc_store(positive_sigmas, len(images), device)
        generator = torch.Generator(device=device)
        generator.manual_seed(
            stable_seed(args.seed, f"sigma-common|{shard}|batch-{batch_number:04d}")
        )
        completed = 0
        while completed < int(args.sigma_draws):
            chunk = min(int(args.mc_chunk), int(args.sigma_draws) - completed)
            noise = torch.randn(
                (chunk,) + tuple(images.shape),
                dtype=images.dtype,
                device=device,
                generator=generator,
            )
            for sigma_group in sigma_groups:
                tensors: list[torch.Tensor] = []
                keys: list[tuple[float, str]] = []
                for sigma in sigma_group:
                    tensors.append((images.unsqueeze(0) + sigma * noise).clamp(0.0, 1.0))
                    keys.append((sigma, "noise_only"))
                    tensors.append((full_base.unsqueeze(0) + sigma * noise).clamp(0.0, 1.0))
                    keys.append((sigma, "full"))
                flattened = torch.cat([tensor.flatten(0, 1) for tensor in tensors], dim=0)
                with torch.inference_mode(), autocast_context(device, args.amp):
                    combined_logits = robust(flattened).float()
                cursor = 0
                block = chunk * len(images)
                for key in keys:
                    logits = combined_logits[cursor : cursor + block].view(
                        chunk, len(images), 1000
                    )
                    cursor += block
                    flat_metrics = metric_tensors(
                        logits.flatten(0, 1),
                        labels.repeat(chunk),
                        reference_k5.repeat(chunk, 1),
                    )
                    for metric, values in flat_metrics.items():
                        counts[key][metric] += values.view(chunk, len(images)).sum(0).double()
                    logit_sums[key] += logits.double().sum(0)
            completed += chunk
        for sigma in positive_sigmas:
            for method, base_spec, active in (
                ("noise_only", noise0, inactive),
                ("full", full0, full_gate),
            ):
                spec_values = asdict(base_spec)
                spec_values.update(
                    {
                        "candidate_id": f"sigma_{common.sigma_tag(sigma)}_{method}",
                        "sigma": sigma,
                        "draws": int(args.sigma_draws),
                    }
                )
                spec = MethodSpec(**spec_values)
                key = (sigma, method)
                mean_logits = (logit_sums[key] / int(args.sigma_draws)).float()
                output.extend(
                    rows_from_statistics(
                        "sigma", shard, spec, labels, paths, reference_k5, active,
                        counts[key], mean_logits
                    )
                )
        print(f"    sigma/{shard}: batch {batch_number}/{len(loader)}", flush=True)
    return output


def _partial_path(output_dir: Path, phase: str, shard: str) -> Path:
    return output_dir / "partials" / phase / f"{shard}.json"


def run_or_resume_shard(
    output_dir: Path,
    phase: str,
    shard: str,
    config_hash: str,
    rows: Sequence[Mapping[str, object]],
    selection_hash: str | None,
    evaluator,
) -> dict[str, Any]:
    path = _partial_path(output_dir, phase, shard)
    expected_paths = sorted(str(row["relative_path"]) for row in rows)
    if path.is_file():
        payload = common.read_json(path)
        if payload.get("protocol_id") != common.PROTOCOL_ID:
            raise AssertionError(f"Partial protocol mismatch: {path}")
        if payload.get("config_hash") != config_hash or payload.get("phase") != phase:
            raise AssertionError(f"Partial invocation mismatch: {path}")
        if payload.get("shard_id") != shard or payload.get("selection_hash") != selection_hash:
            raise AssertionError(f"Partial shard/selection mismatch: {path}")
        observed = sorted({str(row["relative_path"]) for row in payload["examples"]})
        if observed != expected_paths:
            raise AssertionError(f"Partial sample membership mismatch: {path}")
        print(f"RESUME {phase}/{shard}", flush=True)
        return payload
    print(f"RUN {phase}/{shard}", flush=True)
    started = time.monotonic()
    examples = evaluator()
    payload = {
        "protocol_id": common.PROTOCOL_ID,
        "config_hash": config_hash,
        "phase": phase,
        "shard_id": shard,
        "selection_hash": selection_hash,
        "input_examples": len(rows),
        "wall_seconds": time.monotonic() - started,
        "examples": examples,
    }
    common.atomic_json(path, payload)
    print(f"PASS {phase}/{shard}: {len(examples):,} rows", flush=True)
    return payload


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["candidate_id"]), []).append(row)
    output: list[dict[str, Any]] = []
    for candidate_id, group in grouped.items():
        first = group[0]
        paths = {str(row["relative_path"]) for row in group}
        if len(paths) != common.CALIBRATION_EXAMPLES or len(group) != len(paths):
            raise AssertionError(f"Candidate does not have 2,000 unique examples: {candidate_id}")
        def percent(key: str) -> float:
            return 100.0 * float(np.mean([float(row[key]) for row in group]))

        summary = {
                "protocol_id": common.PROTOCOL_ID,
                "phase": first["phase"],
                "method": first["method"],
                "candidate_id": candidate_id,
                "gate_mode": first["gate_mode"],
                "gate_activation_percent": percent("gate_active"),
                "examples": len(group),
                "draws": int(first["draws"]),
                "sigma": float(first["sigma"]),
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
        if "source_method" in first:
            summary["source_method"] = first["source_method"]
        if "source_candidate_id" in first:
            summary["source_candidate_id"] = first["source_candidate_id"]
        output.append(summary)
    return sorted(output, key=lambda row: str(row["candidate_id"]))


def rows_for_candidate(
    rows: Sequence[Mapping[str, Any]], candidate_id: str
) -> list[dict[str, Any]]:
    selected = [dict(row) for row in rows if str(row["candidate_id"]) == candidate_id]
    if len(selected) != common.CALIBRATION_EXAMPLES:
        raise AssertionError(f"Expected 2,000 rows for {candidate_id}; got {len(selected)}")
    return selected


def summary_for_candidate(
    summaries: Sequence[Mapping[str, Any]], candidate_id: str
) -> dict[str, Any]:
    selected = [dict(row) for row in summaries if str(row["candidate_id"]) == candidate_id]
    if len(selected) != 1:
        raise AssertionError(f"Expected one summary for {candidate_id}; got {len(selected)}")
    return selected[0]


def bootstrap_spec(
    group_id: str,
    left: str,
    right: str,
    metric: str,
    left_rows: Sequence[Mapping[str, Any]],
    right_rows: Sequence[Mapping[str, Any]],
    metric_key: str | None = None,
) -> dict[str, Any]:
    key = metric_key or common.PROBABILITY_METRICS[metric]
    return {
        "group_id": group_id,
        "comparison": f"{left}_minus_{right}_{metric}",
        "left": left,
        "right": right,
        "metric": metric,
        "base_metric_key": key,
        "delta": common.aligned_delta(left_rows, right_rows, key),
    }


def alias_rows(rows: Sequence[Mapping[str, Any]], method: str) -> list[dict[str, Any]]:
    output = []
    for source in rows:
        row = dict(source)
        row["source_method"] = row["method"]
        row["source_candidate_id"] = row["candidate_id"]
        row["method"] = method
        row["candidate_id"] = method
        output.append(row)
    return output


def alias_summary(summary: Mapping[str, Any], method: str) -> dict[str, Any]:
    row = dict(summary)
    row["source_method"] = row["method"]
    row["source_candidate_id"] = row["candidate_id"]
    row["method"] = method
    row["candidate_id"] = method
    return row


def reference_quality_decomposition(
    selected_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    by_method = {
        method: common.rows_by_sample(
            [row for row in selected_rows if str(row["method"]) == method]
        )
        for method in ("standard", "robust", "full_sigma0", "noise_only", "full")
    }
    paths = sorted(by_method["standard"])
    if any(sorted(table) != paths for table in by_method.values()):
        raise AssertionError("Selected methods do not share the same samples")
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
                return 100.0 * float(np.mean([float(table[path][key]) for path in group_paths]))

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
        "standard_checkpoint": resolved(args.standard_checkpoint),
        "robust_checkpoint": resolved(args.robust_checkpoint),
        "output_dir": resolved(args.output_dir),
    }
    paths["output_dir"].mkdir(parents=True, exist_ok=True)
    print(f"PROTOCOL_ID={common.PROTOCOL_ID}")
    print(f"sys.executable={sys.executable}")
    print("scope: calibration_dev only; no attack, training, download, or later-split read")

    calibration_rows, _parents = validate_parents(paths)
    shards = outcome_blind_shards(calibration_rows)
    transform, preprocessing = raw_cross_architecture_preprocessing()
    del transform
    calibration_manifest_rows: list[dict[str, Any]] = []
    for shard_number, shard_rows in enumerate(shards):
        for row in shard_rows:
            calibration_manifest_rows.append(
                {**dict(row), "shard_id": common.shard_id(shard_number)}
            )
    calibration_manifest = {
        "protocol_id": common.PROTOCOL_ID,
        "partition": "calibration_dev",
        "examples": common.CALIBRATION_EXAMPLES,
        "classes": 1000,
        "examples_per_class": 2,
        "shards": common.CALIBRATION_SHARDS,
        "examples_per_shard": common.EXAMPLES_PER_SHARD,
        "assignment": "SHA256 rank of relative_path before inference, then contiguous chunks",
        "rows_sha256": common.object_sha256(calibration_manifest_rows),
        "parent_split_sha256": common.FROZEN_SPLIT_SHA256,
        "partitions_read": ["calibration_dev"],
        "partitions_not_read": [
            "preflight_sanity",
            "attack_pilot",
            "attack_confirmatory",
            "clean_confirmation",
        ],
        "attack_outcomes_read": False,
        "preprocessing": preprocessing,
    }
    _write_or_validate_json(
        paths["output_dir"] / "calibration_dev_manifest.json", calibration_manifest
    )
    common.atomic_csv(
        paths["output_dir"] / "calibration_dev_manifest.csv", calibration_manifest_rows
    )

    source_paths = {
        filename: PROJECT_DIR / filename
        for filename in common.NEW_SCRIPT_NAMES
    }
    source_paths.update(
        {filename: PROJECT_DIR / filename for filename in common.EXPECTED_PARENT_SOURCE_HASHES}
    )
    for path in source_paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Required source is missing: {path}")
    config = {
        "protocol_id": common.PROTOCOL_ID,
        "parent_r0_protocol_id": common.PARENT_R0_PROTOCOL_ID,
        "parent_r1_protocol_id": common.PARENT_R1_PROTOCOL_ID,
        "dataset": "ImageNet-1k validation",
        "architecture_roles": {
            "standard_reference": "torchvision ViT-B/16",
            "defended_backbone": "torchvision ResNet-50",
            "cross_architecture_reference": True,
        },
        "experiment_scope": (
            "cross-architecture reference transfer; this is not a robust-ViT "
            "backbone evaluation"
        ),
        "primary_k": common.PRIMARY_K,
        "secondary_k": common.SECONDARY_K,
        "paths": {key: str(value) for key, value in paths.items()},
        "checkpoints": {
            "standard": {
                "path": str(paths["standard_checkpoint"]),
                "sha256": common.STANDARD_CHECKPOINT_SHA256,
                "model_id": "vit_b16_local",
                "role": "standard_reference_top5_only",
                "adversarially_robust_checkpoint": False,
            },
            "robust": {
                "path": str(paths["robust_checkpoint"]),
                "sha256": common.ROBUST_CHECKPOINT_SHA256,
                "state_dict_fingerprint_sha256": common.ROBUST_STATE_FINGERPRINT,
                "model_id": "Salman2020Do_R50",
                "role": "robust_defended_backbone",
                "adversarially_robust_checkpoint": True,
                "training_epsilon_linf_px": 4.0,
            },
        },
        "parent_artifacts": {
            "r1_handoff_sha256": common.R1_HANDOFF_SHA256,
            "r1_report_sha256": common.R1_REPORT_SHA256,
            "r0_preflight_report_sha256": common.file_sha256(
                paths["r0_dir"] / "preflight_report.json"
            ),
            "frozen_split_sha256": common.FROZEN_SPLIT_SHA256,
        },
        "calibration_examples": common.CALIBRATION_EXAMPLES,
        "calibration_shards": common.CALIBRATION_SHARDS,
        "batch_size": int(args.batch_size),
        "mc_chunk": int(args.mc_chunk),
        "sigma_forward_group": int(args.sigma_forward_group),
        "num_workers": int(args.num_workers),
        "gpu_memory_fraction": float(args.gpu_memory_fraction),
        "amp": bool(args.amp),
        "sigma_draws": int(args.sigma_draws),
        "max_clean_drop_pp": float(args.max_clean_drop_pp),
        "bootstrap_resamples": int(args.bootstrap_resamples),
        "seed": int(args.seed),
        "bootstrap_seeds": {
            "strength": int(args.bootstrap_seed_strength),
            "sigma": int(args.bootstrap_seed_sigma),
            "selected": int(args.bootstrap_seed_selected),
        },
        "gate_modes": list(common.GATE_MODES),
        "strength_grid": list(common.STRENGTH_GRID),
        "sigma_grid": list(common.SIGMA_GRID),
        "strength_selection_policy": (
            "maximize deterministic Exact-Top5 among identity plus the frozen ImageNet "
            "factorial grid, subject to paired 95% bootstrap lower bounds >= -2 pp for "
            "Top1 and Top5 accuracy versus robust"
        ),
        "sigma_selection_policy": (
            "largest sigma satisfying all eight paired 95% bootstrap lower-bound "
            "non-inferiority constraints at -2 pp"
        ),
        "common_random_numbers": True,
        "offline": True,
        "downloads_allowed": False,
        "training_allowed": False,
        "attack_outcomes_read": False,
        "partitions_read": ["calibration_dev"],
        "forbidden_partitions_not_read": [
            "attack_pilot",
            "attack_confirmatory",
            "clean_confirmation",
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

    standard, robust, model_audit = load_models(
        paths["standard_checkpoint"], paths["robust_checkpoint"], device
    )
    _write_or_validate_json(paths["output_dir"] / "model_load_audit.json", model_audit)

    strength_payloads = []
    for shard_number, shard_rows in enumerate(shards):
        shard = common.shard_id(shard_number)
        loader = make_loader(
            paths["data_root"], shard_rows, args.batch_size, args.num_workers, device
        )
        strength_payloads.append(
            run_or_resume_shard(
                paths["output_dir"],
                "strength",
                shard,
                config_hash,
                shard_rows,
                None,
                lambda loader=loader, shard=shard: evaluate_strength_shard(
                    shard, loader, standard, robust, device, args
                ),
            )
        )
    strength_rows = [
        dict(row) for payload in strength_payloads for row in payload["examples"]
    ]
    strength_summaries = summarize_rows(strength_rows)
    robust_rows = rows_for_candidate(strength_rows, "identity")
    candidate_summaries = [
        row for row in strength_summaries if row["candidate_id"] not in {"standard"}
    ]
    strength_specs: list[dict[str, Any]] = []
    for summary in candidate_summaries:
        candidate_id = str(summary["candidate_id"])
        candidate_rows = rows_for_candidate(strength_rows, candidate_id)
        for metric in ("top1", "top5", "exact_k5"):
            strength_specs.append(
                bootstrap_spec(
                    candidate_id, candidate_id, "robust", metric,
                    candidate_rows, robust_rows
                )
            )
    print("Computing paired bootstrap for the ImageNet strength frontier", flush=True)
    strength_bootstrap = common.paired_bootstrap_many(
        strength_specs, args.bootstrap_resamples, args.bootstrap_seed_strength
    )
    selected_strength, annotated_strength = common.select_strength_candidate(
        candidate_summaries, strength_bootstrap, args.max_clean_drop_pp
    )
    selected_strength_id = str(selected_strength["candidate_id"])
    annotation_by_id = {str(row["candidate_id"]): row for row in annotated_strength}
    annotated_strength_summaries = []
    for summary in strength_summaries:
        row = dict(summary)
        if row["candidate_id"] != "standard":
            row.update(annotation_by_id[str(row["candidate_id"])])
            row["selected"] = str(row["candidate_id"]) == selected_strength_id
        else:
            row["selected"] = False
        annotated_strength_summaries.append(row)
    common.atomic_csv(paths["output_dir"] / "strength_frontier_summary.csv", annotated_strength_summaries)
    common.atomic_csv(paths["output_dir"] / "strength_frontier_per_example.csv", strength_rows)
    common.atomic_csv(paths["output_dir"] / "strength_paired_bootstrap.csv", strength_bootstrap)
    strength_selection = {
        "protocol_id": common.PROTOCOL_ID,
        "config_hash": config_hash,
        "selection_split": "calibration_dev",
        "selection_policy": config["strength_selection_policy"],
        **selected_strength,
    }
    strength_selection["strength_selection_hash"] = common.object_sha256(strength_selection)
    _write_or_validate_json(
        paths["output_dir"] / "strength_selection.json", strength_selection
    )

    sigma_payloads = []
    for shard_number, shard_rows in enumerate(shards):
        shard = common.shard_id(shard_number)
        loader = make_loader(
            paths["data_root"], shard_rows, args.batch_size, args.num_workers, device
        )
        sigma_payloads.append(
            run_or_resume_shard(
                paths["output_dir"],
                "sigma",
                shard,
                config_hash,
                shard_rows,
                str(strength_selection["strength_selection_hash"]),
                lambda loader=loader, shard=shard: evaluate_sigma_shard(
                    shard, loader, standard, robust, strength_selection, device, args
                ),
            )
        )
    sigma_rows = [dict(row) for payload in sigma_payloads for row in payload["examples"]]
    sigma_summaries = summarize_rows(sigma_rows)
    selected_strength_rows = rows_for_candidate(strength_rows, selected_strength_id)
    noise0_rows = rows_for_candidate(sigma_rows, "sigma_0_noise_only")
    full0_rows = rows_for_candidate(sigma_rows, "sigma_0_full")
    for left_rows, right_rows, label in (
        (noise0_rows, robust_rows, "sigma-zero noise/robust"),
        (full0_rows, selected_strength_rows, "sigma-zero full/selected-strength"),
    ):
        for key in common.PROBABILITY_METRICS.values():
            drift = float(np.max(np.abs(common.aligned_delta(left_rows, right_rows, key))))
            if drift > 1e-10:
                raise AssertionError(f"{label} failed for {key}: {drift}")

    sigma_specs: list[dict[str, Any]] = []
    sigma_combined: list[dict[str, Any]] = []
    sigma_summary_by_id = {str(row["candidate_id"]): row for row in sigma_summaries}
    for sigma in common.SIGMA_GRID:
        tag = common.sigma_tag(sigma)
        group_id = f"sigma_{tag}"
        noise_id = f"sigma_{tag}_noise_only"
        full_id = f"sigma_{tag}_full"
        noise_rows = rows_for_candidate(sigma_rows, noise_id)
        full_rows = rows_for_candidate(sigma_rows, full_id)
        definitions = (
            ("noise_top1_vs_robust", "noise_only", "robust", noise_rows, robust_rows, "top1"),
            ("noise_top5_vs_robust", "noise_only", "robust", noise_rows, robust_rows, "top5"),
            ("noise_exact_k5_vs_robust", "noise_only", "robust", noise_rows, robust_rows, "exact_k5"),
            ("full_top1_vs_robust", "full", "robust", full_rows, robust_rows, "top1"),
            ("full_top5_vs_robust", "full", "robust", full_rows, robust_rows, "top5"),
            ("full_top1_vs_full_sigma0", "full", "full_sigma0", full_rows, full0_rows, "top1"),
            ("full_top5_vs_full_sigma0", "full", "full_sigma0", full_rows, full0_rows, "top5"),
            ("full_exact_k5_vs_full_sigma0", "full", "full_sigma0", full_rows, full0_rows, "exact_k5"),
        )
        for constraint, left, right, left_rows, right_rows, base_metric in definitions:
            spec = bootstrap_spec(
                group_id, left, right, constraint, left_rows, right_rows,
                common.PROBABILITY_METRICS[base_metric]
            )
            spec["base_metric"] = base_metric
            sigma_specs.append(spec)
        noise_summary = sigma_summary_by_id[noise_id]
        full_summary = sigma_summary_by_id[full_id]
        sigma_combined.append(
            {
                "protocol_id": common.PROTOCOL_ID,
                "sigma": float(sigma),
                "draws": int(noise_summary["draws"]),
                "selected_strength_candidate_id": selected_strength_id,
                "noise_top1_accuracy_percent": noise_summary["top1_accuracy_percent"],
                "noise_top5_accuracy_percent": noise_summary["label_coverage_k5_probability_percent"],
                "noise_exact_set_k5_probability_percent": noise_summary["exact_set_k5_probability_percent"],
                "full_top1_accuracy_percent": full_summary["top1_accuracy_percent"],
                "full_top5_accuracy_percent": full_summary["label_coverage_k5_probability_percent"],
                "full_exact_set_k5_probability_percent": full_summary["exact_set_k5_probability_percent"],
            }
        )
    print("Computing paired bootstrap for the ImageNet sigma frontier", flush=True)
    sigma_bootstrap = common.paired_bootstrap_many(
        sigma_specs, args.bootstrap_resamples, args.bootstrap_seed_sigma
    )
    selected_sigma, annotated_sigma = common.select_sigma_candidate(
        sigma_combined, sigma_bootstrap, args.max_clean_drop_pp
    )
    selected_sigma_value = round(float(selected_sigma["sigma"]), 7)
    sigma_annotations = {round(float(row["sigma"]), 7): row for row in annotated_sigma}
    annotated_sigma_summaries = []
    for summary in sigma_summaries:
        row = dict(summary)
        sigma = round(float(row["sigma"]), 7)
        annotation = sigma_annotations[sigma]
        row["selected"] = sigma == selected_sigma_value
        row["jointly_ci_feasible"] = bool(annotation["jointly_ci_feasible"])
        for key, value in annotation.items():
            if key.endswith("_difference_pp") or "_ci95_" in key:
                row[key] = value
        annotated_sigma_summaries.append(row)
    common.atomic_csv(paths["output_dir"] / "sigma_frontier_summary.csv", annotated_sigma_summaries)
    common.atomic_csv(paths["output_dir"] / "sigma_frontier_per_example.csv", sigma_rows)
    common.atomic_csv(paths["output_dir"] / "sigma_paired_bootstrap.csv", sigma_bootstrap)

    standard_rows = rows_for_candidate(strength_rows, "standard")
    selected_noise_id = f"sigma_{common.sigma_tag(selected_sigma_value)}_noise_only"
    selected_full_id = f"sigma_{common.sigma_tag(selected_sigma_value)}_full"
    selected_noise_rows = rows_for_candidate(sigma_rows, selected_noise_id)
    selected_full_rows = rows_for_candidate(sigma_rows, selected_full_id)
    selected_rows = (
        alias_rows(standard_rows, "standard")
        + alias_rows(robust_rows, "robust")
        + alias_rows(selected_strength_rows, "full_sigma0")
        + alias_rows(selected_noise_rows, "noise_only")
        + alias_rows(selected_full_rows, "full")
    )
    selected_summaries = summarize_rows(selected_rows)
    selected_groups = {
        method: [row for row in selected_rows if row["method"] == method]
        for method in ("standard", "robust", "full_sigma0", "noise_only", "full")
    }
    selected_specs: list[dict[str, Any]] = []
    for left, right, metrics in (
        ("noise_only", "robust", ("top1", "top5", "exact_k5", "exact_k2")),
        ("full_sigma0", "robust", ("top1", "top5", "exact_k5", "exact_k2")),
        ("full", "robust", ("top1", "top5", "exact_k5", "exact_k2")),
        ("full", "noise_only", ("top1", "top5", "exact_k5", "exact_k2")),
    ):
        for metric in metrics:
            selected_specs.append(
                bootstrap_spec(
                    "selected", left, right, metric,
                    selected_groups[left], selected_groups[right]
                )
            )
    selected_bootstrap = common.paired_bootstrap_many(
        selected_specs, args.bootstrap_resamples, args.bootstrap_seed_selected
    )
    decomposition = reference_quality_decomposition(selected_rows)
    common.atomic_csv(paths["output_dir"] / "selected_dev_summary.csv", selected_summaries)
    common.atomic_csv(paths["output_dir"] / "selected_dev_per_example.csv", selected_rows)
    common.atomic_csv(paths["output_dir"] / "selected_dev_paired_bootstrap.csv", selected_bootstrap)
    common.atomic_csv(paths["output_dir"] / "reference_quality_decomposition.csv", decomposition)

    positive_strength = int(selected_strength["strength_steps"]) > 0
    positive_sigma = selected_sigma_value > 0.0
    official_clean_ready = bool(
        positive_strength
        and positive_sigma
        and selected_strength["jointly_ci_feasible"]
        and selected_sigma["jointly_ci_feasible"]
    )
    selection = {
        "protocol_id": common.PROTOCOL_ID,
        "config_hash": config_hash,
        "selection_split": "calibration_dev",
        "strength": {
            "candidate_id": selected_strength_id,
            "gate_mode": selected_strength["gate_mode"],
            "alpha_px": float(selected_strength["strength_alpha_px"]),
            "epsilon_px": float(selected_strength["strength_epsilon_px"]),
            "steps": int(selected_strength["strength_steps"]),
            "top1_weight": float(selected_strength["top1_weight"]),
            "set_weight": float(selected_strength["set_weight"]),
            "positive_strength_selected": positive_strength,
            "selected_dev_summary": selected_strength,
        },
        "randomization": {
            "sigma": selected_sigma_value,
            "draws_during_calibration": int(args.sigma_draws),
            "positive_sigma_selected": positive_sigma,
            "selected_dev_constraints": selected_sigma,
        },
        "data_governance": {
            "partitions_read": ["calibration_dev"],
            "attack_outcomes_read": False,
            "training_performed": False,
            "cifar_selected_hyperparameters_transferred_as_fixed_values": False,
            "claim_scope": "cross_architecture_reference_only_not_robust_vit",
        },
    }
    selection["selection_hash"] = common.object_sha256(selection)
    _write_or_validate_json(paths["output_dir"] / "defense_selection_vr1.json", selection)
    blockers = []
    if not positive_strength:
        blockers.append("Identity won the preregistered strengthening frontier")
    if not positive_sigma:
        blockers.append("No positive sigma passed all eight paired-CI constraints")
    final_summary = {
        "protocol_id": common.PROTOCOL_ID,
        "config_hash": config_hash,
        "selection_hash": selection["selection_hash"],
        "status": "PASS",
        "selection": selection,
        "selected_dev_summaries": selected_summaries,
        "selected_dev_paired_bootstrap": selected_bootstrap,
        "official_clean_stage_ready": official_clean_ready,
        "official_clean_stage_blockers": blockers,
        "attack_stage_ready": False,
        "paper_claim_authorized": False,
        "next_stage": (
            "vit_reference_crossarch_clean_confirmation_on_clean_confirmation_partition"
            if official_clean_ready
            else "stop_and_report_failed_vit_reference_crossarch_operating_point"
        ),
        "data_governance": {
            "partitions_read": ["calibration_dev"],
            "partitions_not_read": [
                "preflight_sanity",
                "attack_pilot",
                "attack_confirmatory",
                "clean_confirmation",
            ],
            "no_training": True,
            "no_download": True,
            "no_attack": True,
        },
    }
    common.atomic_json(paths["output_dir"] / "final_summary.json", final_summary)

    table = {row["method"]: row for row in selected_summaries}
    print("\nSelected ImageNet ViT-reference cross-architecture operating point", flush=True)
    print(
        f"  strength={selected_strength_id}: gate={selected_strength['gate_mode']}, "
        f"alpha={float(selected_strength['strength_alpha_px']):.3f}/255, "
        f"epsilon={float(selected_strength['strength_epsilon_px']):.3f}/255, "
        f"steps={int(selected_strength['strength_steps'])}, "
        f"weights=1:{float(selected_strength['set_weight']):g}",
        flush=True,
    )
    print(f"  sigma={selected_sigma_value:.7f}, calibration MC={args.sigma_draws}", flush=True)
    for method in ("standard", "robust", "full_sigma0", "noise_only", "full"):
        row = table[method]
        print(
            f"  {method:>11}: Top1={float(row['top1_accuracy_percent']):.3f}% "
            f"Top5={float(row['label_coverage_k5_probability_percent']):.3f}% "
            f"Exact5={float(row['exact_set_k5_probability_percent']):.3f}% "
            f"Exact2={float(row['exact_set_k2_probability_percent']):.3f}%",
            flush=True,
        )
    print("OFFICIAL_CLEAN_STAGE_READY=" + ("YES" if official_clean_ready else "NO"))
    print("ATTACK_STAGE_READY=NO")
    print("PAPER_CLAIM_AUTHORIZED=NO (calibration only)")
    print(f"Complete: {paths['output_dir']}")


if __name__ == "__main__":
    main()
