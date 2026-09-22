#!/usr/bin/env python3
"""Acquire, normalize, and verify the frozen ImageNet robust ResNet-50 anchor.

R1 is a provenance and identity stage.  It downloads exactly one published
RobustBench checkpoint when absent, converts it to a plain torchvision
ResNet-50 state_dict, evaluates only R0's preflight_sanity partition, and runs
a small label-aware PGD identity check.  It does not train, tune the defense,
read calibration/attack/clean-confirmation outcomes, or download ImageNet.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import sys
import time
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode

from preflight_topk_imagenet_r0 import (
    evaluate_sanity,
    extract_state_dict,
    preprocessing_profile,
    select_architecture,
    torch_load_local,
)
from topk_imagenet_preflight_common import (
    partition_rows,
    read_csv as read_r0_csv,
    validate_frozen_split,
)
from topk_imagenet_robust_anchor_common_r1 import (
    EXPECTED_ARCHITECTURE_ID,
    EXPECTED_FAMILY,
    EXPECTED_NUM_CLASSES,
    EXPECTED_SANITY_EXAMPLES,
    FROZEN_SPLIT_SHA256,
    IMAGENET_MEAN,
    IMAGENET_STD,
    PARENT_PROTOCOL_ID,
    PREPROCESSING_ID,
    PROTOCOL_ID,
    ROBUSTBENCH_EPSILON,
    ROBUSTBENCH_MODEL_ID,
    SCRIPT_NAMES,
    STANDARD_CHECKPOINT_SHA256,
    acquire_official_checkpoint,
    atomic_csv,
    atomic_json,
    human_bytes,
    read_csv,
    read_json,
    script_hashes,
    sha256_file,
    source_provenance,
    stable_hash,
)


DEFAULT_PROJECT_DIR = Path("./projects/ImageNet")
DEFAULT_DATA_ROOT = Path("./datasets/imagenet")
DEFAULT_R0_DIR = Path(
    "./models/imagenet_topk/"
    "paper_topk_imagenet_preflight_r0"
)
DEFAULT_STANDARD_CHECKPOINT = Path(
    "./models/resnet50/imagenet/"
    "resnet50_imagenet.pth"
)
DEFAULT_RAW_CHECKPOINT = Path(
    "./models/resnet50/imagenet/robustbench_raw/"
    "Salman2020Do_R50.pt"
)
DEFAULT_EXPORT_CHECKPOINT = Path(
    "./models/resnet50/imagenet/"
    "salman2020do_r50_linf4.pth"
)
DEFAULT_OUTPUT_DIR = Path(
    "./models/imagenet_topk/"
    "paper_topk_imagenet_robust_anchor_r1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Acquire and verify the official ImageNet robust ResNet-50 anchor"
    )
    parser.add_argument("--project-dir", type=Path, default=DEFAULT_PROJECT_DIR)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--r0-dir", type=Path, default=DEFAULT_R0_DIR)
    parser.add_argument(
        "--standard-checkpoint", type=Path, default=DEFAULT_STANDARD_CHECKPOINT
    )
    parser.add_argument("--raw-checkpoint", type=Path, default=DEFAULT_RAW_CHECKPOINT)
    parser.add_argument(
        "--export-checkpoint", type=Path, default=DEFAULT_EXPORT_CHECKPOINT
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--download-backend", choices=("auto", "gdown", "urllib"), default="auto"
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Require an already-present raw checkpoint; make no network request",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--pgd-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.45)
    parser.add_argument("--pgd-examples", type=int, default=100)
    parser.add_argument("--pgd-steps", type=int, default=10)
    parser.add_argument("--pgd-alpha-255", type=float, default=1.0)
    parser.add_argument("--pgd-seed", type=int, default=91020261)
    parser.add_argument("--minimum-sanity-top1", type=float, default=45.0)
    parser.add_argument("--minimum-sanity-top5", type=float, default=70.0)
    parser.add_argument("--minimum-pgd-top1", type=float, default=10.0)
    parser.add_argument("--minimum-pgd-advantage", type=float, default=10.0)
    return parser.parse_args()


def resolved(path: Path) -> Path:
    return path.expanduser().resolve()


def validate_args(args: argparse.Namespace) -> None:
    if args.eval_batch_size <= 0 or args.pgd_batch_size <= 0:
        raise ValueError("Batch sizes must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if not 0.0 < args.gpu_memory_fraction <= 1.0:
        raise ValueError("--gpu-memory-fraction must be in (0,1]")
    if not 1 <= args.pgd_examples <= EXPECTED_SANITY_EXAMPLES:
        raise ValueError("--pgd-examples must be in [1,1000]")
    if args.pgd_steps <= 0 or args.pgd_alpha_255 <= 0:
        raise ValueError("PGD steps and alpha must be positive")
    for value in (
        args.minimum_sanity_top1,
        args.minimum_sanity_top5,
        args.minimum_pgd_top1,
        args.minimum_pgd_advantage,
    ):
        if not 0.0 <= value <= 100.0:
            raise ValueError("All percentage gates must be in [0,100]")


def configure_device(args: argparse.Namespace) -> torch.device:
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(
            args.gpu_memory_fraction, device.index or 0
        )
    return device


def environment_report(device: torch.device, args: argparse.Namespace) -> dict[str, Any]:
    gpu = None
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        gpu = {
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "total_memory_gib": properties.total_memory / (1024**3),
            "capability": list(torch.cuda.get_device_capability(device)),
        }
    return {
        "sys_executable": sys.executable,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": str(device),
        "gpu": gpu,
        "gpu_memory_fraction": args.gpu_memory_fraction,
    }


def load_and_validate_parent(r0_dir: Path, standard_checkpoint: Path) -> dict[str, Any]:
    parent_manifest = read_json(r0_dir / "run_manifest.json")
    parent_report = read_json(r0_dir / "preflight_report.json")
    split_manifest = read_json(r0_dir / "frozen_imagenet_val_split_manifest.json")
    split_rows = read_r0_csv(r0_dir / "frozen_imagenet_val_split.csv")
    split_check = validate_frozen_split(split_rows)
    if parent_manifest.get("protocol_id") != PARENT_PROTOCOL_ID:
        raise AssertionError("R0 protocol_id mismatch")
    if parent_report.get("preflight_complete") is not True:
        raise AssertionError("R0 preflight is not complete")
    if split_manifest.get("split_sha256") != FROZEN_SPLIT_SHA256:
        raise AssertionError("R0 frozen split manifest hash differs from the R1 lock")
    if split_check.get("split_sha256") != FROZEN_SPLIT_SHA256:
        raise AssertionError("R0 frozen split rows differ from the R1 lock")
    if parent_manifest.get("frozen_split_sha256") != FROZEN_SPLIT_SHA256:
        raise AssertionError("R0 run manifest split hash differs from the R1 lock")
    recorded_sources = parent_manifest.get("script_sha256")
    if not isinstance(recorded_sources, Mapping):
        raise TypeError("R0 source-hash manifest is malformed")
    live_source_dir = Path(__file__).resolve().parent
    for source_name in (
        "topk_imagenet_preflight_common.py",
        "preflight_topk_imagenet_r0.py",
    ):
        source_path = live_source_dir / source_name
        if sha256_file(source_path) != recorded_sources.get(source_name):
            raise AssertionError(f"R0 source changed after its audited run: {source_name}")
    if sha256_file(standard_checkpoint) != STANDARD_CHECKPOINT_SHA256:
        raise AssertionError("Local standard ResNet-50 changed since R0")

    checkpoint_rows = parent_report.get("checkpoints")
    if not isinstance(checkpoint_rows, list):
        raise TypeError("R0 checkpoint report is malformed")
    standard_rows = [
        row
        for row in checkpoint_rows
        if isinstance(row, Mapping) and row.get("checkpoint_id") == "resnet50_local"
    ]
    if len(standard_rows) != 1:
        raise AssertionError("R0 contains no unique resnet50_local checkpoint")
    standard = dict(standard_rows[0])
    if standard.get("status") != "PASS":
        raise AssertionError("R0 standard ResNet-50 did not pass")
    if standard.get("sha256") != STANDARD_CHECKPOINT_SHA256:
        raise AssertionError("R0 standard ResNet-50 SHA mismatch")
    if standard.get("architecture", {}).get("architecture_id") != EXPECTED_ARCHITECTURE_ID:
        raise AssertionError("R0 standard checkpoint is not torchvision ResNet-50")

    sanity_rows = partition_rows(split_rows, "preflight_sanity")
    if len(sanity_rows) != EXPECTED_SANITY_EXAMPLES:
        raise AssertionError("R0 preflight_sanity partition must contain 1000 examples")
    return {
        "manifest": parent_manifest,
        "report": parent_report,
        "split_manifest": split_manifest,
        "split_rows": split_rows,
        "sanity_rows": sanity_rows,
        "standard_report": standard,
    }


def state_dict_fingerprint(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        header = json.dumps(
            [key, str(tensor.dtype), list(tensor.shape)],
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def load_resnet50_checkpoint(path: Path) -> tuple[nn.Module, dict[str, Any]]:
    payload, load_mode = torch_load_local(path, allow_unsafe=False)
    state, route = extract_state_dict(payload)
    model, architecture, match = select_architecture(state)
    if architecture.get("architecture_id") != EXPECTED_ARCHITECTURE_ID:
        raise AssertionError(
            f"Expected {EXPECTED_ARCHITECTURE_ID}, detected {architecture.get('architecture_id')}"
        )
    if architecture.get("family") != EXPECTED_FAMILY:
        raise AssertionError("Detected architecture family is not ResNet-50")
    if float(match.get("parameter_numel_coverage", 0.0)) != 1.0:
        raise AssertionError("Checkpoint does not have 100% ResNet-50 parameter coverage")
    if float(match.get("numel_coverage", 0.0)) != 1.0:
        raise AssertionError("Checkpoint does not have 100% ResNet-50 state coverage")
    if match.get("missing_keys") or match.get("shape_mismatches"):
        raise AssertionError("Checkpoint has missing or shape-mismatched ResNet-50 tensors")
    normalized_state = OrderedDict(
        (key, value.detach().cpu().clone()) for key, value in model.state_dict().items()
    )
    return model, {
        "load_mode": load_mode,
        "state_dict_route": route,
        "source_tensor_count": len(state),
        "architecture": architecture,
        "match": match,
        "normalized_tensor_count": len(normalized_state),
        "state_dict_fingerprint_sha256": state_dict_fingerprint(normalized_state),
        "normalized_state": normalized_state,
    }


def export_normalized_checkpoint(
    raw_state: Mapping[str, torch.Tensor],
    raw_fingerprint: str,
    export_path: Path,
) -> dict[str, object]:
    export_path.parent.mkdir(parents=True, exist_ok=True)
    if export_path.exists():
        _, existing = load_resnet50_checkpoint(export_path)
        existing_fingerprint = str(existing["state_dict_fingerprint_sha256"])
        if existing_fingerprint != raw_fingerprint:
            raise RuntimeError(
                f"Existing exported checkpoint conflicts with the official raw model: {export_path}"
            )
        acquisition = "reused_existing_matching_export"
    else:
        temporary = export_path.with_name(export_path.name + f".tmp.{os.getpid()}")
        torch.save(OrderedDict(raw_state), temporary)
        os.replace(temporary, export_path)
        acquisition = "created_from_verified_raw_checkpoint"
    _, exported = load_resnet50_checkpoint(export_path)
    if exported["state_dict_fingerprint_sha256"] != raw_fingerprint:
        raise AssertionError("Exported checkpoint tensor fingerprint changed")
    return {
        "path": str(export_path),
        "bytes": export_path.stat().st_size,
        "human_size": human_bytes(export_path.stat().st_size),
        "sha256": sha256_file(export_path),
        "state_dict_fingerprint_sha256": raw_fingerprint,
        "tensor_count": exported["normalized_tensor_count"],
        "acquisition": acquisition,
    }


class RawManifestDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        rows: Sequence[Mapping[str, object]],
        transform: Callable,
    ) -> None:
        self.data_root = data_root
        self.rows = list(rows)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        path = self.data_root / str(row["relative_path"])
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, int(row["target"]), str(row["relative_path"])


class NormalizeThenModel(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model((images - self.mean) / self.std)


def raw_preprocessing() -> tuple[Callable, dict[str, object]]:
    transform = transforms.Compose(
        [
            transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )
    return transform, {
        "resize_shorter_side": 256,
        "crop_size": 224,
        "interpolation": "bilinear",
        "range": [0.0, 1.0],
        "normalization_inside_model": True,
    }


def select_pgd_rows(
    sanity_rows: Sequence[Mapping[str, object]], count: int
) -> list[dict[str, object]]:
    def rank(row: Mapping[str, object]) -> str:
        value = f"{PROTOCOL_ID}|pgd_identity|{row['relative_path']}"
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    selected = sorted(sanity_rows, key=rank)[:count]
    return [dict(row) for row in selected]


def pgd_linf(
    model: nn.Module,
    clean: torch.Tensor,
    target: torch.Tensor,
    epsilon: float,
    alpha: float,
    steps: int,
    seed: int,
) -> torch.Tensor:
    cpu_generator = torch.Generator(device="cpu")
    cpu_generator.manual_seed(seed)
    random_start = torch.empty(clean.shape, dtype=clean.dtype, device="cpu")
    random_start.uniform_(-epsilon, epsilon, generator=cpu_generator)
    adversarial = (clean + random_start.to(clean.device)).clamp(0.0, 1.0).detach()
    for _ in range(steps):
        adversarial.requires_grad_(True)
        logits = model(adversarial)
        loss = F.cross_entropy(logits, target, reduction="sum")
        gradient = torch.autograd.grad(loss, adversarial, only_inputs=True)[0]
        adversarial = adversarial.detach() + alpha * gradient.sign()
        delta = (adversarial - clean).clamp(-epsilon, epsilon)
        adversarial = (clean + delta).clamp(0.0, 1.0).detach()
    return adversarial


def evaluate_pgd_identity(
    standard_model: nn.Module,
    robust_model: nn.Module,
    data_root: Path,
    rows: Sequence[Mapping[str, object]],
    device: torch.device,
    args: argparse.Namespace,
    output_path: Path,
) -> dict[str, Any]:
    transform, transform_report = raw_preprocessing()
    loader = DataLoader(
        RawManifestDataset(data_root, rows, transform),
        batch_size=args.pgd_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    models = {
        "standard": NormalizeThenModel(standard_model).to(device).eval(),
        "robust": NormalizeThenModel(robust_model).to(device).eval(),
    }
    for model in models.values():
        model.requires_grad_(False)
    epsilon = ROBUSTBENCH_EPSILON
    alpha = args.pgd_alpha_255 / 255.0
    result_rows: list[dict[str, object]] = []
    maximum_linf = {"standard": 0.0, "robust": 0.0}
    started = time.perf_counter()
    total = 0
    for batch_index, (images, targets, paths) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        batch_outputs: dict[str, dict[str, torch.Tensor]] = {}
        for name, model in models.items():
            with torch.inference_mode():
                clean_pred = model(images).argmax(dim=1)
            adversarial = pgd_linf(
                model,
                images,
                targets,
                epsilon,
                alpha,
                args.pgd_steps,
                args.pgd_seed + 100_003 * batch_index,
            )
            with torch.inference_mode():
                adversarial_pred = model(adversarial).argmax(dim=1)
            linf = (adversarial - images).abs().flatten(1).amax(dim=1)
            maximum_linf[name] = max(maximum_linf[name], float(linf.max().item()))
            batch_outputs[name] = {
                "clean_pred": clean_pred.detach().cpu(),
                "adversarial_pred": adversarial_pred.detach().cpu(),
                "linf": linf.detach().cpu(),
            }
            del adversarial
        targets_cpu = targets.cpu()
        for offset, relative_path in enumerate(paths):
            row: dict[str, object] = {
                "partition": "preflight_sanity",
                "selection": "outcome_blind_hash_rank",
                "relative_path": relative_path,
                "target": int(targets_cpu[offset]),
            }
            for name in ("standard", "robust"):
                clean_pred = int(batch_outputs[name]["clean_pred"][offset])
                adversarial_pred = int(batch_outputs[name]["adversarial_pred"][offset])
                row.update(
                    {
                        f"{name}_clean_pred": clean_pred,
                        f"{name}_clean_correct": int(clean_pred == row["target"]),
                        f"{name}_pgd_pred": adversarial_pred,
                        f"{name}_pgd_correct": int(adversarial_pred == row["target"]),
                        f"{name}_linf": float(batch_outputs[name]["linf"][offset]),
                    }
                )
            result_rows.append(row)
        total += len(paths)
        print(f"    PGD identity sanity: {total}/{len(rows)}", flush=True)
    elapsed = time.perf_counter() - started
    fieldnames = [
        "partition",
        "selection",
        "relative_path",
        "target",
        "standard_clean_pred",
        "standard_clean_correct",
        "standard_pgd_pred",
        "standard_pgd_correct",
        "standard_linf",
        "robust_clean_pred",
        "robust_clean_correct",
        "robust_pgd_pred",
        "robust_pgd_correct",
        "robust_linf",
    ]
    atomic_csv(output_path, result_rows, fieldnames)
    summary: dict[str, Any] = {}
    for name in ("standard", "robust"):
        clean_correct = sum(int(row[f"{name}_clean_correct"]) for row in result_rows)
        pgd_correct = sum(int(row[f"{name}_pgd_correct"]) for row in result_rows)
        summary[name] = {
            "clean_top1_percent": 100.0 * clean_correct / len(result_rows),
            "pgd_top1_percent": 100.0 * pgd_correct / len(result_rows),
            "maximum_linf": maximum_linf[name],
        }
    advantage = summary["robust"]["pgd_top1_percent"] - summary["standard"][
        "pgd_top1_percent"
    ]
    feasible = max(maximum_linf.values()) <= epsilon + 1e-6
    gate = (
        feasible
        and summary["robust"]["pgd_top1_percent"] >= args.minimum_pgd_top1
        and advantage >= args.minimum_pgd_advantage
    )
    return {
        "purpose": (
            "Checkpoint identity/provenance sanity only; not a RobustBench or paper attack result"
        ),
        "paper_claim_authorized": False,
        "partition": "preflight_sanity",
        "selection_rule": "outcome-blind SHA256 rank before inference",
        "examples": len(result_rows),
        "attack": "untargeted label-aware PGD-Linf",
        "epsilon_linf_pixel": epsilon,
        "epsilon_linf_255": 4.0,
        "alpha_linf_pixel": alpha,
        "alpha_linf_255": args.pgd_alpha_255,
        "steps": args.pgd_steps,
        "random_start": True,
        "seed": args.pgd_seed,
        "preprocessing": transform_report,
        "methods": summary,
        "robust_minus_standard_pgd_top1_pp": advantage,
        "minimum_robust_pgd_top1_percent": args.minimum_pgd_top1,
        "minimum_robust_advantage_pp": args.minimum_pgd_advantage,
        "perturbations_feasible": feasible,
        "gate": "PASS" if gate else "FAIL",
        "elapsed_seconds": elapsed,
        "rows_csv": str(output_path),
        "rows_csv_sha256": sha256_file(output_path),
    }


def pair_parent_predictions(
    r0_dir: Path,
    robust_prediction_path: Path,
) -> dict[str, object]:
    standard_rows = read_csv(r0_dir / "sanity_predictions_resnet50_local.csv")
    robust_rows = read_csv(robust_prediction_path)
    if len(standard_rows) != EXPECTED_SANITY_EXAMPLES or len(robust_rows) != len(
        standard_rows
    ):
        raise AssertionError("Sanity prediction row counts differ")
    standard_by_path = {row["relative_path"]: row for row in standard_rows}
    robust_by_path = {row["relative_path"]: row for row in robust_rows}
    if standard_by_path.keys() != robust_by_path.keys():
        raise AssertionError("R0 and R1 sanity predictions use different examples")
    agreement = 0
    both_correct = 0
    standard_only = 0
    robust_only = 0
    both_wrong = 0
    for path, standard in standard_by_path.items():
        robust = robust_by_path[path]
        if standard["target"] != robust["target"]:
            raise AssertionError(f"Target mismatch for {path}")
        agreement += int(standard["predicted"] == robust["predicted"])
        left = int(standard["top1_correct"])
        right = int(robust["top1_correct"])
        both_correct += int(left == 1 and right == 1)
        standard_only += int(left == 1 and right == 0)
        robust_only += int(left == 0 and right == 1)
        both_wrong += int(left == 0 and right == 0)
    return {
        "examples": len(standard_rows),
        "prediction_agreement_percent": 100.0 * agreement / len(standard_rows),
        "both_correct": both_correct,
        "standard_only_correct": standard_only,
        "robust_only_correct": robust_only,
        "both_wrong": both_wrong,
    }


def frozen_run_manifest(
    args: argparse.Namespace,
    paths: Mapping[str, Path],
    environment: Mapping[str, object],
) -> dict[str, Any]:
    manifest = {
        "protocol_id": PROTOCOL_ID,
        "parent_protocol_id": PARENT_PROTOCOL_ID,
        "stage": "published ImageNet robust ResNet-50 acquisition and identity audit",
        "download_scope": "one model checkpoint only",
        "dataset_download_allowed": False,
        "training_allowed": False,
        "defense_tuning_allowed": False,
        "paper_attack_claim_allowed": False,
        "outcomes_read_from_partitions": ["preflight_sanity"],
        "outcome_blind_partitions": [
            "calibration_dev",
            "attack_pilot",
            "attack_confirmatory",
            "clean_confirmation",
        ],
        "frozen_split_sha256": FROZEN_SPLIT_SHA256,
        "source": source_provenance(),
        "paths": {key: str(value) for key, value in paths.items()},
        "arguments": {
            "download_backend": args.download_backend,
            "no_download": args.no_download,
            "device": args.device,
            "eval_batch_size": args.eval_batch_size,
            "pgd_batch_size": args.pgd_batch_size,
            "num_workers": args.num_workers,
            "gpu_memory_fraction": args.gpu_memory_fraction,
            "pgd_examples": args.pgd_examples,
            "pgd_steps": args.pgd_steps,
            "pgd_alpha_255": args.pgd_alpha_255,
            "pgd_seed": args.pgd_seed,
            "minimum_sanity_top1": args.minimum_sanity_top1,
            "minimum_sanity_top5": args.minimum_sanity_top5,
            "minimum_pgd_top1": args.minimum_pgd_top1,
            "minimum_pgd_advantage": args.minimum_pgd_advantage,
        },
        "environment": dict(environment),
        "script_sha256": script_hashes(paths["project_dir"]),
    }
    manifest_path = paths["output_dir"] / "run_manifest.json"
    if manifest_path.exists():
        existing = read_json(manifest_path)
        if stable_hash(existing) != stable_hash(manifest):
            raise RuntimeError(
                f"Existing R1 run manifest differs: {manifest_path}. "
                "Use a new output directory; do not mix protocols."
            )
    else:
        atomic_json(manifest_path, manifest)
    return manifest


def main() -> None:
    args = parse_args()
    validate_args(args)
    paths = {
        "project_dir": resolved(args.project_dir),
        "data_root": resolved(args.data_root),
        "r0_dir": resolved(args.r0_dir),
        "standard_checkpoint": resolved(args.standard_checkpoint),
        "raw_checkpoint": resolved(args.raw_checkpoint),
        "export_checkpoint": resolved(args.export_checkpoint),
        "output_dir": resolved(args.output_dir),
    }
    paths["output_dir"].mkdir(parents=True, exist_ok=True)
    device = configure_device(args)
    environment = environment_report(device, args)
    print(f"PROTOCOL_ID={PROTOCOL_ID}")
    print(f"sys.executable={sys.executable}")
    print(f"write target={paths['output_dir']}")
    print(
        "scope: one published model download; ImageNet download=False, training=False, "
        "defense tuning=False, paper attack claim=False",
        flush=True,
    )
    frozen_run_manifest(args, paths, environment)

    print("[1/6] Locking to the completed R0 dataset/split/checkpoint audit")
    parent = load_and_validate_parent(paths["r0_dir"], paths["standard_checkpoint"])
    sanity_rows = parent["sanity_rows"]
    print(f"PASS R0 split={FROZEN_SPLIT_SHA256[:16]}..., sanity={len(sanity_rows)}")

    print(f"[2/6] Acquiring official RobustBench model {ROBUSTBENCH_MODEL_ID}")
    acquisition = acquire_official_checkpoint(
        paths["raw_checkpoint"],
        backend=args.download_backend,
        allow_download=not args.no_download,
    )
    print(
        f"PASS raw checkpoint: {acquisition['human_size']}, "
        f"sha256={str(acquisition['sha256'])[:16]}..., "
        f"mode={acquisition['acquisition']}",
        flush=True,
    )

    print("[3/6] Recovering exact torchvision ResNet-50 tensors and exporting a plain state_dict")
    robust_model, robust_load = load_resnet50_checkpoint(paths["raw_checkpoint"])
    raw_state = robust_load.pop("normalized_state")
    robust_fingerprint = str(robust_load["state_dict_fingerprint_sha256"])
    export = export_normalized_checkpoint(
        raw_state,
        robust_fingerprint,
        paths["export_checkpoint"],
    )
    if export["sha256"] == STANDARD_CHECKPOINT_SHA256:
        raise AssertionError("Standard and robust checkpoint file hashes are identical")
    standard_model, standard_load = load_resnet50_checkpoint(paths["standard_checkpoint"])
    standard_load.pop("normalized_state")
    if standard_load["state_dict_fingerprint_sha256"] == robust_fingerprint:
        raise AssertionError("Standard and robust checkpoint tensors are identical")
    print(
        f"PASS export={paths['export_checkpoint']} sha256={str(export['sha256'])[:16]}..."
    )

    print("[4/6] Evaluating only the frozen 1,000-example preflight_sanity partition")
    factory = robust_load["architecture"]
    transform, preprocessing = preprocessing_profile(factory)
    if preprocessing.get("resize_shorter_side") != 256 or preprocessing.get("crop_size") != 224:
        raise AssertionError("Unexpected ImageNet preprocessing geometry")
    if list(preprocessing.get("mean", [])) != list(IMAGENET_MEAN):
        raise AssertionError("Unexpected ImageNet mean")
    if list(preprocessing.get("std", [])) != list(IMAGENET_STD):
        raise AssertionError("Unexpected ImageNet std")
    robust_model = robust_model.to(device).eval()
    prediction_path = paths["output_dir"] / "sanity_predictions_salman2020do_r50.csv"
    sanity = evaluate_sanity(
        "salman2020do_r50_linf4",
        robust_model,
        paths["data_root"],
        sanity_rows,
        transform,
        device,
        args.eval_batch_size,
        args.num_workers,
        prediction_path,
    )
    sanity_gate = (
        sanity["top1_percent"] >= args.minimum_sanity_top1
        and sanity["top5_percent"] >= args.minimum_sanity_top5
    )
    paired = pair_parent_predictions(paths["r0_dir"], prediction_path)
    print(
        f"{'PASS' if sanity_gate else 'FAIL'} robust clean sanity: "
        f"Top1={sanity['top1_percent']:.3f}% Top5={sanity['top5_percent']:.3f}%"
    )

    print("[5/6] Running a 100-example PGD identity check (diagnostic, not a paper result)")
    pgd_rows = select_pgd_rows(sanity_rows, args.pgd_examples)
    pgd_report = evaluate_pgd_identity(
        standard_model.to(device).eval(),
        robust_model,
        paths["data_root"],
        pgd_rows,
        device,
        args,
        paths["output_dir"] / "pgd_identity_sanity.csv",
    )
    print(
        f"{pgd_report['gate']} PGD identity: standard="
        f"{pgd_report['methods']['standard']['pgd_top1_percent']:.3f}% robust="
        f"{pgd_report['methods']['robust']['pgd_top1_percent']:.3f}%"
    )

    print("[6/6] Freezing same-family checkpoint roles and calibration handoff")
    blockers: list[str] = []
    if robust_load["architecture"]["architecture_id"] != EXPECTED_ARCHITECTURE_ID:
        blockers.append("robust architecture mismatch")
    if not sanity_gate:
        blockers.append("robust checkpoint clean sanity gate failed")
    if pgd_report["gate"] != "PASS":
        blockers.append("robust checkpoint PGD identity gate failed")
    if acquisition["sha256"] == STANDARD_CHECKPOINT_SHA256:
        blockers.append("raw robust and standard checkpoint hashes are not distinct")
    ready = not blockers
    role_assignment = {
        "protocol_id": PROTOCOL_ID,
        "parent_protocol_id": PARENT_PROTOCOL_ID,
        "status": "FROZEN" if ready else "BLOCKED",
        "frozen_split_sha256": FROZEN_SPLIT_SHA256,
        "standard_reference": {
            "checkpoint_id": "resnet50_local",
            "path": str(paths["standard_checkpoint"]),
            "sha256": STANDARD_CHECKPOINT_SHA256,
            "state_dict_fingerprint_sha256": standard_load[
                "state_dict_fingerprint_sha256"
            ],
            "architecture_id": EXPECTED_ARCHITECTURE_ID,
            "family": EXPECTED_FAMILY,
            "source_role": "fixed standard reference",
        },
        "robust_anchor": {
            "checkpoint_id": "salman2020do_r50_linf4",
            "path": str(paths["export_checkpoint"]),
            "sha256": export["sha256"],
            "raw_path": str(paths["raw_checkpoint"]),
            "raw_sha256": acquisition["sha256"],
            "state_dict_fingerprint_sha256": robust_fingerprint,
            "architecture_id": EXPECTED_ARCHITECTURE_ID,
            "family": EXPECTED_FAMILY,
            "training_provenance": source_provenance(),
        },
        "same_architecture_family": True,
        "distinct_checkpoint_tensors": True,
        "preprocessing": preprocessing,
        "blockers": blockers,
    }
    role_path = paths["output_dir"] / "checkpoint_role_assignment_frozen_r1.json"
    atomic_json(role_path, role_assignment)
    report = {
        "protocol_id": PROTOCOL_ID,
        "status": "PASS" if ready else "BLOCKED",
        "parent": {
            "protocol_id": PARENT_PROTOCOL_ID,
            "r0_dir": str(paths["r0_dir"]),
            "frozen_split_sha256": FROZEN_SPLIT_SHA256,
            "partitions_read": ["preflight_sanity"],
            "partitions_not_read": [
                "calibration_dev",
                "attack_pilot",
                "attack_confirmatory",
                "clean_confirmation",
            ],
        },
        "source": source_provenance(),
        "raw_checkpoint": acquisition,
        "exported_checkpoint": export,
        "robust_checkpoint_load": robust_load,
        "standard_checkpoint": {
            "path": str(paths["standard_checkpoint"]),
            "sha256": STANDARD_CHECKPOINT_SHA256,
            "state_dict_fingerprint_sha256": standard_load[
                "state_dict_fingerprint_sha256"
            ],
            "architecture": standard_load["architecture"],
        },
        "preprocessing": preprocessing,
        "sanity": {
            **sanity,
            "minimum_top1_percent": args.minimum_sanity_top1,
            "minimum_top5_percent": args.minimum_sanity_top5,
            "gate": "PASS" if sanity_gate else "FAIL",
            "purpose": "load/label/preprocessing sanity only; not model selection",
        },
        "paired_with_r0_standard": paired,
        "pgd_identity": pgd_report,
        "role_assignment_path": str(role_path),
        "role_assignment_sha256": sha256_file(role_path),
        "defense_calibration_ready": ready,
        "blockers": blockers,
    }
    report_path = paths["output_dir"] / "robust_anchor_report.json"
    atomic_json(report_path, report)
    handoff = {
        "protocol_id": PROTOCOL_ID,
        "status": "READY" if ready else "BLOCKED",
        "next_stage": "ImageNet ResNet-50 defense calibration on calibration_dev only",
        "data_root": str(paths["data_root"]),
        "r0_dir": str(paths["r0_dir"]),
        "frozen_split_sha256": FROZEN_SPLIT_SHA256,
        "standard_reference_checkpoint": str(paths["standard_checkpoint"]),
        "standard_reference_sha256": STANDARD_CHECKPOINT_SHA256,
        "robust_anchor_checkpoint": str(paths["export_checkpoint"]),
        "robust_anchor_sha256": export["sha256"],
        "calibration_partition": "calibration_dev",
        "calibration_examples": 2000,
        "forbidden_until_later": [
            "attack_pilot",
            "attack_confirmatory",
            "clean_confirmation",
        ],
        "do_not_transfer_cifar_hyperparameters": True,
        "blockers": blockers,
    }
    atomic_json(paths["output_dir"] / "calibration_handoff_r1.json", handoff)

    del standard_model, robust_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"ROLE_ASSIGNMENT={'FROZEN' if ready else 'BLOCKED'}")
    print(f"DEFENSE_CALIBRATION_READY={'YES' if ready else 'NO'}")
    if blockers:
        print("Blockers:")
        for blocker in blockers:
            print(f"  - {blocker}")
    print(f"Complete: {paths['output_dir']}")


if __name__ == "__main__":
    main()
