#!/usr/bin/env python3
"""Offline ImageNet/ResNet-50/ViT-B16 preflight for the Top-k SaTML project.

This stage is deliberately non-adaptive: it freezes all validation partitions
before reading model outcomes, audits the two local checkpoints, evaluates only
the dedicated preflight-sanity partition, and benchmarks safe 224x224 batches.
It never downloads, trains, tunes, attacks, or assigns model roles by guesswork.
"""

from __future__ import annotations

import argparse
import gc
import platform
import statistics
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode

from topk_imagenet_preflight_common import (
    EXPECTED_TRAIN_EXAMPLES,
    EXPECTED_VAL_EXAMPLES,
    NUM_CLASSES,
    PROTOCOL_ID,
    SPLIT_FIELDNAMES,
    SPLIT_SPEC,
    atomic_csv,
    atomic_json,
    build_frozen_split,
    human_bytes,
    is_image_path,
    partition_rows,
    read_csv,
    read_json,
    same_rows,
    sha256_file,
    stable_hash,
    validate_frozen_split,
)


DEFAULT_DATA_ROOT = Path("./datasets/imagenet")
DEFAULT_RESNET50 = Path(
    "./models/resnet50/imagenet/"
    "resnet50_imagenet.pth"
)
DEFAULT_VIT_B16 = Path(
    "./models/vit/imagenet/"
    "vit_b16_imagenet.pth"
)
DEFAULT_OUTPUT = Path(
    "./models/imagenet_topk/"
    "paper_topk_imagenet_preflight_r0"
)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SCRIPT_NAMES = [
    "topk_imagenet_preflight_common.py",
    "preflight_topk_imagenet_r0.py",
    "audit_topk_imagenet_preflight_r0.py",
    "test_topk_imagenet_preflight_r0.py",
    "run_topk_imagenet_preflight_r0.py",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict offline ImageNet Top-k preflight; no download, tuning, or attack"
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--resnet50-checkpoint", type=Path, default=DEFAULT_RESNET50)
    parser.add_argument("--vit-b16-checkpoint", type=Path, default=DEFAULT_VIT_B16)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split-seed", type=int, default=9102026)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.45)
    parser.add_argument(
        "--memory-batch-sizes",
        default="1,2,4,8,16,24,32",
        help="Ascending 224x224 batches used for forward/backward capacity audit",
    )
    parser.add_argument("--skip-memory-sweep", action="store_true")
    parser.add_argument(
        "--allow-unsafe-checkpoint-load",
        action="store_true",
        help="Allow weights_only=False only for locally trusted legacy checkpoints",
    )
    parser.add_argument("--minimum-sanity-top1", type=float, default=20.0)
    parser.add_argument("--minimum-sanity-top5", type=float, default=40.0)
    return parser.parse_args()


def parse_batch_sizes(value: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("--memory-batch-sizes must be comma-separated integers") from exc
    if not values or any(item <= 0 for item in values):
        raise ValueError("Memory batch sizes must be positive")
    if values != sorted(set(values)):
        raise ValueError("Memory batch sizes must be unique and ascending")
    return values


def resolve_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "data_root": args.data_root.expanduser().resolve(),
        "train": (args.data_root.expanduser().resolve() / "train"),
        "val": (args.data_root.expanduser().resolve() / "val"),
        "devkit": (args.data_root.expanduser().resolve() / "devkit"),
        "resnet50": args.resnet50_checkpoint.expanduser().resolve(),
        "vit_b16": args.vit_b16_checkpoint.expanduser().resolve(),
        "output": args.output_dir.expanduser().resolve(),
    }


def image_files_under(path: Path) -> list[Path]:
    return sorted(
        item for item in path.rglob("*") if item.is_file() and is_image_path(item)
    )


def top_level_class_inventory(path: Path) -> tuple[dict[str, int], int, int]:
    if not path.is_dir():
        raise FileNotFoundError(path)
    counts: dict[str, int] = {}
    symlinks = 0
    for directory in sorted(item for item in path.iterdir() if item.is_dir()):
        files = image_files_under(directory)
        if files:
            counts[directory.name] = len(files)
            symlinks += sum(int(item.is_symlink()) for item in files)
    return counts, sum(counts.values()), symlinks


def find_devkit_file(devkit: Path, filename: str) -> Path:
    matches = sorted(item for item in devkit.rglob(filename) if item.is_file())
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one {filename} under {devkit}; found {len(matches)}"
        )
    return matches[0]


def load_devkit_id_to_wnid(meta_path: Path) -> dict[int, str]:
    try:
        import numpy as np
        from scipy.io import loadmat
    except ImportError as exc:
        raise RuntimeError(
            "Flat ImageNet val layout requires scipy to read devkit/meta.mat. "
            "No data will be rearranged automatically."
        ) from exc

    payload = loadmat(str(meta_path), squeeze_me=True, struct_as_record=False)
    raw = payload.get("synsets")
    if raw is None:
        raise KeyError(f"synsets is missing from {meta_path}")
    mapping: dict[int, str] = {}
    for item in np.atleast_1d(raw).ravel():
        try:
            class_id = int(item.ILSVRC2012_ID)
            wnid = str(item.WNID)
            children = int(item.num_children)
        except AttributeError:
            values = list(np.atleast_1d(item).ravel())
            if len(values) < 5:
                continue
            class_id = int(np.asarray(values[0]).item())
            wnid = str(np.asarray(values[1]).item())
            children = int(np.asarray(values[4]).item())
        if children == 0 and wnid.startswith("n"):
            mapping[class_id] = wnid
    if len(mapping) != NUM_CLASSES:
        raise AssertionError(
            f"Expected 1000 leaf synsets in {meta_path}; recovered {len(mapping)}"
        )
    return mapping


def enumerate_validation(
    data_root: Path, val_dir: Path, devkit: Path
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    immediate_dirs = sorted(item for item in val_dir.iterdir() if item.is_dir())
    structured: dict[str, list[str]] = {}
    for directory in immediate_dirs:
        files = image_files_under(directory)
        if files:
            structured[directory.name] = [
                str(item.relative_to(data_root)) for item in files
            ]
    if structured:
        if len(structured) != NUM_CLASSES:
            raise AssertionError(
                f"Structured val has {len(structured)} non-empty class directories, expected 1000"
            )
        return structured, {
            "layout": "class_directories",
            "ground_truth_path": None,
            "meta_path": None,
        }

    flat_files = sorted(
        item for item in val_dir.iterdir() if item.is_file() and is_image_path(item)
    )
    if len(flat_files) != EXPECTED_VAL_EXAMPLES:
        raise AssertionError(
            f"Flat val has {len(flat_files)} images, expected {EXPECTED_VAL_EXAMPLES}"
        )
    ground_truth_path = find_devkit_file(
        devkit, "ILSVRC2012_validation_ground_truth.txt"
    )
    meta_path = find_devkit_file(devkit, "meta.mat")
    labels = [
        int(line.strip())
        for line in ground_truth_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(labels) != EXPECTED_VAL_EXAMPLES:
        raise AssertionError(
            f"Ground truth has {len(labels)} labels, expected {EXPECTED_VAL_EXAMPLES}"
        )
    id_to_wnid = load_devkit_id_to_wnid(meta_path)
    samples: dict[str, list[str]] = defaultdict(list)
    for path, class_id in zip(flat_files, labels):
        if class_id not in id_to_wnid:
            raise AssertionError(f"Unknown devkit class id {class_id}")
        samples[id_to_wnid[class_id]].append(str(path.relative_to(data_root)))
    return dict(samples), {
        "layout": "flat_with_devkit_ground_truth",
        "ground_truth_path": str(ground_truth_path),
        "ground_truth_sha256": sha256_file(ground_truth_path),
        "meta_path": str(meta_path),
        "meta_sha256": sha256_file(meta_path),
    }


def verify_decoding(data_root: Path, rows: Sequence[Mapping[str, object]]) -> dict[str, Any]:
    selected = sorted(rows, key=lambda row: int(row["target"]))[:100]
    widths: list[int] = []
    heights: list[int] = []
    modes: Counter[str] = Counter()
    for row in selected:
        path = data_root / str(row["relative_path"])
        with Image.open(path) as image:
            image.load()
            rgb = image.convert("RGB")
            widths.append(rgb.width)
            heights.append(rgb.height)
            modes[rgb.mode] += 1
    return {
        "decoded": len(selected),
        "minimum_width": min(widths),
        "minimum_height": min(heights),
        "maximum_width": max(widths),
        "maximum_height": max(heights),
        "modes": dict(sorted(modes.items())),
    }


def audit_and_freeze_dataset(
    paths: dict[str, Path], split_seed: int
) -> tuple[dict[str, Any], list[dict[str, object]]]:
    for name in ("data_root", "train", "val", "devkit"):
        if not paths[name].is_dir():
            raise FileNotFoundError(f"ImageNet {name} directory is missing: {paths[name]}")

    train_counts, train_total, train_symlinks = top_level_class_inventory(paths["train"])
    if len(train_counts) != NUM_CLASSES:
        raise AssertionError(f"ImageNet train has {len(train_counts)} classes, expected 1000")
    if train_total != EXPECTED_TRAIN_EXAMPLES:
        raise AssertionError(
            f"ImageNet train has {train_total} images, expected {EXPECTED_TRAIN_EXAMPLES}"
        )

    samples_by_class, layout = enumerate_validation(
        paths["data_root"], paths["val"], paths["devkit"]
    )
    val_total = sum(len(items) for items in samples_by_class.values())
    if val_total != EXPECTED_VAL_EXAMPLES:
        raise AssertionError(f"ImageNet val has {val_total} images, expected 50000")
    if set(samples_by_class) != set(train_counts):
        missing_train = sorted(set(samples_by_class) - set(train_counts))[:10]
        missing_val = sorted(set(train_counts) - set(samples_by_class))[:10]
        raise AssertionError(
            f"Train/val class mapping mismatch: val-only={missing_train}, train-only={missing_val}"
        )

    rows = build_frozen_split(samples_by_class, split_seed)
    split_summary = validate_frozen_split(rows)
    output = paths["output"]
    split_csv = output / "frozen_imagenet_val_split.csv"
    if split_csv.exists():
        existing = read_csv(split_csv)
        if not same_rows(existing, rows):
            raise RuntimeError(
                f"Refusing to change an existing frozen split: {split_csv}. "
                "Use a new output directory for a new protocol."
            )
    else:
        atomic_csv(split_csv, rows, SPLIT_FIELDNAMES)

    split_manifest = {
        "protocol_id": PROTOCOL_ID,
        "created_before_model_evaluation": True,
        "outcome_blind": True,
        "split_seed": split_seed,
        "split_spec_per_class": dict(SPLIT_SPEC),
        "split_csv": str(split_csv),
        **split_summary,
    }
    manifest_path = output / "frozen_imagenet_val_split_manifest.json"
    if manifest_path.exists():
        existing = read_json(manifest_path)
        immutable = ("protocol_id", "split_seed", "split_spec_per_class", "split_sha256")
        for key in immutable:
            if existing.get(key) != split_manifest.get(key):
                raise RuntimeError(f"Frozen split manifest conflict at {key}")
    else:
        atomic_json(manifest_path, split_manifest)

    decode = verify_decoding(paths["data_root"], partition_rows(rows, "preflight_sanity"))
    dataset_report = {
        "status": "PASS",
        "dataset": "ImageNet-1K / ILSVRC2012",
        "data_root": str(paths["data_root"]),
        "train_dir": str(paths["train"]),
        "val_dir": str(paths["val"]),
        "devkit_dir": str(paths["devkit"]),
        "train_classes": len(train_counts),
        "train_examples": train_total,
        "train_min_per_class": min(train_counts.values()),
        "train_max_per_class": max(train_counts.values()),
        "train_symlinked_images": train_symlinks,
        "val_classes": len(samples_by_class),
        "val_examples": val_total,
        "val_per_class_values": sorted({len(items) for items in samples_by_class.values()}),
        "val_layout": layout,
        "decode_audit": decode,
        "class_names_sha256": stable_hash(sorted(samples_by_class)),
        "split": split_manifest,
    }
    return dataset_report, rows


def torch_load_local(path: Path, allow_unsafe: bool) -> tuple[Any, str]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True), "weights_only=True"
    except TypeError:
        # PyTorch releases before weights_only support.  This preserves backward
        # compatibility for the user's local checkpoint and is reported.
        return torch.load(path, map_location="cpu"), "legacy torch.load"
    except Exception as exc:
        if not allow_unsafe:
            raise RuntimeError(
                f"Safe weights-only load failed for {path}: {exc}. If and only if this "
                "is a trusted local checkpoint, rerun with --allow-unsafe-checkpoint-load."
            ) from exc
        return (
            torch.load(path, map_location="cpu", weights_only=False),
            "weights_only=False (explicitly authorized)",
        )


def tensor_mapping_score(value: object) -> int:
    if not isinstance(value, Mapping):
        return -1
    return sum(int(torch.is_tensor(item)) for item in value.values())


def extract_state_dict(payload: object) -> tuple[dict[str, torch.Tensor], str]:
    if isinstance(payload, torch.nn.Module):
        return dict(payload.state_dict()), "serialized_module.state_dict"
    if not isinstance(payload, Mapping):
        raise TypeError(f"Checkpoint payload is {type(payload).__name__}, not mapping-like")

    candidates: list[tuple[int, str, Mapping[str, object]]] = []

    def visit(value: object, route: str, depth: int) -> None:
        if depth > 3 or not isinstance(value, Mapping):
            return
        score = tensor_mapping_score(value)
        if score > 0:
            candidates.append((score, route, value))
        priority = (
            "state_dict",
            "model_state_dict",
            "model",
            "net",
            "network",
            "module",
            "backbone",
            "ema",
        )
        for key in priority:
            child = value.get(key)
            if isinstance(child, Mapping):
                visit(child, f"{route}.{key}" if route else key, depth + 1)

    visit(payload, "root", 0)
    if not candidates:
        raise ValueError("No tensor state_dict could be recovered from checkpoint")
    _, route, selected = max(candidates, key=lambda item: item[0])
    state = {str(key): value for key, value in selected.items() if torch.is_tensor(value)}
    if not state:
        raise ValueError("Recovered state_dict is empty")
    return state, route


PREFIX_CHAINS = (
    "",
    "module.",
    "model.",
    "net.",
    "network.",
    "backbone.",
    "module.model.",
    "module.net.",
    "model.module.",
    "model.backbone.",
    "module.backbone.",
)


def state_variants(state: Mapping[str, torch.Tensor]) -> list[tuple[str, dict[str, torch.Tensor]]]:
    variants: list[tuple[str, dict[str, torch.Tensor]]] = []
    seen: set[tuple[str, ...]] = set()
    for prefix in PREFIX_CHAINS:
        candidate = {
            (key[len(prefix) :] if prefix and key.startswith(prefix) else key): value
            for key, value in state.items()
        }
        signature = tuple(sorted(candidate))
        if signature not in seen:
            seen.add(signature)
            variants.append((prefix or "identity", candidate))
    return variants


def model_factories() -> list[dict[str, Any]]:
    factories: list[dict[str, Any]] = []
    if hasattr(torchvision.models, "resnet50"):
        factories.append({
            "architecture_id": "torchvision_resnet50",
            "family": "resnet50",
            "factory": lambda: torchvision.models.resnet50(weights=None),
            "classifier_keys": {"fc.weight", "fc.bias"},
            "interpolation": "bilinear",
        })
    if hasattr(torchvision.models, "vit_b_16"):
        factories.append({
            "architecture_id": "torchvision_vit_b_16",
            "family": "vit_b16",
            "factory": lambda: torchvision.models.vit_b_16(weights=None),
            "classifier_keys": {"heads.head.weight", "heads.head.bias"},
            "interpolation": "bicubic",
        })
    try:
        import timm
    except ImportError:
        timm = None
    if timm is not None:
        factories.append(
            {
                "architecture_id": "timm_vit_base_patch16_224",
                "family": "vit_b16",
                "factory": lambda: timm.create_model(
                    "vit_base_patch16_224", pretrained=False, num_classes=NUM_CLASSES
                ),
                "classifier_keys": {"head.weight", "head.bias"},
                "interpolation": "bicubic",
            }
        )
    return factories


def match_state_to_model(
    state: Mapping[str, torch.Tensor], model: torch.nn.Module, classifier_keys: set[str]
) -> dict[str, Any]:
    expected = model.state_dict()
    parameter_keys = set(dict(model.named_parameters()))
    matched: dict[str, torch.Tensor] = {}
    mismatched: list[dict[str, object]] = []
    unexpected: list[str] = []
    for key, value in state.items():
        target = expected.get(key)
        if target is None:
            unexpected.append(key)
        elif tuple(value.shape) == tuple(target.shape):
            matched[key] = value
        else:
            mismatched.append(
                {
                    "key": key,
                    "checkpoint_shape": list(value.shape),
                    "model_shape": list(target.shape),
                }
            )

    expected_numel = sum(value.numel() for value in expected.values())
    parameter_numel = sum(value.numel() for value in model.parameters())
    matched_numel = sum(value.numel() for value in matched.values())
    matched_parameter_numel = sum(
        value.numel() for key, value in matched.items() if key in parameter_keys
    )
    missing = sorted(set(expected) - set(matched))
    return {
        "matched": matched,
        "matched_tensors": len(matched),
        "expected_tensors": len(expected),
        "tensor_coverage": len(matched) / max(1, len(expected)),
        "matched_numel": matched_numel,
        "expected_numel": expected_numel,
        "numel_coverage": matched_numel / max(1, expected_numel),
        "parameter_numel_coverage": matched_parameter_numel / max(1, parameter_numel),
        "classifier_keys_expected": sorted(classifier_keys),
        "classifier_keys_matched": sorted(classifier_keys.intersection(matched)),
        "classifier_complete": classifier_keys.issubset(matched),
        "missing_keys": missing,
        "unexpected_keys": sorted(unexpected),
        "shape_mismatches": mismatched,
    }


def select_architecture(
    state: Mapping[str, torch.Tensor]
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    trials: list[tuple[tuple[float, float, int], dict[str, Any], dict[str, Any], torch.nn.Module]] = []
    factory_errors: list[str] = []
    for factory in model_factories():
        try:
            model = factory["factory"]()
        except Exception as exc:
            factory_errors.append(
                f"{factory['architecture_id']}: {type(exc).__name__}: {exc}"
            )
            continue
        for prefix, variant in state_variants(state):
            match = match_state_to_model(variant, model, factory["classifier_keys"])
            score = (
                float(match["parameter_numel_coverage"]),
                float(match["numel_coverage"]),
                int(match["matched_tensors"]),
            )
            public = {key: value for key, value in match.items() if key != "matched"}
            trials.append((score, factory, {"prefix_transform": prefix, **public}, model))
        # Only one model instance per factory is retained in trials; clear the
        # repeated references after selection below.
    if not trials:
        raise RuntimeError(f"No local model factory is available: {factory_errors}")
    _, best_factory, best_public, best_model = max(trials, key=lambda item: item[0])
    best_variant = dict(state)
    selected_prefix = str(best_public["prefix_transform"])
    if selected_prefix != "identity":
        best_variant = {
            (
                key[len(selected_prefix) :]
                if key.startswith(selected_prefix)
                else key
            ): value
            for key, value in state.items()
        }
    full_match = match_state_to_model(
        best_variant, best_model, best_factory["classifier_keys"]
    )
    load_ready = (
        full_match["parameter_numel_coverage"] >= 0.995
        and full_match["numel_coverage"] >= 0.995
        and full_match["classifier_complete"]
    )
    public_match = {key: value for key, value in full_match.items() if key != "matched"}
    public_match["prefix_transform"] = selected_prefix
    public_match["load_ready"] = load_ready
    trial_summary = sorted(
        [
            {
                "architecture_id": item[1]["architecture_id"],
                **item[2],
            }
            for item in trials
        ],
        key=lambda row: (
            float(row["parameter_numel_coverage"]),
            float(row["numel_coverage"]),
        ),
        reverse=True,
    )[:6]
    if not load_ready:
        raise RuntimeError(
            "No architecture reached 99.5% parameter/numel coverage with a complete "
            f"1000-class head. Best={best_factory['architecture_id']} "
            f"parameter={full_match['parameter_numel_coverage']:.6f}, "
            f"numel={full_match['numel_coverage']:.6f}. "
            f"Top trials={trial_summary}"
        )
    best_model.load_state_dict(full_match["matched"], strict=False)
    factory_public = {
        key: value for key, value in best_factory.items() if key != "factory"
    }
    factory_public["classifier_keys"] = sorted(factory_public["classifier_keys"])
    return best_model, factory_public, {
        **public_match,
        "top_architecture_trials": trial_summary,
    }


def preprocessing_profile(factory: Mapping[str, object]) -> tuple[Callable, dict[str, Any]]:
    interpolation_name = str(factory["interpolation"])
    interpolation = (
        InterpolationMode.BICUBIC
        if interpolation_name == "bicubic"
        else InterpolationMode.BILINEAR
    )
    transform = transforms.Compose(
        [
            transforms.Resize(256, interpolation=interpolation),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    profile = {
        "selection_rule": "architecture_default; not selected by accuracy",
        "resize_shorter_side": 256,
        "crop_size": 224,
        "interpolation": interpolation_name,
        "mean": list(IMAGENET_MEAN),
        "std": list(IMAGENET_STD),
    }
    return transform, profile


class ManifestImageDataset(Dataset):
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


@torch.inference_mode()
def evaluate_sanity(
    checkpoint_id: str,
    model: torch.nn.Module,
    data_root: Path,
    rows: Sequence[Mapping[str, object]],
    transform: Callable,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    output_path: Path,
) -> dict[str, Any]:
    loader = DataLoader(
        ManifestImageDataset(data_root, rows, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    model.eval()
    prediction_rows: list[dict[str, object]] = []
    total = 0
    top1_correct = 0
    top5_correct = 0
    started = time.perf_counter()
    for batch_index, (images, targets, relative_paths) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        if logits.ndim != 2 or logits.shape[1] != NUM_CLASSES:
            raise AssertionError(f"Expected [N,1000] logits, got {tuple(logits.shape)}")
        if not torch.isfinite(logits).all():
            raise AssertionError(f"Non-finite logits from {checkpoint_id}")
        top5 = logits.topk(5, dim=1).indices
        predicted = top5[:, 0]
        top1_mask = predicted.eq(targets)
        top5_mask = top5.eq(targets[:, None]).any(dim=1)
        total += targets.numel()
        top1_correct += int(top1_mask.sum().item())
        top5_correct += int(top5_mask.sum().item())
        logits_cpu = logits.detach().float().cpu()
        targets_cpu = targets.cpu()
        top5_cpu = top5.cpu()
        for offset, relative_path in enumerate(relative_paths):
            vector = logits_cpu[offset]
            prediction_rows.append(
                {
                    "checkpoint_id": checkpoint_id,
                    "partition": "preflight_sanity",
                    "relative_path": relative_path,
                    "target": int(targets_cpu[offset]),
                    "predicted": int(top5_cpu[offset, 0]),
                    "top1_correct": int(top1_mask[offset].item()),
                    "top5_correct": int(top5_mask[offset].item()),
                    "logit_min": float(vector.min().item()),
                    "logit_max": float(vector.max().item()),
                    "logit_mean": float(vector.mean().item()),
                    "logit_std": float(vector.std(unbiased=False).item()),
                }
            )
        if batch_index % 5 == 0 or total == len(rows):
            print(f"    {checkpoint_id}: sanity {total}/{len(rows)}", flush=True)
    elapsed = time.perf_counter() - started
    fields = [
        "checkpoint_id",
        "partition",
        "relative_path",
        "target",
        "predicted",
        "top1_correct",
        "top5_correct",
        "logit_min",
        "logit_max",
        "logit_mean",
        "logit_std",
    ]
    atomic_csv(output_path, prediction_rows, fields)
    return {
        "partition": "preflight_sanity",
        "examples": total,
        "top1_correct": top1_correct,
        "top5_correct": top5_correct,
        "top1_percent": 100.0 * top1_correct / total,
        "top5_percent": 100.0 * top5_correct / total,
        "elapsed_seconds": elapsed,
        "examples_per_second": total / elapsed,
        "prediction_csv": str(output_path),
        "prediction_csv_sha256": sha256_file(output_path),
    }


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_memory(
    model: torch.nn.Module,
    device: torch.device,
    batch_sizes: Sequence[int],
) -> dict[str, Any]:
    if device.type != "cuda":
        return {"status": "SKIPPED_CPU", "input_size": [3, 224, 224], "rows": []}
    model.eval()
    rows: list[dict[str, object]] = []
    inference_open = True
    backward_open = True
    for batch_size in batch_sizes:
        row: dict[str, object] = {"batch_size": batch_size}
        if inference_open:
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                tensor = torch.randn(batch_size, 3, 224, 224, device=device)
                with torch.inference_mode():
                    _ = model(tensor)
                    synchronize(device)
                    times: list[float] = []
                    for _ in range(3):
                        start = time.perf_counter()
                        output = model(tensor)
                        synchronize(device)
                        times.append(time.perf_counter() - start)
                if output.ndim != 2 or output.shape[1] != NUM_CLASSES:
                    raise AssertionError(f"Unexpected benchmark output {tuple(output.shape)}")
                row.update(
                    {
                        "forward_status": "PASS",
                        "forward_median_ms": 1000.0 * statistics.median(times),
                        "forward_peak_mib": torch.cuda.max_memory_allocated(device) / (1024**2),
                    }
                )
                del tensor, output
            except torch.cuda.OutOfMemoryError:
                row["forward_status"] = "OOM"
                inference_open = False
                backward_open = False
                torch.cuda.empty_cache()
        else:
            row["forward_status"] = "SKIPPED_AFTER_OOM"

        if row["forward_status"] == "PASS" and backward_open:
            try:
                torch.cuda.empty_cache()
                model.zero_grad(set_to_none=True)
                torch.cuda.reset_peak_memory_stats(device)
                tensor = torch.randn(
                    batch_size, 3, 224, 224, device=device, requires_grad=True
                )
                start = time.perf_counter()
                output = model(tensor)
                output.float().mean().backward()
                synchronize(device)
                row.update(
                    {
                        "backward_status": "PASS",
                        "backward_ms": 1000.0 * (time.perf_counter() - start),
                        "backward_peak_mib": torch.cuda.max_memory_allocated(device) / (1024**2),
                    }
                )
                del tensor, output
                model.zero_grad(set_to_none=True)
            except torch.cuda.OutOfMemoryError:
                row["backward_status"] = "OOM"
                backward_open = False
                model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
        elif row["forward_status"] == "PASS":
            row["backward_status"] = "SKIPPED_AFTER_OOM"
        else:
            row["backward_status"] = "NOT_RUN"
        rows.append(row)
        print(
            f"    memory batch={batch_size}: forward={row['forward_status']} "
            f"backward={row['backward_status']}",
            flush=True,
        )
        gc.collect()
        torch.cuda.empty_cache()

    safe_forward = [
        int(row["batch_size"]) for row in rows if row["forward_status"] == "PASS"
    ]
    safe_backward = [
        int(row["batch_size"]) for row in rows if row["backward_status"] == "PASS"
    ]
    return {
        "status": "PASS",
        "input_size": [3, 224, 224],
        "rows": rows,
        "maximum_tested_forward_batch": max(safe_forward, default=0),
        "maximum_tested_backward_batch": max(safe_backward, default=0),
        "note": "These are tested-safe points under the configured process memory fraction, not absolute maxima.",
    }


def audit_checkpoint(
    checkpoint_id: str,
    path: Path,
    args: argparse.Namespace,
    data_root: Path,
    sanity_rows: Sequence[Mapping[str, object]],
    device: torch.device,
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "checkpoint_id": checkpoint_id,
        "path": str(path),
        "exists": path.is_file(),
    }
    if not path.is_file():
        return {**base, "status": "BLOCKED", "error": "checkpoint file is missing"}
    base.update(
        {
            "bytes": path.stat().st_size,
            "human_size": human_bytes(path.stat().st_size),
            "sha256": sha256_file(path),
        }
    )
    model: torch.nn.Module | None = None
    try:
        payload, load_mode = torch_load_local(path, args.allow_unsafe_checkpoint_load)
        state, state_route = extract_state_dict(payload)
        model, factory, match = select_architecture(state)
        transform, transform_profile = preprocessing_profile(factory)
        model = model.to(device)
        output_path = args.output_dir.expanduser().resolve() / f"sanity_predictions_{checkpoint_id}.csv"
        sanity = evaluate_sanity(
            checkpoint_id,
            model,
            data_root,
            sanity_rows,
            transform,
            device,
            args.eval_batch_size,
            args.num_workers,
            output_path,
        )
        sanity_pass = (
            sanity["top1_percent"] >= args.minimum_sanity_top1
            and sanity["top5_percent"] >= args.minimum_sanity_top5
        )
        memory_error = None
        try:
            memory = (
                {"status": "SKIPPED_BY_FLAG", "rows": []}
                if args.skip_memory_sweep
                else benchmark_memory(model, device, parse_batch_sizes(args.memory_batch_sizes))
            )
        except Exception as exc:
            memory_error = f"{type(exc).__name__}: {exc}"
            memory = {"status": "ERROR", "rows": [], "error": memory_error}
        checkpoint_pass = sanity_pass and memory_error is None
        return {
            **base,
            "status": "PASS" if checkpoint_pass else "BLOCKED",
            "load_mode": load_mode,
            "state_dict_route": state_route,
            "state_dict_tensor_count": len(state),
            "architecture": factory,
            "state_dict_match": match,
            "preprocessing": transform_profile,
            "sanity": {
                **sanity,
                "minimum_top1_percent": args.minimum_sanity_top1,
                "minimum_top5_percent": args.minimum_sanity_top5,
                "sanity_gate": "PASS" if sanity_pass else "FAIL",
                "purpose": "load/label/normalization sanity only; not model selection",
            },
            "memory_latency_224": memory,
            "provenance_role": "UNRESOLVED",
            "error": (
                memory_error
                if memory_error is not None
                else (None if sanity_pass else "clean sanity floor failed")
            ),
        }
    except Exception as exc:
        return {
            **base,
            "status": "BLOCKED",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    finally:
        if model is not None:
            del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def role_template(checkpoints: Sequence[Mapping[str, object]]) -> dict[str, Any]:
    inventory = []
    for item in checkpoints:
        inventory.append(
            {
                "checkpoint_id": item["checkpoint_id"],
                "sha256": item.get("sha256"),
                "detected_family": (
                    item.get("architecture", {}).get("family")
                    if isinstance(item.get("architecture"), Mapping)
                    else None
                ),
                "audit_status": item.get("status"),
            }
        )
    return {
        "protocol_id": PROTOCOL_ID,
        "status": "UNRESOLVED",
        "standard_reference_checkpoint_id": None,
        "standard_reference_checkpoint_sha256": None,
        "robust_anchor_checkpoint_id": None,
        "robust_anchor_checkpoint_sha256": None,
        "robust_anchor_training_provenance": None,
        "require_distinct_checkpoints": True,
        "require_same_architecture_family": True,
        "checkpoint_inventory": inventory,
        "instructions": [
            "Do not infer roles from filenames or clean accuracy.",
            "The robust anchor must have documented robustness/training provenance.",
            "Standard and anchor checkpoints must be distinct and from one architecture family.",
            "Set status to FROZEN only after the provenance is verified.",
        ],
    }


def validate_roles(
    assignment: Mapping[str, object], checkpoints: Sequence[Mapping[str, object]]
) -> tuple[str, list[str]]:
    blockers: list[str] = []
    if assignment.get("protocol_id") != PROTOCOL_ID:
        blockers.append("role assignment protocol_id mismatch")
    if assignment.get("status") != "FROZEN":
        blockers.append("checkpoint roles are not explicitly FROZEN")
    standard_id = assignment.get("standard_reference_checkpoint_id")
    anchor_id = assignment.get("robust_anchor_checkpoint_id")
    inventory = {str(item["checkpoint_id"]): item for item in checkpoints}
    if standard_id not in inventory:
        blockers.append("standard reference checkpoint id is missing or unknown")
    if anchor_id not in inventory:
        blockers.append("robust anchor checkpoint id is missing or unknown")
    if standard_id == anchor_id and standard_id is not None:
        blockers.append("standard reference and robust anchor must be distinct checkpoints")
    if standard_id in inventory and inventory[str(standard_id)].get("status") != "PASS":
        blockers.append("standard reference checkpoint did not pass preflight")
    if anchor_id in inventory and inventory[str(anchor_id)].get("status") != "PASS":
        blockers.append("robust anchor checkpoint did not pass preflight")
    standard_hash = assignment.get("standard_reference_checkpoint_sha256")
    anchor_hash = assignment.get("robust_anchor_checkpoint_sha256")
    if standard_id in inventory and standard_hash != inventory[str(standard_id)].get("sha256"):
        blockers.append("standard reference checkpoint SHA256 is absent or stale")
    if anchor_id in inventory and anchor_hash != inventory[str(anchor_id)].get("sha256"):
        blockers.append("robust anchor checkpoint SHA256 is absent or stale")
    if standard_id in inventory and anchor_id in inventory:
        left = inventory[str(standard_id)].get("architecture", {})
        right = inventory[str(anchor_id)].get("architecture", {})
        left_family = left.get("family") if isinstance(left, Mapping) else None
        right_family = right.get("family") if isinstance(right, Mapping) else None
        if left_family != right_family:
            blockers.append(
                f"architecture-family mismatch: standard={left_family}, anchor={right_family}"
            )
    provenance = assignment.get("robust_anchor_training_provenance")
    if not isinstance(provenance, str) or len(provenance.strip()) < 20:
        blockers.append("robust anchor training provenance is absent or too vague")
    return ("PASS" if not blockers else "BLOCKED"), blockers


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
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": str(device),
        "gpu": gpu,
        "gpu_memory_fraction": args.gpu_memory_fraction,
    }


def script_hashes(project_dir: Path) -> dict[str, str]:
    hashes = {}
    for name in SCRIPT_NAMES:
        path = project_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Required R0 source is missing: {path}")
        hashes[name] = sha256_file(path)
    return hashes


def freeze_run_manifest(
    args: argparse.Namespace,
    paths: Mapping[str, Path],
    dataset_report: Mapping[str, object],
) -> dict[str, Any]:
    project_dir = Path(__file__).resolve().parent
    manifest = {
        "protocol_id": PROTOCOL_ID,
        "stage": "offline ImageNet dataset/checkpoint/role/memory preflight",
        "offline": True,
        "downloads_allowed": False,
        "training_allowed": False,
        "tuning_allowed": False,
        "attack_allowed": False,
        "split_seed": args.split_seed,
        "split_spec_per_class": dict(SPLIT_SPEC),
        "data_root": str(paths["data_root"]),
        "resnet50_checkpoint": str(paths["resnet50"]),
        "vit_b16_checkpoint": str(paths["vit_b16"]),
        "output_dir": str(paths["output"]),
        "frozen_split_sha256": dataset_report["split"]["split_sha256"],
        "script_sha256": script_hashes(project_dir),
        "arguments": {
            "device": args.device,
            "eval_batch_size": args.eval_batch_size,
            "num_workers": args.num_workers,
            "gpu_memory_fraction": args.gpu_memory_fraction,
            "memory_batch_sizes": parse_batch_sizes(args.memory_batch_sizes),
            "skip_memory_sweep": args.skip_memory_sweep,
            "minimum_sanity_top1": args.minimum_sanity_top1,
            "minimum_sanity_top5": args.minimum_sanity_top5,
            "allow_unsafe_checkpoint_load": args.allow_unsafe_checkpoint_load,
        },
    }
    manifest_path = paths["output"] / "run_manifest.json"
    if manifest_path.exists():
        existing = read_json(manifest_path)
        if stable_hash(existing) != stable_hash(manifest):
            raise RuntimeError(
                f"Existing run manifest differs: {manifest_path}. Use a new output directory; "
                "do not mix protocols."
            )
    else:
        atomic_json(manifest_path, manifest)
    return manifest


def main() -> None:
    args = parse_args()
    if args.eval_batch_size <= 0 or args.num_workers < 0:
        raise ValueError("Invalid batch size or worker count")
    if not 0.0 < args.gpu_memory_fraction <= 1.0:
        raise ValueError("--gpu-memory-fraction must be in (0,1]")
    for value in (args.minimum_sanity_top1, args.minimum_sanity_top5):
        if not 0.0 <= value <= 100.0:
            raise ValueError("Sanity thresholds must be in [0,100]")
    parse_batch_sizes(args.memory_batch_sizes)

    paths = resolve_paths(args)
    paths["output"].mkdir(parents=True, exist_ok=True)
    print(f"PROTOCOL_ID={PROTOCOL_ID}")
    print(f"sys.executable={sys.executable}")
    print(f"write target={paths['output']}")
    print("offline=True downloads=False training=False tuning=False attack=False")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(
            args.gpu_memory_fraction, device.index or 0
        )
    environment = environment_report(device, args)
    print(
        f"torch={environment['torch']} torchvision={environment['torchvision']} "
        f"cuda={environment['cuda_runtime']} gpu={environment['gpu']}",
        flush=True,
    )

    print("[1/4] Auditing local ImageNet and freezing all partitions before outcomes")
    dataset_report, frozen_rows = audit_and_freeze_dataset(paths, args.split_seed)
    freeze_run_manifest(args, paths, dataset_report)
    sanity_rows = partition_rows(frozen_rows, "preflight_sanity")
    print(
        f"PASS dataset: train={dataset_report['train_examples']}, "
        f"val={dataset_report['val_examples']}, sanity={len(sanity_rows)}"
    )

    print("[2/4] Auditing ResNet-50 checkpoint")
    resnet_report = audit_checkpoint(
        "resnet50_local",
        paths["resnet50"],
        args,
        paths["data_root"],
        sanity_rows,
        device,
    )
    print(
        f"{resnet_report['status']} resnet50_local: "
        f"{resnet_report.get('sanity', {}).get('top1_percent', 'NA')} Top-1, "
        f"architecture={resnet_report.get('architecture', {}).get('architecture_id', 'UNRESOLVED')}"
    )

    print("[3/4] Auditing ViT-B/16 checkpoint")
    vit_report = audit_checkpoint(
        "vit_b16_local",
        paths["vit_b16"],
        args,
        paths["data_root"],
        sanity_rows,
        device,
    )
    print(
        f"{vit_report['status']} vit_b16_local: "
        f"{vit_report.get('sanity', {}).get('top1_percent', 'NA')} Top-1, "
        f"architecture={vit_report.get('architecture', {}).get('architecture_id', 'UNRESOLVED')}"
    )

    print("[4/4] Enforcing explicit standard/robust role assignment")
    checkpoint_reports = [resnet_report, vit_report]
    role_path = paths["output"] / "checkpoint_role_assignment.json"
    if not role_path.exists():
        atomic_json(role_path, role_template(checkpoint_reports))
    assignment = read_json(role_path)
    role_status, role_blockers = validate_roles(assignment, checkpoint_reports)

    checkpoint_blockers = [
        f"{item['checkpoint_id']}: {item.get('error', 'sanity/load gate failed')}"
        for item in checkpoint_reports
        if item["status"] != "PASS"
    ]
    blockers = checkpoint_blockers + role_blockers
    calibration_ready = not blockers
    report = {
        "protocol_id": PROTOCOL_ID,
        "status": "PASS_READY" if calibration_ready else "PASS_WITH_BLOCKERS",
        "preflight_complete": True,
        "offline": True,
        "downloads_attempted": False,
        "outcomes_read_from_partitions": ["preflight_sanity"],
        "outcome_blind_partitions": [
            name for name in SPLIT_SPEC if name != "preflight_sanity"
        ],
        "dataset": dataset_report,
        "checkpoints": checkpoint_reports,
        "role_assignment_path": str(role_path),
        "role_assignment_status": role_status,
        "calibration_ready": calibration_ready,
        "blockers": blockers,
        "environment": environment,
    }
    report_path = paths["output"] / "preflight_report.json"
    atomic_json(report_path, report)

    print("IMAGENET_PREFLIGHT_COMPLETE=YES")
    print(f"ROLE_ASSIGNMENT={role_status}")
    print(f"DEFENSE_CALIBRATION_READY={'YES' if calibration_ready else 'NO'}")
    if blockers:
        print("Blockers:")
        for blocker in blockers:
            print(f"  - {blocker}")
    print(f"Complete: {paths['output']}")


if __name__ == "__main__":
    main()
