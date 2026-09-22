#!/usr/bin/env python3
"""Train-only calibration of the actual Top-k defense on CIFAR-100.

This script deliberately does *not* train or select a neural-network checkpoint.
It imports the already completed CIFAR-100 R2 model code, freezes the R2
standard reference and adversarial anchor, then calibrates the two inference
components used by the paper's actual defense:

1. reference-conditioned input strengthening; and
2. Gaussian input randomization.

Selection is restricted to two disjoint, balanced subsets of the frozen R2
training split.  The frozen 5,000-example R2 validation split is evaluated only
after all hyperparameters have been locked.  The official CIFAR-100 test set and
all attack outcomes are never read.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

import topk_cifar100_defense_calibration_common as common
import topk_cifar100_r2_common as r2_common
import train_resnet20_teacher_top2_adversarial_cifar100_r2 as r2_anchor


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_R2_ROOT = Path(
    "./models/resnet20/"
    "paper_topk_cifar100_external_validity_r2"
)
DEFAULT_OUTPUT = Path(
    "./models/resnet20/"
    "paper_topk_cifar100_actual_defense_calibration_train2k_r1"
)


@dataclass(frozen=True)
class MethodSpec:
    name: str
    model: str
    strength_alpha_px: float = 0.0
    strength_epsilon_px: float = 0.0
    strength_steps: int = 0
    sigma: float = 0.0
    draws: int = 1


class IndexedDataset(Dataset):
    def __init__(self, dataset: Dataset, indices: Sequence[int]):
        self.dataset = dataset
        self.indices = [int(value) for value in indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int):
        index = self.indices[position]
        image, label = self.dataset[index]
        return image, int(label), index


def parse_float_list(text: str) -> tuple[float, ...]:
    values = tuple(float(value.strip()) for value in text.split(",") if value.strip())
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated float")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train-only calibration of the actual CIFAR-100 Top-k defense"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./datasets/cifar-100-python"),
    )
    parser.add_argument("--r2-root", type=Path, default=DEFAULT_R2_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--mc-chunk", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026090401)
    parser.add_argument("--tune-examples", type=int, default=1000)
    parser.add_argument("--confirm-examples", type=int, default=1000)
    parser.add_argument("--validation-examples", type=int, default=5000)
    parser.add_argument("--sigma-tune-draws", type=int, default=50)
    parser.add_argument("--locked-eval-draws", type=int, default=100)
    parser.add_argument("--top1-weight", type=float, default=1.0)
    parser.add_argument("--set-weight", type=float, default=4.0)
    parser.add_argument("--max-clean-drop-pp", type=float, default=2.0)
    parser.add_argument("--locked-mc-tolerance-pp", type=float, default=0.5)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026090402)
    parser.add_argument(
        "--sigma-grid",
        type=parse_float_list,
        default=common.SIGMA_GRID,
        help="Comma-separated raw-pixel Gaussian sigmas; must include 0",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use CUDA autocast for output forwards (strength gradients stay FP32)",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in ("tune_examples", "confirm_examples", "validation_examples"):
        value = int(getattr(args, name))
        if value <= 0 or value % 100:
            raise ValueError(f"--{name.replace('_', '-')} must be positive and divisible by 100")
    if args.tune_examples + args.confirm_examples > 45000:
        raise ValueError("Tune and confirmation subsets exceed the frozen training split")
    if args.validation_examples > 5000:
        raise ValueError("Validation examples cannot exceed the frozen R2 validation split")
    if min(args.batch_size, args.mc_chunk, args.sigma_tune_draws, args.locked_eval_draws) < 1:
        raise ValueError("Batch sizes and MC draw counts must be positive")
    if args.top1_weight < 0 or args.set_weight < 0 or args.top1_weight + args.set_weight <= 0:
        raise ValueError("Strengthening weights must be non-negative with positive sum")
    if args.max_clean_drop_pp < 0 or args.locked_mc_tolerance_pp < 0:
        raise ValueError("Utility tolerances must be non-negative")
    sigmas = tuple(round(float(value), 7) for value in args.sigma_grid)
    if any(value < 0 for value in sigmas) or len(sigmas) != len(set(sigmas)):
        raise ValueError("Sigma grid must contain unique non-negative values")
    if 0.0 not in sigmas:
        raise ValueError("Sigma grid must include 0 so feasibility is guaranteed")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def autocast_context(device: torch.device, enabled: bool):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", enabled=enabled)
    return contextlib.nullcontext()


def stable_noise_seed(base_seed: int, phase: str) -> int:
    value = int(common.object_sha256({"seed": base_seed, "phase": phase})[:15], 16)
    return value % (2**63 - 1)


def load_frozen_split(path: Path) -> dict[str, Any]:
    payload = common.read_json(path)
    if payload.get("protocol_id") != common.PARENT_PROTOCOL_ID:
        raise AssertionError(f"Unexpected parent protocol: {payload.get('protocol_id')}")
    if payload.get("dataset") != "CIFAR-100":
        raise AssertionError("Frozen split is not CIFAR-100")
    train = [int(value) for value in payload["train_indices"]]
    validation = [int(value) for value in payload["validation_indices"]]
    if len(train) != 45000 or len(validation) != 5000:
        raise AssertionError("Frozen R2 split must contain 45,000/5,000 indices")
    if set(train) & set(validation) or len(set(train + validation)) != 50000:
        raise AssertionError("Frozen R2 train/validation indices are not a partition")
    train_hash = common.indices_sha256(train)
    validation_hash = common.indices_sha256(validation)
    if train_hash != payload.get("train_indices_sha256") or train_hash != common.EXPECTED_TRAIN_HASH:
        raise AssertionError(f"Frozen training-index hash mismatch: {train_hash}")
    if (
        validation_hash != payload.get("validation_indices_sha256")
        or validation_hash != common.EXPECTED_VALIDATION_HASH
    ):
        raise AssertionError(f"Frozen validation-index hash mismatch: {validation_hash}")
    return payload


def balanced_disjoint_subsets(
    targets: Sequence[int],
    candidate_indices: Sequence[int],
    tune_examples: int,
    confirm_examples: int,
    seed: int,
) -> tuple[list[int], list[int]]:
    target_array = np.asarray(targets, dtype=np.int64)
    candidates = np.asarray(candidate_indices, dtype=np.int64)
    rng = np.random.default_rng(seed)
    tune_per_class = tune_examples // 100
    confirm_per_class = confirm_examples // 100
    tune: list[int] = []
    confirm: list[int] = []
    for class_id in range(100):
        class_indices = candidates[target_array[candidates] == class_id].copy()
        rng.shuffle(class_indices)
        needed = tune_per_class + confirm_per_class
        if len(class_indices) < needed:
            raise ValueError(f"Class {class_id} has only {len(class_indices)} available samples")
        tune.extend(int(value) for value in class_indices[:tune_per_class])
        confirm.extend(int(value) for value in class_indices[tune_per_class:needed])
    rng.shuffle(tune)
    rng.shuffle(confirm)
    if set(tune) & set(confirm):
        raise AssertionError("Tune and confirmation subsets overlap")
    return tune, confirm


def select_balanced_prefix(
    targets: Sequence[int], candidate_indices: Sequence[int], size: int
) -> list[int]:
    target_array = np.asarray(targets, dtype=np.int64)
    per_class = size // 100
    counts = np.zeros(100, dtype=np.int64)
    selected: list[int] = []
    for raw_index in candidate_indices:
        index = int(raw_index)
        label = int(target_array[index])
        if counts[label] < per_class:
            selected.append(index)
            counts[label] += 1
    if not np.all(counts == per_class):
        raise AssertionError(f"Could not create balanced validation subset: {counts.tolist()}")
    return selected


def make_loader(
    dataset: Dataset,
    indices: Sequence[int],
    args: argparse.Namespace,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        IndexedDataset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )


def freeze_model(model: nn.Module) -> nn.Module:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def load_models(
    standard_weights: Path, anchor_weights: Path, device: torch.device
) -> tuple[nn.Module, nn.Module]:
    standard = r2_anchor.make_model().to(device)
    anchor = r2_anchor.make_model().to(device)
    r2_anchor.load_model_weights(standard, standard_weights, device, "R2 standard reference")
    r2_anchor.load_model_weights(anchor, anchor_weights, device, "R2 adversarial anchor")
    return freeze_model(standard), freeze_model(anchor)


def ranked_top2(logits: torch.Tensor) -> torch.Tensor:
    return logits.topk(2, dim=1, largest=True, sorted=True).indices


def sorted_top2(logits: torch.Tensor) -> torch.Tensor:
    return ranked_top2(logits).sort(dim=1).values


def strengthening_objective(
    logits: torch.Tensor,
    reference_top2: torch.Tensor,
    top1_weight: float,
    set_weight: float,
) -> torch.Tensor:
    selected = logits.gather(1, reference_top2)
    reference_top1 = reference_top2[:, :1]
    not_top1 = torch.ones_like(logits, dtype=torch.bool)
    not_top1.scatter_(1, reference_top1, False)
    strongest_not_top1 = logits.masked_fill(~not_top1, float("-inf")).max(1).values
    outside = torch.ones_like(logits, dtype=torch.bool)
    outside.scatter_(1, reference_top2, False)
    strongest_outside = logits.masked_fill(~outside, float("-inf")).max(1).values
    top1_margin = selected[:, 0] - strongest_not_top1
    set_margin = selected.min(1).values - strongest_outside
    return float(top1_weight) * top1_margin + float(set_weight) * set_margin


def strengthen_inputs(
    clean: torch.Tensor,
    reference_top2: torch.Tensor,
    anchor: nn.Module,
    alpha_px: float,
    epsilon_px: float,
    steps: int,
    top1_weight: float,
    set_weight: float,
) -> torch.Tensor:
    if steps == 0:
        return clean.detach()
    alpha = float(alpha_px) / 255.0
    epsilon = float(epsilon_px) / 255.0
    strengthened = clean.detach().clone()
    for _ in range(int(steps)):
        with torch.enable_grad():
            variable = strengthened.detach().requires_grad_(True)
            logits = anchor(variable)
            objective = strengthening_objective(
                logits, reference_top2, top1_weight, set_weight
            )
            gradient = torch.autograd.grad(objective.sum(), variable, only_inputs=True)[0]
        strengthened = strengthened + alpha * gradient.sign()
        delta = (strengthened - clean).clamp(-epsilon, epsilon)
        strengthened = (clean + delta).clamp(0.0, 1.0).detach()
    return strengthened


def metric_tensors(
    logits: torch.Tensor,
    labels: torch.Tensor,
    reference_ranked: torch.Tensor,
) -> dict[str, torch.Tensor]:
    predictions = logits.argmax(1)
    predicted_set = sorted_top2(logits)
    reference_set = reference_ranked.sort(1).values
    return {
        "top1_accuracy": predictions.eq(labels).float(),
        "reference_top1_match": predictions.eq(reference_ranked[:, 0]).float(),
        "exact_set": predicted_set.eq(reference_set).all(1).float(),
        "label_coverage": predicted_set.eq(labels[:, None]).any(1).float(),
    }


def boundary_margin(logits: torch.Tensor, reference_ranked: torch.Tensor) -> torch.Tensor:
    selected = logits.gather(1, reference_ranked)
    outside = torch.ones_like(logits, dtype=torch.bool)
    outside.scatter_(1, reference_ranked, False)
    strongest_outside = logits.masked_fill(~outside, float("-inf")).max(1).values
    return selected.min(1).values - strongest_outside


def summarize_examples(
    phase: str, method: MethodSpec, examples: Sequence[Mapping[str, Any]], wall_seconds: float
) -> dict[str, Any]:
    def mean_percent(key: str) -> float:
        return 100.0 * float(np.mean([float(row[key]) for row in examples]))

    return {
        "protocol_id": common.PROTOCOL_ID,
        "phase": phase,
        "method": method.name,
        "examples": len(examples),
        "draws": int(method.draws),
        "sigma": float(method.sigma),
        "strength_alpha_px": float(method.strength_alpha_px),
        "strength_epsilon_px": float(method.strength_epsilon_px),
        "strength_steps": int(method.strength_steps),
        "top1_accuracy_percent": mean_percent("top1_accuracy_probability"),
        "reference_top1_match_probability_percent": mean_percent(
            "reference_top1_match_probability"
        ),
        "exact_set_probability_percent": mean_percent("exact_set_probability"),
        "label_coverage_probability_percent": mean_percent("label_coverage_probability"),
        "mean_logits_top1_accuracy_percent": mean_percent("mean_logits_top1_accuracy"),
        "mean_logits_exact_set_percent": mean_percent("mean_logits_exact_set"),
        "mean_boundary_margin": float(
            np.mean([float(row["mean_boundary_margin"]) for row in examples])
        ),
        "wall_seconds": float(wall_seconds),
    }


def evaluate_methods(
    phase: str,
    loader: DataLoader,
    methods: Sequence[MethodSpec],
    standard: nn.Module,
    anchor: nn.Module,
    device: torch.device,
    args: argparse.Namespace,
    noise_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate method probabilities, sharing Gaussian draws when possible."""

    if len({method.name for method in methods}) != len(methods):
        raise ValueError("Method names must be unique within a condition")
    started = time.monotonic()
    rows_by_method: dict[str, list[dict[str, Any]]] = {method.name: [] for method in methods}
    generator = torch.Generator(device=device)
    generator.manual_seed(int(noise_seed))

    for batch_number, (images, labels, indices) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        indices_list = [int(value) for value in indices.tolist()]
        # ``reference_ranked`` is consumed by the differentiable strengthening
        # objective below.  Building it under inference_mode would turn it into
        # an inference tensor, which PyTorch refuses to save for backward in
        # ``gather``/``scatter``.  no_grad keeps the reference frozen while
        # producing an ordinary tensor that is safe in the autograd graph.
        with torch.no_grad(), autocast_context(device, args.amp):
            standard_logits = standard(images).float()
            reference_ranked = ranked_top2(standard_logits)

        strength_cache: dict[tuple[float, float, int], torch.Tensor] = {}
        for method in methods:
            if method.strength_steps > 0:
                key = (
                    float(method.strength_alpha_px),
                    float(method.strength_epsilon_px),
                    int(method.strength_steps),
                )
                if key not in strength_cache:
                    strength_cache[key] = strengthen_inputs(
                        images,
                        reference_ranked,
                        anchor,
                        method.strength_alpha_px,
                        method.strength_epsilon_px,
                        method.strength_steps,
                        args.top1_weight,
                        args.set_weight,
                    )

        count_store: dict[str, dict[str, torch.Tensor]] = {}
        logit_store: dict[str, torch.Tensor] = {}
        for method in methods:
            count_store[method.name] = {
                name: torch.zeros(len(images), device=device, dtype=torch.float64)
                for name in ("top1_accuracy", "reference_top1_match", "exact_set", "label_coverage")
            }
            logit_store[method.name] = torch.zeros(
                len(images), 100, device=device, dtype=torch.float64
            )

        deterministic = [method for method in methods if method.sigma == 0.0]
        for method in deterministic:
            if method.model == "standard" and method.strength_steps == 0:
                logits = standard_logits
            else:
                base = strength_cache.get(
                    (
                        float(method.strength_alpha_px),
                        float(method.strength_epsilon_px),
                        int(method.strength_steps),
                    ),
                    images,
                )
                model = standard if method.model == "standard" else anchor
                with torch.inference_mode(), autocast_context(device, args.amp):
                    logits = model(base).float()
            metrics = metric_tensors(logits, labels, reference_ranked)
            for name, values in metrics.items():
                count_store[method.name][name] += values.double()
            logit_store[method.name] += logits.double()

        stochastic_groups: dict[tuple[float, int], list[MethodSpec]] = {}
        for method in methods:
            if method.sigma > 0:
                stochastic_groups.setdefault((float(method.sigma), int(method.draws)), []).append(method)
        for (sigma, draws), group in sorted(stochastic_groups.items()):
            completed = 0
            while completed < draws:
                chunk = min(args.mc_chunk, draws - completed)
                noise = torch.randn(
                    (chunk,) + tuple(images.shape),
                    device=device,
                    dtype=images.dtype,
                    generator=generator,
                )
                for method in group:
                    base = strength_cache.get(
                        (
                            float(method.strength_alpha_px),
                            float(method.strength_epsilon_px),
                            int(method.strength_steps),
                        ),
                        images,
                    )
                    noisy = (base.unsqueeze(0) + sigma * noise).clamp(0.0, 1.0)
                    flat = noisy.flatten(0, 1)
                    model = standard if method.model == "standard" else anchor
                    with torch.inference_mode(), autocast_context(device, args.amp):
                        logits = model(flat).float().view(chunk, len(images), 100)
                    predictions = logits.argmax(2)
                    sets = logits.topk(2, dim=2, largest=True, sorted=False).indices.sort(2).values
                    reference_set = reference_ranked.sort(1).values
                    count_store[method.name]["top1_accuracy"] += (
                        predictions.eq(labels[None, :]).sum(0).double()
                    )
                    count_store[method.name]["reference_top1_match"] += (
                        predictions.eq(reference_ranked[None, :, 0]).sum(0).double()
                    )
                    count_store[method.name]["exact_set"] += (
                        sets.eq(reference_set[None, :, :]).all(2).sum(0).double()
                    )
                    count_store[method.name]["label_coverage"] += (
                        sets.eq(labels[None, :, None]).any(2).sum(0).double()
                    )
                    logit_store[method.name] += logits.double().sum(0)
                completed += chunk

        reference_ranked_cpu = reference_ranked.detach().cpu().numpy()
        labels_cpu = labels.detach().cpu().numpy()
        for method in methods:
            effective_draws = int(method.draws if method.sigma > 0 else 1)
            mean_logits = (logit_store[method.name] / effective_draws).float()
            mean_metrics = metric_tensors(mean_logits, labels, reference_ranked)
            margins = boundary_margin(mean_logits, reference_ranked)
            counts = count_store[method.name]
            for position, sample_index in enumerate(indices_list):
                ref = reference_ranked_cpu[position]
                rows_by_method[method.name].append(
                    {
                        "protocol_id": common.PROTOCOL_ID,
                        "phase": phase,
                        "method": method.name,
                        "sample_index": sample_index,
                        "label": int(labels_cpu[position]),
                        "reference_top1": int(ref[0]),
                        "reference_set": f"{min(ref)}|{max(ref)}",
                        "draws": effective_draws,
                        "sigma": float(method.sigma),
                        "strength_alpha_px": float(method.strength_alpha_px),
                        "strength_epsilon_px": float(method.strength_epsilon_px),
                        "strength_steps": int(method.strength_steps),
                        "top1_accuracy_probability": float(
                            counts["top1_accuracy"][position].item() / effective_draws
                        ),
                        "reference_top1_match_probability": float(
                            counts["reference_top1_match"][position].item() / effective_draws
                        ),
                        "exact_set_probability": float(
                            counts["exact_set"][position].item() / effective_draws
                        ),
                        "label_coverage_probability": float(
                            counts["label_coverage"][position].item() / effective_draws
                        ),
                        "mean_logits_top1_accuracy": float(
                            mean_metrics["top1_accuracy"][position].item()
                        ),
                        "mean_logits_exact_set": float(mean_metrics["exact_set"][position].item()),
                        "mean_boundary_margin": float(margins[position].item()),
                    }
                )
        if batch_number % 10 == 0 or batch_number == len(loader):
            print(
                f"    {phase}: batch {batch_number}/{len(loader)}, "
                f"methods={','.join(method.name for method in methods)}",
                flush=True,
            )

    wall_seconds = time.monotonic() - started
    examples = [row for method in methods for row in rows_by_method[method.name]]
    summaries = [
        summarize_examples(phase, method, rows_by_method[method.name], wall_seconds)
        for method in methods
    ]
    return summaries, examples


def partial_path(output_dir: Path, condition_id: str) -> Path:
    return output_dir / "partials" / f"{condition_id}.json"


def load_partial(
    output_dir: Path,
    condition_id: str,
    config_hash: str,
    expected_methods: Sequence[str],
    expected_examples_per_method: int,
) -> dict[str, Any] | None:
    path = partial_path(output_dir, condition_id)
    if not path.is_file():
        return None
    payload = common.read_json(path)
    if payload.get("protocol_id") != common.PROTOCOL_ID:
        raise AssertionError(f"Partial protocol mismatch: {path}")
    if payload.get("config_hash") != config_hash or payload.get("condition_id") != condition_id:
        raise AssertionError(f"Partial belongs to a different frozen run: {path}")
    summaries = payload.get("summaries", [])
    examples = payload.get("examples", [])
    if {row.get("method") for row in summaries} != set(expected_methods):
        raise AssertionError(f"Partial method mismatch: {path}")
    for method in expected_methods:
        method_rows = [row for row in examples if row.get("method") == method]
        if len(method_rows) != expected_examples_per_method:
            raise AssertionError(f"Partial row count mismatch for {method}: {path}")
        if len({int(row["sample_index"]) for row in method_rows}) != len(method_rows):
            raise AssertionError(f"Duplicate sample rows for {method}: {path}")
    print(f"RESUME {condition_id}", flush=True)
    return payload


def run_condition(
    output_dir: Path,
    condition_id: str,
    config_hash: str,
    phase: str,
    loader: DataLoader,
    methods: Sequence[MethodSpec],
    standard: nn.Module,
    anchor: nn.Module,
    device: torch.device,
    args: argparse.Namespace,
    noise_seed: int,
    expected_examples: int,
) -> dict[str, Any]:
    cached = load_partial(
        output_dir,
        condition_id,
        config_hash,
        [method.name for method in methods],
        expected_examples,
    )
    if cached is not None:
        return cached
    print(f"RUN {condition_id}", flush=True)
    summaries, examples = evaluate_methods(
        phase, loader, methods, standard, anchor, device, args, noise_seed
    )
    payload = {
        "protocol_id": common.PROTOCOL_ID,
        "config_hash": config_hash,
        "condition_id": condition_id,
        "phase": phase,
        "methods": [asdict(method) for method in methods],
        "summaries": summaries,
        "examples": examples,
    }
    common.atomic_json(partial_path(output_dir, condition_id), payload)
    print(
        "PASS " + condition_id + ": " + ", ".join(
            f"{row['method']} Top1={row['top1_accuracy_percent']:.3f}% "
            f"Exact={row['exact_set_probability_percent']:.3f}%"
            for row in summaries
        ),
        flush=True,
    )
    return payload


def summary_for(payload: Mapping[str, Any], method: str) -> dict[str, Any]:
    matches = [dict(row) for row in payload["summaries"] if row["method"] == method]
    if len(matches) != 1:
        raise AssertionError(f"Expected one summary for {method}; got {len(matches)}")
    return matches[0]


def write_or_validate_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.is_file():
        existing = common.read_json(path)
        if existing != payload:
            raise AssertionError(f"Frozen file differs from this invocation: {path}")
    else:
        common.atomic_json(path, payload)


def bootstrap_validation(
    validation_examples: Sequence[Mapping[str, Any]],
    resamples: int,
    seed: int,
) -> list[dict[str, Any]]:
    if resamples <= 0:
        return []
    metrics = {
        "top1": "top1_accuracy_probability",
        "exact_set": "exact_set_probability",
    }
    comparisons = (
        ("noise_only", "robust"),
        ("full", "robust"),
        ("full", "noise_only"),
        ("full_sigma0", "robust"),
    )
    methods = sorted({str(row["method"]) for row in validation_examples})
    by_method: dict[str, dict[int, Mapping[str, Any]]] = {
        method: {
            int(row["sample_index"]): row
            for row in validation_examples
            if row["method"] == method
        }
        for method in methods
    }
    common_indices = sorted(next(iter(by_method.values())))
    if any(sorted(rows) != common_indices for rows in by_method.values()):
        raise AssertionError("Validation methods do not share identical sample indices")
    columns: list[tuple[str, str, str, np.ndarray]] = []
    for left, right in comparisons:
        for metric_name, key in metrics.items():
            delta = np.asarray(
                [
                    float(by_method[left][index][key]) - float(by_method[right][index][key])
                    for index in common_indices
                ],
                dtype=np.float64,
            )
            columns.append((left, right, metric_name, delta))
    matrix = np.stack([column[3] for column in columns], axis=1)
    rng = np.random.default_rng(seed)
    boot = np.empty((resamples, len(columns)), dtype=np.float64)
    cursor = 0
    while cursor < resamples:
        count = min(200, resamples - cursor)
        sampled_indices = rng.integers(0, len(common_indices), size=(count, len(common_indices)))
        boot[cursor:cursor + count] = matrix[sampled_indices].mean(axis=1)
        cursor += count
    output: list[dict[str, Any]] = []
    for position, (left, right, metric_name, delta) in enumerate(columns):
        point = 100.0 * float(delta.mean())
        low, high = 100.0 * np.quantile(boot[:, position], [0.025, 0.975])
        status = "positive" if low > 0 else "negative" if high < 0 else "statistically_indistinguishable"
        output.append(
            {
                "comparison": f"{left}_minus_{right}_{metric_name}",
                "left": left,
                "right": right,
                "metric": metric_name,
                "difference_pp": point,
                "ci95_low_pp": float(low),
                "ci95_high_pp": float(high),
                "status": status,
                "examples": len(common_indices),
                "resamples": resamples,
                "seed": seed,
            }
        )
    return output


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    seed_everything(args.seed)

    r2_root = args.r2_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    standard_weights = r2_root / "standard" / "resnet20_standard_best_validation_weights.pt"
    anchor_weights = r2_root / "teacher_top2_at" / "resnet20_teacher_top2_best_composite_weights.pt"
    split_path = r2_root / "frozen_split.json"
    pipeline_manifest_path = r2_root / "pipeline_manifest.json"
    for path in (standard_weights, anchor_weights, split_path, pipeline_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required R2 artifact is missing: {path}")

    standard_sha256 = common.file_sha256(standard_weights)
    anchor_sha256 = common.file_sha256(anchor_weights)
    if standard_sha256 != common.EXPECTED_STANDARD_SHA256:
        raise AssertionError(f"R2 standard checkpoint hash mismatch: {standard_sha256}")
    if anchor_sha256 != common.EXPECTED_ANCHOR_SHA256:
        raise AssertionError(f"R2 anchor checkpoint hash mismatch: {anchor_sha256}")

    parent_manifest = common.read_json(pipeline_manifest_path)
    if parent_manifest.get("protocol_id") != common.PARENT_PROTOCOL_ID:
        raise AssertionError("R2 pipeline manifest belongs to another protocol")
    expected_parent_scripts = {
        "topk_cifar100_r2_common.py": common.EXPECTED_R2_COMMON_SHA256,
        "train_resnet20_teacher_top2_adversarial_cifar100_r2.py": (
            common.EXPECTED_R2_ANCHOR_SOURCE_SHA256
        ),
    }
    for filename, expected_hash in expected_parent_scripts.items():
        if parent_manifest.get("script_sha256", {}).get(filename) != expected_hash:
            raise AssertionError(f"R2 manifest has an unexpected hash for {filename}")
        if common.file_sha256(PROJECT_DIR / filename) != expected_hash:
            raise AssertionError(f"Local R2 source differs from the frozen manifest: {filename}")

    split = load_frozen_split(split_path)
    dataset_root = r2_anchor.resolve_torchvision_root(args.data_dir)
    dataset = torchvision.datasets.CIFAR100(
        root=str(dataset_root), train=True, download=False, transform=transforms.ToTensor()
    )
    if len(dataset) != 50000 or len(dataset.targets) != 50000:
        raise AssertionError("Unexpected CIFAR-100 training-set size")
    if r2_common.labels_sha256(dataset.targets) != split.get("labels_sha256"):
        raise AssertionError("Local CIFAR-100 labels differ from the frozen R2 split")
    r2_train, r2_validation, _ = r2_common.load_frozen_split(
        split_path, dataset.targets, seed=9103, val_size=5000
    )
    if r2_train != split["train_indices"] or r2_validation != split["validation_indices"]:
        raise AssertionError("R2 common loader did not reproduce the frozen split")
    tune_indices, confirm_indices = balanced_disjoint_subsets(
        dataset.targets,
        split["train_indices"],
        args.tune_examples,
        args.confirm_examples,
        args.seed,
    )
    validation_indices = select_balanced_prefix(
        dataset.targets, split["validation_indices"], args.validation_examples
    )
    if (set(tune_indices) | set(confirm_indices)) & set(validation_indices):
        raise AssertionError("Calibration subsets overlap frozen validation")

    subset_manifest = {
        "protocol_id": common.PROTOCOL_ID,
        "parent_protocol_id": common.PARENT_PROTOCOL_ID,
        "selection_data": "frozen R2 training split only",
        "confirmation_used_for_retuning": False,
        "validation_used_for_retuning": False,
        "official_test_read": False,
        "attack_outcomes_read": False,
        "seed": args.seed,
        "tune_indices": tune_indices,
        "tune_indices_sha256": common.indices_sha256(tune_indices),
        "confirm_indices": confirm_indices,
        "confirm_indices_sha256": common.indices_sha256(confirm_indices),
        "validation_indices": validation_indices,
        "validation_indices_sha256": common.indices_sha256(validation_indices),
        "parent_train_indices_sha256": split["train_indices_sha256"],
        "parent_validation_indices_sha256": split["validation_indices_sha256"],
    }
    write_or_validate_json(output_dir / "subset_manifest.json", subset_manifest)

    evaluator_path = Path(__file__).resolve()
    dependency_paths = {
        "evaluator": evaluator_path,
        "calibration_common": PROJECT_DIR / "topk_cifar100_defense_calibration_common.py",
        "r2_common": PROJECT_DIR / "topk_cifar100_r2_common.py",
        "r2_anchor": PROJECT_DIR / "train_resnet20_teacher_top2_adversarial_cifar100_r2.py",
    }
    for name, path in dependency_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Required code dependency {name} is missing: {path}")
    config = {
        "protocol_id": common.PROTOCOL_ID,
        "parent_protocol_id": common.PARENT_PROTOCOL_ID,
        "dataset": "CIFAR-100",
        "architecture": "ResNet-20",
        "primary_k": 2,
        "data_dir": str(args.data_dir.expanduser().resolve()),
        "r2_root": str(r2_root),
        "output_dir": str(output_dir),
        "device": str(device),
        "batch_size": args.batch_size,
        "mc_chunk": args.mc_chunk,
        "num_workers": args.num_workers,
        "amp": bool(args.amp),
        "seed": args.seed,
        "tune_examples": args.tune_examples,
        "confirm_examples": args.confirm_examples,
        "validation_examples": args.validation_examples,
        "sigma_tune_draws": args.sigma_tune_draws,
        "locked_eval_draws": args.locked_eval_draws,
        "top1_weight": args.top1_weight,
        "set_weight": args.set_weight,
        "max_clean_drop_pp": args.max_clean_drop_pp,
        "locked_mc_tolerance_pp": args.locked_mc_tolerance_pp,
        "strength_epsilon_px": 8.0,
        "strength_grid": list(common.STRENGTH_GRID),
        "sigma_grid": sorted(round(float(value), 7) for value in args.sigma_grid),
        "bootstrap_resamples": args.bootstrap_resamples,
        "bootstrap_seed": args.bootstrap_seed,
        "selection_policy": {
            "strength": "maximum tune Exact-Set2 subject to <=2 pp Top-1 drop vs robust; ties by Top-1, fewer steps, lower alpha",
            "sigma": "largest predeclared sigma jointly satisfying <=2 pp Top-1 and Exact-Set2 drop for Full and Noise-only",
            "confirmation": "diagnostic gate only; no retuning",
            "validation": "locked evaluation only; no retuning",
        },
        "checkpoints": {
            "standard": {"path": str(standard_weights), "sha256": standard_sha256},
            "anchor": {"path": str(anchor_weights), "sha256": anchor_sha256},
        },
        "split": {
            "path": str(split_path),
            "sha256": common.file_sha256(split_path),
            "train_indices_sha256": split["train_indices_sha256"],
            "validation_indices_sha256": split["validation_indices_sha256"],
        },
        "parent_pipeline_manifest": {
            "path": str(pipeline_manifest_path),
            "sha256": common.file_sha256(pipeline_manifest_path),
        },
        "script_sha256": {
            name: common.file_sha256(path) for name, path in dependency_paths.items()
        },
        "offline": True,
        "downloads_allowed": False,
        "official_test_read": False,
        "attack_outcomes_read": False,
    }
    config_hash = common.object_sha256(config)
    config["config_hash"] = config_hash
    write_or_validate_json(output_dir / "run_config.json", config)

    standard, anchor = load_models(standard_weights, anchor_weights, device)
    tune_loader = make_loader(dataset, tune_indices, args, device)
    confirm_loader = make_loader(dataset, confirm_indices, args, device)
    validation_loader = make_loader(dataset, validation_indices, args, device)

    calibration_examples: list[dict[str, Any]] = []
    baseline_payload = run_condition(
        output_dir,
        "tune_robust",
        config_hash,
        "tune",
        tune_loader,
        [MethodSpec("robust", "anchor")],
        standard,
        anchor,
        device,
        args,
        stable_noise_seed(args.seed, "tune_robust"),
        args.tune_examples,
    )
    robust_tune = summary_for(baseline_payload, "robust")
    calibration_examples.extend(baseline_payload["examples"])

    strength_rows: list[dict[str, Any]] = [robust_tune]
    strength_payloads: list[dict[str, Any]] = []
    for candidate in common.STRENGTH_GRID:
        condition_id = common.strength_candidate_id(candidate)
        method = MethodSpec(
            "full_sigma0",
            "anchor",
            strength_alpha_px=float(candidate["alpha_px"]),
            strength_epsilon_px=float(candidate["epsilon_px"]),
            strength_steps=int(candidate["steps"]),
        )
        payload = run_condition(
            output_dir,
            condition_id,
            config_hash,
            "tune",
            tune_loader,
            [method],
            standard,
            anchor,
            device,
            args,
            stable_noise_seed(args.seed, condition_id),
            args.tune_examples,
        )
        strength_payloads.append(payload)
        strength_rows.append(summary_for(payload, "full_sigma0"))
        calibration_examples.extend(payload["examples"])

    selected_strength, annotated_strength = common.select_strength_candidate(
        strength_rows[1:], robust_tune, args.max_clean_drop_pp
    )
    for row in strength_rows:
        row["selected"] = bool(
            row["method"] == "full_sigma0"
            and float(row["strength_alpha_px"]) == float(selected_strength["strength_alpha_px"])
            and int(row["strength_steps"]) == int(selected_strength["strength_steps"])
        )
        if row["method"] == "robust":
            row["feasible"] = True
            row["top1_floor_percent"] = float(robust_tune["top1_accuracy_percent"]) - args.max_clean_drop_pp
        else:
            match = next(
                item for item in annotated_strength
                if float(item["strength_alpha_px"]) == float(row["strength_alpha_px"])
                and int(item["strength_steps"]) == int(row["strength_steps"])
            )
            row["feasible"] = match["feasible"]
            row["top1_floor_percent"] = match["top1_floor_percent"]

    selected_alpha = float(selected_strength["strength_alpha_px"])
    selected_epsilon = float(selected_strength["strength_epsilon_px"])
    selected_steps = int(selected_strength["strength_steps"])
    sigma_rows: list[dict[str, Any]] = []
    sigma_payloads: list[dict[str, Any]] = []
    for sigma in sorted(round(float(value), 7) for value in args.sigma_grid):
        condition_id = common.sigma_candidate_id(sigma)
        draws = args.sigma_tune_draws if sigma > 0 else 1
        methods = [
            MethodSpec("noise_only", "anchor", sigma=sigma, draws=draws),
            MethodSpec(
                "full",
                "anchor",
                strength_alpha_px=selected_alpha,
                strength_epsilon_px=selected_epsilon,
                strength_steps=selected_steps,
                sigma=sigma,
                draws=draws,
            ),
        ]
        payload = run_condition(
            output_dir,
            condition_id,
            config_hash,
            "tune",
            tune_loader,
            methods,
            standard,
            anchor,
            device,
            args,
            stable_noise_seed(args.seed, "sigma_grid_common_random_numbers"),
            args.tune_examples,
        )
        sigma_payloads.append(payload)
        sigma_rows.extend(dict(row) for row in payload["summaries"])
        calibration_examples.extend(payload["examples"])

    selected_sigma, annotated_sigma = common.select_sigma_candidate(
        sigma_rows, robust_tune, selected_strength, args.max_clean_drop_pp
    )
    sigma0_noise = next(
        row for row in sigma_rows
        if float(row["sigma"]) == 0.0 and row["method"] == "noise_only"
    )
    sigma0_full = next(
        row for row in sigma_rows
        if float(row["sigma"]) == 0.0 and row["method"] == "full"
    )
    for metric in common.PRIMARY_METRICS:
        if abs(float(sigma0_noise[metric]) - float(robust_tune[metric])) > 1e-10:
            raise AssertionError(f"sigma=0 noise-only failed to reproduce robust for {metric}")
        if abs(float(sigma0_full[metric]) - float(selected_strength[metric])) > 1e-10:
            raise AssertionError(f"sigma=0 Full failed to reproduce strengthening for {metric}")
    selected_sigma_value = float(selected_sigma["sigma"])
    for row in sigma_rows:
        annotation = next(
            item for item in annotated_sigma if float(item["sigma"]) == float(row["sigma"])
        )
        row["jointly_feasible"] = annotation["feasible"]
        row["selected"] = float(row["sigma"]) == selected_sigma_value
        for key, value in annotation.items():
            if key.endswith("_drop_pp"):
                row[key] = value

    selection = {
        "protocol_id": common.PROTOCOL_ID,
        "config_hash": config_hash,
        "selection_split": "tune subset of frozen R2 training indices",
        "confirmation_used_for_retuning": False,
        "validation_used_for_retuning": False,
        "official_test_read": False,
        "attack_outcomes_read": False,
        "strength": {
            "alpha_px": selected_alpha,
            "epsilon_px": selected_epsilon,
            "steps": selected_steps,
            "top1_weight": args.top1_weight,
            "set_weight": args.set_weight,
            "selected_tune_summary": selected_strength,
        },
        "randomization": {
            "sigma": selected_sigma_value,
            "positive_sigma_selected": selected_sigma_value > 0,
            "selected_tune_constraints": selected_sigma,
        },
    }
    selection["selection_hash"] = common.object_sha256(selection)
    write_or_validate_json(output_dir / "calibration_selection.json", selection)

    locked_methods = [
        MethodSpec("standard", "standard"),
        MethodSpec("robust", "anchor"),
        MethodSpec(
            "full_sigma0",
            "anchor",
            strength_alpha_px=selected_alpha,
            strength_epsilon_px=selected_epsilon,
            strength_steps=selected_steps,
        ),
        MethodSpec(
            "noise_only",
            "anchor",
            sigma=selected_sigma_value,
            draws=args.locked_eval_draws if selected_sigma_value > 0 else 1,
        ),
        MethodSpec(
            "full",
            "anchor",
            strength_alpha_px=selected_alpha,
            strength_epsilon_px=selected_epsilon,
            strength_steps=selected_steps,
            sigma=selected_sigma_value,
            draws=args.locked_eval_draws if selected_sigma_value > 0 else 1,
        ),
    ]
    locked_payloads: list[dict[str, Any]] = []
    for phase, loader, example_count in (
        ("confirmation", confirm_loader, args.confirm_examples),
        ("validation", validation_loader, args.validation_examples),
    ):
        payload = run_condition(
            output_dir,
            f"locked_{phase}",
            config_hash,
            phase,
            loader,
            locked_methods,
            standard,
            anchor,
            device,
            args,
            stable_noise_seed(args.seed, f"locked_{phase}_common_random_numbers"),
            example_count,
        )
        locked_payloads.append(payload)

    locked_summaries = [dict(row) for payload in locked_payloads for row in payload["summaries"]]
    locked_examples = [dict(row) for payload in locked_payloads for row in payload["examples"]]
    gates = {
        phase: common.gate_locked_phase(
            [row for row in locked_summaries if row["phase"] == phase],
            args.max_clean_drop_pp,
            args.locked_mc_tolerance_pp,
        )
        for phase in ("confirmation", "validation")
    }
    bootstrap_rows = bootstrap_validation(
        [row for row in locked_examples if row["phase"] == "validation"],
        args.bootstrap_resamples,
        args.bootstrap_seed,
    )

    common.atomic_csv(output_dir / "strength_calibration_summary.csv", strength_rows)
    common.atomic_csv(output_dir / "sigma_calibration_summary.csv", sigma_rows)
    common.atomic_csv(output_dir / "locked_clean_summary.csv", locked_summaries)
    common.atomic_csv(output_dir / "locked_clean_per_example.csv", locked_examples)
    common.atomic_csv(output_dir / "calibration_per_example.csv", calibration_examples)
    if bootstrap_rows:
        common.atomic_csv(output_dir / "validation_paired_bootstrap.csv", bootstrap_rows)

    attack_gate = bool(
        selected_sigma_value > 0
        and gates["confirmation"]["pass"]
        and gates["validation"]["pass"]
    )
    final_summary = {
        "protocol_id": common.PROTOCOL_ID,
        "config_hash": config_hash,
        "selection_hash": selection["selection_hash"],
        "status": "PASS",
        "attack_stage_ready": attack_gate,
        "attack_stage_blockers": [
            message
            for condition, message in (
                (selected_sigma_value <= 0, "No positive Gaussian sigma met the frozen utility constraints"),
                (not gates["confirmation"]["pass"], "Locked confirmation utility gate failed"),
                (not gates["validation"]["pass"], "Locked validation utility gate failed"),
            )
            if condition
        ],
        "selection": selection,
        "locked_gates": gates,
        "locked_summaries": locked_summaries,
        "validation_paired_bootstrap": bootstrap_rows,
        "data_governance": {
            "hyperparameter_selection": "train tune subset only",
            "confirmation_retuning": False,
            "validation_retuning": False,
            "official_test_read": False,
            "attack_outcomes_read": False,
        },
    }
    common.atomic_json(output_dir / "final_summary.json", final_summary)

    print("\nSelected CIFAR-100 defense parameters", flush=True)
    print(
        f"  strength: alpha={selected_alpha:.3f}/255, epsilon={selected_epsilon:.3f}/255, "
        f"steps={selected_steps}, weights={args.top1_weight:g}:{args.set_weight:g}",
        flush=True,
    )
    print(f"  sigma: {selected_sigma_value:.7f}", flush=True)
    for phase in ("confirmation", "validation"):
        print(f"  {phase} utility gate: {'PASS' if gates[phase]['pass'] else 'FAIL'}", flush=True)
    print(
        "ATTACK_STAGE_READY=" + ("YES" if attack_gate else "NO"),
        flush=True,
    )
    print(f"Complete: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
