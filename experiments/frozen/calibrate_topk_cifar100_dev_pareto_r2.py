#!/usr/bin/env python3
"""CI-aware clean Pareto calibration for the actual CIFAR-100 Top-k defense.

This is a deliberately declared redesign after the R1 clean-transfer gate failed.
The complete frozen 5,000-example R2 validation split is now treated as a
development set.  It selects between global and disagreement-conditioned input
strengthening, then selects Gaussian sigma using paired percentile-bootstrap
non-inferiority constraints.  This stage loads only CIFAR-100 ``train=True``;
it never loads the official test set or any attack result.
"""

from __future__ import annotations

import argparse
import contextlib
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

import calibrate_topk_cifar100_actual_defense_r1 as r1
import topk_cifar100_dev_pareto_r2_common as common


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = Path("./datasets/cifar-100-python")
DEFAULT_R2_ROOT = Path(
    "./models/resnet20/"
    "paper_topk_cifar100_external_validity_r2"
)
DEFAULT_R1_DIR = Path(
    "./models/resnet20/"
    "paper_topk_cifar100_actual_defense_calibration_train2k_r1"
)
DEFAULT_OUTPUT = Path(
    "./models/resnet20/"
    "paper_topk_cifar100_actual_defense_dev_pareto_r2"
)


@dataclass(frozen=True)
class MethodSpec:
    name: str
    model: str
    candidate_id: str
    gate_mode: str = "none"
    strength_alpha_px: float = 0.0
    strength_epsilon_px: float = 0.0
    strength_steps: int = 0
    sigma: float = 0.0
    draws: int = 1


class IndexedDataset(Dataset):
    def __init__(self, dataset: Dataset, indices: Sequence[int]):
        self.dataset = dataset
        self.indices = [int(index) for index in indices]

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
    parser = argparse.ArgumentParser(description="CIFAR-100 actual-defense dev Pareto R2")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--r2-root", type=Path, default=DEFAULT_R2_ROOT)
    parser.add_argument("--r1-dir", type=Path, default=DEFAULT_R1_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--mc-chunk", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026090403)
    parser.add_argument("--dev-examples", type=int, default=5000)
    parser.add_argument("--sigma-draws", type=int, default=100)
    parser.add_argument("--max-clean-drop-pp", type=float, default=2.0)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed-strength", type=int, default=2026090404)
    parser.add_argument("--bootstrap-seed-sigma", type=int, default=2026090405)
    parser.add_argument("--bootstrap-seed-selected", type=int, default=2026090406)
    parser.add_argument(
        "--sigma-grid",
        type=parse_float_list,
        default=common.SIGMA_GRID,
        help="Comma-separated Gaussian sigmas in [0,1] image units; must include zero",
    )
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.dev_examples != 5000:
        raise ValueError("The frozen primary protocol requires all 5,000 R2 dev examples")
    if min(args.batch_size, args.mc_chunk, args.sigma_draws) <= 0:
        raise ValueError("Batch sizes and MC draws must be positive")
    if args.max_clean_drop_pp < 0:
        raise ValueError("The non-inferiority margin must be non-negative")
    if args.bootstrap_resamples < 1000:
        raise ValueError("Primary calibration requires at least 1,000 bootstrap resamples")
    sigmas = tuple(round(float(value), 7) for value in args.sigma_grid)
    if len(sigmas) != len(set(sigmas)) or any(value < 0 for value in sigmas):
        raise ValueError("Sigma values must be unique and non-negative")
    if 0.0 not in sigmas:
        raise ValueError("Sigma grid must include zero")
    if tuple(sorted(sigmas)) != tuple(common.SIGMA_GRID):
        raise ValueError("Primary R2 protocol requires the frozen sigma grid")


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


def stable_seed(base_seed: int, label: str) -> int:
    value = int(common.object_sha256({"seed": int(base_seed), "label": str(label)})[:15], 16)
    return value % (2**63 - 1)


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


def _strength_key(method: MethodSpec) -> tuple[float, float, int]:
    return (
        float(method.strength_alpha_px),
        float(method.strength_epsilon_px),
        int(method.strength_steps),
    )


def _base_for_method(
    method: MethodSpec,
    images: torch.Tensor,
    strength_cache: Mapping[tuple[float, float, int], torch.Tensor],
    disagreement: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if method.strength_steps == 0:
        return images, torch.zeros(len(images), dtype=torch.bool, device=images.device)
    strengthened = strength_cache[_strength_key(method)]
    if method.gate_mode == "global":
        active = torch.ones(len(images), dtype=torch.bool, device=images.device)
        return strengthened, active
    if method.gate_mode == "disagreement":
        active = disagreement
        view = active.view(-1, 1, 1, 1)
        return torch.where(view, strengthened, images), active
    raise ValueError(f"Strengthening method has invalid gate mode: {method.gate_mode}")


def _summarize(
    phase: str,
    method: MethodSpec,
    examples: Sequence[Mapping[str, Any]],
    wall_seconds: float,
) -> dict[str, Any]:
    def mean_percent(key: str) -> float:
        return 100.0 * float(np.mean([float(row[key]) for row in examples]))

    return {
        "protocol_id": common.PROTOCOL_ID,
        "phase": phase,
        "method": method.name,
        "candidate_id": method.candidate_id,
        "model": method.model,
        "gate_mode": method.gate_mode,
        "gate_activation_percent": mean_percent("gate_active"),
        "examples": len(examples),
        "draws": int(method.draws if method.sigma > 0 else 1),
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
    """Evaluate deterministic/stochastic methods with common random numbers."""

    if len({method.name for method in methods}) != len(methods):
        raise ValueError("Method names must be unique inside a condition")
    if len({method.candidate_id for method in methods}) != len(methods):
        raise ValueError("Candidate IDs must be unique inside a condition")
    started = time.monotonic()
    rows_by_method: dict[str, list[dict[str, Any]]] = {method.name: [] for method in methods}
    generator = torch.Generator(device=device)
    generator.manual_seed(int(noise_seed))

    for batch_number, (images, labels, indices) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        indices_list = [int(value) for value in indices.tolist()]
        # no_grad, rather than inference_mode, intentionally creates an ordinary
        # reference tensor that can be consumed by the strengthening autograd graph.
        with torch.no_grad(), autocast_context(device, args.amp):
            standard_logits = standard(images).float()
            anchor_clean_logits = anchor(images).float()
            reference_ranked = r1.ranked_top2(standard_logits).clone()
            disagreement = anchor_clean_logits.argmax(1).ne(standard_logits.argmax(1))

        strength_cache: dict[tuple[float, float, int], torch.Tensor] = {}
        for method in methods:
            if method.strength_steps <= 0:
                continue
            key = _strength_key(method)
            if key not in strength_cache:
                strength_cache[key] = r1.strengthen_inputs(
                    images,
                    reference_ranked,
                    anchor,
                    method.strength_alpha_px,
                    method.strength_epsilon_px,
                    method.strength_steps,
                    1.0,
                    4.0,
                )

        bases: dict[str, torch.Tensor] = {}
        gates: dict[str, torch.Tensor] = {}
        for method in methods:
            base, active = _base_for_method(
                method, images, strength_cache, disagreement
            )
            bases[method.name] = base
            gates[method.name] = active

        count_store: dict[str, dict[str, torch.Tensor]] = {}
        logit_store: dict[str, torch.Tensor] = {}
        for method in methods:
            count_store[method.name] = {
                metric: torch.zeros(len(images), device=device, dtype=torch.float64)
                for metric in ("top1_accuracy", "reference_top1_match", "exact_set", "label_coverage")
            }
            logit_store[method.name] = torch.zeros(
                len(images), 100, device=device, dtype=torch.float64
            )

        for method in (item for item in methods if item.sigma == 0.0):
            if method.model == "standard" and method.strength_steps == 0:
                logits = standard_logits
            elif method.model == "anchor" and method.strength_steps == 0:
                logits = anchor_clean_logits
            else:
                model = standard if method.model == "standard" else anchor
                with torch.inference_mode(), autocast_context(device, args.amp):
                    logits = model(bases[method.name]).float()
            metrics = r1.metric_tensors(logits, labels, reference_ranked)
            for metric, values in metrics.items():
                count_store[method.name][metric] += values.double()
            logit_store[method.name] += logits.double()

        stochastic_groups: dict[tuple[float, int], list[MethodSpec]] = {}
        for method in methods:
            if method.sigma > 0:
                stochastic_groups.setdefault(
                    (float(method.sigma), int(method.draws)), []
                ).append(method)
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
                    noisy = (bases[method.name].unsqueeze(0) + sigma * noise).clamp(0.0, 1.0)
                    flat = noisy.flatten(0, 1)
                    model = standard if method.model == "standard" else anchor
                    with torch.inference_mode(), autocast_context(device, args.amp):
                        logits = model(flat).float().view(chunk, len(images), 100)
                    predictions = logits.argmax(2)
                    predicted_sets = (
                        logits.topk(2, dim=2, largest=True, sorted=False)
                        .indices.sort(2)
                        .values
                    )
                    reference_set = reference_ranked.sort(1).values
                    count_store[method.name]["top1_accuracy"] += (
                        predictions.eq(labels[None, :]).sum(0).double()
                    )
                    count_store[method.name]["reference_top1_match"] += (
                        predictions.eq(reference_ranked[None, :, 0]).sum(0).double()
                    )
                    count_store[method.name]["exact_set"] += (
                        predicted_sets.eq(reference_set[None, :, :]).all(2).sum(0).double()
                    )
                    count_store[method.name]["label_coverage"] += (
                        predicted_sets.eq(labels[None, :, None]).any(2).sum(0).double()
                    )
                    logit_store[method.name] += logits.double().sum(0)
                completed += chunk

        reference_cpu = reference_ranked.detach().cpu().numpy()
        labels_cpu = labels.detach().cpu().numpy()
        for method in methods:
            effective_draws = int(method.draws if method.sigma > 0 else 1)
            mean_logits = (logit_store[method.name] / effective_draws).float()
            mean_metrics = r1.metric_tensors(mean_logits, labels, reference_ranked)
            margins = r1.boundary_margin(mean_logits, reference_ranked)
            counts = count_store[method.name]
            active_cpu = gates[method.name].detach().cpu().numpy()
            for position, sample_index in enumerate(indices_list):
                reference = reference_cpu[position]
                rows_by_method[method.name].append(
                    {
                        "protocol_id": common.PROTOCOL_ID,
                        "phase": phase,
                        "method": method.name,
                        "candidate_id": method.candidate_id,
                        "sample_index": sample_index,
                        "label": int(labels_cpu[position]),
                        "reference_top1": int(reference[0]),
                        "reference_set": f"{min(reference)}|{max(reference)}",
                        "gate_mode": method.gate_mode,
                        "gate_active": int(bool(active_cpu[position])),
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
                        "mean_logits_exact_set": float(
                            mean_metrics["exact_set"][position].item()
                        ),
                        "mean_boundary_margin": float(margins[position].item()),
                    }
                )
        if batch_number % 20 == 0 or batch_number == len(loader):
            print(
                f"    {phase}: batch {batch_number}/{len(loader)}, "
                f"methods={','.join(method.name for method in methods)}",
                flush=True,
            )

    elapsed = time.monotonic() - started
    rows = [row for method in methods for row in rows_by_method[method.name]]
    summaries = [
        _summarize(phase, method, rows_by_method[method.name], elapsed)
        for method in methods
    ]
    return summaries, rows


def _partial_path(output_dir: Path, condition_id: str) -> Path:
    return output_dir / "partials" / f"{condition_id}.json"


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
) -> dict[str, Any]:
    path = _partial_path(output_dir, condition_id)
    if path.is_file():
        payload = common.read_json(path)
        if payload.get("protocol_id") != common.PROTOCOL_ID:
            raise AssertionError(f"Partial protocol mismatch: {path}")
        if payload.get("config_hash") != config_hash or payload.get("condition_id") != condition_id:
            raise AssertionError(f"Partial belongs to another frozen invocation: {path}")
        expected = {method.name for method in methods}
        if {row["method"] for row in payload["summaries"]} != expected:
            raise AssertionError(f"Partial method mismatch: {path}")
        counts = {
            method: sum(row["method"] == method for row in payload["examples"])
            for method in expected
        }
        if set(counts.values()) != {args.dev_examples}:
            raise AssertionError(f"Partial example count mismatch: {path}: {counts}")
        print(f"RESUME {condition_id}", flush=True)
        return payload

    print(f"RUN {condition_id}", flush=True)
    summaries, rows = evaluate_methods(
        phase, loader, methods, standard, anchor, device, args, noise_seed
    )
    payload = {
        "protocol_id": common.PROTOCOL_ID,
        "config_hash": config_hash,
        "condition_id": condition_id,
        "phase": phase,
        "methods": [asdict(method) for method in methods],
        "summaries": summaries,
        "examples": rows,
    }
    common.atomic_json(path, payload)
    print(
        "PASS "
        + condition_id
        + ": "
        + ", ".join(
            f"{row['method']} Top1={row['top1_accuracy_percent']:.3f}% "
            f"Exact={row['exact_set_probability_percent']:.3f}%"
            for row in summaries
        ),
        flush=True,
    )
    return payload


def _summary(payload: Mapping[str, Any], method: str) -> dict[str, Any]:
    rows = [dict(row) for row in payload["summaries"] if row["method"] == method]
    if len(rows) != 1:
        raise AssertionError(f"Expected exactly one summary for {method}; got {len(rows)}")
    return rows[0]


def _examples(payload: Mapping[str, Any], method: str) -> list[dict[str, Any]]:
    rows = [dict(row) for row in payload["examples"] if row["method"] == method]
    if len(rows) == 0:
        raise AssertionError(f"No examples for method {method}")
    return rows


def _write_or_validate_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.is_file():
        if common.read_json(path) != payload:
            raise AssertionError(f"Frozen file differs from current invocation: {path}")
    else:
        common.atomic_json(path, payload)


def _bootstrap_specs(
    group_id: str,
    left_name: str,
    right_name: str,
    left_rows: Sequence[Mapping[str, Any]],
    right_rows: Sequence[Mapping[str, Any]],
    metrics: Sequence[str],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for metric in metrics:
        key = common.PROBABILITY_METRICS[metric]
        output.append(
            {
                "group_id": group_id,
                "comparison": f"{left_name}_minus_{right_name}_{metric}",
                "left": left_name,
                "right": right_name,
                "metric": metric,
                "delta": common.aligned_delta(left_rows, right_rows, key),
            }
        )
    return output


def _alias_rows(
    rows: Sequence[Mapping[str, Any]], alias: str
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        row["source_method"] = row["method"]
        row["method"] = alias
        output.append(row)
    return output


def _alias_summary(summary: Mapping[str, Any], alias: str) -> dict[str, Any]:
    row = dict(summary)
    row["source_method"] = row["method"]
    row["method"] = alias
    return row


def reference_quality_decomposition(
    selected_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    methods = sorted({str(row["method"]) for row in selected_rows})
    by_method = {
        method: common.rows_by_index(
            [row for row in selected_rows if str(row["method"]) == method]
        )
        for method in methods
    }
    indices = sorted(by_method["standard"])
    if any(sorted(rows) != indices for rows in by_method.values()):
        raise AssertionError("Selected methods do not share samples for decomposition")
    standard_correct = {
        index: float(by_method["standard"][index]["top1_accuracy_probability"]) == 1.0
        for index in indices
    }
    groups = {
        "all": indices,
        "standard_correct": [index for index in indices if standard_correct[index]],
        "standard_wrong": [index for index in indices if not standard_correct[index]],
    }
    output: list[dict[str, Any]] = []
    for group, group_indices in groups.items():
        if not group_indices:
            raise AssertionError(f"Empty reference-quality group: {group}")
        for method in ("standard", "robust", "full_sigma0", "noise_only", "full"):
            rows = by_method[method]
            output.append(
                {
                    "protocol_id": common.PROTOCOL_ID,
                    "group": group,
                    "method": method,
                    "examples": len(group_indices),
                    "top1_accuracy_percent": 100.0
                    * float(
                        np.mean(
                            [float(rows[index]["top1_accuracy_probability"]) for index in group_indices]
                        )
                    ),
                    "reference_top1_match_probability_percent": 100.0
                    * float(
                        np.mean(
                            [
                                float(rows[index]["reference_top1_match_probability"])
                                for index in group_indices
                            ]
                        )
                    ),
                    "exact_set_probability_percent": 100.0
                    * float(
                        np.mean(
                            [float(rows[index]["exact_set_probability"]) for index in group_indices]
                        )
                    ),
                    "label_coverage_probability_percent": 100.0
                    * float(
                        np.mean(
                            [
                                float(rows[index]["label_coverage_probability"])
                                for index in group_indices
                            ]
                        )
                    ),
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
    r1_dir = args.r1_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    standard_weights = r2_root / "standard" / "resnet20_standard_best_validation_weights.pt"
    anchor_weights = r2_root / "teacher_top2_at" / "resnet20_teacher_top2_best_composite_weights.pt"
    split_path = r2_root / "frozen_split.json"
    pipeline_manifest_path = r2_root / "pipeline_manifest.json"
    r1_final_path = r1_dir / "final_summary.json"
    required_paths = (
        standard_weights,
        anchor_weights,
        split_path,
        pipeline_manifest_path,
        r1_final_path,
    )
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Required frozen artifact is missing: {path}")

    fixed_hashes = {
        standard_weights: common.EXPECTED_STANDARD_SHA256,
        anchor_weights: common.EXPECTED_ANCHOR_SHA256,
        r1_final_path: common.EXPECTED_R1_FINAL_SUMMARY_SHA256,
    }
    for path, expected in fixed_hashes.items():
        actual = common.file_sha256(path)
        if actual != expected:
            raise AssertionError(f"Frozen artifact hash mismatch: {path}: {actual}")

    r1_final = common.read_json(r1_final_path)
    if r1_final.get("protocol_id") != common.PARENT_R1_PROTOCOL_ID:
        raise AssertionError("R1 diagnostic belongs to another protocol")
    if r1_final.get("attack_stage_ready") is not False:
        raise AssertionError("R2 redesign is only valid after the declared R1 gate failure")
    if r1_final.get("data_governance", {}).get("official_test_read") is not False:
        raise AssertionError("R1 defense calibration unexpectedly read the official test set")

    parent_manifest = common.read_json(pipeline_manifest_path)
    if parent_manifest.get("protocol_id") != common.PARENT_R2_PROTOCOL_ID:
        raise AssertionError("Unexpected R2 model-preparation protocol")
    if parent_manifest.get("dataset") != "CIFAR-100":
        raise AssertionError("Parent model preparation is not CIFAR-100")

    dependency_paths = {
        "evaluator": Path(__file__).resolve(),
        "dev_pareto_common": PROJECT_DIR / "topk_cifar100_dev_pareto_r2_common.py",
        "r1_evaluator": PROJECT_DIR / "calibrate_topk_cifar100_actual_defense_r1.py",
        "r1_common": PROJECT_DIR / "topk_cifar100_defense_calibration_common.py",
        "r2_common": PROJECT_DIR / "topk_cifar100_r2_common.py",
        "r2_anchor": PROJECT_DIR / "train_resnet20_teacher_top2_adversarial_cifar100_r2.py",
    }
    expected_source_hashes = {
        "r1_evaluator": common.EXPECTED_R1_EVALUATOR_SHA256,
        "r1_common": common.EXPECTED_R1_COMMON_SHA256,
        "r2_common": common.EXPECTED_R2_COMMON_SHA256,
        "r2_anchor": common.EXPECTED_R2_ANCHOR_SOURCE_SHA256,
    }
    for name, path in dependency_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Required source is missing: {path}")
        if name in expected_source_hashes and common.file_sha256(path) != expected_source_hashes[name]:
            raise AssertionError(f"Frozen source hash mismatch: {name}: {path}")

    split = r1.load_frozen_split(split_path)
    dataset_root = r1.r2_anchor.resolve_torchvision_root(args.data_dir)
    dataset = torchvision.datasets.CIFAR100(
        root=str(dataset_root), train=True, download=False, transform=transforms.ToTensor()
    )
    if len(dataset) != 50000 or len(dataset.targets) != 50000:
        raise AssertionError("Unexpected CIFAR-100 training-set size")
    if r1.r2_common.labels_sha256(dataset.targets) != split["labels_sha256"]:
        raise AssertionError("Local CIFAR-100 labels differ from the frozen R2 split")
    dev_indices = [int(index) for index in split["validation_indices"]]
    if len(dev_indices) != args.dev_examples or len(set(dev_indices)) != args.dev_examples:
        raise AssertionError("Frozen R2 dev split must contain 5,000 unique samples")
    counts = np.bincount(np.asarray(dataset.targets)[dev_indices], minlength=100)
    if not np.all(counts == 50):
        raise AssertionError("Frozen R2 dev split is not balanced at 50 examples per class")

    dev_manifest = {
        "protocol_id": common.PROTOCOL_ID,
        "parent_r1_protocol_id": common.PARENT_R1_PROTOCOL_ID,
        "parent_r2_protocol_id": common.PARENT_R2_PROTOCOL_ID,
        "selection_split": "complete frozen R2 validation split, reclassified as development after R1 failure",
        "dev_indices": dev_indices,
        "dev_indices_sha256": common.indices_sha256(dev_indices),
        "expected_parent_validation_indices_sha256": common.EXPECTED_VALIDATION_HASH,
        "official_test_read_by_this_stage": False,
        "official_test_used_for_selection": False,
        "attack_outcomes_read": False,
        "r1_failure_used_to_declare_redesign": True,
    }
    _write_or_validate_json(output_dir / "dev_manifest.json", dev_manifest)

    config = {
        "protocol_id": common.PROTOCOL_ID,
        "parent_r1_protocol_id": common.PARENT_R1_PROTOCOL_ID,
        "parent_r2_protocol_id": common.PARENT_R2_PROTOCOL_ID,
        "dataset": "CIFAR-100",
        "architecture": "ResNet-20",
        "primary_k": 2,
        "data_dir": str(args.data_dir.expanduser().resolve()),
        "r2_root": str(r2_root),
        "r1_dir": str(r1_dir),
        "output_dir": str(output_dir),
        "device": str(device),
        "batch_size": args.batch_size,
        "mc_chunk": args.mc_chunk,
        "num_workers": args.num_workers,
        "amp": bool(args.amp),
        "seed": args.seed,
        "dev_examples": args.dev_examples,
        "sigma_draws": args.sigma_draws,
        "max_clean_drop_pp": args.max_clean_drop_pp,
        "bootstrap_resamples": args.bootstrap_resamples,
        "bootstrap_seeds": {
            "strength": args.bootstrap_seed_strength,
            "sigma": args.bootstrap_seed_sigma,
            "selected": args.bootstrap_seed_selected,
        },
        "gate_modes": list(common.GATE_MODES),
        "strength_grid": list(common.STRENGTH_GRID),
        "sigma_grid": list(common.SIGMA_GRID),
        "strength_selection_policy": (
            "maximize deterministic Exact-Set2 among identity/global/disagreement candidates "
            "whose paired two-sided 95% bootstrap lower bound for Top1 minus robust is >= -2 pp"
        ),
        "sigma_selection_policy": (
            "largest sigma whose paired two-sided 95% bootstrap lower bounds are >= -2 pp "
            "for noise Top1/Exact vs robust, full Top1 vs robust, and full Top1/Exact vs full-sigma0"
        ),
        "checkpoints": {
            "standard": {"path": str(standard_weights), "sha256": common.file_sha256(standard_weights)},
            "anchor": {"path": str(anchor_weights), "sha256": common.file_sha256(anchor_weights)},
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
        "r1_failure_summary": {
            "path": str(r1_final_path),
            "sha256": common.file_sha256(r1_final_path),
        },
        "script_sha256": {
            name: common.file_sha256(path) for name, path in dependency_paths.items()
        },
        "offline": True,
        "downloads_allowed": False,
        "official_test_read_by_this_stage": False,
        "official_test_used_for_selection": False,
        "attack_outcomes_read": False,
    }
    config_hash = common.object_sha256(config)
    config["config_hash"] = config_hash
    _write_or_validate_json(output_dir / "run_config.json", config)

    standard, anchor = r1.load_models(standard_weights, anchor_weights, device)
    loader = make_loader(dataset, dev_indices, args, device)

    baseline_payload = run_condition(
        output_dir,
        "dev_baselines",
        config_hash,
        "dev",
        loader,
        [
            MethodSpec("standard", "standard", "standard"),
            MethodSpec("robust", "anchor", "identity"),
        ],
        standard,
        anchor,
        device,
        args,
        stable_seed(args.seed, "dev_baselines"),
    )
    standard_summary = _summary(baseline_payload, "standard")
    robust_summary = _summary(baseline_payload, "robust")
    standard_examples = _examples(baseline_payload, "standard")
    robust_examples = _examples(baseline_payload, "robust")

    strength_payloads: list[dict[str, Any]] = []
    for candidate in common.STRENGTH_GRID:
        condition_id = common.strength_condition_id(candidate)
        methods = [
            MethodSpec(
                name=f"{gate_mode}_full_sigma0",
                model="anchor",
                candidate_id=common.strength_candidate_id(gate_mode, candidate),
                gate_mode=gate_mode,
                strength_alpha_px=float(candidate["alpha_px"]),
                strength_epsilon_px=float(candidate["epsilon_px"]),
                strength_steps=int(candidate["steps"]),
            )
            for gate_mode in common.GATE_MODES
        ]
        strength_payloads.append(
            run_condition(
                output_dir,
                condition_id,
                config_hash,
                "dev_strength",
                loader,
                methods,
                standard,
                anchor,
                device,
                args,
                stable_seed(args.seed, condition_id),
            )
        )

    strength_candidate_summaries: list[dict[str, Any]] = []
    identity_summary = dict(robust_summary)
    identity_summary.update(
        {
            "candidate_id": "identity",
            "gate_mode": "none",
            "strength_alpha_px": 0.0,
            "strength_epsilon_px": 0.0,
            "strength_steps": 0,
        }
    )
    strength_candidate_summaries.append(identity_summary)
    strength_bootstrap_specs = _bootstrap_specs(
        "identity", "identity", "robust", robust_examples, robust_examples, ("top1", "exact_set")
    )
    for payload in strength_payloads:
        for method in ("global_full_sigma0", "disagreement_full_sigma0"):
            summary = _summary(payload, method)
            examples = _examples(payload, method)
            strength_candidate_summaries.append(summary)
            strength_bootstrap_specs.extend(
                _bootstrap_specs(
                    str(summary["candidate_id"]),
                    str(summary["candidate_id"]),
                    "robust",
                    examples,
                    robust_examples,
                    ("top1", "exact_set"),
                )
            )
    print("Computing paired bootstrap for the strength frontier", flush=True)
    strength_bootstrap = common.paired_bootstrap_many(
        strength_bootstrap_specs,
        args.bootstrap_resamples,
        args.bootstrap_seed_strength,
    )
    selected_strength, annotated_strength = common.select_strength_candidate(
        strength_candidate_summaries,
        strength_bootstrap,
        args.max_clean_drop_pp,
    )
    selected_strength_id = str(selected_strength["candidate_id"])
    annotated_by_id = {str(row["candidate_id"]): row for row in annotated_strength}
    strength_summary_rows = [dict(standard_summary)]
    for source in strength_candidate_summaries:
        row = dict(annotated_by_id[str(source["candidate_id"])])
        row["selected"] = str(row["candidate_id"]) == selected_strength_id
        strength_summary_rows.append(row)
    strength_examples = [dict(row) for row in baseline_payload["examples"]]
    for payload in strength_payloads:
        strength_examples.extend(dict(row) for row in payload["examples"])
    common.atomic_csv(output_dir / "strength_frontier_summary.csv", strength_summary_rows)
    common.atomic_csv(output_dir / "strength_frontier_per_example.csv", strength_examples)
    common.atomic_csv(output_dir / "strength_paired_bootstrap.csv", strength_bootstrap)

    selected_gate = str(selected_strength["gate_mode"])
    selected_alpha = float(selected_strength["strength_alpha_px"])
    selected_epsilon = float(selected_strength["strength_epsilon_px"])
    selected_steps = int(selected_strength["strength_steps"])
    sigma_payloads: list[dict[str, Any]] = []
    for sigma in common.SIGMA_GRID:
        condition_id = common.sigma_condition_id(sigma)
        draws = args.sigma_draws if sigma > 0 else 1
        methods = [
            MethodSpec(
                "noise_only",
                "anchor",
                f"{condition_id}_noise_only",
                sigma=float(sigma),
                draws=draws,
            ),
            MethodSpec(
                "full",
                "anchor",
                f"{condition_id}_full",
                gate_mode=selected_gate,
                strength_alpha_px=selected_alpha,
                strength_epsilon_px=selected_epsilon,
                strength_steps=selected_steps,
                sigma=float(sigma),
                draws=draws,
            ),
        ]
        sigma_payloads.append(
            run_condition(
                output_dir,
                condition_id,
                config_hash,
                "dev_sigma",
                loader,
                methods,
                standard,
                anchor,
                device,
                args,
                stable_seed(args.seed, "sigma_grid_common_random_numbers"),
            )
        )

    sigma0_payload = next(
        payload for payload in sigma_payloads if float(_summary(payload, "full")["sigma"]) == 0.0
    )
    sigma0_noise_examples = _examples(sigma0_payload, "noise_only")
    full0_examples = _examples(sigma0_payload, "full")
    full0_summary = _summary(sigma0_payload, "full")
    selected_strength_reference_examples = (
        robust_examples
        if selected_strength_id == "identity"
        else next(
            _examples(payload, method)
            for payload in strength_payloads
            for method in ("global_full_sigma0", "disagreement_full_sigma0")
            if _summary(payload, method)["candidate_id"] == selected_strength_id
        )
    )
    for left_rows, right_rows, label in (
        (sigma0_noise_examples, robust_examples, "sigma=0 Noise-only/Robust"),
        (full0_examples, selected_strength_reference_examples, "sigma=0 Full/strength"),
    ):
        for metric_key in common.PROBABILITY_METRICS.values():
            drift = np.max(np.abs(common.aligned_delta(left_rows, right_rows, metric_key)))
            if float(drift) > 1e-10:
                raise AssertionError(f"{label} reproduction failed for {metric_key}: {drift}")
    sigma_bootstrap_specs: list[dict[str, Any]] = []
    sigma_combined_rows: list[dict[str, Any]] = []
    for payload in sigma_payloads:
        noise_summary = _summary(payload, "noise_only")
        full_summary = _summary(payload, "full")
        noise_examples = _examples(payload, "noise_only")
        full_examples = _examples(payload, "full")
        sigma = round(float(noise_summary["sigma"]), 7)
        if abs(sigma - float(full_summary["sigma"])) > 1e-12:
            raise AssertionError("Full/Noise sigma mismatch")
        group_id = common.sigma_condition_id(sigma)
        constraint_specs = (
            (
                "noise_top1_vs_robust",
                "noise_only",
                "robust",
                noise_examples,
                robust_examples,
                "top1",
            ),
            (
                "noise_exact_set_vs_robust",
                "noise_only",
                "robust",
                noise_examples,
                robust_examples,
                "exact_set",
            ),
            (
                "full_top1_vs_robust",
                "full",
                "robust",
                full_examples,
                robust_examples,
                "top1",
            ),
            (
                "full_top1_vs_full_sigma0",
                "full",
                "full_sigma0",
                full_examples,
                full0_examples,
                "top1",
            ),
            (
                "full_exact_set_vs_full_sigma0",
                "full",
                "full_sigma0",
                full_examples,
                full0_examples,
                "exact_set",
            ),
        )
        for constraint, left, right, left_rows, right_rows, base_metric in constraint_specs:
            sigma_bootstrap_specs.append(
                {
                    "group_id": group_id,
                    "comparison": f"{constraint}_{group_id}",
                    "left": left,
                    "right": right,
                    "metric": constraint,
                    "base_metric": base_metric,
                    "delta": common.aligned_delta(
                        left_rows, right_rows, common.PROBABILITY_METRICS[base_metric]
                    ),
                }
            )
        sigma_combined_rows.append(
            {
                "protocol_id": common.PROTOCOL_ID,
                "sigma": sigma,
                "draws": int(noise_summary["draws"]),
                "selected_strength_candidate_id": selected_strength_id,
                "selected_gate_mode": selected_gate,
                "noise_top1_accuracy_percent": float(noise_summary["top1_accuracy_percent"]),
                "noise_exact_set_probability_percent": float(
                    noise_summary["exact_set_probability_percent"]
                ),
                "full_top1_accuracy_percent": float(full_summary["top1_accuracy_percent"]),
                "full_exact_set_probability_percent": float(
                    full_summary["exact_set_probability_percent"]
                ),
            }
        )

    print("Computing paired bootstrap for the sigma frontier", flush=True)
    sigma_bootstrap = common.paired_bootstrap_many(
        sigma_bootstrap_specs,
        args.bootstrap_resamples,
        args.bootstrap_seed_sigma,
    )
    selected_sigma, annotated_sigma = common.select_sigma_candidate(
        sigma_combined_rows,
        sigma_bootstrap,
        args.max_clean_drop_pp,
    )
    selected_sigma_value = float(selected_sigma["sigma"])
    sigma_annotation = {round(float(row["sigma"]), 7): row for row in annotated_sigma}
    sigma_summary_rows: list[dict[str, Any]] = []
    sigma_examples: list[dict[str, Any]] = []
    for payload in sigma_payloads:
        sigma = round(float(_summary(payload, "noise_only")["sigma"]), 7)
        annotation = sigma_annotation[sigma]
        for method_summary in payload["summaries"]:
            row = dict(method_summary)
            row["jointly_ci_feasible"] = bool(annotation["jointly_ci_feasible"])
            row["selected"] = sigma == selected_sigma_value
            for key, value in annotation.items():
                if key.endswith("_difference_pp") or "_ci95_" in key:
                    row[key] = value
            sigma_summary_rows.append(row)
        sigma_examples.extend(dict(row) for row in payload["examples"])
    common.atomic_csv(output_dir / "sigma_frontier_summary.csv", sigma_summary_rows)
    common.atomic_csv(output_dir / "sigma_frontier_per_example.csv", sigma_examples)
    common.atomic_csv(output_dir / "sigma_paired_bootstrap.csv", sigma_bootstrap)

    selected_strength_payload = None
    selected_strength_method = None
    if selected_strength_id == "identity":
        selected_strength_examples = robust_examples
        selected_strength_summary = identity_summary
    else:
        for payload in strength_payloads:
            for method in ("global_full_sigma0", "disagreement_full_sigma0"):
                if _summary(payload, method)["candidate_id"] == selected_strength_id:
                    selected_strength_payload = payload
                    selected_strength_method = method
                    break
            if selected_strength_payload is not None:
                break
        if selected_strength_payload is None or selected_strength_method is None:
            raise AssertionError("Could not locate selected strength payload")
        selected_strength_examples = _examples(selected_strength_payload, selected_strength_method)
        selected_strength_summary = _summary(selected_strength_payload, selected_strength_method)

    selected_sigma_payload = next(
        payload
        for payload in sigma_payloads
        if round(float(_summary(payload, "noise_only")["sigma"]), 7)
        == round(selected_sigma_value, 7)
    )
    selected_noise_examples = _examples(selected_sigma_payload, "noise_only")
    selected_full_examples = _examples(selected_sigma_payload, "full")
    selected_noise_summary = _summary(selected_sigma_payload, "noise_only")
    selected_full_summary = _summary(selected_sigma_payload, "full")

    selected_rows = (
        _alias_rows(standard_examples, "standard")
        + _alias_rows(robust_examples, "robust")
        + _alias_rows(selected_strength_examples, "full_sigma0")
        + _alias_rows(selected_noise_examples, "noise_only")
        + _alias_rows(selected_full_examples, "full")
    )
    selected_summaries = [
        _alias_summary(standard_summary, "standard"),
        _alias_summary(robust_summary, "robust"),
        _alias_summary(selected_strength_summary, "full_sigma0"),
        _alias_summary(selected_noise_summary, "noise_only"),
        _alias_summary(selected_full_summary, "full"),
    ]
    selected_maps = {
        method: [row for row in selected_rows if row["method"] == method]
        for method in ("standard", "robust", "full_sigma0", "noise_only", "full")
    }
    selected_bootstrap_specs: list[dict[str, Any]] = []
    for left, right, metrics in (
        ("noise_only", "robust", ("top1", "exact_set")),
        ("full_sigma0", "robust", ("top1", "exact_set")),
        ("full", "robust", ("top1", "exact_set")),
        ("full", "noise_only", ("exact_set",)),
    ):
        selected_bootstrap_specs.extend(
            _bootstrap_specs(
                "selected",
                left,
                right,
                selected_maps[left],
                selected_maps[right],
                metrics,
            )
        )
    selected_bootstrap = common.paired_bootstrap_many(
        selected_bootstrap_specs,
        args.bootstrap_resamples,
        args.bootstrap_seed_selected,
    )
    decomposition = reference_quality_decomposition(selected_rows)
    common.atomic_csv(output_dir / "selected_dev_summary.csv", selected_summaries)
    common.atomic_csv(output_dir / "selected_dev_per_example.csv", selected_rows)
    common.atomic_csv(output_dir / "selected_dev_paired_bootstrap.csv", selected_bootstrap)
    common.atomic_csv(output_dir / "reference_quality_decomposition.csv", decomposition)

    positive_strength = selected_steps > 0 and selected_gate in common.GATE_MODES
    positive_sigma = selected_sigma_value > 0
    official_test_stage_ready = bool(
        positive_strength
        and positive_sigma
        and selected_strength["ci_feasible"]
        and selected_sigma["jointly_ci_feasible"]
    )
    selection = {
        "protocol_id": common.PROTOCOL_ID,
        "config_hash": config_hash,
        "selection_split": "complete frozen R2 validation split, now declared development",
        "official_test_read_by_this_stage": False,
        "official_test_used_for_selection": False,
        "attack_outcomes_read": False,
        "strength": {
            "candidate_id": selected_strength_id,
            "gate_mode": selected_gate,
            "alpha_px": selected_alpha,
            "epsilon_px": selected_epsilon,
            "steps": selected_steps,
            "top1_weight": 1.0,
            "set_weight": 4.0,
            "positive_strength_selected": positive_strength,
            "selected_dev_summary": selected_strength,
        },
        "randomization": {
            "sigma": selected_sigma_value,
            "positive_sigma_selected": positive_sigma,
            "selected_dev_constraints": selected_sigma,
        },
    }
    selection["selection_hash"] = common.object_sha256(selection)
    _write_or_validate_json(output_dir / "dev_pareto_selection.json", selection)

    blockers = []
    if not positive_strength:
        blockers.append("No positive strengthening candidate passed the paired-CI Top-1 constraint")
    if not positive_sigma:
        blockers.append("No positive sigma passed all five paired-CI constraints")
    final_summary = {
        "protocol_id": common.PROTOCOL_ID,
        "config_hash": config_hash,
        "selection_hash": selection["selection_hash"],
        "status": "PASS",
        "official_test_stage_ready": official_test_stage_ready,
        "official_test_stage_blockers": blockers,
        "attack_stage_ready": False,
        "attack_stage_blocker": (
            "The selected operating point requires one frozen clean evaluation on the official "
            "CIFAR-100 test set before any attack is authorized"
        ),
        "selection": selection,
        "selected_dev_summaries": selected_summaries,
        "selected_dev_paired_bootstrap": selected_bootstrap,
        "reference_quality_decomposition": decomposition,
        "data_governance": {
            "r2_validation_reclassified_as_development": True,
            "official_test_read_by_this_stage": False,
            "official_test_used_for_selection": False,
            "attack_outcomes_read": False,
            "no_retraining": True,
        },
        "next_stage": (
            "frozen_official_test_clean_confirmation"
            if official_test_stage_ready
            else "stop_and_report_clean_transfer_failure"
        ),
    }
    common.atomic_json(output_dir / "final_summary.json", final_summary)

    print("\nSelected CIFAR-100 dev-Pareto operating point", flush=True)
    print(
        f"  strength={selected_strength_id}: gate={selected_gate}, "
        f"alpha={selected_alpha:.3f}/255, epsilon={selected_epsilon:.3f}/255, "
        f"steps={selected_steps}",
        flush=True,
    )
    print(
        f"  strength Top1 delta={float(selected_strength['top1_difference_pp']):+.3f} pp "
        f"[{float(selected_strength['top1_ci95_low_pp']):+.3f}, "
        f"{float(selected_strength['top1_ci95_high_pp']):+.3f}]",
        flush=True,
    )
    print(f"  sigma={selected_sigma_value:.7f}", flush=True)
    for row in selected_summaries:
        print(
            f"  {row['method']:>11}: Top1={float(row['top1_accuracy_percent']):.3f}% "
            f"Exact={float(row['exact_set_probability_percent']):.3f}%",
            flush=True,
        )
    print(
        "OFFICIAL_TEST_STAGE_READY=" + ("YES" if official_test_stage_ready else "NO"),
        flush=True,
    )
    print("ATTACK_STAGE_READY=NO", flush=True)
    print(f"Complete: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
