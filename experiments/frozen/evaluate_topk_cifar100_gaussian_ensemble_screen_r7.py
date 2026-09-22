#!/usr/bin/env python3
"""Compute-matched Gaussian logit-ensemble screening for CIFAR-100 R5.

This is an explicitly post-hoc/exploratory screening experiment.  It does not
change the frozen R5 protocol and it does not claim adaptive robustness.  It:

1. benchmarks the deployed Full defense and Gaussian ensembles on the same GPU;
2. chooses B* from latency only;
3. evaluates clean inputs and saved R5 adversarial candidates with an *inner*
   B-draw Gaussian ensemble and an independent *outer* Monte-Carlo evaluation;
4. reports mean-logit aggregation (primary) and mean-probability aggregation
   (secondary), using common random numbers for every B.

Place this file beside the existing CIFAR-100 project scripts, or pass
--project-dir so these modules can be imported:

  calibrate_topk_cifar100_actual_defense_r1.py
  topk_cifar100_attack_pilot_r4_common.py
  search_topk_cifar100_attack_pilot_r4.py

The large candidate.pt files are expected to remain in the R5 output tree on
the experiment server; they are not bundled with this script.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn


SCRIPT_VERSION = "cifar100_gaussian_ensemble_screen_r7_v1"
DEFAULT_R5_ROOT = Path(
    "./models/resnet20/"
    "paper_topk_cifar100_confirmatory_attack_k2_q5k_r5"
)
DEFAULT_OUTPUT_NAME = "paper_topk_cifar100_gaussian_ensemble_screen_k2_r7"
NUM_CLASSES = 100
TOP_K = 2

METRIC_NAMES = (
    "top1_accuracy_probability",
    "reference_top1_match_probability",
    "exact_set_probability",
    "label_coverage_probability",
    "boundary_margin_mean",
)


def parse_int_tuple(text: str) -> tuple[int, ...]:
    try:
        values = tuple(int(value.strip()) for value in text.split(",") if value.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected comma-separated integers") from error
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("Values must be positive integers")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("Values must be unique")
    return tuple(sorted(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CIFAR-100 compute-matched Gaussian ensemble screening"
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing the existing CIFAR-100 experiment modules",
    )
    parser.add_argument(
        "--r5-root",
        type=Path,
        default=None,
        help="R5 root containing confirmatory_protocol.json and search/*/candidate.pt",
    )
    parser.add_argument(
        "--protocol-json",
        type=Path,
        default=None,
        help="Optional explicit R5 confirmatory_protocol.json",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--stage", choices=("latency", "screen", "all"), default="all"
    )
    parser.add_argument("--b-values", type=parse_int_tuple, default=(1, 2, 4, 8, 16, 32))
    parser.add_argument(
        "--sigma",
        type=float,
        default=None,
        help="Raw-pixel Gaussian sigma; default is frozen R5 sigma",
    )
    parser.add_argument("--examples", type=int, default=200)
    parser.add_argument("--outer-draws", type=int, default=100)
    parser.add_argument("--eval-batch-size", type=int, default=5)
    parser.add_argument(
        "--outer-chunk",
        type=int,
        default=2,
        help="Number of independent ensemble responses produced together",
    )
    parser.add_argument(
        "--forward-microbatch",
        type=int,
        default=512,
        help="Maximum noisy images per anchor forward; 0 disables microbatching",
    )
    parser.add_argument("--candidate-method", default="noise_only")
    parser.add_argument("--candidate-attack", default=None)
    parser.add_argument(
        "--candidate-seeds",
        type=parse_int_tuple,
        default=None,
        help="Comma-separated R5 search seeds; default uses every frozen seed",
    )
    parser.add_argument("--subset-seed", type=int, default=2026091801)
    parser.add_argument("--noise-seed", type=int, default=2026091802)
    parser.add_argument(
        "--latency-batch-sizes", type=parse_int_tuple, default=(1, 25)
    )
    parser.add_argument("--selection-batch-size", type=int, default=1)
    parser.add_argument("--latency-warmup", type=int, default=20)
    parser.add_argument("--latency-repeats", type=int, default=100)
    parser.add_argument("--latency-tolerance", type=float, default=0.05)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.45)
    parser.add_argument(
        "--skip-checkpoint-hash",
        action="store_true",
        help="Skip SHA-256 verification recorded by the frozen R5 protocol",
    )
    parser.add_argument(
        "--overwrite-cache",
        action="store_true",
        help="Recompute a matching condition instead of resuming its cache",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.examples <= 0 or args.examples % NUM_CLASSES:
        raise ValueError("--examples must be positive and divisible by 100")
    for name in (
        "outer_draws",
        "eval_batch_size",
        "outer_chunk",
        "latency_warmup",
        "latency_repeats",
        "selection_batch_size",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.forward_microbatch < 0:
        raise ValueError("--forward-microbatch cannot be negative")
    if args.sigma is not None and args.sigma < 0:
        raise ValueError("--sigma cannot be negative")
    if not 0 < args.gpu_memory_fraction <= 1:
        raise ValueError("--gpu-memory-fraction must be in (0,1]")
    if not 0 <= args.latency_tolerance <= 1:
        raise ValueError("--latency-tolerance must be in [0,1]")
    if args.selection_batch_size not in args.latency_batch_sizes:
        raise ValueError("--selection-batch-size must appear in --latency-batch-sizes")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def object_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json(path: Path, payload: Any) -> None:
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
    seen: set[str] = set()
    for row in materialized:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(materialized)
    os.replace(temporary, path)


def atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def autocast_off(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", enabled=False)
    return contextlib.nullcontext()


def load_project_modules(project_dir: Path):
    resolved = project_dir.expanduser().resolve()
    if str(resolved) not in sys.path:
        sys.path.insert(0, str(resolved))
    try:
        import calibrate_topk_cifar100_actual_defense_r1 as r1
        import topk_cifar100_attack_pilot_r4_common as r4_common
        from search_topk_cifar100_attack_pilot_r4 import FrozenDefense
    except ImportError as error:
        raise RuntimeError(
            "Could not import the existing CIFAR-100 modules. Place this script "
            "beside the project scripts or pass --project-dir. Original error: "
            f"{error}"
        ) from error
    return r1, r4_common, FrozenDefense


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    if args.protocol_json is not None:
        protocol_path = args.protocol_json.expanduser().resolve()
        r5_root = (
            args.r5_root.expanduser().resolve()
            if args.r5_root is not None
            else protocol_path.parent
        )
    else:
        r5_root = (
            args.r5_root.expanduser().resolve()
            if args.r5_root is not None
            else DEFAULT_R5_ROOT
        )
        protocol_path = r5_root / "confirmatory_protocol.json"
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else r5_root.parent / DEFAULT_OUTPUT_NAME
    )
    return r5_root, protocol_path, output_dir


def verify_checkpoint(path: Path, expected_sha256: str | None, skip: bool) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    if skip or not expected_sha256:
        return
    observed = file_sha256(path)
    if observed != expected_sha256:
        raise AssertionError(
            f"Checkpoint SHA-256 mismatch for {path}: {observed} != {expected_sha256}"
        )


def candidate_paths(
    *,
    protocol: Mapping[str, Any],
    r5_root: Path,
    method: str,
    attack_override: str | None,
    seeds_override: Sequence[int] | None,
) -> tuple[str, tuple[int, ...], list[Path]]:
    method_attacks = protocol.get("method_attack", {})
    attack = attack_override or method_attacks.get(method)
    if not attack:
        raise KeyError(
            f"No frozen attack found for method={method}; pass --candidate-attack"
        )
    seeds = tuple(
        int(value)
        for value in (
            seeds_override
            if seeds_override is not None
            else protocol.get("search_seeds", ())
        )
    )
    if not seeds:
        raise ValueError("No candidate search seeds were supplied or found in protocol")
    paths = [
        r5_root / "search" / method / str(attack) / f"seed_{seed}" / "candidate.pt"
        for seed in seeds
    ]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        joined = "\n  ".join(str(path) for path in missing)
        raise FileNotFoundError(
            "The large R5 candidate payloads are missing. Run this on the experiment "
            f"server where candidate.pt files are retained:\n  {joined}"
        )
    return str(attack), seeds, paths


def load_candidate(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {"clean", "adversarial", "labels", "sample_indices", "reference_top2"}
    missing = required - set(payload)
    if missing:
        raise AssertionError(f"{path} misses candidate tensors: {sorted(missing)}")
    result = dict(payload)
    for key in required:
        if not torch.is_tensor(result[key]):
            raise TypeError(f"{path}: {key} is not a tensor")
        result[key] = result[key].detach().cpu().contiguous()
    clean = result["clean"]
    adversarial = result["adversarial"]
    examples = len(clean)
    if clean.shape != adversarial.shape or clean.ndim != 4 or clean.shape[1:] != (3, 32, 32):
        raise AssertionError(f"{path}: invalid clean/adversarial shapes")
    if result["labels"].shape != (examples,):
        raise AssertionError(f"{path}: invalid labels shape")
    if result["sample_indices"].shape != (examples,):
        raise AssertionError(f"{path}: invalid sample_indices shape")
    if result["reference_top2"].shape != (examples, TOP_K):
        raise AssertionError(f"{path}: invalid reference_top2 shape")
    if not torch.isfinite(clean).all() or not torch.isfinite(adversarial).all():
        raise AssertionError(f"{path}: non-finite pixels")
    for key in ("clean", "adversarial"):
        tensor = result[key]
        if float(tensor.min()) < -1e-7 or float(tensor.max()) > 1.0 + 1e-7:
            raise AssertionError(f"{path}: {key} pixels outside [0,1]")
    return result


def validate_candidate_alignment(payloads: Sequence[Mapping[str, Any]]) -> None:
    if not payloads:
        raise ValueError("Expected at least one candidate payload")
    first = payloads[0]
    for position, payload in enumerate(payloads[1:], start=1):
        for key in ("clean", "labels", "sample_indices", "reference_top2"):
            if not torch.equal(first[key], payload[key]):
                raise AssertionError(
                    f"Candidate payload {position} is not aligned on tensor {key}"
                )


def balanced_subset_positions(
    labels: torch.Tensor, examples: int, seed: int, num_classes: int = NUM_CLASSES
) -> torch.Tensor:
    labels_np = labels.detach().cpu().numpy().astype(np.int64, copy=False)
    if examples <= 0 or examples % num_classes:
        raise ValueError("examples must be positive and divisible by num_classes")
    per_class = examples // num_classes
    rng = np.random.default_rng(int(seed))
    chosen: list[int] = []
    for class_id in range(num_classes):
        positions = np.flatnonzero(labels_np == class_id)
        if len(positions) < per_class:
            raise ValueError(
                f"Class {class_id} has {len(positions)} candidates; need {per_class}"
            )
        rng.shuffle(positions)
        chosen.extend(int(value) for value in positions[:per_class])
    chosen.sort()
    return torch.tensor(chosen, dtype=torch.long)


def forward_model(
    model: nn.Module,
    images: torch.Tensor,
    device: torch.device,
    microbatch: int,
) -> torch.Tensor:
    if images.ndim != 4:
        raise ValueError("forward_model expects [N,C,H,W]")
    step = len(images) if microbatch <= 0 else int(microbatch)
    pieces: list[torch.Tensor] = []
    with torch.inference_mode(), autocast_off(device):
        for low in range(0, len(images), step):
            pieces.append(model(images[low : low + step]).float())
    return torch.cat(pieces, dim=0)


def prefix_aggregates(
    logits: torch.Tensor, b_values: Sequence[int]
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """Aggregate [outer,B,N,C] draws using nested common-random prefixes."""

    if logits.ndim != 4:
        raise ValueError("Expected logits with shape [outer,B,N,C]")
    if max(b_values) > logits.shape[1]:
        raise ValueError("A requested B exceeds available inner draws")
    logit_prefix = logits.cumsum(dim=1)
    probability_prefix = logits.softmax(dim=-1).cumsum(dim=1)
    mean_logits = {
        int(b): logit_prefix[:, int(b) - 1] / float(b) for b in b_values
    }
    mean_probabilities = {
        int(b): probability_prefix[:, int(b) - 1] / float(b) for b in b_values
    }
    return mean_logits, mean_probabilities


def empty_metric_store(examples: int) -> dict[str, torch.Tensor]:
    return {
        name: torch.zeros(examples, dtype=torch.float64) for name in METRIC_NAMES
    }


def update_metric_store(
    store: dict[str, torch.Tensor],
    scores: torch.Tensor,
    labels: torch.Tensor,
    reference_top2: torch.Tensor,
) -> None:
    """Accumulate metrics for scores shaped [outer,N,C]."""

    if scores.ndim != 3:
        raise ValueError("scores must have shape [outer,N,C]")
    outer, examples, classes = scores.shape
    if classes != NUM_CLASSES:
        raise ValueError(f"Expected {NUM_CLASSES} classes, got {classes}")
    reference_set = reference_top2.sort(dim=1).values
    predictions = scores.argmax(dim=2)
    predicted_set = (
        scores.topk(TOP_K, dim=2, largest=True, sorted=False)
        .indices.sort(dim=2)
        .values
    )
    top1 = predictions.eq(labels[None]).sum(dim=0).double().cpu()
    ref_top1 = predictions.eq(reference_top2[None, :, 0]).sum(dim=0).double().cpu()
    exact = predicted_set.eq(reference_set[None]).all(dim=2).sum(dim=0).double().cpu()
    coverage = (
        predicted_set.eq(labels[None, :, None]).any(dim=2).sum(dim=0).double().cpu()
    )

    gather_index = reference_top2[None].expand(outer, -1, -1)
    selected = scores.gather(2, gather_index)
    outside = torch.ones(
        (outer, examples, classes), dtype=torch.bool, device=scores.device
    )
    outside.scatter_(2, gather_index, False)
    strongest_outside = scores.masked_fill(~outside, float("-inf")).max(dim=2).values
    margin = selected.min(dim=2).values - strongest_outside

    store["top1_accuracy_probability"] += top1
    store["reference_top1_match_probability"] += ref_top1
    store["exact_set_probability"] += exact
    store["label_coverage_probability"] += coverage
    store["boundary_margin_mean"] += margin.sum(dim=0).double().cpu()


def deterministic_metrics(
    scores: torch.Tensor, labels: torch.Tensor, reference_top2: torch.Tensor
) -> dict[str, torch.Tensor]:
    store = empty_metric_store(len(scores))
    update_metric_store(store, scores.unsqueeze(0), labels, reference_top2)
    return store


def finalize_metric_store(
    store: Mapping[str, torch.Tensor], divisor: int
) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().double() / float(divisor)
        for key, value in store.items()
    }


def evaluate_condition(
    *,
    images_cpu: torch.Tensor,
    labels_cpu: torch.Tensor,
    reference_top2_cpu: torch.Tensor,
    sample_indices_cpu: torch.Tensor,
    anchor: nn.Module,
    full_defense: Any,
    device: torch.device,
    sigma: float,
    full_sigma: float,
    b_values: Sequence[int],
    outer_draws: int,
    batch_size: int,
    outer_chunk: int,
    forward_microbatch: int,
    random_seed: int,
) -> dict[str, dict[str, torch.Tensor]]:
    """Evaluate robust, Full, and Gaussian ensembles on one fixed input set."""

    del sample_indices_cpu  # identity is checked by the caller/cache
    examples = len(images_cpu)
    maximum_b = max(int(value) for value in b_values)
    method_stores: dict[str, dict[str, torch.Tensor]] = {
        "full": empty_metric_store(examples)
    }
    for b in b_values:
        method_stores[f"gaussian_mean_logits_b{b}"] = empty_metric_store(examples)
        method_stores[f"gaussian_mean_probabilities_b{b}"] = empty_metric_store(examples)
    robust_values = {name: torch.zeros(examples, dtype=torch.float64) for name in METRIC_NAMES}

    generator = torch.Generator(device=device)
    generator.manual_seed(int(random_seed))
    for low in range(0, examples, batch_size):
        high = min(low + batch_size, examples)
        images = images_cpu[low:high].to(device)
        labels = labels_cpu[low:high].to(device)
        reference = reference_top2_cpu[low:high].to(device)

        robust_logits = forward_model(anchor, images, device, forward_microbatch)
        robust_batch = deterministic_metrics(robust_logits, labels, reference)
        for name in METRIC_NAMES:
            robust_values[name][low:high] = robust_batch[name]

        # Full's deterministic reference/strengthening is cached exactly as in
        # the existing MC evaluator; the response distribution is unchanged.
        full_base = full_defense.base(images)

        local_stores: dict[str, dict[str, torch.Tensor]] = {
            method: empty_metric_store(len(images)) for method in method_stores
        }
        completed = 0
        while completed < outer_draws:
            current_outer = min(outer_chunk, outer_draws - completed)
            noise = torch.randn(
                (current_outer, maximum_b) + tuple(images.shape),
                device=device,
                dtype=images.dtype,
                generator=generator,
            )
            noisy = (images[None, None] + float(sigma) * noise).clamp(0.0, 1.0)
            flat_logits = forward_model(
                anchor,
                noisy.flatten(0, 2),
                device,
                forward_microbatch,
            )
            logits = flat_logits.view(
                current_outer, maximum_b, len(images), NUM_CLASSES
            )
            mean_logits, mean_probabilities = prefix_aggregates(logits, b_values)
            for b in b_values:
                update_metric_store(
                    local_stores[f"gaussian_mean_logits_b{b}"],
                    mean_logits[int(b)],
                    labels,
                    reference,
                )
                update_metric_store(
                    local_stores[f"gaussian_mean_probabilities_b{b}"],
                    mean_probabilities[int(b)],
                    labels,
                    reference,
                )

            # Pair Full with the first Gaussian draw of every outer response.
            full_noisy = (
                full_base[None] + float(full_sigma) * noise[:, 0]
            ).clamp(0.0, 1.0)
            full_logits = forward_model(
                anchor,
                full_noisy.flatten(0, 1),
                device,
                forward_microbatch,
            ).view(current_outer, len(images), NUM_CLASSES)
            update_metric_store(
                local_stores["full"], full_logits, labels, reference
            )
            completed += current_outer

        for method, store in local_stores.items():
            finalized = finalize_metric_store(store, outer_draws)
            for name in METRIC_NAMES:
                method_stores[method][name][low:high] = finalized[name]
        print(
            f"    evaluated {high}/{examples} examples; outer={outer_draws}, "
            f"Bmax={maximum_b}",
            flush=True,
        )

    return {"robust": robust_values, **method_stores}


def condition_cache_identity(
    *,
    condition: str,
    images: torch.Tensor,
    labels: torch.Tensor,
    reference_top2: torch.Tensor,
    sample_indices: torch.Tensor,
    anchor_sha256: str,
    standard_sha256: str,
    frozen_defense: Mapping[str, Any],
    sigma: float,
    full_sigma: float,
    b_values: Sequence[int],
    args: argparse.Namespace,
    random_seed: int,
) -> dict[str, Any]:
    return {
        "script_version": SCRIPT_VERSION,
        "condition": condition,
        "tensor_sha256": {
            "images": tensor_sha256(images),
            "labels": tensor_sha256(labels),
            "reference_top2": tensor_sha256(reference_top2),
            "sample_indices": tensor_sha256(sample_indices),
        },
        "checkpoint_sha256": {
            "anchor": anchor_sha256,
            "standard": standard_sha256,
        },
        "frozen_defense": dict(frozen_defense),
        "sigma": float(sigma),
        "full_sigma": float(full_sigma),
        "b_values": [int(value) for value in b_values],
        "outer_draws": int(args.outer_draws),
        "eval_batch_size": int(args.eval_batch_size),
        "outer_chunk": int(args.outer_chunk),
        "forward_microbatch": int(args.forward_microbatch),
        "random_seed": int(random_seed),
        "aggregation_primary": "mean_logits",
        "aggregation_secondary": "mean_probabilities",
    }


def validate_cached_results(payload: Mapping[str, Any], identity: Mapping[str, Any]) -> None:
    if payload.get("identity") != identity:
        raise AssertionError("Cache identity mismatch")
    results = payload.get("results")
    if not isinstance(results, Mapping) or not results:
        raise AssertionError("Cache results are absent")
    examples = len(next(iter(next(iter(results.values())).values())))
    for method, metrics in results.items():
        if set(metrics) != set(METRIC_NAMES):
            raise AssertionError(f"Cache metric mismatch for {method}")
        for name, values in metrics.items():
            if values.shape != (examples,) or not torch.isfinite(values).all():
                raise AssertionError(f"Invalid cached tensor: {method}/{name}")


def cache_or_evaluate_condition(
    *,
    output_dir: Path,
    identity: Mapping[str, Any],
    overwrite: bool,
    evaluator_kwargs: Mapping[str, Any],
) -> tuple[dict[str, dict[str, torch.Tensor]], float, Path]:
    identity_hash = object_sha256(identity)
    safe_condition = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in str(identity["condition"])
    )
    path = output_dir / "cache" / f"{safe_condition}_{identity_hash[:16]}.pt"
    if path.is_file() and not overwrite:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        validate_cached_results(payload, identity)
        print(f"  RESUME {path.name}", flush=True)
        return dict(payload["results"]), float(payload["wall_seconds"]), path
    started = time.monotonic()
    results = evaluate_condition(**evaluator_kwargs)
    wall_seconds = float(time.monotonic() - started)
    payload = {
        "identity": dict(identity),
        "identity_hash": identity_hash,
        "results": results,
        "wall_seconds": wall_seconds,
    }
    validate_cached_results(payload, identity)
    atomic_torch_save(path, payload)
    return results, wall_seconds, path


def method_metadata(method: str) -> tuple[str, int | str]:
    if method == "robust":
        return "deterministic", 0
    if method == "full":
        return "single_noisy_full", 1
    prefix_logits = "gaussian_mean_logits_b"
    prefix_probabilities = "gaussian_mean_probabilities_b"
    if method.startswith(prefix_logits):
        return "mean_logits", int(method[len(prefix_logits) :])
    if method.startswith(prefix_probabilities):
        return "mean_probabilities", int(method[len(prefix_probabilities) :])
    return "unknown", ""


def result_rows(
    *,
    condition: str,
    candidate_seed: int | str,
    sample_indices: torch.Tensor,
    labels: torch.Tensor,
    reference_top2: torch.Tensor,
    results: Mapping[str, Mapping[str, torch.Tensor]],
    sigma: float,
    full_sigma: float,
    outer_draws: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_example: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for method, metrics in sorted(results.items()):
        aggregation, b_value = method_metadata(method)
        method_sigma = (
            0.0 if method == "robust" else float(full_sigma) if method == "full" else float(sigma)
        )
        row = {
            "condition": condition,
            "candidate_seed": candidate_seed,
            "method": method,
            "aggregation": aggregation,
            "B": b_value,
            "sigma": method_sigma,
            "examples": len(sample_indices),
            "outer_draws": 1 if method == "robust" else int(outer_draws),
        }
        for name in METRIC_NAMES:
            scale = 1.0 if name == "boundary_margin_mean" else 100.0
            row[name + ("" if scale == 1.0 else "_percent")] = (
                scale * float(metrics[name].mean().item())
            )
        summary.append(row)
        for position in range(len(sample_indices)):
            reference = [int(value) for value in reference_top2[position].tolist()]
            item = {
                "condition": condition,
                "candidate_seed": candidate_seed,
                "sample_index": int(sample_indices[position]),
                "label": int(labels[position]),
                "reference_top1": reference[0],
                "reference_set": f"{min(reference)}|{max(reference)}",
                "method": method,
                "aggregation": aggregation,
                "B": b_value,
                "sigma": method_sigma,
                "outer_draws": 1 if method == "robust" else int(outer_draws),
            }
            for name in METRIC_NAMES:
                item[name] = float(metrics[name][position].item())
            per_example.append(item)
    return per_example, summary


def aggregate_replay_results(
    replay_results: Sequence[Mapping[str, Mapping[str, torch.Tensor]]]
) -> dict[str, dict[str, torch.Tensor]]:
    if not replay_results:
        raise ValueError("No replay results to aggregate")
    methods = set(replay_results[0])
    if any(set(result) != methods for result in replay_results[1:]):
        raise AssertionError("Replay method sets differ across seeds")
    aggregated: dict[str, dict[str, torch.Tensor]] = {}
    for method in sorted(methods):
        aggregated[method] = {}
        for metric in METRIC_NAMES:
            values = torch.stack(
                [result[method][metric].double() for result in replay_results], dim=0
            )
            aggregated[method][metric] = values.mean(dim=0)
    return aggregated


def ensemble_logits_once(
    *,
    anchor: nn.Module,
    images: torch.Tensor,
    sigma: float,
    b_value: int,
    device: torch.device,
    forward_microbatch: int,
) -> torch.Tensor:
    noise = torch.randn(
        (int(b_value),) + tuple(images.shape),
        device=device,
        dtype=images.dtype,
    )
    noisy = (images[None] + float(sigma) * noise).clamp(0.0, 1.0)
    logits = forward_model(
        anchor, noisy.flatten(0, 1), device, forward_microbatch
    ).view(int(b_value), len(images), NUM_CLASSES)
    return logits.mean(dim=0)


def latency_samples(
    callable_fn: Any,
    *,
    warmup: int,
    repeats: int,
    device: torch.device,
) -> tuple[list[float], float]:
    for _ in range(int(warmup)):
        output = callable_fn()
        del output
    cuda_sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    values: list[float] = []
    for _ in range(int(repeats)):
        cuda_sync(device)
        started = time.perf_counter()
        output = callable_fn()
        cuda_sync(device)
        values.append(1000.0 * (time.perf_counter() - started))
        del output
    peak_mb = (
        float(torch.cuda.max_memory_allocated(device)) / (1024.0**2)
        if device.type == "cuda"
        else float("nan")
    )
    return values, peak_mb


def summarize_latency(
    method: str,
    b_value: int | str,
    batch_size: int,
    values: Sequence[float],
    peak_memory_mb: float,
) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "method": method,
        "B": b_value,
        "batch_size": int(batch_size),
        "repeats": len(values),
        "latency_mean_ms": float(array.mean()),
        "latency_std_ms": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "latency_p50_ms": float(np.percentile(array, 50)),
        "latency_p95_ms": float(np.percentile(array, 95)),
        "latency_min_ms": float(array.min()),
        "peak_memory_mb": float(peak_memory_mb),
    }


def benchmark_latency(
    *,
    clean_images_cpu: torch.Tensor,
    anchor: nn.Module,
    full_defense: Any,
    device: torch.device,
    sigma: float,
    b_values: Sequence[int],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for batch_size in args.latency_batch_sizes:
        if len(clean_images_cpu) < batch_size:
            raise ValueError(
                f"Need at least {batch_size} clean examples for latency benchmarking"
            )
        images = clean_images_cpu[:batch_size].to(device)
        print(f"  Latency batch={batch_size}: robust", flush=True)
        values, peak = latency_samples(
            lambda: forward_model(
                anchor, images, device, args.forward_microbatch
            ),
            warmup=args.latency_warmup,
            repeats=args.latency_repeats,
            device=device,
        )
        rows.append(summarize_latency("robust", 0, batch_size, values, peak))

        print(f"  Latency batch={batch_size}: Full", flush=True)
        values, peak = latency_samples(
            lambda: full_defense.actual(images),
            warmup=args.latency_warmup,
            repeats=args.latency_repeats,
            device=device,
        )
        rows.append(summarize_latency("full", 1, batch_size, values, peak))

        for b_value in b_values:
            print(
                f"  Latency batch={batch_size}: Gaussian ensemble B={b_value}",
                flush=True,
            )
            values, peak = latency_samples(
                lambda b=int(b_value): ensemble_logits_once(
                    anchor=anchor,
                    images=images,
                    sigma=sigma,
                    b_value=b,
                    device=device,
                    forward_microbatch=args.forward_microbatch,
                ),
                warmup=args.latency_warmup,
                repeats=args.latency_repeats,
                device=device,
            )
            rows.append(
                summarize_latency(
                    "gaussian_mean_logits", int(b_value), batch_size, values, peak
                )
            )
    robust_by_batch = {
        int(row["batch_size"]): float(row["latency_p50_ms"])
        for row in rows
        if row["method"] == "robust"
    }
    full_by_batch = {
        int(row["batch_size"]): float(row["latency_p50_ms"])
        for row in rows
        if row["method"] == "full"
    }
    for row in rows:
        batch_size = int(row["batch_size"])
        row["p50_x_robust"] = float(row["latency_p50_ms"]) / robust_by_batch[batch_size]
        row["p50_x_full"] = float(row["latency_p50_ms"]) / full_by_batch[batch_size]
    return rows


def select_compute_matched_b(
    rows: Sequence[Mapping[str, Any]],
    *,
    selection_batch_size: int,
    tolerance: float,
) -> dict[str, Any]:
    full_rows = [
        row
        for row in rows
        if row["method"] == "full" and int(row["batch_size"]) == selection_batch_size
    ]
    ensemble_rows = [
        row
        for row in rows
        if row["method"] == "gaussian_mean_logits"
        and int(row["batch_size"]) == selection_batch_size
    ]
    if len(full_rows) != 1 or not ensemble_rows:
        raise ValueError("Incomplete latency rows for B selection")
    target = float(full_rows[0]["latency_p50_ms"])
    cap = target * (1.0 + float(tolerance))
    eligible = [row for row in ensemble_rows if float(row["latency_p50_ms"]) <= cap]
    if eligible:
        selected = max(eligible, key=lambda row: int(row["B"]))
        rule = "largest_B_with_p50_not_above_full_times_1_plus_tolerance"
    else:
        selected = min(
            ensemble_rows,
            key=lambda row: abs(float(row["latency_p50_ms"]) - target),
        )
        rule = "closest_p50_no_B_met_cap"
    closest = min(
        ensemble_rows,
        key=lambda row: abs(float(row["latency_p50_ms"]) - target),
    )
    return {
        "selection_batch_size": int(selection_batch_size),
        "tolerance": float(tolerance),
        "full_target_p50_ms": target,
        "latency_cap_ms": cap,
        "selected_B": int(selected["B"]),
        "selected_p50_ms": float(selected["latency_p50_ms"]),
        "selected_rule": rule,
        "closest_B": int(closest["B"]),
        "closest_p50_ms": float(closest["latency_p50_ms"]),
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    r5_root, protocol_path, output_dir = resolve_paths(args)
    if not protocol_path.is_file():
        raise FileNotFoundError(f"R5 protocol not found: {protocol_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = read_json(protocol_path)
    if protocol.get("dataset") != "CIFAR-100" or int(protocol.get("top_k", -1)) != TOP_K:
        raise AssertionError("Expected the frozen CIFAR-100 Top-2 R5 protocol")

    sigma = (
        float(args.sigma)
        if args.sigma is not None
        else float(protocol["frozen_defense"]["sigma"])
    )
    if sigma < 0:
        raise ValueError("Resolved sigma cannot be negative")
    seed_everything(args.subset_seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, device)

    attack, candidate_seeds, paths = candidate_paths(
        protocol=protocol,
        r5_root=r5_root,
        method=args.candidate_method,
        attack_override=args.candidate_attack,
        seeds_override=args.candidate_seeds,
    )
    print("Loading R5 candidate payloads", flush=True)
    payloads = [load_candidate(path) for path in paths]
    validate_candidate_alignment(payloads)
    first = payloads[0]
    positions = balanced_subset_positions(
        first["labels"], args.examples, args.subset_seed
    )
    clean = first["clean"].index_select(0, positions)
    labels = first["labels"].index_select(0, positions).long()
    sample_indices = first["sample_indices"].index_select(0, positions).long()
    reference_top2 = first["reference_top2"].index_select(0, positions).long()

    checkpoints = protocol["checkpoints"]
    standard_path = Path(checkpoints["standard"]["path"])
    anchor_path = Path(checkpoints["anchor"]["path"])
    verify_checkpoint(
        standard_path,
        checkpoints["standard"].get("sha256"),
        args.skip_checkpoint_hash,
    )
    verify_checkpoint(
        anchor_path,
        checkpoints["anchor"].get("sha256"),
        args.skip_checkpoint_hash,
    )
    r1, r4_common, FrozenDefense = load_project_modules(args.project_dir)
    print("Loading frozen standard and robust-anchor checkpoints", flush=True)
    standard, anchor = r1.load_models(standard_path, anchor_path, device)
    full_defense = FrozenDefense(
        r4_common.method_spec("full"), standard, anchor, device
    )

    run_config = {
        "script_version": SCRIPT_VERSION,
        "exploratory_only": True,
        "adaptive_attack_result": False,
        "r5_protocol_id": protocol.get("protocol_id"),
        "r5_protocol_hash": protocol.get("protocol_hash"),
        "r5_root": str(r5_root),
        "protocol_json": str(protocol_path),
        "output_dir": str(output_dir),
        "device": str(device),
        "candidate_method": args.candidate_method,
        "candidate_attack": attack,
        "candidate_seeds": list(candidate_seeds),
        "candidate_paths": [str(path) for path in paths],
        "examples": int(args.examples),
        "sample_indices_sha256": tensor_sha256(sample_indices),
        "b_values": [int(value) for value in args.b_values],
        "sigma": float(sigma),
        "full_sigma": float(full_defense.sigma),
        "outer_draws": int(args.outer_draws),
        "subset_seed": int(args.subset_seed),
        "noise_seed": int(args.noise_seed),
        "eval_batch_size": int(args.eval_batch_size),
        "outer_chunk": int(args.outer_chunk),
        "forward_microbatch": int(args.forward_microbatch),
        "latency_batch_sizes": list(args.latency_batch_sizes),
        "selection_batch_size": int(args.selection_batch_size),
        "latency_warmup": int(args.latency_warmup),
        "latency_repeats": int(args.latency_repeats),
        "latency_tolerance": float(args.latency_tolerance),
        "checkpoint_sha256": {
            "standard": checkpoints["standard"].get("sha256"),
            "anchor": checkpoints["anchor"].get("sha256"),
        },
        "frozen_defense": protocol["frozen_defense"],
        "inner_B_definition": "independent_Gaussian_draws_averaged_inside_one_API_response",
        "outer_MC_definition": "independent_API_responses_used_only_for_evaluation",
        "primary_aggregation": "mean_logits",
        "secondary_aggregation": "mean_probabilities",
    }
    run_config["config_hash"] = object_sha256(run_config)
    atomic_json(output_dir / "run_config.json", run_config)

    latency_rows: list[dict[str, Any]] = []
    latency_selection: dict[str, Any] | None = None
    if args.stage in {"latency", "all"}:
        print("\n[1/2] Same-session latency benchmark", flush=True)
        latency_rows = benchmark_latency(
            clean_images_cpu=clean,
            anchor=anchor,
            full_defense=full_defense,
            device=device,
            sigma=sigma,
            b_values=args.b_values,
            args=args,
        )
        latency_selection = select_compute_matched_b(
            latency_rows,
            selection_batch_size=args.selection_batch_size,
            tolerance=args.latency_tolerance,
        )
        atomic_csv(output_dir / "latency.csv", latency_rows)
        atomic_json(output_dir / "latency_selection.json", latency_selection)
        print(
            f"  Selected compute-matched B*={latency_selection['selected_B']} "
            f"at batch={args.selection_batch_size}",
            flush=True,
        )

    per_example_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    cache_files: list[str] = []
    if args.stage in {"screen", "all"}:
        print("\n[2/2] Clean and saved-candidate screening", flush=True)
        base_evaluator = {
            "labels_cpu": labels,
            "reference_top2_cpu": reference_top2,
            "sample_indices_cpu": sample_indices,
            "anchor": anchor,
            "full_defense": full_defense,
            "device": device,
            "sigma": sigma,
            "full_sigma": float(full_defense.sigma),
            "b_values": args.b_values,
            "outer_draws": args.outer_draws,
            "batch_size": args.eval_batch_size,
            "outer_chunk": args.outer_chunk,
            "forward_microbatch": args.forward_microbatch,
        }

        clean_seed = int(args.noise_seed)
        clean_identity = condition_cache_identity(
            condition="clean",
            images=clean,
            labels=labels,
            reference_top2=reference_top2,
            sample_indices=sample_indices,
            anchor_sha256=str(checkpoints["anchor"].get("sha256", "")),
            standard_sha256=str(checkpoints["standard"].get("sha256", "")),
            frozen_defense=protocol["frozen_defense"],
            sigma=sigma,
            full_sigma=float(full_defense.sigma),
            b_values=args.b_values,
            args=args,
            random_seed=clean_seed,
        )
        print("  Condition: clean", flush=True)
        clean_results, clean_wall, cache_path = cache_or_evaluate_condition(
            output_dir=output_dir,
            identity=clean_identity,
            overwrite=args.overwrite_cache,
            evaluator_kwargs={
                **base_evaluator,
                "images_cpu": clean,
                "random_seed": clean_seed,
            },
        )
        cache_files.append(str(cache_path))
        rows, summaries = result_rows(
            condition="clean",
            candidate_seed="",
            sample_indices=sample_indices,
            labels=labels,
            reference_top2=reference_top2,
            results=clean_results,
            sigma=sigma,
            full_sigma=float(full_defense.sigma),
            outer_draws=args.outer_draws,
        )
        for row in summaries:
            row["wall_seconds"] = clean_wall
        per_example_rows.extend(rows)
        summary_rows.extend(summaries)

        replay_results: list[dict[str, dict[str, torch.Tensor]]] = []
        replay_walls: list[float] = []
        for order, (candidate_seed, payload, path) in enumerate(
            zip(candidate_seeds, payloads, paths), start=1
        ):
            adversarial = payload["adversarial"].index_select(0, positions)
            condition = f"replay_{args.candidate_method}_{attack}_seed_{candidate_seed}"
            condition_seed = int(args.noise_seed + 1000 * order)
            identity = condition_cache_identity(
                condition=condition,
                images=adversarial,
                labels=labels,
                reference_top2=reference_top2,
                sample_indices=sample_indices,
                anchor_sha256=str(checkpoints["anchor"].get("sha256", "")),
                standard_sha256=str(checkpoints["standard"].get("sha256", "")),
                frozen_defense=protocol["frozen_defense"],
                sigma=sigma,
                full_sigma=float(full_defense.sigma),
                b_values=args.b_values,
                args=args,
                random_seed=condition_seed,
            )
            print(f"  Condition: {condition}", flush=True)
            results, wall_seconds, cache_path = cache_or_evaluate_condition(
                output_dir=output_dir,
                identity=identity,
                overwrite=args.overwrite_cache,
                evaluator_kwargs={
                    **base_evaluator,
                    "images_cpu": adversarial,
                    "random_seed": condition_seed,
                },
            )
            cache_files.append(str(cache_path))
            replay_results.append(results)
            replay_walls.append(wall_seconds)
            rows, summaries = result_rows(
                condition="candidate_replay",
                candidate_seed=int(candidate_seed),
                sample_indices=sample_indices,
                labels=labels,
                reference_top2=reference_top2,
                results=results,
                sigma=sigma,
                full_sigma=float(full_defense.sigma),
                outer_draws=args.outer_draws,
            )
            for row in summaries:
                row["candidate_file"] = str(path)
                row["wall_seconds"] = wall_seconds
            per_example_rows.extend(rows)
            summary_rows.extend(summaries)

        replay_mean = aggregate_replay_results(replay_results)
        rows, summaries = result_rows(
            condition="candidate_replay_mean_over_seeds",
            candidate_seed="mean",
            sample_indices=sample_indices,
            labels=labels,
            reference_top2=reference_top2,
            results=replay_mean,
            sigma=sigma,
            full_sigma=float(full_defense.sigma),
            outer_draws=args.outer_draws,
        )
        for row in summaries:
            row["wall_seconds"] = float(sum(replay_walls))
        per_example_rows.extend(rows)
        summary_rows.extend(summaries)
        atomic_csv(output_dir / "screening_per_example.csv", per_example_rows)
        atomic_csv(output_dir / "screening_summary.csv", summary_rows)

    final_summary = {
        "script_version": SCRIPT_VERSION,
        "config_hash": run_config["config_hash"],
        "exploratory_only": True,
        "adaptive_attack_result": False,
        "warning": (
            "Candidate replay is a transfer/screening diagnostic. It cannot support a "
            "robustness claim until the Gaussian ensemble is attacked directly."
        ),
        "latency_selection": latency_selection,
        "latency_rows": len(latency_rows),
        "screening_summary_rows": len(summary_rows),
        "screening_per_example_rows": len(per_example_rows),
        "cache_files": cache_files,
        "next_gate": (
            "Run a direct adaptive pilot only if the latency-matched mean-logit "
            "ensemble is competitive with Full on clean/candidate replay."
        ),
    }
    atomic_json(output_dir / "final_summary.json", final_summary)
    print(f"\nDONE: {output_dir}", flush=True)
    if latency_selection is not None:
        print(
            f"Compute-matched B*: {latency_selection['selected_B']} "
            f"(Full p50={latency_selection['full_target_p50_ms']:.3f} ms, "
            f"ensemble p50={latency_selection['selected_p50_ms']:.3f} ms)",
            flush=True,
        )
    print(
        "IMPORTANT: replay results are exploratory transfer diagnostics, not adaptive robustness.",
        flush=True,
    )


if __name__ == "__main__":
    main()
