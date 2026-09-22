#!/usr/bin/env python3
"""Paper-facing clean utility and inference-cost evaluation on all CIFAR-10 test images.

This protocol is deliberately separate from the 200-image confirmatory attack
holdout.  It evaluates the complete canonical CIFAR-10 test split and never
loads, selects, or tunes on attack candidates.

Primary clean metrics are true-label Top-1 accuracy and per-request Exact
Top-2 set fidelity to the clean Standard model.  Standard and Robust are
deterministic.  Noise-only and Full use 100 independent Gaussian draws per
image.  Full and Noise-only use common random numbers within each resume-safe
shard.  Paired bootstrap intervals are over the 10,000 test images.

The evaluator also benchmarks uncached, single-request inference.  A
semantics-preserving measurement optimization reuses Full's deterministic
strengthening result across Monte-Carlo draws, but the latency benchmark calls
the complete defense independently for every request.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_ROOT = Path("./models/resnet20")
DEFAULT_DATA_DIR = Path(
    "./datasets/cifar-10-batches-py"
)
DEFAULT_STANDARD_WEIGHTS = MODEL_ROOT / "cifar10/resnet20_cifar10.pth"
DEFAULT_DUAL_WEIGHTS = (
    MODEL_ROOT
    / "fine_frontier_top1_set2/resnet20_staged_epsilon_final_weights.pt"
)
DEFAULT_OUTPUT_DIR = (
    MODEL_ROOT / "paper_topk_full_clean_utility_cifar10_k2_mc100_r1"
)

PROTOCOL = "topk_full_clean_utility_cifar10_v1"
EXAMPLES = 10_000
CLASSES = 10
EXAMPLES_PER_CLASS = 1_000
K = 2
EVAL_DRAWS = 100
BATCH_SIZE = 25
SHARD_SIZE = 500
SHARD_COUNT = EXAMPLES // SHARD_SIZE
EVAL_SEED = 2026082902
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260830
BOOTSTRAP_CHUNK = 32
RANK_TEMPERATURE = 0.5
SIGMA = 0.02144277648925781
STRENGTH_STEPS = 2
STRENGTH_ALPHA = 0.005
STRENGTH_EPSILON = 0.03
TOP1_WEIGHT = 1.0
SET_WEIGHT = 4.0
GPU_MEMORY_FRACTION = 0.45
BENCHMARK_BATCH_SIZES = (1, 25)
BENCHMARK_WARMUPS = 20
BENCHMARK_BLOCKS = 10
BENCHMARK_REPEATS_PER_BLOCK = 20
SOURCES = ("standard", "robust", "noise_only", "full")

DEPENDENCY_HASHES = {
    "evaluate_general_topk_defense_v6_cifar10.py": (
        "bfbe2815e60bed15cbeb205b60579f84ead96c63be4c19cde49e0cb71676a8df"
    ),
    "evaluate_final_top1_set2_defense_cifar10.py": (
        "6c2cbe148ff7fcaf7de02b830ccaf47a79d27d25f4c142981f635dca96e6a6c4"
    ),
    "evaluate_top1_set2_gated_cifar10.py": (
        "e65e8073e6fb7817455e1f904758943600c50a6d15887147a76a32ff0082d581"
    ),
    "train_resnet20_dual_backbone_top1_set2_cifar10.py": (
        "cbfcad9db073f36f3b3861c057633944cb3efd9ac0e9e71894b0c7bc2fb58b0b"
    ),
}

SOURCE_DEFINITIONS = {
    "standard": {
        "stochastic": False,
        "draws": 1,
        "standard_forwards": 1,
        "robust_forwards": 0,
        "input_gradient_steps": 0,
        "deployed_models": "standard",
        "implementation": "standard_resnet20",
    },
    "robust": {
        "stochastic": False,
        "draws": 1,
        "standard_forwards": 0,
        "robust_forwards": 1,
        "input_gradient_steps": 0,
        "deployed_models": "robust",
        "implementation": "standalone_robust_pixel_backbone",
    },
    "noise_only": {
        "stochastic": True,
        "draws": EVAL_DRAWS,
        "standard_forwards": 0,
        "robust_forwards": 1,
        "input_gradient_steps": 0,
        "deployed_models": "robust",
        "implementation": "optimized_noise_only_equivalent_to_strength_steps_0",
    },
    "full": {
        "stochastic": True,
        "draws": EVAL_DRAWS,
        "standard_forwards": 1,
        "robust_forwards": 3,
        "input_gradient_steps": 2,
        "deployed_models": "standard+robust",
        "implementation": "targeted_strengthening_2step_plus_gaussian",
    },
}

SUMMARY_FILE = "full_clean_utility_summary.csv"
EXAMPLES_FILE = "full_clean_utility_per_example.csv"
COMPARISONS_FILE = "full_clean_utility_comparisons.csv"
CLASSWISE_FILE = "full_clean_utility_classwise.csv"
LATENCY_FILE = "full_clean_utility_latency.csv"
FINAL_FILE = "full_clean_utility_summary.json"
CONFIG_FILE = "run_config_full_clean_utility.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate clean Top-k utility and inference cost on CIFAR-10 test."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--standard-weights", type=Path, default=DEFAULT_STANDARD_WEIGHTS
    )
    parser.add_argument("--dual-weights", type=Path, default=DEFAULT_DUAL_WEIGHTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--eval-draws", type=int, default=EVAL_DRAWS)
    parser.add_argument("--shard-size", type=int, default=SHARD_SIZE)
    parser.add_argument(
        "--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES
    )
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--eval-seed", type=int, default=EVAL_SEED)
    parser.add_argument(
        "--gpu-memory-fraction", type=float, default=GPU_MEMORY_FRACTION
    )
    parser.add_argument("--progress-every-batches", type=int, default=5)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Load data/models and verify optimized implementations without evaluation.",
    )
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with tempfile.NamedTemporaryFile(
        "w", newline="", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def acquire_lock(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / ".full_clean_utility.lock"
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.seek(0)
        owner = handle.read().strip() or "unknown"
        raise SystemExit(f"Full clean utility evaluation is active (PID {owner}).") from exc
    handle.seek(0)
    handle.truncate()
    handle.write(str(os.getpid()))
    handle.flush()
    return handle


def close(a: Any, b: Any, tolerance: float = 2e-6) -> bool:
    return abs(float(a) - float(b)) <= tolerance


def validate_dependencies() -> dict[str, str]:
    observed: dict[str, str] = {}
    for name, expected in DEPENDENCY_HASHES.items():
        path = PROJECT_DIR / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing bundled dependency: {path}")
        current = sha256_file(path)
        observed[name] = current
        if current != expected:
            raise ValueError(f"Bundled dependency hash differs for {name}: {current}")
    return observed


def validate_args(args: argparse.Namespace) -> None:
    fixed = {
        "batch_size": BATCH_SIZE,
        "eval_draws": EVAL_DRAWS,
        "shard_size": SHARD_SIZE,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "eval_seed": EVAL_SEED,
    }
    mismatches = {
        key: (getattr(args, key), value)
        for key, value in fixed.items()
        if getattr(args, key) != value
    }
    if mismatches:
        raise ValueError(f"Paper protocol arguments are frozen: {mismatches}")
    if args.progress_every_batches < 1:
        raise ValueError("progress_every_batches must be positive")
    if not 0 < args.gpu_memory_fraction <= 1:
        raise ValueError("gpu_memory_fraction must be in (0,1]")
    for path in (args.standard_weights, args.dual_weights):
        if not path.expanduser().is_file():
            raise FileNotFoundError(f"Missing model checkpoint: {path}")


def dataset_manifest(dataset: Any) -> dict[str, Any]:
    data = np.ascontiguousarray(dataset.data, dtype=np.uint8)
    targets = np.asarray(dataset.targets, dtype=np.int64)
    if data.shape != (EXAMPLES, 32, 32, 3) or targets.shape != (EXAMPLES,):
        raise ValueError(
            f"Unexpected CIFAR-10 test shapes: data={data.shape}, targets={targets.shape}"
        )
    counts = {str(label): int(np.sum(targets == label)) for label in range(CLASSES)}
    if any(value != EXAMPLES_PER_CLASS for value in counts.values()):
        raise ValueError(f"CIFAR-10 test labels are not balanced: {counts}")
    return {
        "dataset": "CIFAR-10",
        "split": "test",
        "examples": EXAMPLES,
        "canonical_indices": "0..9999",
        "class_counts": counts,
        "images_uint8_sha256": sha256_bytes(data.tobytes()),
        "targets_int64_sha256": sha256_bytes(targets.tobytes()),
    }


def tensor_shard(torch: Any, dataset: Any, shard_index: int):
    if not 0 <= shard_index < SHARD_COUNT:
        raise ValueError(f"Invalid shard index: {shard_index}")
    low = shard_index * SHARD_SIZE
    high = low + SHARD_SIZE
    array = np.ascontiguousarray(dataset.data[low:high], dtype=np.uint8)
    images = torch.from_numpy(array).permute(0, 3, 1, 2).float().div_(255.0)
    labels = torch.as_tensor(dataset.targets[low:high], dtype=torch.long)
    indices = torch.arange(low, high, dtype=torch.long)
    if len(images) != SHARD_SIZE:
        raise RuntimeError("Incomplete CIFAR-10 shard")
    return images, labels, indices


def parameter_count(model: Any) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def build_models(evaluator: Any, torch: Any, args: argparse.Namespace, sample: Any):
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no GPU is available")
        torch.cuda.set_device(device)
        index = device.index if device.index is not None else 0
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, index)

    standard_model, standard_arch = evaluator.legacy.load_resnet20(
        args.standard_weights,
        device,
        evaluator.legacy.STANDARD_CIFAR10_STD,
    )
    dual = evaluator.base.load_dual_defense(args.dual_weights, device)
    legacy_robust = evaluator.base.RobustPixelModel(dual).to(device).eval()

    class StandaloneRobustPixelModel(torch.nn.Module):
        def __init__(self, robust_branch: Any, mean: Any, std: Any):
            super().__init__()
            self.robust_branch = robust_branch
            self.register_buffer("mean", mean.detach().clone())
            self.register_buffer("std", std.detach().clone())

        def forward(self, x: Any):
            return self.robust_branch((x - self.mean) / self.std)

    robust_model = StandaloneRobustPixelModel(
        dual.robust_branch, dual.mean, dual.std
    ).to(device).eval()
    for parameter in robust_model.parameters():
        parameter.requires_grad_(False)

    class PaperTargetedGaussianDefense:
        def __init__(self, steps: int):
            self.standard_model = standard_model
            self.gradient_model = robust_model
            self.output_model = robust_model
            self.top_k = K
            self.alpha = STRENGTH_ALPHA
            self.epsilon = STRENGTH_EPSILON
            self.steps = int(steps)
            self.sigma = SIGMA
            self.top1_weight = TOP1_WEIGHT
            self.set_weight = SET_WEIGHT

        def objective(self, logits: Any, reference_topk: Any):
            selected = logits.gather(1, reference_topk)
            reference_top1 = reference_topk[:, :1]
            top1_logit = selected[:, 0]
            not_top1 = torch.ones_like(logits, dtype=torch.bool)
            not_top1.scatter_(1, reference_top1, False)
            strongest_not_top1 = logits.masked_fill(
                ~not_top1, float("-inf")
            ).max(dim=1).values
            outside = torch.ones_like(logits, dtype=torch.bool)
            outside.scatter_(1, reference_topk, False)
            weakest_selected = selected.min(dim=1).values
            strongest_outside = logits.masked_fill(
                ~outside, float("-inf")
            ).max(dim=1).values
            top1_margin = top1_logit - strongest_not_top1
            set_margin = weakest_selected - strongest_outside
            return self.top1_weight * top1_margin + self.set_weight * set_margin

        def prepare_for_mc(self, x: Any):
            clean = x.detach()
            if self.steps == 0:
                return clean
            with torch.no_grad():
                reference_topk = self.standard_model(clean).topk(
                    self.top_k, dim=1
                ).indices
            strengthened = clean.clone()
            for _ in range(self.steps):
                with torch.enable_grad():
                    variable = strengthened.detach().requires_grad_(True)
                    objective = self.objective(
                        self.gradient_model(variable), reference_topk
                    )
                    gradient = torch.autograd.grad(
                        objective.sum(), variable, only_inputs=True
                    )[0]
                strengthened = strengthened + self.alpha * gradient.sign()
                delta = (strengthened - clean).clamp(-self.epsilon, self.epsilon)
                strengthened = (clean + delta).clamp(0.0, 1.0).detach()
            return strengthened

        def sample_prepared(self, prepared: Any):
            noised = (
                prepared + self.sigma * torch.randn_like(prepared)
            ).clamp(0.0, 1.0)
            with torch.no_grad():
                return self.output_model(noised)

        def __call__(self, x: Any):
            return self.sample_prepared(self.prepare_for_mc(x))

    noise_actual = PaperTargetedGaussianDefense(steps=0)
    full_actual = PaperTargetedGaussianDefense(steps=STRENGTH_STEPS)
    source_objects = {
        "standard": evaluator.base.DefenseSpec(
            "standard", evaluator.base.make_actual(standard_model), standard_model, 0.0
        ),
        "robust": evaluator.base.DefenseSpec(
            "robust", evaluator.base.make_actual(robust_model), robust_model, 0.0
        ),
        "noise_only": evaluator.base.DefenseSpec(
            "noise_only", noise_actual, robust_model, SIGMA
        ),
        "full": evaluator.base.DefenseSpec("full", full_actual, robust_model, SIGMA),
    }

    sample = sample.to(device)
    evaluator.base.seed_everything(EVAL_SEED)
    with torch.no_grad():
        legacy_robust_logits = legacy_robust(sample).detach()
        standalone_logits = robust_model(sample).detach()
    robust_diff = float(
        (legacy_robust_logits - standalone_logits).abs().max().item()
    )

    legacy_noise = evaluator.TopKTargetedGaussianDefense(
        standard_model=standard_model,
        gradient_model=robust_model,
        output_model=robust_model,
        top_k=K,
        alpha=STRENGTH_ALPHA,
        epsilon=STRENGTH_EPSILON,
        steps=0,
        sigma=SIGMA,
        top1_weight=TOP1_WEIGHT,
        set_weight=SET_WEIGHT,
    )
    legacy_full = evaluator.TopKTargetedGaussianDefense(
        standard_model=standard_model,
        gradient_model=robust_model,
        output_model=robust_model,
        top_k=K,
        alpha=STRENGTH_ALPHA,
        epsilon=STRENGTH_EPSILON,
        steps=STRENGTH_STEPS,
        sigma=SIGMA,
        top1_weight=TOP1_WEIGHT,
        set_weight=SET_WEIGHT,
    )
    evaluator.base.seed_everything(EVAL_SEED + 1)
    old_noise = legacy_noise(sample).detach()
    evaluator.base.seed_everything(EVAL_SEED + 1)
    new_noise = noise_actual(sample).detach()
    noise_diff = float((old_noise - new_noise).abs().max().item())
    evaluator.base.seed_everything(EVAL_SEED + 2)
    old_full = legacy_full(sample).detach()
    evaluator.base.seed_everything(EVAL_SEED + 2)
    new_full = full_actual(sample).detach()
    full_diff = float((old_full - new_full).abs().max().item())
    if max(robust_diff, noise_diff, full_diff) > 1e-6:
        raise RuntimeError(
            "Optimized deployment equivalence failed: "
            f"robust={robust_diff}, noise={noise_diff}, full={full_diff}"
        )
    equivalence = {
        "examples": len(sample),
        "tolerance": 1e-6,
        "robust_standalone_max_abs_logit_diff": robust_diff,
        "noise_optimized_max_abs_logit_diff": noise_diff,
        "full_refactor_max_abs_logit_diff": full_diff,
        "passed": True,
    }
    counts = {
        "standard_parameters": parameter_count(standard_model),
        "robust_parameters": parameter_count(robust_model),
        "full_deployed_parameters": (
            parameter_count(standard_model) + parameter_count(robust_model)
        ),
    }
    del legacy_noise, legacy_full, legacy_robust, dual
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    return device, source_objects, standard_arch, equivalence, counts


def summary_from_values(values: Mapping[str, Any]) -> dict[str, float]:
    percentage = {
        "top1",
        "coverage",
        "exact_set",
        "overlap",
        "jaccard",
        "weighted_membership",
        "rank_overlap",
        "reference_top1_match",
        "hidden_topk_failure",
    }
    return {
        name: (100.0 if name in percentage else 1.0)
        * float(tensor.mean().item())
        for name, tensor in values.items()
    }


def evaluate_in_batches(
    evaluator: Any,
    torch: Any,
    defense: Any,
    images: Any,
    labels: Any,
    reference_logits: Any,
    *,
    draws: int,
    seed: int,
    source_id: str,
    progress_every: int,
):
    evaluator.base.seed_everything(seed)
    bundles = []
    total_batches = (len(images) + BATCH_SIZE - 1) // BATCH_SIZE
    for number, low in enumerate(range(0, len(images), BATCH_SIZE), start=1):
        high = min(low + BATCH_SIZE, len(images))
        batch = images[low:high]
        if draws > 1 and hasattr(defense.actual, "prepare_for_mc"):
            prepared = defense.actual.prepare_for_mc(batch)
            draw_logits = torch.stack(
                [
                    defense.actual.sample_prepared(prepared).detach().cpu()
                    for _ in range(draws)
                ],
                dim=0,
            )
        else:
            draw_logits = evaluator.collect_actual_draws(
                defense, batch, high - low, draws
            )
        bundles.append(
            evaluator.evaluate_draws(
                draw_logits,
                labels[low:high],
                reference_logits[low:high],
                K,
                RANK_TEMPERATURE,
                True,
            )
        )
        if number % progress_every == 0 or number == total_batches:
            print(
                f"    {source_id}: batch {number}/{total_batches}, draws={draws}",
                flush=True,
            )
    values = {
        name: torch.cat([bundle.values[name] for bundle in bundles], dim=0)
        for name in bundles[0].values
    }
    mean_values = {
        name: torch.cat(
            [bundle.mean_logits_values[name] for bundle in bundles], dim=0
        )
        for name in bundles[0].mean_logits_values
    }
    return evaluator.EvaluationBundle(
        summary=summary_from_values(values),
        values=values,
        reference_set=torch.cat(
            [bundle.reference_set for bundle in bundles], dim=0
        ),
        first_draw_set=torch.cat(
            [bundle.first_draw_set for bundle in bundles], dim=0
        ),
        mean_logits_set=torch.cat(
            [bundle.mean_logits_set for bundle in bundles], dim=0
        ),
        mean_logits_summary=summary_from_values(mean_values),
        mean_logits_values=mean_values,
        draws=draws,
    )


def partial_paths(output_dir: Path, source_id: str, shard_index: int):
    root = output_dir / "partials" / source_id
    stem = f"shard_{shard_index:02d}"
    return root / f"{stem}.summary.json", root / f"{stem}.per_example.csv"


def validate_partial(
    output_dir: Path,
    source_id: str,
    shard_index: int,
    config_hash: str,
):
    summary_path, rows_path = partial_paths(output_dir, source_id, shard_index)
    if not summary_path.exists() and not rows_path.exists():
        return None
    if not summary_path.is_file() or not rows_path.is_file():
        raise RuntimeError(
            f"Incomplete partial for {source_id}/shard_{shard_index:02d}"
        )
    summary = read_json(summary_path)
    rows = read_csv(rows_path)
    low = shard_index * SHARD_SIZE
    high = low + SHARD_SIZE
    expected = {
        "protocol_version": PROTOCOL,
        "source_id": source_id,
        "shard_index": shard_index,
        "shard_count": SHARD_COUNT,
        "start_index": low,
        "end_index_exclusive": high,
        "examples": SHARD_SIZE,
        "eval_draws": SOURCE_DEFINITIONS[source_id]["draws"],
        "batch_size": BATCH_SIZE,
        "eval_seed": EVAL_SEED + shard_index,
        "config_sha256": config_hash,
    }
    mismatches = {
        key: (summary.get(key), value)
        for key, value in expected.items()
        if summary.get(key) != value
    }
    if mismatches or len(rows) != SHARD_SIZE:
        raise RuntimeError(
            f"Partial mismatch for {source_id}/shard_{shard_index:02d}: {mismatches}"
        )
    expected_positions = list(range(low, high))
    if [int(row["position"]) for row in rows] != expected_positions:
        raise RuntimeError("Partial positions are not canonical")
    if [int(row["sample_index"]) for row in rows] != expected_positions:
        raise RuntimeError("Partial sample indices are not canonical")
    if any(row.get("source_id") != source_id for row in rows):
        raise RuntimeError("Partial source ID differs")
    return summary, rows


def evaluate_shard_source(
    evaluator: Any,
    torch: Any,
    source_id: str,
    defense: Any,
    images: Any,
    labels: Any,
    indices: Any,
    reference_logits: Any,
    *,
    shard_index: int,
    output_dir: Path,
    config_hash: str,
    progress_every: int,
):
    seed = EVAL_SEED + shard_index
    draws = int(SOURCE_DEFINITIONS[source_id]["draws"])
    started = time.perf_counter()
    if source_id == "standard":
        evaluation = evaluator.evaluate_draws(
            reference_logits.unsqueeze(0),
            labels,
            reference_logits,
            K,
            RANK_TEMPERATURE,
            True,
        )
    else:
        evaluation = evaluate_in_batches(
            evaluator,
            torch,
            defense,
            images,
            labels,
            reference_logits,
            draws=draws,
            seed=seed,
            source_id=f"{source_id}/shard_{shard_index:02d}",
            progress_every=progress_every,
        )
    wall_seconds = time.perf_counter() - started
    summary: dict[str, Any] = {
        "protocol_version": PROTOCOL,
        "source_id": source_id,
        "implementation": SOURCE_DEFINITIONS[source_id]["implementation"],
        "condition": "clean_full_test",
        "k": K,
        "shard_index": shard_index,
        "shard_count": SHARD_COUNT,
        "start_index": shard_index * SHARD_SIZE,
        "end_index_exclusive": (shard_index + 1) * SHARD_SIZE,
        "examples": SHARD_SIZE,
        "eval_draws": draws,
        "batch_size": BATCH_SIZE,
        "eval_seed": seed,
        "common_random_numbers_noise_full": source_id in {"noise_only", "full"},
        "deterministic_preprocessing_reused_across_mc_draws": source_id == "full",
        "wall_seconds": wall_seconds,
        "config_sha256": config_hash,
    }
    summary.update(evaluation.summary)
    for name, value in evaluation.mean_logits_summary.items():
        summary[f"mean_logits_{name}"] = value
    raw_rows = evaluator.per_example_rows(
        evaluation=evaluation,
        labels=labels,
        dataset_indices=indices,
        defense=source_id,
        condition="clean_full_test",
        k=K,
        data_seed=0,
        attack_seed=0,
        reeval_seed=seed,
        run_tag="full_clean_utility_cifar10",
        shard_count=SHARD_COUNT,
        shard_index=shard_index,
        attack="none",
        attack_goal="none",
        epsilon_px=0.0,
        requested_budget=0,
        logical_budget=0,
        attack_eot=0,
        attack_api_queries=0,
        candidate_file="",
    )
    low = shard_index * SHARD_SIZE
    rows = []
    for local_position, row in enumerate(raw_rows):
        global_position = low + local_position
        enriched = {
            "protocol_version": PROTOCOL,
            "source_id": source_id,
            "implementation": SOURCE_DEFINITIONS[source_id]["implementation"],
            "common_random_numbers_noise_full": source_id in {"noise_only", "full"},
            "deterministic_preprocessing_reused_across_mc_draws": source_id == "full",
            **row,
        }
        enriched["position"] = global_position
        rows.append(enriched)
    summary_path, rows_path = partial_paths(output_dir, source_id, shard_index)
    atomic_write_json(summary_path, summary)
    atomic_write_csv(rows_path, rows)
    return summary, rows


PERCENT_COLUMNS = {
    "top1": "top1_probability",
    "coverage": "coverage_probability",
    "exact_set": "exact_set_probability",
    "overlap": "overlap",
    "jaccard": "jaccard",
    "weighted_membership": "weighted_membership",
    "reference_top1_match": "reference_top1_match_probability",
    "hidden_topk_failure": "hidden_topk_failure_probability",
}


def aggregate_source(
    source_id: str,
    rows: Sequence[Mapping[str, Any]],
    partial_summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: int(row["position"]))
    if len(ordered) != EXAMPLES:
        raise ValueError(f"Expected {EXAMPLES} rows for {source_id}")
    summary: dict[str, Any] = {
        "protocol_version": PROTOCOL,
        "source_id": source_id,
        "implementation": SOURCE_DEFINITIONS[source_id]["implementation"],
        "condition": "clean_full_test",
        "k": K,
        "examples": EXAMPLES,
        "eval_draws": SOURCE_DEFINITIONS[source_id]["draws"],
        "batch_size": BATCH_SIZE,
        "shard_count": SHARD_COUNT,
        "measurement_wall_seconds": float(
            sum(float(row["wall_seconds"]) for row in partial_summaries)
        ),
        "measurement_reuses_deterministic_preprocessing": source_id == "full",
    }
    arrays: dict[str, np.ndarray] = {}
    for metric, column in PERCENT_COLUMNS.items():
        values = np.array([float(row[column]) for row in ordered], dtype=np.float64)
        arrays[metric] = values
        summary[metric] = 100.0 * float(values.mean())
    summary["boundary_margin"] = float(
        np.mean([float(row["boundary_margin"]) for row in ordered])
    )
    summary["reference_boundary_margin"] = float(
        np.mean([float(row["reference_boundary_margin"]) for row in ordered])
    )
    for metric, column in (
        ("top1", "mean_logits_top1"),
        ("exact_set", "mean_logits_exact_set"),
        ("overlap", "mean_logits_overlap"),
    ):
        summary[f"mean_logits_{metric}"] = 100.0 * float(
            np.mean([float(row[column]) for row in ordered])
        )
    exact = arrays["exact_set"]
    draws = int(SOURCE_DEFINITIONS[source_id]["draws"])
    summary["mc_only_standard_error_pp"] = 100.0 * float(
        np.sqrt(np.sum(exact * (1.0 - exact)))
        / (len(exact) * np.sqrt(max(1, draws)))
    )
    return summary


def metric_array(rows: Sequence[Mapping[str, Any]], metric: str) -> np.ndarray:
    column = PERCENT_COLUMNS[metric]
    ordered = sorted(rows, key=lambda row: int(row["position"]))
    return np.array([float(row[column]) for row in ordered], dtype=np.float64)


def paired_bootstrap_many(
    deltas: Mapping[str, np.ndarray],
    *,
    resamples: int,
    seed: int,
    chunk_size: int = BOOTSTRAP_CHUNK,
) -> dict[str, tuple[float, float, float]]:
    names = list(deltas)
    if not names:
        return {}
    lengths = {len(np.asarray(deltas[name])) for name in names}
    if len(lengths) != 1 or next(iter(lengths)) < 2:
        raise ValueError("Paired bootstrap arrays must share a length >= 2")
    matrix = np.column_stack(
        [np.asarray(deltas[name], dtype=np.float64) for name in names]
    )
    n = matrix.shape[0]
    rng = np.random.default_rng(seed)
    sampled = np.empty((resamples, len(names)), dtype=np.float64)
    for low in range(0, resamples, chunk_size):
        high = min(low + chunk_size, resamples)
        indices = rng.integers(0, n, size=(high - low, n), dtype=np.int32)
        sampled[low:high] = matrix[indices].mean(axis=1)
    output: dict[str, tuple[float, float, float]] = {}
    for column, name in enumerate(names):
        low, high = np.quantile(sampled[:, column], [0.025, 0.975])
        output[name] = (float(matrix[:, column].mean()), float(low), float(high))
    return output


def comparison_specs() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    pairs = (
        ("robust", "standard", "Robust minus Standard"),
        ("noise_only", "standard", "Noise-only minus Standard"),
        ("full", "standard", "Full minus Standard"),
        ("noise_only", "robust", "Noise-only minus Robust"),
        ("full", "robust", "Full minus Robust"),
        ("full", "noise_only", "Full minus Noise-only"),
    )
    for metric in ("top1", "exact_set"):
        for left, right, label in pairs:
            rows.append(
                {
                    "comparison": f"clean_{left}_minus_{right}_{metric}",
                    "metric": metric,
                    "left_source_id": left,
                    "right_source_id": right,
                    "label": label,
                }
            )
    return rows


def build_comparisons(rows_by_source: Mapping[str, Sequence[Mapping[str, Any]]]):
    specs = comparison_specs()
    arrays = {
        (source, metric): metric_array(rows_by_source[source], metric)
        for source in SOURCES
        for metric in ("top1", "exact_set")
    }
    deltas = {
        spec["comparison"]: arrays[(spec["left_source_id"], spec["metric"])]
        - arrays[(spec["right_source_id"], spec["metric"])]
        for spec in specs
    }
    intervals = paired_bootstrap_many(
        deltas, resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED
    )
    output = []
    for spec in specs:
        estimate, low, high = intervals[spec["comparison"]]
        left = arrays[(spec["left_source_id"], spec["metric"])]
        right = arrays[(spec["right_source_id"], spec["metric"])]
        verdict = (
            "positive"
            if low > 0
            else "negative"
            if high < 0
            else "statistically_indistinguishable"
        )
        output.append(
            {
                "protocol_version": PROTOCOL,
                "comparison": spec["comparison"],
                "metric": spec["metric"],
                "left_source_id": spec["left_source_id"],
                "right_source_id": spec["right_source_id"],
                "left_value_pct": 100.0 * float(left.mean()),
                "right_value_pct": 100.0 * float(right.mean()),
                "left_minus_right_pp": 100.0 * estimate,
                "ci95_low_pp": 100.0 * low,
                "ci95_high_pp": 100.0 * high,
                "verdict": verdict,
                "positive_delta_means": f"{spec['label']} is positive",
                "inference_scope": (
                    "ordinary paired bootstrap over all 10,000 canonical CIFAR-10 "
                    "test images; common resample indices across comparisons"
                ),
                "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                "bootstrap_seed": BOOTSTRAP_SEED,
            }
        )
    return output


def build_classwise(rows_by_source: Mapping[str, Sequence[Mapping[str, Any]]]):
    output = []
    for source_id in SOURCES:
        rows = rows_by_source[source_id]
        for label in range(CLASSES):
            selected = [row for row in rows if int(row["true_label"]) == label]
            if len(selected) != EXAMPLES_PER_CLASS:
                raise ValueError(
                    f"Class {label}/{source_id} has {len(selected)} examples"
                )
            output.append(
                {
                    "protocol_version": PROTOCOL,
                    "source_id": source_id,
                    "true_label": label,
                    "examples": len(selected),
                    "top1_pct": 100.0
                    * float(np.mean([float(row["top1_probability"]) for row in selected])),
                    "exact_set_pct": 100.0
                    * float(
                        np.mean(
                            [float(row["exact_set_probability"]) for row in selected]
                        )
                    ),
                    "overlap_pct": 100.0
                    * float(np.mean([float(row["overlap"]) for row in selected])),
                }
            )
    return output


def validate_latency_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    if len(rows) != len(SOURCES) * len(BENCHMARK_BATCH_SIZES):
        raise ValueError("Latency benchmark grid is incomplete")
    expected = {(source, batch) for source in SOURCES for batch in BENCHMARK_BATCH_SIZES}
    observed = {(str(row["source_id"]), int(row["batch_size"])) for row in rows}
    if observed != expected:
        raise ValueError("Latency benchmark source/batch grid differs")
    for row in rows:
        if float(row["wall_latency_ms_per_batch_mean"]) <= 0:
            raise ValueError("Latency must be positive")
        if int(row["benchmark_blocks"]) != BENCHMARK_BLOCKS:
            raise ValueError("Latency benchmark block count differs")


def benchmark_sources(
    torch: Any,
    device: Any,
    source_objects: Mapping[str, Any],
    images: Any,
    parameter_counts: Mapping[str, int],
):
    rows: list[dict[str, Any]] = []
    for batch_size in BENCHMARK_BATCH_SIZES:
        batch = images[:batch_size].to(device)
        for source_index, source_id in enumerate(SOURCES):
            actual = source_objects[source_id].actual
            torch.manual_seed(EVAL_SEED + 1000 + source_index)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(EVAL_SEED + 1000 + source_index)
            for _ in range(BENCHMARK_WARMUPS):
                actual(batch)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            wall_blocks: list[float] = []
            cuda_blocks: list[float] = []
            for _ in range(BENCHMARK_BLOCKS):
                if device.type == "cuda":
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                started = time.perf_counter()
                for _ in range(BENCHMARK_REPEATS_PER_BLOCK):
                    actual(batch)
                if device.type == "cuda":
                    end_event.record()
                    torch.cuda.synchronize(device)
                elapsed = 1000.0 * (time.perf_counter() - started)
                wall_blocks.append(elapsed / BENCHMARK_REPEATS_PER_BLOCK)
                if device.type == "cuda":
                    cuda_blocks.append(
                        start_event.elapsed_time(end_event)
                        / BENCHMARK_REPEATS_PER_BLOCK
                    )
            incremental_peak_mib: float | None = None
            absolute_peak_mib: float | None = None
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                before = int(torch.cuda.memory_allocated(device))
                actual(batch)
                torch.cuda.synchronize(device)
                peak = int(torch.cuda.max_memory_allocated(device))
                incremental_peak_mib = max(0, peak - before) / (1024.0**2)
                absolute_peak_mib = peak / (1024.0**2)
            wall = np.asarray(wall_blocks, dtype=np.float64)
            cuda = np.asarray(cuda_blocks, dtype=np.float64)
            definition = SOURCE_DEFINITIONS[source_id]
            deployed_parameters = (
                parameter_counts["full_deployed_parameters"]
                if source_id == "full"
                else parameter_counts["standard_parameters"]
                if source_id == "standard"
                else parameter_counts["robust_parameters"]
            )
            rows.append(
                {
                    "protocol_version": PROTOCOL,
                    "source_id": source_id,
                    "implementation": definition["implementation"],
                    "batch_size": batch_size,
                    "benchmark_input_start_index": 0,
                    "benchmark_input_end_index_exclusive": batch_size,
                    "benchmark_warmups": BENCHMARK_WARMUPS,
                    "benchmark_blocks": BENCHMARK_BLOCKS,
                    "repeats_per_block": BENCHMARK_REPEATS_PER_BLOCK,
                    "single_request_uncached": True,
                    "includes_host_to_device_transfer": False,
                    "wall_latency_ms_per_batch_mean": float(wall.mean()),
                    "wall_latency_ms_per_batch_sd": float(wall.std(ddof=1)),
                    "wall_latency_ms_per_batch_median": float(np.median(wall)),
                    "wall_latency_ms_per_batch_p95": float(np.quantile(wall, 0.95)),
                    "wall_latency_ms_per_image_mean": float(wall.mean() / batch_size),
                    "throughput_images_per_second": float(
                        1000.0 * batch_size / wall.mean()
                    ),
                    "cuda_latency_ms_per_batch_mean": (
                        float(cuda.mean()) if len(cuda) else ""
                    ),
                    "cuda_latency_ms_per_batch_sd": (
                        float(cuda.std(ddof=1)) if len(cuda) > 1 else ""
                    ),
                    "incremental_peak_allocated_mib": (
                        incremental_peak_mib if incremental_peak_mib is not None else ""
                    ),
                    "absolute_peak_allocated_mib": (
                        absolute_peak_mib if absolute_peak_mib is not None else ""
                    ),
                    "deployed_parameter_count": deployed_parameters,
                    "standard_forwards_per_request": definition["standard_forwards"],
                    "robust_forwards_per_request": definition["robust_forwards"],
                    "input_gradient_steps_per_request": definition[
                        "input_gradient_steps"
                    ],
                    "wall_latency_ratio_vs_robust": "",
                }
            )
    by_batch = {
        int(row["batch_size"]): float(row["wall_latency_ms_per_batch_mean"])
        for row in rows
        if row["source_id"] == "robust"
    }
    for row in rows:
        row["wall_latency_ratio_vs_robust"] = float(
            float(row["wall_latency_ms_per_batch_mean"])
            / by_batch[int(row["batch_size"])]
        )
    validate_latency_rows(rows)
    return rows


def normalized_config(
    args: argparse.Namespace,
    dataset_info: Mapping[str, Any],
    dependency_hashes: Mapping[str, str],
    standard_arch: str,
    equivalence: Mapping[str, Any],
    parameter_counts: Mapping[str, int],
) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL,
        "output_dir": str(args.output_dir.expanduser().resolve()),
        "data_dir": str(args.data_dir.expanduser().resolve()),
        "standard_weights": str(args.standard_weights.expanduser().resolve()),
        "dual_weights": str(args.dual_weights.expanduser().resolve()),
        "standard_weights_sha256": sha256_file(args.standard_weights.expanduser()),
        "dual_weights_sha256": sha256_file(args.dual_weights.expanduser()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "dependency_sha256": dict(dependency_hashes),
        "device": args.device,
        "gpu_memory_fraction": args.gpu_memory_fraction,
        "dataset": dict(dataset_info),
        "examples": EXAMPLES,
        "k": K,
        "eval_draws": EVAL_DRAWS,
        "batch_size": BATCH_SIZE,
        "shard_size": SHARD_SIZE,
        "shard_count": SHARD_COUNT,
        "eval_seed": EVAL_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_method": "ordinary paired image bootstrap",
        "rank_temperature": RANK_TEMPERATURE,
        "sources": list(SOURCES),
        "source_definitions": SOURCE_DEFINITIONS,
        "defense_parameters": {
            "top_k": K,
            "sigma": SIGMA,
            "strength_steps": STRENGTH_STEPS,
            "strength_alpha": STRENGTH_ALPHA,
            "strength_epsilon": STRENGTH_EPSILON,
            "top1_weight": TOP1_WEIGHT,
            "set_weight": SET_WEIGHT,
            "reference_model": "standard",
            "output_model": "standalone robust pixel backbone",
        },
        "standard_architecture": standard_arch,
        "deployment_equivalence_preflight": dict(equivalence),
        "parameter_counts": dict(parameter_counts),
        "measurement_optimization": (
            "Full deterministic strengthening is reused across MC draws only for "
            "distribution measurement; single-request latency remains uncached."
        ),
        "benchmark": {
            "batch_sizes": list(BENCHMARK_BATCH_SIZES),
            "warmups": BENCHMARK_WARMUPS,
            "blocks": BENCHMARK_BLOCKS,
            "repeats_per_block": BENCHMARK_REPEATS_PER_BLOCK,
            "includes_host_to_device_transfer": False,
            "single_request_uncached": True,
        },
        "confirmatory_holdout_used": False,
    }


def sample_key_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    text = "\n".join(
        f"{row['sample_index']},{row['true_label']},{row['reference_set']}"
        for row in sorted(rows, key=lambda row: int(row["position"]))
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    args = parse_args()
    validate_args(args)
    dependency_hashes = validate_dependencies()

    import torch
    import torchvision

    sys.path.insert(0, str(PROJECT_DIR))
    import evaluate_general_topk_defense_v6_cifar10 as evaluator

    root = evaluator.legacy.resolve_torchvision_root(args.data_dir)
    dataset = torchvision.datasets.CIFAR10(
        root=str(root), train=False, download=False
    )
    dataset_info = dataset_manifest(dataset)
    sample_images, _, _ = tensor_shard(torch, dataset, 0)
    (
        device,
        source_objects,
        standard_arch,
        equivalence,
        parameter_counts,
    ) = build_models(evaluator, torch, args, sample_images[:8])
    print(
        "PASS deployment equivalence: "
        f"robust={equivalence['robust_standalone_max_abs_logit_diff']:.3g}, "
        f"noise={equivalence['noise_optimized_max_abs_logit_diff']:.3g}, "
        f"full={equivalence['full_refactor_max_abs_logit_diff']:.3g}",
        flush=True,
    )
    print(
        f"PASS CIFAR-10 full test: {EXAMPLES} canonical images, "
        f"{EXAMPLES_PER_CLASS}/class",
        flush=True,
    )
    if args.preflight_only:
        print("Full clean utility GPU preflight complete; no results were written.")
        return

    output_dir = args.output_dir.expanduser().resolve()
    lock = acquire_lock(output_dir)
    try:
        config = normalized_config(
            args,
            dataset_info,
            dependency_hashes,
            standard_arch,
            equivalence,
            parameter_counts,
        )
        config_path = output_dir / CONFIG_FILE
        if config_path.is_file():
            existing = read_json(config_path)
            if existing != config:
                raise RuntimeError(
                    "Existing output uses a different protocol/config; use a fresh directory."
                )
        else:
            atomic_write_json(config_path, config)
        config_hash = sha256_file(config_path)

        print(
            f"Full clean utility: images={EXAMPLES}, MC={EVAL_DRAWS}, "
            f"shards={SHARD_COUNT}, batch={BATCH_SIZE}",
            flush=True,
        )
        for shard_index in range(SHARD_COUNT):
            completed = {
                source_id: validate_partial(
                    output_dir, source_id, shard_index, config_hash
                )
                for source_id in SOURCES
            }
            if all(value is not None for value in completed.values()):
                print(
                    f"PASS reusing complete shard {shard_index + 1}/{SHARD_COUNT}",
                    flush=True,
                )
                continue
            images_cpu, labels_cpu, indices = tensor_shard(
                torch, dataset, shard_index
            )
            images = images_cpu.to(device)
            with torch.no_grad():
                reference_logits = evaluator.base.compute_logits_batched(
                    source_objects["standard"], images, BATCH_SIZE, 1
                ).detach().cpu()
            for source_id in SOURCES:
                if completed[source_id] is not None:
                    print(
                        f"PASS reusing {source_id}/shard_{shard_index:02d}",
                        flush=True,
                    )
                    continue
                print(
                    f"[{shard_index + 1}/{SHARD_COUNT}] Evaluating {source_id}",
                    flush=True,
                )
                summary, _ = evaluate_shard_source(
                    evaluator,
                    torch,
                    source_id,
                    source_objects[source_id],
                    images,
                    labels_cpu,
                    indices,
                    reference_logits,
                    shard_index=shard_index,
                    output_dir=output_dir,
                    config_hash=config_hash,
                    progress_every=args.progress_every_batches,
                )
                print(
                    f"PASS {source_id}/shard_{shard_index:02d}: "
                    f"Top1={summary['top1']:.3f}% "
                    f"Exact-set={summary['exact_set']:.3f}%",
                    flush=True,
                )
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            del images, images_cpu, labels_cpu, indices, reference_logits

        rows_by_source: dict[str, list[dict[str, str]]] = {}
        summaries = []
        all_rows: list[dict[str, str]] = []
        for source_id in SOURCES:
            source_rows: list[dict[str, str]] = []
            partial_summaries: list[dict[str, Any]] = []
            for shard_index in range(SHARD_COUNT):
                partial = validate_partial(
                    output_dir, source_id, shard_index, config_hash
                )
                if partial is None:
                    raise RuntimeError("A required partial disappeared")
                partial_summary, partial_rows = partial
                partial_summaries.append(partial_summary)
                source_rows.extend(partial_rows)
            source_rows.sort(key=lambda row: int(row["position"]))
            if [int(row["position"]) for row in source_rows] != list(range(EXAMPLES)):
                raise RuntimeError(f"{source_id} final positions differ")
            rows_by_source[source_id] = source_rows
            all_rows.extend(source_rows)
            summaries.append(
                aggregate_source(source_id, source_rows, partial_summaries)
            )

        canonical = [
            (
                row["sample_index"],
                row["true_label"],
                row["reference_set"],
            )
            for row in rows_by_source["standard"]
        ]
        for source_id in SOURCES[1:]:
            current = [
                (row["sample_index"], row["true_label"], row["reference_set"])
                for row in rows_by_source[source_id]
            ]
            if current != canonical:
                raise RuntimeError(f"Sample/reference alignment failed for {source_id}")

        comparisons = build_comparisons(rows_by_source)
        classwise = build_classwise(rows_by_source)
        atomic_write_csv(output_dir / SUMMARY_FILE, summaries)
        atomic_write_csv(output_dir / EXAMPLES_FILE, all_rows)
        atomic_write_csv(output_dir / COMPARISONS_FILE, comparisons)
        atomic_write_csv(output_dir / CLASSWISE_FILE, classwise)

        latency_path = output_dir / LATENCY_FILE
        if latency_path.is_file():
            latency = read_csv(latency_path)
            validate_latency_rows(latency)
            print("PASS reusing completed latency benchmark", flush=True)
        else:
            benchmark_images, _, _ = tensor_shard(torch, dataset, 0)
            print("Running uncached single-request latency benchmark", flush=True)
            latency = benchmark_sources(
                torch,
                device,
                source_objects,
                benchmark_images,
                parameter_counts,
            )
            atomic_write_csv(latency_path, latency)

        summary_by_id = {row["source_id"]: row for row in summaries}
        comparison_by_id = {row["comparison"]: row for row in comparisons}
        final = {
            "protocol_version": PROTOCOL,
            "examples": EXAMPLES,
            "sources": list(SOURCES),
            "eval_draws_stochastic": EVAL_DRAWS,
            "summary_rows": len(summaries),
            "per_example_rows": len(all_rows),
            "comparison_rows": len(comparisons),
            "classwise_rows": len(classwise),
            "latency_rows": len(latency),
            "sample_key_sha256": sample_key_hash(rows_by_source["standard"]),
            "clean_top1_accuracy_pct": {
                source_id: float(summary_by_id[source_id]["top1"])
                for source_id in SOURCES
            },
            "clean_exact_set_pct": {
                source_id: float(summary_by_id[source_id]["exact_set"])
                for source_id in SOURCES
            },
            "primary_comparisons": {
                name: comparison_by_id[name]
                for name in (
                    "clean_noise_only_minus_robust_top1",
                    "clean_full_minus_robust_top1",
                    "clean_full_minus_noise_only_top1",
                    "clean_noise_only_minus_robust_exact_set",
                    "clean_full_minus_robust_exact_set",
                    "clean_full_minus_noise_only_exact_set",
                )
            },
            "measurement_scope": (
                "Clean utility on all 10,000 canonical CIFAR-10 test images. "
                "Stochastic metrics use 100 per-request draws and paired image "
                "bootstrap intervals. This run does not access the confirmatory holdout."
            ),
            "latency_scope": (
                "Uncached single-request model execution on resident GPU tensors; "
                "host-to-device transfer and data loading are excluded."
            ),
        }
        atomic_write_json(output_dir / FINAL_FILE, final)

        print("\nFull-test clean utility", flush=True)
        for source_id in SOURCES:
            row = summary_by_id[source_id]
            print(
                f"{source_id:>10}: Top1={float(row['top1']):.3f}% "
                f"Exact-set={float(row['exact_set']):.3f}%",
                flush=True,
            )
        print("\nPrimary paired comparisons", flush=True)
        for name, row in final["primary_comparisons"].items():
            print(
                f"{name}: {float(row['left_minus_right_pp']):+.3f} pp "
                f"[{float(row['ci95_low_pp']):+.3f}, "
                f"{float(row['ci95_high_pp']):+.3f}] -> {row['verdict']}",
                flush=True,
            )
        print(f"Full clean utility complete: {output_dir}", flush=True)
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    main()
