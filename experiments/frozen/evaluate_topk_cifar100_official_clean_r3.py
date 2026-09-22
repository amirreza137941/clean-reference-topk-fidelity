#!/usr/bin/env python3
"""Frozen official-test clean confirmation for the CIFAR-100 Top-2 defense.

The defense operating point is loaded from the completed dev-Pareto R2 stage.
No model is trained, no hyperparameter is searched, and no attack outcome is
read.  The complete official CIFAR-100 test set is evaluated once with a fixed
MC seed; resumable shards are merely pieces of that single frozen evaluation.
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

import calibrate_topk_cifar100_dev_pareto_r2 as dev
import topk_cifar100_official_clean_r3_common as common


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = Path("./datasets/cifar-100-python")
DEFAULT_R2_ROOT = Path(
    "./models/resnet20/"
    "paper_topk_cifar100_external_validity_r2"
)
DEFAULT_DEV_DIR = Path(
    "./models/resnet20/"
    "paper_topk_cifar100_actual_defense_dev_pareto_r2"
)
DEFAULT_OUTPUT = Path(
    "./models/resnet20/"
    "paper_topk_cifar100_official_clean_confirmation_k2_mc100_r3"
)

TEST_EXAMPLES = 10000
SHARD_COUNT = 20
MC_DRAWS = 100
NONINFERIORITY_MARGIN_PP = 2.0
SEED = 2026090501
BOOTSTRAP_SEED = 2026090502
ATTACK_SPLIT_SEED = 2026090503
BOOTSTRAP_RESAMPLES = 10000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frozen CIFAR-100 official-test clean confirmation R3"
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--r2-root", type=Path, default=DEFAULT_R2_ROOT)
    parser.add_argument("--dev-dir", type=Path, default=DEFAULT_DEV_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--mc-chunk", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.45)
    parser.add_argument("--test-examples", type=int, default=TEST_EXAMPLES)
    parser.add_argument("--shard-count", type=int, default=SHARD_COUNT)
    parser.add_argument("--mc-draws", type=int, default=MC_DRAWS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--attack-split-seed", type=int, default=ATTACK_SPLIT_SEED)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    frozen = {
        "test_examples": TEST_EXAMPLES,
        "shard_count": SHARD_COUNT,
        "mc_draws": MC_DRAWS,
        "seed": SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "attack_split_seed": ATTACK_SPLIT_SEED,
        "batch_size": 25,
        "mc_chunk": 20,
    }
    for name, expected in frozen.items():
        if int(getattr(args, name)) != expected:
            raise ValueError(f"Frozen R3 protocol requires --{name.replace('_', '-')} {expected}")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if not 0.0 < args.gpu_memory_fraction <= 1.0:
        raise ValueError("--gpu-memory-fraction must be in (0,1]")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _required_sources() -> dict[str, Path]:
    return {
        "official_evaluator": Path(__file__).resolve(),
        "official_common": PROJECT_DIR / "topk_cifar100_official_clean_r3_common.py",
        "dev_evaluator": PROJECT_DIR / "calibrate_topk_cifar100_dev_pareto_r2.py",
        "dev_common": PROJECT_DIR / "topk_cifar100_dev_pareto_r2_common.py",
        "r1_evaluator": PROJECT_DIR / "calibrate_topk_cifar100_actual_defense_r1.py",
        "r1_common": PROJECT_DIR / "topk_cifar100_defense_calibration_common.py",
        "r2_common": PROJECT_DIR / "topk_cifar100_r2_common.py",
        "r2_anchor": PROJECT_DIR / "train_resnet20_teacher_top2_adversarial_cifar100_r2.py",
    }


def load_and_validate_parents(args: argparse.Namespace) -> dict[str, Any]:
    r2_root = args.r2_root.expanduser().resolve()
    dev_dir = args.dev_dir.expanduser().resolve()
    paths = {
        "standard_weights": r2_root / "standard" / "resnet20_standard_best_validation_weights.pt",
        "anchor_weights": r2_root / "teacher_top2_at" / "resnet20_teacher_top2_best_composite_weights.pt",
        "pipeline_manifest": r2_root / "pipeline_manifest.json",
        "standard_test_metrics": r2_root / "standard" / "final_test_metrics.json",
        "anchor_test_metrics": r2_root / "teacher_top2_at" / "final_test_metrics.json",
        "dev_selection": dev_dir / "dev_pareto_selection.json",
        "dev_final": dev_dir / "final_summary.json",
        "dev_config": dev_dir / "run_config.json",
    }
    sources = _required_sources()
    missing = [str(path) for path in list(paths.values()) + list(sources.values()) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required frozen artifacts are missing:\n  " + "\n  ".join(missing))

    if common.file_sha256(paths["standard_weights"]) != common.EXPECTED_STANDARD_SHA256:
        raise AssertionError("Frozen standard checkpoint hash mismatch")
    if common.file_sha256(paths["anchor_weights"]) != common.EXPECTED_ANCHOR_SHA256:
        raise AssertionError("Frozen anchor checkpoint hash mismatch")

    selection = common.read_json(paths["dev_selection"])
    dev_final = common.read_json(paths["dev_final"])
    dev_config = common.read_json(paths["dev_config"])
    common.validate_frozen_selection(selection, dev_final, dev_config)
    config_payload = dict(dev_config)
    claimed_config_hash = str(config_payload.pop("config_hash"))
    if common.object_sha256(config_payload) != claimed_config_hash:
        raise AssertionError("Parent dev run_config self-hash mismatch")

    expected_parent_sources = {
        "evaluator": sources["dev_evaluator"],
        "dev_pareto_common": sources["dev_common"],
        "r1_evaluator": sources["r1_evaluator"],
        "r1_common": sources["r1_common"],
        "r2_common": sources["r2_common"],
        "r2_anchor": sources["r2_anchor"],
    }
    if set(dev_config["script_sha256"]) != set(expected_parent_sources):
        raise AssertionError("Parent dev source-hash key set changed")
    for name, path in expected_parent_sources.items():
        if common.file_sha256(path) != str(dev_config["script_sha256"][name]):
            raise AssertionError(f"Frozen parent source changed: {name}: {path}")

    pipeline = common.read_json(paths["pipeline_manifest"])
    standard_test = common.read_json(paths["standard_test_metrics"])
    anchor_test = common.read_json(paths["anchor_test_metrics"])
    if pipeline.get("protocol_id") != common.PARENT_MODEL_PROTOCOL_ID:
        raise AssertionError("Unexpected parent model-preparation protocol")
    for payload, label in ((standard_test, "standard"), (anchor_test, "anchor")):
        if payload.get("protocol_id") != common.PARENT_MODEL_PROTOCOL_ID:
            raise AssertionError(f"Unexpected {label} parent test protocol")
        if payload.get("test_used_for_selection") is not False:
            raise AssertionError(f"Parent {label} checkpoint used official test for selection")
        if payload.get("official_full_test") is not True:
            raise AssertionError(f"Parent {label} did not report the complete official test")

    return {
        "paths": paths,
        "sources": sources,
        "selection": selection,
        "dev_final": dev_final,
        "dev_config": dev_config,
        "pipeline": pipeline,
        "standard_test": standard_test,
        "anchor_test": anchor_test,
    }


def selected_methods(selection: Mapping[str, Any], draws: int) -> list[dev.MethodSpec]:
    strength = selection["strength"]
    sigma = float(selection["randomization"]["sigma"])
    return [
        dev.MethodSpec("standard", "standard", "standard"),
        dev.MethodSpec("robust", "anchor", "identity"),
        dev.MethodSpec(
            "full_sigma0",
            "anchor",
            str(strength["candidate_id"]),
            gate_mode=str(strength["gate_mode"]),
            strength_alpha_px=float(strength["alpha_px"]),
            strength_epsilon_px=float(strength["epsilon_px"]),
            strength_steps=int(strength["steps"]),
        ),
        dev.MethodSpec(
            "noise_only",
            "anchor",
            "noise_only_sigma_0p015",
            sigma=sigma,
            draws=int(draws),
        ),
        dev.MethodSpec(
            "full",
            "anchor",
            "global_a1p275_s2_sigma_0p015",
            gate_mode=str(strength["gate_mode"]),
            strength_alpha_px=float(strength["alpha_px"]),
            strength_epsilon_px=float(strength["epsilon_px"]),
            strength_steps=int(strength["steps"]),
            sigma=sigma,
            draws=int(draws),
        ),
    ]


def build_test_manifest(dataset: Any, args: argparse.Namespace) -> dict[str, Any]:
    labels = [int(value) for value in dataset.targets]
    indices = list(range(len(dataset)))
    partitions = common.balanced_attack_partitions(labels, args.attack_split_seed)
    manifest: dict[str, Any] = {
        "protocol_id": common.PROTOCOL_ID,
        "dataset": "CIFAR-100",
        "split": "official_test",
        "train_flag": False,
        "download": False,
        "examples": len(indices),
        "test_indices": indices,
        "test_indices_sha256": common.sequence_sha256(indices),
        "labels_sha256": common.sequence_sha256(labels),
        "class_counts": np.bincount(np.asarray(labels), minlength=100).astype(int).tolist(),
        "defense_parameters_frozen_before_test_read": True,
        "official_test_previously_read_only_for_parent_checkpoint_reporting": True,
        "official_test_used_for_parent_checkpoint_selection": False,
        "official_test_used_for_defense_parameter_selection": False,
        "first_evaluation_of_selected_defense_operating_point": True,
        "attack_outcomes_read": False,
        "future_attack_partitions": {
            "seed": int(args.attack_split_seed),
            "rule": "label-only balanced sampling before any CIFAR-100 attack outcome",
            "pilot": {
                "examples": len(partitions["pilot"]),
                "per_class": 2,
                "indices": partitions["pilot"],
                "indices_sha256": common.sequence_sha256(partitions["pilot"]),
            },
            "confirmatory": {
                "examples": len(partitions["confirmatory"]),
                "per_class": 10,
                "indices": partitions["confirmatory"],
                "indices_sha256": common.sequence_sha256(partitions["confirmatory"]),
            },
            "disjoint": True,
        },
    }
    manifest["manifest_hash"] = common.object_sha256(manifest)
    return manifest


def _rewrite_protocol(
    rows: Sequence[Mapping[str, Any]], shard_id: int, config_hash: str
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        row["protocol_id"] = common.PROTOCOL_ID
        row["phase"] = "official_test_clean"
        row["shard_id"] = int(shard_id)
        row["config_hash"] = config_hash
        output.append(row)
    return output


def partial_path(output_dir: Path, shard_id: int) -> Path:
    return output_dir / "partials" / f"official_test_shard_{shard_id:02d}.json"


def validate_partial(
    payload: Mapping[str, Any],
    config_hash: str,
    shard_id: int,
    expected_indices: Sequence[int],
    methods: Sequence[dev.MethodSpec],
) -> None:
    condition_id = f"official_test_shard_{shard_id:02d}"
    if payload.get("protocol_id") != common.PROTOCOL_ID:
        raise AssertionError(f"Partial protocol mismatch: {condition_id}")
    if payload.get("config_hash") != config_hash or payload.get("condition_id") != condition_id:
        raise AssertionError(f"Partial identity mismatch: {condition_id}")
    if int(payload.get("shard_id")) != shard_id:
        raise AssertionError(f"Partial shard mismatch: {condition_id}")
    if [int(value) for value in payload.get("sample_indices", [])] != list(expected_indices):
        raise AssertionError(f"Partial index mismatch: {condition_id}")
    expected_specs = [asdict(method) for method in methods]
    if payload.get("methods") != expected_specs:
        raise AssertionError(f"Partial method specification mismatch: {condition_id}")
    summaries = payload.get("summaries", [])
    examples = payload.get("examples", [])
    if {str(row["method"]) for row in summaries} != set(common.METHODS):
        raise AssertionError(f"Partial summary method mismatch: {condition_id}")
    for method in common.METHODS:
        rows = [row for row in examples if str(row["method"]) == method]
        if [int(row["sample_index"]) for row in rows] != list(expected_indices):
            raise AssertionError(f"Partial rows mismatch: {condition_id}/{method}")


def run_shard(
    output_dir: Path,
    config_hash: str,
    shard_id: int,
    sample_indices: Sequence[int],
    dataset: Any,
    methods: Sequence[dev.MethodSpec],
    standard: torch.nn.Module,
    anchor: torch.nn.Module,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    path = partial_path(output_dir, shard_id)
    if path.is_file():
        payload = common.read_json(path)
        validate_partial(payload, config_hash, shard_id, sample_indices, methods)
        print(f"RESUME official_test_shard_{shard_id:02d}", flush=True)
        return payload

    condition_id = f"official_test_shard_{shard_id:02d}"
    print(f"RUN {condition_id}", flush=True)
    loader = dev.make_loader(dataset, sample_indices, args, device)
    noise_seed = dev.stable_seed(args.seed, f"official_test_common_mc_shard_{shard_id:02d}")
    summaries, rows = dev.evaluate_methods(
        "official_test_clean", loader, methods, standard, anchor, device, args, noise_seed
    )
    summaries = _rewrite_protocol(summaries, shard_id, config_hash)
    rows = _rewrite_protocol(rows, shard_id, config_hash)
    payload = {
        "protocol_id": common.PROTOCOL_ID,
        "config_hash": config_hash,
        "condition_id": condition_id,
        "shard_id": int(shard_id),
        "sample_indices": [int(value) for value in sample_indices],
        "sample_indices_sha256": common.sequence_sha256(sample_indices),
        "noise_seed": int(noise_seed),
        "methods": [asdict(method) for method in methods],
        "summaries": summaries,
        "examples": rows,
        "wall_seconds": float(summaries[0]["wall_seconds"]),
    }
    validate_partial(payload, config_hash, shard_id, sample_indices, methods)
    common.atomic_json(path, payload)
    print(
        "PASS "
        + condition_id
        + ": "
        + ", ".join(
            f"{row['method']} Top1={float(row['top1_accuracy_percent']):.3f}% "
            f"Exact={float(row['exact_set_probability_percent']):.3f}%"
            for row in summaries
        ),
        flush=True,
    )
    return payload


def method_rows(rows: Sequence[Mapping[str, Any]], method: str) -> list[dict[str, Any]]:
    selected = [dict(row) for row in rows if str(row["method"]) == method]
    selected.sort(key=lambda row: int(row["sample_index"]))
    if len(selected) != TEST_EXAMPLES:
        raise AssertionError(f"Expected {TEST_EXAMPLES} rows for {method}; got {len(selected)}")
    return selected


def summarize_method(
    rows: Sequence[Mapping[str, Any]],
    method: dev.MethodSpec,
    total_wall_seconds: float,
) -> dict[str, Any]:
    def mean_percent(key: str) -> float:
        return 100.0 * float(np.mean([float(row[key]) for row in rows]))

    return {
        "protocol_id": common.PROTOCOL_ID,
        "phase": "official_test_clean",
        "method": method.name,
        "candidate_id": method.candidate_id,
        "model": method.model,
        "gate_mode": method.gate_mode,
        "gate_activation_percent": mean_percent("gate_active"),
        "examples": len(rows),
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
            np.mean([float(row["mean_boundary_margin"]) for row in rows])
        ),
        "total_gpu_stage_wall_seconds": float(total_wall_seconds),
    }


def bootstrap_comparisons(
    grouped: Mapping[str, Sequence[Mapping[str, Any]]], args: argparse.Namespace
) -> list[dict[str, Any]]:
    pairs = (
        ("standard", "robust", ("top1",)),
        ("noise_only", "robust", ("top1", "exact_set")),
        ("full_sigma0", "robust", ("top1", "exact_set")),
        ("full", "robust", ("top1", "exact_set")),
        ("full", "noise_only", ("top1", "exact_set")),
        ("full", "full_sigma0", ("top1", "exact_set")),
    )
    specs: list[dict[str, Any]] = []
    for left, right, metrics in pairs:
        for metric in metrics:
            key = common.PROBABILITY_METRICS[metric]
            specs.append(
                {
                    "group_id": "official_test",
                    "comparison": f"{left}_minus_{right}_{metric}",
                    "left": left,
                    "right": right,
                    "metric": metric,
                    "delta": common.aligned_delta(grouped[left], grouped[right], key),
                }
            )
    return common.paired_bootstrap_many(
        specs, args.bootstrap_resamples, args.bootstrap_seed
    )


def reference_decomposition(
    grouped: Mapping[str, Sequence[Mapping[str, Any]]]
) -> list[dict[str, Any]]:
    maps = {method: common.rows_by_index(rows) for method, rows in grouped.items()}
    indices = sorted(maps["standard"])
    standard_correct = {
        index: float(maps["standard"][index]["top1_accuracy_probability"]) == 1.0
        for index in indices
    }
    groups = {
        "all": indices,
        "standard_correct": [index for index in indices if standard_correct[index]],
        "standard_wrong": [index for index in indices if not standard_correct[index]],
    }
    output: list[dict[str, Any]] = []
    for group, selected in groups.items():
        if not selected:
            raise AssertionError(f"Empty decomposition group: {group}")
        for method in common.METHODS:
            rows = maps[method]
            row: dict[str, Any] = {
                "protocol_id": common.PROTOCOL_ID,
                "group": group,
                "method": method,
                "examples": len(selected),
            }
            for name, key in common.PROBABILITY_METRICS.items():
                row[f"{name}_percent"] = 100.0 * float(
                    np.mean([float(rows[index][key]) for index in selected])
                )
            output.append(row)
    return output


def classwise_summary(
    grouped: Mapping[str, Sequence[Mapping[str, Any]]]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for class_id in range(100):
        for method in common.METHODS:
            selected = [row for row in grouped[method] if int(row["label"]) == class_id]
            if len(selected) != 100:
                raise AssertionError(f"Class count mismatch for class={class_id}, method={method}")
            output.append(
                {
                    "protocol_id": common.PROTOCOL_ID,
                    "class_id": class_id,
                    "method": method,
                    "examples": len(selected),
                    "top1_accuracy_percent": 100.0
                    * float(np.mean([float(row["top1_accuracy_probability"]) for row in selected])),
                    "exact_set_probability_percent": 100.0
                    * float(np.mean([float(row["exact_set_probability"]) for row in selected])),
                    "label_coverage_probability_percent": 100.0
                    * float(np.mean([float(row["label_coverage_probability"]) for row in selected])),
                }
            )
    return output


def validate_parent_test_reproduction(
    summaries: Mapping[str, Mapping[str, Any]], parents: Mapping[str, Any]
) -> dict[str, Any]:
    expected = {
        "standard_top1": float(parents["standard_test"]["test_clean"]["top1"]),
        "robust_top1": float(parents["anchor_test"]["test_clean"]["student_top1"]),
        "robust_exact_set": float(parents["anchor_test"]["test_clean"]["set2_exact"]),
    }
    observed = {
        "standard_top1": float(summaries["standard"]["top1_accuracy_percent"]),
        "robust_top1": float(summaries["robust"]["top1_accuracy_percent"]),
        "robust_exact_set": float(summaries["robust"]["exact_set_probability_percent"]),
    }
    drift = {key: observed[key] - expected[key] for key in expected}
    if any(abs(value) > 1e-9 for value in drift.values()):
        raise AssertionError(f"Parent deterministic official-test anchors did not reproduce: {drift}")
    return {"expected": expected, "observed": observed, "drift_pp": drift, "passed": True}


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, device=device)
    seed_everything(args.seed)
    # Attributes consumed by the frozen R2 evaluator.
    args.amp = False
    args.top1_weight = common.EXPECTED_SELECTION["top1_weight"]
    args.set_weight = common.EXPECTED_SELECTION["set_weight"]

    parents = load_and_validate_parents(args)
    selection = parents["selection"]
    methods = selected_methods(selection, args.mc_draws)

    dataset_root = dev.r1.r2_anchor.resolve_torchvision_root(args.data_dir)
    dataset = torchvision.datasets.CIFAR100(
        root=str(dataset_root),
        train=False,
        download=False,
        transform=transforms.ToTensor(),
    )
    if len(dataset) != TEST_EXAMPLES or len(dataset.targets) != TEST_EXAMPLES:
        raise AssertionError("Unexpected CIFAR-100 official-test size")
    counts = np.bincount(np.asarray(dataset.targets), minlength=100)
    if not np.all(counts == 100):
        raise AssertionError("Official CIFAR-100 test is not balanced at 100 examples per class")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_test_manifest(dataset, args)
    common.write_or_validate_json(output_dir / "official_test_manifest.json", manifest)

    paths = parents["paths"]
    sources = parents["sources"]
    config: dict[str, Any] = {
        "protocol_id": common.PROTOCOL_ID,
        "parent_dev_protocol_id": common.PARENT_DEV_PROTOCOL_ID,
        "parent_model_protocol_id": common.PARENT_MODEL_PROTOCOL_ID,
        "dataset": "CIFAR-100",
        "architecture": "ResNet-20",
        "primary_k": 2,
        "split": "complete official test",
        "test_examples": TEST_EXAMPLES,
        "data_dir": str(args.data_dir.expanduser().resolve()),
        "r2_root": str(args.r2_root.expanduser().resolve()),
        "dev_dir": str(args.dev_dir.expanduser().resolve()),
        "output_dir": str(output_dir),
        "device": args.device,
        "batch_size": args.batch_size,
        "mc_chunk": args.mc_chunk,
        "num_workers": args.num_workers,
        "gpu_memory_fraction": args.gpu_memory_fraction,
        "amp": False,
        "shard_count": args.shard_count,
        "shard_size": TEST_EXAMPLES // args.shard_count,
        "mc_draws": args.mc_draws,
        "seed": args.seed,
        "bootstrap_resamples": args.bootstrap_resamples,
        "bootstrap_seed": args.bootstrap_seed,
        "attack_split_seed": args.attack_split_seed,
        "noninferiority_margin_pp": NONINFERIORITY_MARGIN_PP,
        "methods": [asdict(method) for method in methods],
        "selection": {
            key: common.EXPECTED_SELECTION[key] for key in common.EXPECTED_SELECTION
        },
        "parent_dev_config_hash": common.EXPECTED_DEV_CONFIG_HASH,
        "parent_dev_selection_hash": common.EXPECTED_DEV_SELECTION_HASH,
        "checkpoints": {
            "standard": {
                "path": str(paths["standard_weights"]),
                "sha256": common.EXPECTED_STANDARD_SHA256,
            },
            "anchor": {
                "path": str(paths["anchor_weights"]),
                "sha256": common.EXPECTED_ANCHOR_SHA256,
            },
        },
        "parent_artifacts": {
            name: {"path": str(path), "sha256": common.file_sha256(path)}
            for name, path in paths.items()
            if name not in {"standard_weights", "anchor_weights"}
        },
        "script_sha256": {name: common.file_sha256(path) for name, path in sources.items()},
        "official_evaluation_manifest": {
            "path": str(output_dir / "official_test_manifest.json"),
            "sha256": common.file_sha256(output_dir / "official_test_manifest.json"),
            "manifest_hash": manifest["manifest_hash"],
        },
        "offline": True,
        "downloads_allowed": False,
        "no_training": True,
        "no_hyperparameter_search": True,
        "official_test_used_for_defense_selection": False,
        "attack_outcomes_read": False,
        "future_attack_splits_declared_before_attack_outcomes": True,
    }
    config_hash = common.object_sha256(config)
    config["config_hash"] = config_hash
    common.write_or_validate_json(output_dir / "run_config.json", config)

    standard, anchor = dev.r1.load_models(
        paths["standard_weights"], paths["anchor_weights"], device
    )
    shard_size = TEST_EXAMPLES // args.shard_count
    payloads: list[dict[str, Any]] = []
    run_started = time.monotonic()
    for shard_id in range(args.shard_count):
        start = shard_id * shard_size
        stop = start + shard_size
        payloads.append(
            run_shard(
                output_dir,
                config_hash,
                shard_id,
                list(range(start, stop)),
                dataset,
                methods,
                standard,
                anchor,
                device,
                args,
            )
        )
    orchestration_wall_seconds = time.monotonic() - run_started

    all_rows = [dict(row) for payload in payloads for row in payload["examples"]]
    grouped = {method: method_rows(all_rows, method) for method in common.METHODS}
    all_rows = [row for method in common.METHODS for row in grouped[method]]
    total_gpu_stage_wall_seconds = sum(float(payload["wall_seconds"]) for payload in payloads)
    summaries = [
        summarize_method(grouped[method.name], method, total_gpu_stage_wall_seconds)
        for method in methods
    ]
    summary_map = {str(row["method"]): row for row in summaries}
    reproduction = validate_parent_test_reproduction(summary_map, parents)

    print("Computing preregistered paired bootstrap", flush=True)
    comparisons = bootstrap_comparisons(grouped, args)
    decisions, attack_ready = common.evaluate_gate_decisions(
        comparisons, NONINFERIORITY_MARGIN_PP
    )
    decomposition = reference_decomposition(grouped)
    classwise = classwise_summary(grouped)

    common.atomic_csv(output_dir / "official_clean_per_example.csv", all_rows)
    common.atomic_csv(output_dir / "official_clean_summary.csv", summaries)
    common.atomic_csv(output_dir / "official_clean_paired_bootstrap.csv", comparisons)
    common.atomic_csv(output_dir / "official_clean_gate_decisions.csv", decisions)
    common.atomic_csv(output_dir / "official_clean_reference_decomposition.csv", decomposition)
    common.atomic_csv(output_dir / "official_clean_classwise.csv", classwise)

    attack_handoff = {
        "protocol_id": common.PROTOCOL_ID,
        "config_hash": config_hash,
        "authorized": bool(attack_ready),
        "authorization_basis": "all seven preregistered official clean gates passed",
        "if_not_authorized": "stop and report external-validity failure; do not retune on official test",
        "frozen_defense": {key: common.EXPECTED_SELECTION[key] for key in common.EXPECTED_SELECTION},
        "checkpoints": config["checkpoints"],
        "future_attack_partitions": manifest["future_attack_partitions"],
        "next_stage": (
            "adaptive_black_box_pilot_on_frozen_200_example_partition"
            if attack_ready
            else "stop_without_official_test_retuning"
        ),
        "attack_protocol_requirements": [
            "Use the 200-example pilot partition for implementation checks and attack-family choice.",
            "Freeze attack settings before reading outcomes on the disjoint 1,000-example confirmatory partition.",
            "Treat strengthening and Gaussian randomization as part of the queried oracle.",
            "Count every EOT, refresh, racing, and confirmation draw against the API-query budget.",
            "Re-evaluate saved candidates with a common independent MC=1000 estimator.",
            "Do not alter the defense operating point after this official clean evaluation.",
        ],
    }
    attack_handoff["handoff_hash"] = common.object_sha256(attack_handoff)
    common.atomic_json(output_dir / "attack_handoff.json", attack_handoff)

    final_summary = {
        "protocol_id": common.PROTOCOL_ID,
        "config_hash": config_hash,
        "status": "PASS",
        "official_clean_confirmation": "PASS" if attack_ready else "FAIL",
        "attack_stage_ready": bool(attack_ready),
        "summaries": summaries,
        "paired_comparisons": comparisons,
        "gate_decisions": decisions,
        "parent_test_reproduction": reproduction,
        "reference_quality_decomposition": decomposition,
        "orchestration_wall_seconds": float(orchestration_wall_seconds),
        "data_governance": {
            "defense_parameters_frozen_on_dev": True,
            "official_test_used_for_defense_selection": False,
            "no_retraining": True,
            "no_retuning": True,
            "attack_outcomes_read": False,
            "future_attack_partitions_predeclared": True,
        },
        "next_stage": attack_handoff["next_stage"],
    }
    common.atomic_json(output_dir / "final_summary.json", final_summary)

    print("\nCIFAR-100 official-test clean confirmation", flush=True)
    for row in summaries:
        print(
            f"  {row['method']:>11}: Top1={float(row['top1_accuracy_percent']):.3f}% "
            f"Exact={float(row['exact_set_probability_percent']):.3f}%",
            flush=True,
        )
    print("PASS deterministic parent official-test anchors reproduce exactly", flush=True)
    print("Preregistered clean gates", flush=True)
    for row in decisions:
        print(
            f"  {row['gate']}: {float(row['difference_pp']):+.3f} pp "
            f"[{float(row['ci95_low_pp']):+.3f}, {float(row['ci95_high_pp']):+.3f}] "
            f"-> {'PASS' if row['passed'] else 'FAIL'}",
            flush=True,
        )
    print(
        "OFFICIAL_CLEAN_CONFIRMATION=" + ("PASS" if attack_ready else "FAIL"),
        flush=True,
    )
    print("ATTACK_STAGE_READY=" + ("YES" if attack_ready else "NO"), flush=True)
    print(f"Complete: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
