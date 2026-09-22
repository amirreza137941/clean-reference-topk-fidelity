#!/usr/bin/env python3
"""Common-MC=1000 evaluation of all pre-registered holdout candidates."""

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
from dataclasses import dataclass
from pathlib import Path
from statistics import stdev
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_ROOT = Path("./models/resnet20")
DEFAULT_STANDARD_WEIGHTS = MODEL_ROOT / "cifar10/resnet20_cifar10.pth"
DEFAULT_DUAL_WEIGHTS = (
    MODEL_ROOT / "fine_frontier_top1_set2/resnet20_staged_epsilon_final_weights.pt"
)
PROTOCOL = "topk_confirmatory_holdout_common_mc1000_v1"
ATTACK_PROTOCOL = "topk_confirmatory_holdout_attack_v1"
MANIFEST_PROTOCOL = "topk_confirmatory_holdout_manifest_v1"
SELECTION_SEED = 2026083001
REEVAL_SEED = 2026083002
BOOTSTRAP_SEED = 20260830
BOOTSTRAP_RESAMPLES = 50000
SEEDS = (7171, 8181, 9191)
EXAMPLES = 200
K = 2
EPSILON_PX = 2.0
API_BUDGET = 5000
EVAL_DRAWS = 1000
SIGMA = 0.02144277648925781
BATCH_SIZE = 5
MODE_SPECS = {
    "full": {"strength_steps": 2, "attack": "SquareAdaptive"},
    "noise_only": {"strength_steps": 0, "attack": "SquareRacing"},
}
SUMMARY_FILE = "confirmatory_mc1000_summary.csv"
EXAMPLES_FILE = "confirmatory_mc1000_per_example.csv"
COMPARISONS_FILE = "confirmatory_mc1000_comparisons.csv"
FINAL_FILE = "confirmatory_mc1000_summary.json"
CONFIG_FILE = "run_config_confirmatory_mc1000.json"
SOURCE_IDS = (
    "clean_standard",
    "clean_robust",
    "clean_proposed",
    *(f"attack_robust_seed_{seed}" for seed in SEEDS),
    *(f"attack_proposed_seed_{seed}" for seed in SEEDS),
)
PERCENTAGE_METRICS = {
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


@dataclass(frozen=True)
class CandidateSpec:
    source_id: str
    defense: str
    launcher_seed: int
    seed_dir: Path
    config: Mapping[str, Any]
    source_row: Mapping[str, str]
    candidate_path: Path
    candidate_sha256: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=tuple(MODE_SPECS), required=True)
    parser.add_argument("--attack-root", type=Path, required=True)
    parser.add_argument("--holdout-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--standard-weights", type=Path, default=DEFAULT_STANDARD_WEIGHTS)
    parser.add_argument("--dual-weights", type=Path, default=DEFAULT_DUAL_WEIGHTS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--eval-draws", type=int, default=EVAL_DRAWS)
    parser.add_argument("--reeval-seed", type=int, default=REEVAL_SEED)
    parser.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.45)
    parser.add_argument("--rank-temperature", type=float, default=0.5)
    parser.add_argument("--progress-every-batches", type=int, default=5)
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def csv_text(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return ""
    import io

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    atomic_write_text(path, csv_text(rows))


def acquire_lock(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    handle = (output_dir / ".confirmatory_mc1000.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise SystemExit("Confirmatory MC=1000 evaluation is already active") from exc
    handle.seek(0)
    handle.truncate()
    handle.write(str(os.getpid()))
    handle.flush()
    return handle


def resolve_under(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Candidate path escapes its seed directory: {relative}") from exc
    return candidate


def manifest_samples(path: Path):
    manifest = read_json(path)
    expected = {
        "protocol_version": MANIFEST_PROTOCOL,
        "dataset": "CIFAR-10",
        "split": "test",
        "selection_seed": SELECTION_SEED,
        "examples": EXAMPLES,
        "examples_per_class": 20,
        "discovery_examples": 100,
        "discovery_overlap": 0,
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Holdout manifest mismatch: {mismatches}")
    samples = list(manifest.get("samples", []))
    if len(samples) != EXAMPLES:
        raise ValueError("Holdout manifest sample grid is incomplete")
    positions = [int(item["position"]) for item in samples]
    indices = [int(item["sample_index"]) for item in samples]
    labels = [int(item["true_label"]) for item in samples]
    excluded = {int(value) for value in manifest["excluded_discovery_indices"]}
    if positions != list(range(EXAMPLES)) or len(set(indices)) != EXAMPLES:
        raise ValueError("Holdout manifest positions/indices are invalid")
    if set(indices) & excluded:
        raise ValueError("Holdout manifest overlaps discovery")
    if {label: labels.count(label) for label in range(10)} != {
        label: 20 for label in range(10)
    }:
        raise ValueError("Holdout manifest is not class-balanced")
    return manifest, indices, labels


def unique_attack_row(rows, *, defense: str, attack: str):
    selected = [
        row
        for row in rows
        if row["defense"] == defense
        and row["attack"] == attack
        and int(row["k"]) == K
        and int(row["requested_budget"]) == API_BUDGET
        and abs(float(row["epsilon_px"]) - EPSILON_PX) < 1e-12
    ]
    if len(selected) != 1:
        raise ValueError(
            f"Expected one {defense}/{attack} Q={API_BUDGET} row, got {len(selected)}"
        )
    return selected[0]


def build_candidate_specs(
    attack_root: Path,
    *,
    mode: str,
    manifest_hash: str,
) -> tuple[dict[str, Any], list[CandidateSpec]]:
    attack_root = attack_root.expanduser().resolve()
    attack_manifest_path = attack_root / "confirmatory_attack_manifest.json"
    attack_manifest = read_json(attack_manifest_path)
    mode_spec = MODE_SPECS[mode]
    expected_manifest = {
        "protocol_version": ATTACK_PROTOCOL,
        "mode": mode,
        "strength_steps": mode_spec["strength_steps"],
        "attack": mode_spec["attack"],
        "seeds": list(SEEDS),
        "examples": EXAMPLES,
        "k": K,
        "epsilon_px": EPSILON_PX,
        "api_budget": API_BUDGET,
        "holdout_manifest_sha256": manifest_hash,
    }
    mismatches = {
        key: (attack_manifest.get(key), value)
        for key, value in expected_manifest.items()
        if attack_manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Attack-root manifest mismatch: {mismatches}")
    specs: list[CandidateSpec] = []
    for seed in SEEDS:
        seed_dir = attack_root / f"seed_{seed}"
        config_path = seed_dir / "run_config_topk_v6.json"
        rows_path = seed_dir / "attack_results_topk_v6.csv"
        examples_path = seed_dir / "attack_per_example_topk_v6.csv"
        for path in (config_path, rows_path, examples_path):
            if not path.is_file():
                raise FileNotFoundError(f"Missing seed artifact: {path}")
        config = read_json(config_path)
        expected_config = {
            "protocol_version": ATTACK_PROTOCOL,
            "evaluator_sha256": attack_manifest["evaluator_sha256"],
            "seed": seed,
            "data_seed": SELECTION_SEED,
            "attack_examples": EXAMPLES,
            "clean_examples": EXAMPLES,
            "top_k": [K],
            "defenses": ["robust", "proposed"],
            "attacks": [mode_spec["attack"]],
            "attack_goals": ["exact_set"],
            "attack_epsilons_px": [EPSILON_PX],
            "budgets": [API_BUDGET],
            "budget_unit": "api_queries",
            "eval_draws": 100,
            "clean_eval_draws": 100,
            "fixed_sigma": SIGMA,
            "strength_steps": mode_spec["strength_steps"],
            "fixed_subset_manifest_sha256": manifest_hash,
            "batch_size": BATCH_SIZE,
            "save_candidates": True,
            "candidate_dtype": "float32",
        }
        config_mismatches = {
            key: (config.get(key), value)
            for key, value in expected_config.items()
            if config.get(key) != value
        }
        if config_mismatches:
            raise ValueError(f"Seed {seed} config mismatch: {config_mismatches}")
        rows = read_csv(rows_path)
        example_rows = read_csv(examples_path)
        if len(rows) != 2 or len(example_rows) != 2 * EXAMPLES:
            raise ValueError(f"Seed {seed} attack grid is incomplete")
        for defense in ("robust", "proposed"):
            row = unique_attack_row(rows, defense=defense, attack=mode_spec["attack"])
            if int(row["attack_api_queries"]) != API_BUDGET:
                raise ValueError(f"Seed {seed}/{defense} query accounting differs")
            candidate_path = resolve_under(seed_dir, row["candidate_file"])
            if not candidate_path.is_file():
                raise FileNotFoundError(f"Missing candidate: {candidate_path}")
            specs.append(
                CandidateSpec(
                    source_id=f"attack_{defense}_seed_{seed}",
                    defense=defense,
                    launcher_seed=seed,
                    seed_dir=seed_dir,
                    config=config,
                    source_row=dict(row),
                    candidate_path=candidate_path,
                    candidate_sha256=sha256_file(candidate_path),
                )
            )
    return attack_manifest, specs


def load_payload(
    torch: Any,
    spec: CandidateSpec,
    manifest_hash: str,
    mode: str,
):
    try:
        raw = torch.load(spec.candidate_path, map_location="cpu", weights_only=True)
    except TypeError:
        raw = torch.load(spec.candidate_path, map_location="cpu")
    except Exception as exc:
        print(f"weights_only load failed for trusted artifact: {exc}", flush=True)
        raw = torch.load(spec.candidate_path, map_location="cpu", weights_only=False)
    if not isinstance(raw, Mapping):
        raise TypeError(f"Candidate is not a mapping: {spec.candidate_path}")
    required = {"adversarial", "clean", "labels", "dataset_indices", "metadata"}
    missing = required - set(raw)
    if missing:
        raise ValueError(f"Candidate misses {sorted(missing)}: {spec.candidate_path}")
    adversarial = raw["adversarial"].float()
    clean = raw["clean"].float()
    labels = raw["labels"].long()
    indices = raw["dataset_indices"].long()
    metadata = raw["metadata"]
    if tuple(adversarial.shape) != (EXAMPLES, 3, 32, 32):
        raise ValueError(f"Candidate shape differs: {spec.candidate_path}")
    if clean.shape != adversarial.shape or tuple(labels.shape) != (EXAMPLES,):
        raise ValueError("Candidate clean/label shapes differ")
    if tuple(indices.shape) != (EXAMPLES,):
        raise ValueError("Candidate index shape differs")
    if not bool(torch.isfinite(adversarial).all()) or not bool(torch.isfinite(clean).all()):
        raise ValueError("Candidate contains non-finite pixels")
    if float(adversarial.min()) < -1e-7 or float(adversarial.max()) > 1.0 + 1e-7:
        raise ValueError("Candidate leaves [0,1]")
    linf_px = 255.0 * float((adversarial - clean).abs().amax())
    if linf_px > EPSILON_PX + 2.6e-5:
        raise ValueError(f"Candidate violates Linf: {linf_px:.8f}px")
    if not isinstance(metadata, Mapping):
        raise ValueError("Candidate metadata is not a mapping")
    expected_metadata = {
        "protocol_version": ATTACK_PROTOCOL,
        "attack_engine_protocol": "topk_v6_perdraw_square_racing",
        "fixed_subset_manifest_protocol": MANIFEST_PROTOCOL,
        "fixed_subset_manifest_sha256": manifest_hash,
        "defense": spec.defense,
        "attack": MODE_SPECS[mode]["attack"],
        "attack_goal": "exact_set",
        "k": K,
        "requested_budget": API_BUDGET,
        "attack_api_queries": API_BUDGET,
        "data_seed": SELECTION_SEED,
    }
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected_metadata.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Candidate metadata mismatch: {mismatches}")
    return {
        "adversarial": adversarial,
        "clean": clean,
        "labels": labels,
        "dataset_indices": indices,
        "metadata": dict(metadata),
        "linf_px": linf_px,
    }


def summary_from_values(values: Mapping[str, Any]) -> dict[str, float]:
    return {
        name: (100.0 if name in PERCENTAGE_METRICS else 1.0)
        * float(tensor.mean().item())
        for name, tensor in values.items()
    }


def evaluate_with_progress(
    evaluator: Any,
    defense: Any,
    images: Any,
    labels: Any,
    reference_logits: Any,
    *,
    source_id: str,
    args: argparse.Namespace,
):
    evaluator.base.seed_everything(args.reeval_seed)
    draws = args.eval_draws if defense.sigma > 0 else 1
    bundles = []
    total_batches = (len(images) + args.batch_size - 1) // args.batch_size
    for batch_number, low in enumerate(
        range(0, len(images), args.batch_size), start=1
    ):
        high = min(low + args.batch_size, len(images))
        draw_logits = evaluator.collect_actual_draws(
            defense, images[low:high], high - low, draws
        )
        bundles.append(
            evaluator.evaluate_draws(
                draw_logits,
                labels[low:high].detach().cpu(),
                reference_logits[low:high].detach().cpu(),
                K,
                args.rank_temperature,
                True,
            )
        )
        if (
            batch_number % args.progress_every_batches == 0
            or batch_number == total_batches
        ):
            print(
                f"    {source_id}: batch {batch_number}/{total_batches}, "
                f"draws={draws}",
                flush=True,
            )
    torch = evaluator.torch
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
        reference_set=torch.cat([bundle.reference_set for bundle in bundles]),
        first_draw_set=torch.cat([bundle.first_draw_set for bundle in bundles]),
        mean_logits_set=torch.cat([bundle.mean_logits_set for bundle in bundles]),
        mean_logits_summary=summary_from_values(mean_values),
        mean_logits_values=mean_values,
        draws=draws,
    )


def partial_paths(output_dir: Path, source_id: str):
    return (
        output_dir / f"partials/{source_id}.summary.json",
        output_dir / f"partials/{source_id}.per_example.csv",
    )


def evaluate_source(
    *,
    evaluator: Any,
    source_id: str,
    condition: str,
    defense_name: str,
    defense: Any,
    images: Any,
    clean: Any,
    labels: Any,
    indices: Any,
    reference_logits: Any,
    mode: str,
    manifest_hash: str,
    candidate_spec: CandidateSpec | None,
    output_dir: Path,
    args: argparse.Namespace,
):
    candidate_hash = candidate_spec.candidate_sha256 if candidate_spec else ""
    summary_path, examples_path = partial_paths(output_dir, source_id)
    if summary_path.is_file() and examples_path.is_file():
        summary = read_json(summary_path)
        rows = read_csv(examples_path)
        expected = {
            "protocol_version": PROTOCOL,
            "source_id": source_id,
            "mode": mode,
            "condition": condition,
            "defense": defense_name,
            "holdout_manifest_sha256": manifest_hash,
            "source_candidate_sha256": candidate_hash,
            "reeval_seed": args.reeval_seed,
            "reeval_draws_requested": args.eval_draws,
            "reeval_draws_actual": args.eval_draws if defense_name == "proposed" else 1,
            "batch_size": args.batch_size,
        }
        mismatches = {
            key: (summary.get(key), value)
            for key, value in expected.items()
            if summary.get(key) != value
        }
        if mismatches or len(rows) != EXAMPLES:
            raise RuntimeError(f"Partial mismatch for {source_id}: {mismatches}")
        print(f"PASS reusing partial {source_id}", flush=True)
        return summary, rows

    started = time.perf_counter()
    evaluation = evaluate_with_progress(
        evaluator,
        defense,
        images,
        labels,
        reference_logits,
        source_id=source_id,
        args=args,
    )
    wall_seconds = time.perf_counter() - started
    exact = evaluation.values["exact_set"].detach().cpu().numpy()
    mc_se = 100.0 * float(
        np.sqrt(np.sum(exact * (1.0 - exact)))
        / (len(exact) * np.sqrt(max(1, evaluation.draws)))
    )
    summary: dict[str, Any] = {
        "protocol_version": PROTOCOL,
        "source_id": source_id,
        "mode": mode,
        "condition": condition,
        "defense": defense_name,
        "holdout_manifest_sha256": manifest_hash,
        "source_candidate_sha256": candidate_hash,
        "source_candidate_file": (
            str(candidate_spec.candidate_path) if candidate_spec else ""
        ),
        "launcher_seed": candidate_spec.launcher_seed if candidate_spec else "",
        "source_attack_seed": (
            int(candidate_spec.source_row["attack_seed"])
            if candidate_spec
            else 0
        ),
        "source_attack": candidate_spec.source_row["attack"] if candidate_spec else "none",
        "candidate_linf_px": (
            255.0 * float((images - clean).abs().amax())
            if candidate_spec
            else 0.0
        ),
        "data_seed": SELECTION_SEED,
        "k": K,
        "examples": EXAMPLES,
        "reeval_seed": args.reeval_seed,
        "reeval_draws_requested": args.eval_draws,
        "reeval_draws_actual": evaluation.draws,
        "measurement_draws_counted_in_attack_budget": False,
        "common_random_numbers": True,
        "batch_size": args.batch_size,
        "wall_seconds": wall_seconds,
        "mc_only_standard_error_pp": mc_se,
    }
    summary.update(evaluation.summary)
    for name, value in evaluation.mean_logits_summary.items():
        summary[f"mean_logits_{name}"] = value
    attack_seed = int(candidate_spec.source_row["attack_seed"]) if candidate_spec else 0
    rows = evaluator.per_example_rows(
        evaluation=evaluation,
        labels=labels,
        dataset_indices=indices,
        defense=defense_name,
        condition=condition,
        k=K,
        data_seed=SELECTION_SEED,
        attack_seed=attack_seed,
        reeval_seed=args.reeval_seed,
        run_tag=f"confirmatory_mc1000_{mode}",
        shard_count=1,
        shard_index=0,
        attack=candidate_spec.source_row["attack"] if candidate_spec else "none",
        attack_goal="exact_set" if candidate_spec else "none",
        epsilon_px=EPSILON_PX if candidate_spec else 0.0,
        requested_budget=API_BUDGET if candidate_spec else 0,
        logical_budget=(
            candidate_spec.source_row["logical_budget"] if candidate_spec else 0
        ),
        attack_eot=candidate_spec.source_row["attack_eot"] if candidate_spec else 0,
        attack_api_queries=API_BUDGET if candidate_spec else 0,
        candidate_file=(str(candidate_spec.candidate_path) if candidate_spec else ""),
    )
    enriched = [
        {
            "protocol_version": PROTOCOL,
            "source_id": source_id,
            "mode": mode,
            "holdout_manifest_sha256": manifest_hash,
            "source_candidate_sha256": candidate_hash,
            "launcher_seed": candidate_spec.launcher_seed if candidate_spec else "",
            "measurement_draws_counted_in_attack_budget": False,
            "common_random_numbers": True,
            **row,
        }
        for row in rows
    ]
    atomic_write_text(summary_path, json.dumps(summary, indent=2, sort_keys=True))
    atomic_write_csv(examples_path, enriched)
    return summary, enriched


def metric_array(rows, source_id: str, metric: str):
    column = {"top1": "top1_probability", "exact_set": "exact_set_probability"}[metric]
    selected = sorted(
        (row for row in rows if row["source_id"] == source_id),
        key=lambda row: int(row["position"]),
    )
    if len(selected) != EXAMPLES:
        raise ValueError(f"Expected {EXAMPLES} rows for {source_id}")
    return [row["sample_id"] for row in selected], np.array(
        [float(row[column]) for row in selected], dtype=np.float64
    )


def paired_bootstrap(delta: np.ndarray, resamples: int, seed: int):
    if delta.ndim != 1 or len(delta) != EXAMPLES:
        raise ValueError("Paired bootstrap requires the 200-image vector")
    rng = np.random.default_rng(seed)
    parts = []
    for start in range(0, resamples, 5000):
        count = min(5000, resamples - start)
        indices = rng.integers(0, len(delta), size=(count, len(delta)))
        parts.append(delta[indices].mean(axis=1))
    samples = np.concatenate(parts)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(delta.mean()), float(low), float(high)


def comparison_row(
    *,
    name: str,
    family: str,
    metric: str,
    left_id: str,
    right_id: str,
    left: np.ndarray,
    right: np.ndarray,
    seed: int,
    positive_means: str,
    inference_scope: str,
):
    estimate, low, high = paired_bootstrap(left - right, BOOTSTRAP_RESAMPLES, seed)
    verdict = (
        "positive"
        if low > 0
        else "negative"
        if high < 0
        else "statistically_indistinguishable"
    )
    return {
        "comparison": name,
        "family": family,
        "metric": metric,
        "left_source_id": left_id,
        "right_source_id": right_id,
        "left_value_pct": 100.0 * float(left.mean()),
        "right_value_pct": 100.0 * float(right.mean()),
        "left_minus_right_pp": 100.0 * estimate,
        "ci95_low_pp": 100.0 * low,
        "ci95_high_pp": 100.0 * high,
        "positive_delta_means": positive_means,
        "verdict": verdict,
        "inference_scope": inference_scope,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": seed,
    }


def build_comparisons(rows):
    arrays: dict[tuple[str, str], np.ndarray] = {}
    canonical = None
    for source_id in SOURCE_IDS:
        for metric in ("top1", "exact_set"):
            ids, values = metric_array(rows, source_id, metric)
            if canonical is None:
                canonical = ids
            elif ids != canonical:
                raise ValueError(f"Sample order differs for {source_id}/{metric}")
            arrays[(source_id, metric)] = values
    result = []
    offset = 0
    scope = "paired bootstrap over the pre-registered 200-image holdout"

    def add(name, family, metric, left_id, right_id, left, right, meaning, extra_scope=""):
        nonlocal offset
        result.append(
            comparison_row(
                name=name,
                family=family,
                metric=metric,
                left_id=left_id,
                right_id=right_id,
                left=left,
                right=right,
                seed=BOOTSTRAP_SEED + offset,
                positive_means=meaning,
                inference_scope=scope + extra_scope,
            )
        )
        offset += 1

    add(
        "clean_proposed_minus_standard_top1", "clean", "top1",
        "clean_proposed", "clean_standard",
        arrays[("clean_proposed", "top1")], arrays[("clean_standard", "top1")],
        "Proposed has higher clean Top-1 accuracy",
    )
    add(
        "clean_robust_minus_standard_top1", "clean", "top1",
        "clean_robust", "clean_standard",
        arrays[("clean_robust", "top1")], arrays[("clean_standard", "top1")],
        "Robust has higher clean Top-1 accuracy",
    )
    for metric in ("top1", "exact_set"):
        add(
            f"clean_proposed_minus_robust_{metric}", "clean", metric,
            "clean_proposed", "clean_robust",
            arrays[("clean_proposed", metric)], arrays[("clean_robust", metric)],
            "Proposed has higher clean performance",
        )

    for seed in SEEDS:
        proposed_id = f"attack_proposed_seed_{seed}"
        robust_id = f"attack_robust_seed_{seed}"
        for metric in ("top1", "exact_set"):
            proposed_attack = arrays[(proposed_id, metric)]
            robust_attack = arrays[(robust_id, metric)]
            proposed_clean = arrays[("clean_proposed", metric)]
            robust_clean = arrays[("clean_robust", metric)]
            add(
                f"seed_{seed}_attack_proposed_minus_robust_{metric}",
                "per_seed_attacked_absolute", metric, proposed_id, robust_id,
                proposed_attack, robust_attack,
                "Proposed has higher attacked performance",
                "; one pre-registered observed attack seed",
            )
            add(
                f"seed_{seed}_proposed_attack_minus_clean_{metric}",
                "per_seed_clean_to_attack", metric, proposed_id, "clean_proposed",
                proposed_attack, proposed_clean,
                "Positive means improvement; negative means degradation",
                "; one pre-registered observed attack seed",
            )
            add(
                f"seed_{seed}_robust_attack_minus_clean_{metric}",
                "per_seed_clean_to_attack", metric, robust_id, "clean_robust",
                robust_attack, robust_clean,
                "Positive means improvement; negative means degradation",
                "; one pre-registered observed attack seed",
            )
            add(
                f"seed_{seed}_difference_in_drops_{metric}",
                "per_seed_difference_in_drops", metric,
                f"{proposed_id}-clean_proposed", f"{robust_id}-clean_robust",
                proposed_attack - proposed_clean, robust_attack - robust_clean,
                "Proposed degrades less than Robust",
                "; one pre-registered observed attack seed",
            )

    for metric in ("top1", "exact_set"):
        proposed_mean = np.stack(
            [arrays[(f"attack_proposed_seed_{seed}", metric)] for seed in SEEDS]
        ).mean(axis=0)
        robust_mean = np.stack(
            [arrays[(f"attack_robust_seed_{seed}", metric)] for seed in SEEDS]
        ).mean(axis=0)
        proposed_clean = arrays[("clean_proposed", metric)]
        robust_clean = arrays[("clean_robust", metric)]
        extra = "; per-image mean over three pre-registered observed attack seeds"
        add(
            f"three_seed_mean_attack_proposed_minus_robust_{metric}",
            "primary_three_seed_mean", metric,
            "attack_proposed_three_seed_mean", "attack_robust_three_seed_mean",
            proposed_mean, robust_mean,
            "Proposed has higher attacked performance", extra,
        )
        add(
            f"three_seed_mean_proposed_attack_minus_clean_{metric}",
            "primary_three_seed_mean", metric,
            "attack_proposed_three_seed_mean", "clean_proposed",
            proposed_mean, proposed_clean,
            "Positive means improvement; negative means degradation", extra,
        )
        add(
            f"three_seed_mean_robust_attack_minus_clean_{metric}",
            "primary_three_seed_mean", metric,
            "attack_robust_three_seed_mean", "clean_robust",
            robust_mean, robust_clean,
            "Positive means improvement; negative means degradation", extra,
        )
        add(
            f"three_seed_mean_difference_in_drops_{metric}",
            "primary_three_seed_mean", metric,
            "proposed_three_seed_mean_drop", "robust_three_seed_mean_drop",
            proposed_mean - proposed_clean, robust_mean - robust_clean,
            "Proposed degrades less than Robust", extra,
        )
    if len(result) != 36:
        raise RuntimeError(f"Comparison-grid error: {len(result)}")
    return result


def validate_args(args):
    if args.batch_size != BATCH_SIZE:
        raise ValueError("Confirmatory common-MC protocol is fixed at batch_size=5")
    if args.eval_draws != EVAL_DRAWS or args.reeval_seed != REEVAL_SEED:
        raise ValueError("Confirmatory common-MC draw count/seed is immutable")
    if args.bootstrap_resamples != BOOTSTRAP_RESAMPLES or args.bootstrap_seed != BOOTSTRAP_SEED:
        raise ValueError("Confirmatory bootstrap protocol is immutable")
    if abs(args.rank_temperature - 0.5) > 1e-12:
        raise ValueError("rank_temperature must remain 0.5")
    if args.progress_every_batches < 1:
        raise ValueError("progress_every_batches must be positive")
    required = [
        args.standard_weights,
        args.dual_weights,
        args.holdout_manifest,
        args.attack_root / "confirmatory_attack_manifest.json",
        PROJECT_DIR / "evaluate_topk_confirmatory_holdout_cifar10.py",
        PROJECT_DIR / "evaluate_final_top1_set2_defense_cifar10.py",
        PROJECT_DIR / "evaluate_top1_set2_gated_cifar10.py",
    ]
    missing = [str(path) for path in required if not path.expanduser().is_file()]
    if missing:
        raise FileNotFoundError("Missing re-evaluation inputs:\n  " + "\n  ".join(missing))


def main():
    args = parse_args()
    validate_args(args)
    output_dir = args.output_dir.expanduser().resolve()
    lock = acquire_lock(output_dir)
    try:
        manifest_path = args.holdout_manifest.expanduser().resolve()
        manifest, manifest_indices, manifest_labels = manifest_samples(manifest_path)
        manifest_hash = sha256_file(manifest_path)
        attack_manifest, specs = build_candidate_specs(
            args.attack_root, mode=args.mode, manifest_hash=manifest_hash
        )

        import torch

        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        sys.path.insert(0, str(PROJECT_DIR))
        import evaluate_topk_confirmatory_holdout_cifar10 as evaluator

        payloads = {
            spec.source_id: load_payload(torch, spec, manifest_hash, args.mode)
            for spec in specs
        }
        first = payloads[specs[0].source_id]
        baseline = {
            "clean": first["clean"],
            "labels": first["labels"],
            "dataset_indices": first["dataset_indices"],
        }
        if baseline["dataset_indices"].tolist() != manifest_indices:
            raise ValueError("Candidate order differs from the holdout manifest")
        if baseline["labels"].tolist() != manifest_labels:
            raise ValueError("Candidate labels differ from the holdout manifest")
        for spec in specs[1:]:
            payload = payloads[spec.source_id]
            if not torch.equal(payload["clean"], baseline["clean"]):
                raise ValueError(f"Clean tensor differs for {spec.source_id}")
            if not torch.equal(payload["labels"], baseline["labels"]):
                raise ValueError(f"Labels differ for {spec.source_id}")
            if not torch.equal(payload["dataset_indices"], baseline["dataset_indices"]):
                raise ValueError(f"Indices differ for {spec.source_id}")

        device = torch.device(args.device)
        if device.type == "cuda":
            torch.cuda.set_device(device)
            index = device.index if device.index is not None else 0
            torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, index)
            torch.cuda.reset_peak_memory_stats(device)
        standard_model, standard_arch = evaluator.legacy.load_resnet20(
            args.standard_weights, device, evaluator.legacy.STANDARD_CIFAR10_STD
        )
        dual_model = evaluator.base.load_dual_defense(args.dual_weights, device)
        robust_model = evaluator.base.RobustPixelModel(dual_model).to(device).eval()
        proposed_actual = evaluator.TopKTargetedGaussianDefense(
            standard_model=standard_model,
            gradient_model=robust_model,
            output_model=robust_model,
            top_k=K,
            alpha=0.005,
            epsilon=0.03,
            steps=MODE_SPECS[args.mode]["strength_steps"],
            sigma=SIGMA,
            top1_weight=1.0,
            set_weight=4.0,
        )
        defenses = {
            "standard": evaluator.base.DefenseSpec(
                "standard", evaluator.base.make_actual(standard_model), standard_model, 0.0
            ),
            "robust": evaluator.base.DefenseSpec(
                "robust", evaluator.base.make_actual(robust_model), robust_model, 0.0
            ),
            "proposed": evaluator.base.DefenseSpec(
                "proposed", proposed_actual, robust_model, SIGMA
            ),
        }
        clean_device = baseline["clean"].to(device)
        labels_device = baseline["labels"].to(device)
        reference_logits = evaluator.base.compute_logits_batched(
            defenses["standard"], clean_device, args.batch_size, 1
        ).detach().cpu()
        reference = torch.topk(reference_logits, K, dim=1).indices

        config = {
            "protocol_version": PROTOCOL,
            "mode": args.mode,
            "attack_root": str(args.attack_root.expanduser().resolve()),
            "attack_manifest_sha256": sha256_file(
                args.attack_root / "confirmatory_attack_manifest.json"
            ),
            "holdout_manifest": str(manifest_path),
            "holdout_manifest_sha256": manifest_hash,
            "output_dir": str(output_dir),
            "standard_weights": str(args.standard_weights.expanduser().resolve()),
            "dual_weights": str(args.dual_weights.expanduser().resolve()),
            "standard_weights_sha256": sha256_file(args.standard_weights),
            "dual_weights_sha256": sha256_file(args.dual_weights),
            "evaluator_sha256": sha256_file(
                PROJECT_DIR / "evaluate_topk_confirmatory_holdout_cifar10.py"
            ),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "attack_protocol": ATTACK_PROTOCOL,
            "attack": MODE_SPECS[args.mode]["attack"],
            "strength_steps": MODE_SPECS[args.mode]["strength_steps"],
            "seeds": list(SEEDS),
            "data_seed": SELECTION_SEED,
            "examples": EXAMPLES,
            "k": K,
            "epsilon_px": EPSILON_PX,
            "api_budget": API_BUDGET,
            "eval_draws": args.eval_draws,
            "reeval_seed": args.reeval_seed,
            "batch_size": args.batch_size,
            "bootstrap_resamples": args.bootstrap_resamples,
            "bootstrap_seed": args.bootstrap_seed,
            "common_random_numbers": True,
            "measurement_draws_counted_in_attack_budget": False,
            "candidate_source_ids": [spec.source_id for spec in specs],
        }
        config_path = output_dir / CONFIG_FILE
        if config_path.is_file():
            if read_json(config_path) != config:
                raise RuntimeError("Existing MC=1000 config differs; use a fresh output")
        else:
            atomic_write_text(config_path, json.dumps(config, indent=2, sort_keys=True))

        print(f"Standard checkpoint architecture: {standard_arch}", flush=True)
        plan = [
            ("clean_standard", "clean", "standard", None),
            ("clean_robust", "clean", "robust", None),
            ("clean_proposed", "clean", "proposed", None),
        ]
        spec_by_id = {spec.source_id: spec for spec in specs}
        for seed in SEEDS:
            plan.append(
                (f"attack_robust_seed_{seed}", "attack", "robust", spec_by_id[f"attack_robust_seed_{seed}"])
            )
        for seed in SEEDS:
            plan.append(
                (f"attack_proposed_seed_{seed}", "attack", "proposed", spec_by_id[f"attack_proposed_seed_{seed}"])
            )

        summaries = []
        all_rows = []
        for position, (source_id, condition, defense_name, spec) in enumerate(plan, start=1):
            images = baseline["clean"] if spec is None else payloads[source_id]["adversarial"]
            summary, rows = evaluate_source(
                evaluator=evaluator,
                source_id=source_id,
                condition=condition,
                defense_name=defense_name,
                defense=defenses[defense_name],
                images=images.to(device),
                clean=baseline["clean"].to(device),
                labels=labels_device,
                indices=baseline["dataset_indices"],
                reference_logits=reference_logits,
                mode=args.mode,
                manifest_hash=manifest_hash,
                candidate_spec=spec,
                output_dir=output_dir,
                args=args,
            )
            current = sorted(rows, key=lambda row: int(row["position"]))
            if [int(row["sample_index"]) for row in current] != manifest_indices:
                raise RuntimeError(f"Per-example order differs for {source_id}")
            if [int(row["true_label"]) for row in current] != manifest_labels:
                raise RuntimeError(f"Per-example labels differ for {source_id}")
            if [json.loads(row["reference_set"]) for row in current] != reference.tolist():
                raise RuntimeError(f"Reference set differs for {source_id}")
            summaries.append(summary)
            all_rows.extend(current)
            print(
                f"PASS [{position}/{len(plan)}] {source_id}: "
                f"Top1={float(summary['top1']):.4f}% "
                f"Exact={float(summary['exact_set']):.4f}%",
                flush=True,
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if tuple(row["source_id"] for row in summaries) != SOURCE_IDS:
            raise RuntimeError("Final source ordering differs")
        comparisons = build_comparisons(all_rows)
        atomic_write_csv(output_dir / SUMMARY_FILE, summaries)
        atomic_write_csv(output_dir / EXAMPLES_FILE, all_rows)
        atomic_write_csv(output_dir / COMPARISONS_FILE, comparisons)

        summary_by_id = {row["source_id"]: row for row in summaries}
        per_seed = {}
        for defense in ("robust", "proposed"):
            values = [
                float(summary_by_id[f"attack_{defense}_seed_{seed}"]["exact_set"])
                for seed in SEEDS
            ]
            per_seed[defense] = {
                "values_pct": values,
                "mean_pct": float(np.mean(values)),
                "sd_pp": float(stdev(values)),
                "lowest_survival_seed": SEEDS[int(np.argmin(values))],
                "lowest_survival_pct": float(min(values)),
            }
        comparison_by_name = {row["comparison"]: row for row in comparisons}
        final = {
            "protocol_version": PROTOCOL,
            "mode": args.mode,
            "holdout_manifest_sha256": manifest_hash,
            "source_count": len(summaries),
            "per_example_rows": len(all_rows),
            "comparison_count": len(comparisons),
            "clean_top1_accuracy_pct": {
                source: float(summary_by_id[source]["top1"])
                for source in ("clean_standard", "clean_robust", "clean_proposed")
            },
            "clean_exact_set_pct": {
                source: float(summary_by_id[source]["exact_set"])
                for source in ("clean_standard", "clean_robust", "clean_proposed")
            },
            "attacked_exact_set_across_seeds": per_seed,
            "primary_three_seed_mean_comparisons": {
                metric: {
                    name: comparison_by_name[name]
                    for name in comparison_by_name
                    if name.startswith("three_seed_mean_") and name.endswith(metric)
                }
                for metric in ("top1", "exact_set")
            },
            "inference_scope": (
                "Confirmatory inference is paired over the immutable 200-image "
                "holdout after per-image averaging of three pre-registered attack "
                "seeds. Seed SD is descriptive because only three seeds were run."
            ),
        }
        atomic_write_text(output_dir / FINAL_FILE, json.dumps(final, indent=2, sort_keys=True))
        print(f"Confirmatory common-MC=1000 complete: {output_dir}", flush=True)
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    main()
