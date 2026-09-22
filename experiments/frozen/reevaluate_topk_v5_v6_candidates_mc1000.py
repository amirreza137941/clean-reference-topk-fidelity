#!/usr/bin/env python3
"""Common-random-number re-evaluation of saved V5/V6 Top-k candidates.

This diagnostic performs no attack optimization.  It loads the adversarial
candidate tensors already produced by V5 and V6 and evaluates every candidate
with the same model, sample order, batch size, re-evaluation seed, and Monte
Carlo draw count.  The goal is to distinguish a genuinely stronger V5
candidate from finite-MC noise or a candidate-selection artifact.

The final evaluation draws are measurement draws and are deliberately reported
separately from the original Q=5000 attack budget.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_ROOT = Path("./models/resnet20")
DEFAULT_STANDARD_WEIGHTS = (
    DEFAULT_MODEL_ROOT / "cifar10/resnet20_cifar10.pth"
)
DEFAULT_DUAL_WEIGHTS = (
    DEFAULT_MODEL_ROOT
    / "fine_frontier_top1_set2/resnet20_staged_epsilon_final_weights.pt"
)
DEFAULT_V5_FULL = DEFAULT_MODEL_ROOT / "paper_topk_v5_square_full_q5k_independent"
DEFAULT_V5_NOISE = (
    DEFAULT_MODEL_ROOT / "paper_topk_v5_square_noise_only_q5k_independent"
)
DEFAULT_V6_FULL = DEFAULT_MODEL_ROOT / "paper_topk_v6_racing_full_k2_q5k_seed3"
DEFAULT_V6_NOISE = (
    DEFAULT_MODEL_ROOT / "paper_topk_v6_racing_noise_only_k2_q5k_seed3"
)
V6_SEEDS = (4041, 5051, 6061)
EXPECTED_DATA_SEED = 3031
EXPECTED_K = 2
EXPECTED_EPSILON_PX = 2.0
EXPECTED_BUDGET = 5000
EXPECTED_EXAMPLES = 100
EXPECTED_SIGMA = 0.02144277648925781
EXPECTED_V5_PROTOCOL = "topk_v5_reference_square_refresh_confirm"
EXPECTED_V6_PROTOCOL = "topk_v6_perdraw_square_racing"
REEVAL_PROTOCOL = "topk_v5_v6_saved_candidate_common_mc1000_r2"
CANONICAL_CLEAN_SOURCE_ID = "v6_proposed_seed_4041"
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
    source_version: str
    source_protocol: str
    source_dir: Path
    source_row: Mapping[str, str]
    candidate_path: Path

    @property
    def defense(self) -> str:
        return str(self.source_row["defense"])

    @property
    def attack(self) -> str:
        return str(self.source_row["attack"])

    @property
    def attack_seed(self) -> int:
        return int(self.source_row["attack_seed"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-evaluate saved V5/V6 Top-k candidates with common MC draws."
    )
    parser.add_argument("--mode", choices=("full", "noise_only"), required=True)
    parser.add_argument("--v5-dir", type=Path, required=True)
    parser.add_argument("--v6-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--standard-weights", type=Path, default=DEFAULT_STANDARD_WEIGHTS)
    parser.add_argument("--dual-weights", type=Path, default=DEFAULT_DUAL_WEIGHTS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--eval-draws", type=int, default=1000)
    parser.add_argument("--reeval-seed", type=int, default=2026082801)
    parser.add_argument("--bootstrap-resamples", type=int, default=50000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260827)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.45)
    parser.add_argument("--rank-temperature", type=float, default=0.5)
    parser.add_argument("--progress-every-batches", type=int, default=1)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_under(root: Path, relative: str) -> Path:
    root = root.expanduser().resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Candidate path escapes its result directory: {relative}") from exc
    return candidate


def select_unique_row(
    rows: Sequence[Mapping[str, str]],
    *,
    defense: str,
    attack: str,
    k: int = EXPECTED_K,
    budget: int = EXPECTED_BUDGET,
) -> Mapping[str, str]:
    matches = [
        row
        for row in rows
        if row.get("defense") == defense
        and row.get("attack") == attack
        and int(row.get("k", -1)) == k
        and int(row.get("requested_budget", -1)) == budget
        and row.get("attack_goal") == "exact_set"
        and abs(float(row.get("epsilon_px", "nan")) - EXPECTED_EPSILON_PX) < 1e-12
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one row for defense={defense}, attack={attack}, k={k}, "
            f"Q={budget}; found {len(matches)}"
        )
    row = matches[0]
    if int(row.get("attack_api_queries", -1)) != budget:
        raise ValueError("Source row does not record the exact attack API budget")
    if not row.get("candidate_file"):
        raise ValueError("Source row has no candidate_file")
    return row


def validate_config(
    config: Mapping[str, Any],
    *,
    protocol: str,
    mode: str,
    expected_seed: int | None,
    standard_weights: Path,
    dual_weights: Path,
) -> None:
    expected_steps = 2 if mode == "full" else 0
    expected = {
        "protocol_version": protocol,
        "data_seed": EXPECTED_DATA_SEED,
        "attack_examples": EXPECTED_EXAMPLES,
        "attack_goals": ["exact_set"],
        "strength_steps": expected_steps,
        "strength_alpha": 0.005,
        "strength_epsilon": 0.03,
        "top1_weight": 1.0,
        "set_weight": 4.0,
        "fixed_sigma": EXPECTED_SIGMA,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Source config mismatch for {key}: {config.get(key)!r} != {value!r}")
    if EXPECTED_K not in [int(item) for item in config.get("top_k", [])]:
        raise ValueError("Source config does not include k=2")
    if EXPECTED_BUDGET not in [int(item) for item in config.get("budgets", [])]:
        raise ValueError("Source config does not include Q=5000")
    epsilons = [float(item) for item in config.get("attack_epsilons_px", [])]
    if not any(abs(item - EXPECTED_EPSILON_PX) < 1e-12 for item in epsilons):
        raise ValueError("Source config does not include epsilon=2/255")
    if expected_seed is not None and int(config.get("seed", -1)) != expected_seed:
        raise ValueError("Source config records the wrong attack seed")
    for key, current in (
        ("standard_weights", standard_weights),
        ("dual_weights", dual_weights),
    ):
        configured = Path(str(config.get(key, ""))).expanduser().resolve()
        if configured != current.expanduser().resolve():
            raise ValueError(f"Current {key} differs from the source run: {configured} != {current}")


def source_spec(
    *,
    source_id: str,
    source_version: str,
    source_protocol: str,
    source_dir: Path,
    row: Mapping[str, str],
) -> CandidateSpec:
    path = resolve_under(source_dir, row["candidate_file"])
    if not path.is_file():
        extra = (
            " The compact ZIP archives may omit candidate tensors; use the original "
            "result directory on the GPU server."
        )
        raise FileNotFoundError(f"Missing saved candidate tensor: {path}.{extra}")
    return CandidateSpec(
        source_id=source_id,
        source_version=source_version,
        source_protocol=source_protocol,
        source_dir=source_dir.resolve(),
        source_row=dict(row),
        candidate_path=path,
    )


def build_candidate_specs(
    mode: str,
    v5_dir: Path,
    v6_root: Path,
    standard_weights: Path,
    dual_weights: Path,
) -> list[CandidateSpec]:
    v5_dir = v5_dir.expanduser().resolve()
    v6_root = v6_root.expanduser().resolve()
    v5_config = read_json(v5_dir / "run_config_topk_v5.json")
    validate_config(
        v5_config,
        protocol=EXPECTED_V5_PROTOCOL,
        mode=mode,
        expected_seed=4041,
        standard_weights=standard_weights,
        dual_weights=dual_weights,
    )
    v5_rows = read_csv(v5_dir / "attack_results_topk_v5.csv")
    strongest_attack = "SquareAdaptive" if mode == "full" else "SquareRef"
    specs = [
        source_spec(
            source_id="v5_proposed_strongest",
            source_version="V5",
            source_protocol=EXPECTED_V5_PROTOCOL,
            source_dir=v5_dir,
            row=select_unique_row(
                v5_rows, defense="proposed", attack=strongest_attack
            ),
        )
    ]
    for seed in V6_SEEDS:
        seed_dir = v6_root / f"seed_{seed}"
        config = read_json(seed_dir / "run_config_topk_v6.json")
        validate_config(
            config,
            protocol=EXPECTED_V6_PROTOCOL,
            mode=mode,
            expected_seed=seed,
            standard_weights=standard_weights,
            dual_weights=dual_weights,
        )
        rows = read_csv(seed_dir / "attack_results_topk_v6.csv")
        specs.append(
            source_spec(
                source_id=f"v6_proposed_seed_{seed}",
                source_version="V6",
                source_protocol=EXPECTED_V6_PROTOCOL,
                source_dir=seed_dir,
                row=select_unique_row(rows, defense="proposed", attack="SquareRacing"),
            )
        )

    # Deterministic anchors are almost free and verify that the common evaluator
    # and model/reference path are identical across source versions and modes.
    specs.append(
        source_spec(
            source_id="v5_robust_anchor",
            source_version="V5",
            source_protocol=EXPECTED_V5_PROTOCOL,
            source_dir=v5_dir,
            row=select_unique_row(v5_rows, defense="robust", attack="SquareRef"),
        )
    )
    for seed in V6_SEEDS:
        seed_dir = v6_root / f"seed_{seed}"
        rows = read_csv(seed_dir / "attack_results_topk_v6.csv")
        specs.append(
            source_spec(
                source_id=f"v6_robust_seed_{seed}",
                source_version="V6",
                source_protocol=EXPECTED_V6_PROTOCOL,
                source_dir=seed_dir,
                row=select_unique_row(rows, defense="robust", attack="SquareRacing"),
            )
        )
    return specs


def paired_bootstrap(
    delta: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> tuple[float, float, float]:
    delta = np.asarray(delta, dtype=np.float64)
    if delta.ndim != 1 or len(delta) < 2:
        raise ValueError("paired bootstrap requires a one-dimensional sample")
    rng = np.random.default_rng(seed)
    means: list[np.ndarray] = []
    batch = 5000
    for start in range(0, resamples, batch):
        count = min(batch, resamples - start)
        indices = rng.integers(0, len(delta), size=(count, len(delta)))
        means.append(delta[indices].mean(axis=1))
    samples = np.concatenate(means)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(delta.mean()), float(low), float(high)


def classify_attack_difference(low: float, high: float) -> str:
    # Delta is V6 survival minus V5 survival; lower survival means stronger attack.
    if low > 0:
        return "V5_stronger"
    if high < 0:
        return "V6_stronger"
    return "statistically_indistinguishable"


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def csv_text(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return ""
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    import io

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    atomic_write_text(path, csv_text(rows))


def acquire_lock(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / ".candidate_reeval.lock"
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.seek(0)
        owner = handle.read().strip() or "unknown"
        raise SystemExit(f"Candidate re-evaluation is already active (PID {owner}).") from exc
    handle.seek(0)
    handle.truncate()
    handle.write(str(os.getpid()))
    handle.flush()
    return handle


def load_payload(torch: Any, spec: CandidateSpec) -> Mapping[str, Any]:
    try:
        payload = torch.load(spec.candidate_path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(spec.candidate_path, map_location="cpu")
    except Exception as exc:
        print(
            f"weights_only load failed for trusted research artifact {spec.candidate_path}: {exc}",
            flush=True,
        )
        payload = torch.load(spec.candidate_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError(f"Candidate payload is not a mapping: {spec.candidate_path}")
    # V5 predates preservation of the clean tensor in candidate files.  V6
    # stores it, so V5 candidates are paired with a validated canonical V6
    # clean tensor after exact label/index checks.
    required = {"adversarial", "labels", "dataset_indices", "metadata"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Candidate payload misses {sorted(missing)}: {spec.candidate_path}")
    return payload


def validate_payload(
    torch: Any,
    spec: CandidateSpec,
    payload: Mapping[str, Any],
    baseline: Mapping[str, Any] | None,
) -> dict[str, Any]:
    adversarial = payload["adversarial"].float()
    has_stored_clean = "clean" in payload
    if has_stored_clean:
        clean = payload["clean"].float()
    elif baseline is not None:
        clean = baseline["clean"].float()
    else:
        raise ValueError(
            f"Candidate has no clean tensor and no canonical clean source is "
            f"available: {spec.candidate_path}"
        )
    labels = payload["labels"].long()
    indices = payload["dataset_indices"].long()
    metadata = payload["metadata"]
    if tuple(adversarial.shape) != (EXPECTED_EXAMPLES, 3, 32, 32):
        raise ValueError(f"Unexpected adversarial shape in {spec.candidate_path}")
    if adversarial.shape != clean.shape:
        raise ValueError("Adversarial and clean tensors have different shapes")
    if tuple(labels.shape) != (EXPECTED_EXAMPLES,) or tuple(indices.shape) != (
        EXPECTED_EXAMPLES,
    ):
        raise ValueError("Candidate labels/indices have unexpected shapes")
    if not bool(torch.isfinite(adversarial).all()) or not bool(torch.isfinite(clean).all()):
        raise ValueError("Candidate payload contains non-finite pixels")
    if float(adversarial.min()) < -1e-7 or float(adversarial.max()) > 1.0 + 1e-7:
        raise ValueError("Adversarial candidate leaves [0,1]")
    linf = float((adversarial - clean).abs().amax())
    if linf > EXPECTED_EPSILON_PX / 255.0 + 1e-7:
        raise ValueError(f"Linf violation: {linf * 255.0:.8f} pixels")
    if not isinstance(metadata, Mapping):
        raise ValueError("Candidate metadata is not a mapping")
    if str(metadata.get("protocol_version")) != spec.source_protocol:
        raise ValueError("Candidate metadata records the wrong protocol version")
    if int(metadata.get("data_seed", -1)) != EXPECTED_DATA_SEED:
        raise ValueError("Candidate metadata records the wrong data seed")
    for key in ("defense", "attack", "attack_goal"):
        if str(metadata.get(key)) != str(spec.source_row[key]):
            raise ValueError(f"Candidate metadata mismatch for {key}")
    if int(metadata.get("k", -1)) != EXPECTED_K:
        raise ValueError("Candidate metadata records the wrong k")
    if int(metadata.get("attack_api_queries", -1)) != EXPECTED_BUDGET:
        raise ValueError("Candidate metadata records the wrong API query count")
    if baseline is not None:
        if not torch.equal(indices, baseline["dataset_indices"]):
            raise ValueError("Candidate sample indices differ from the common subset")
        if not torch.equal(labels, baseline["labels"]):
            raise ValueError("Candidate labels differ from the common subset")
        if not torch.equal(clean, baseline["clean"]):
            raise ValueError("Candidate clean tensors differ from the common subset")
    return {
        "adversarial": adversarial,
        "clean": clean,
        "labels": labels,
        "dataset_indices": indices,
        "metadata": dict(metadata),
        "linf_px": linf * 255.0,
        "clean_tensor_source": (
            "stored_in_candidate"
            if has_stored_clean
            else f"reused_from_{CANONICAL_CLEAN_SOURCE_ID}"
        ),
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
    k: int,
    batch_size: int,
    requested_draws: int,
    tau: float,
    seed: int,
    source_id: str,
    progress_every_batches: int,
) -> Any:
    evaluator.base.seed_everything(seed)
    draws = requested_draws if defense.sigma > 0 else 1
    bundles = []
    total_batches = (len(images) + batch_size - 1) // batch_size
    for batch_number, low in enumerate(range(0, len(images), batch_size), start=1):
        high = min(low + batch_size, len(images))
        draw_logits = evaluator.collect_actual_draws(
            defense, images[low:high], high - low, draws
        )
        bundles.append(
            evaluator.evaluate_draws(
                draw_logits,
                labels[low:high].detach().cpu(),
                reference_logits[low:high].detach().cpu(),
                k,
                tau,
                True,
            )
        )
        if batch_number % progress_every_batches == 0 or batch_number == total_batches:
            print(
                f"    {source_id}: batch {batch_number}/{total_batches}, "
                f"draws={draws} complete",
                flush=True,
            )
    torch = evaluator.torch
    values = {
        name: torch.cat([bundle.values[name] for bundle in bundles], dim=0)
        for name in bundles[0].values
    }
    mean_values = {
        name: torch.cat([bundle.mean_logits_values[name] for bundle in bundles], dim=0)
        for name in bundles[0].mean_logits_values
    }
    return evaluator.EvaluationBundle(
        summary=summary_from_values(values),
        values=values,
        reference_set=torch.cat([bundle.reference_set for bundle in bundles], dim=0),
        first_draw_set=torch.cat([bundle.first_draw_set for bundle in bundles], dim=0),
        mean_logits_set=torch.cat([bundle.mean_logits_set for bundle in bundles], dim=0),
        mean_logits_summary=summary_from_values(mean_values),
        mean_logits_values=mean_values,
        draws=draws,
    )


def partial_paths(output_dir: Path, source_id: str) -> tuple[Path, Path]:
    partial = output_dir / "partials"
    return partial / f"{source_id}.summary.json", partial / f"{source_id}.per_example.csv"


def read_completed_partial(
    output_dir: Path, source_id: str, expected_rows: int = EXPECTED_EXAMPLES
) -> tuple[dict[str, Any], list[dict[str, str]]] | None:
    summary_path, examples_path = partial_paths(output_dir, source_id)
    if not summary_path.exists() and not examples_path.exists():
        return None
    if not summary_path.is_file() or not examples_path.is_file():
        raise RuntimeError(f"Incomplete partial output for {source_id}; preserve it for audit")
    summary = read_json(summary_path)
    examples = read_csv(examples_path)
    if summary.get("source_id") != source_id or len(examples) != expected_rows:
        raise RuntimeError(f"Invalid partial output for {source_id}")
    if any(row.get("source_id") != source_id for row in examples):
        raise RuntimeError(f"Partial per-example source mismatch for {source_id}")
    return summary, examples


def evaluate_one(
    *,
    evaluator: Any,
    torch: Any,
    spec: CandidateSpec,
    payload: Mapping[str, Any],
    defense: Any,
    reference_logits: Any,
    mode: str,
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidate_hash = sha256_file(spec.candidate_path)
    started = time.perf_counter()
    evaluation = evaluate_with_progress(
        evaluator,
        defense,
        payload["adversarial"].to(torch.device(args.device)),
        payload["labels"].to(torch.device(args.device)),
        reference_logits,
        k=EXPECTED_K,
        batch_size=args.batch_size,
        requested_draws=args.eval_draws,
        tau=args.rank_temperature,
        seed=args.reeval_seed,
        source_id=spec.source_id,
        progress_every_batches=args.progress_every_batches,
    )
    wall_seconds = time.perf_counter() - started
    exact_values = evaluation.values["exact_set"].detach().cpu().numpy()
    mc_se = 100.0 * float(
        np.sqrt(np.sum(exact_values * (1.0 - exact_values)))
        / (len(exact_values) * np.sqrt(max(1, evaluation.draws)))
    )
    summary: dict[str, Any] = {
        "source_id": spec.source_id,
        "mode": mode,
        "source_version": spec.source_version,
        "source_protocol": spec.source_protocol,
        "source_attack": spec.attack,
        "source_defense": spec.defense,
        "source_attack_seed": spec.attack_seed,
        "source_reeval_seed": int(spec.source_row["reeval_seed"]),
        "source_eval_draws": int(spec.source_row["eval_draws"]),
        "source_reported_exact_set": float(spec.source_row["exact_set"]),
        "source_candidate_file": str(spec.source_row["candidate_file"]),
        "source_candidate_sha256": candidate_hash,
        "candidate_linf_px": payload["linf_px"],
        "clean_tensor_source": payload["clean_tensor_source"],
        "data_seed": EXPECTED_DATA_SEED,
        "k": EXPECTED_K,
        "epsilon_px": EXPECTED_EPSILON_PX,
        "original_attack_api_queries": EXPECTED_BUDGET,
        "reeval_seed": args.reeval_seed,
        "reeval_draws_requested": args.eval_draws,
        "reeval_draws_actual": evaluation.draws,
        "reeval_draws_counted_in_attack_budget": False,
        "common_random_numbers": True,
        "examples": EXPECTED_EXAMPLES,
        "batch_size": args.batch_size,
        "wall_seconds": wall_seconds,
        "mc_only_standard_error_pp": mc_se,
    }
    summary.update(evaluation.summary)
    for name, value in evaluation.mean_logits_summary.items():
        summary[f"mean_logits_{name}"] = value
    rows = evaluator.per_example_rows(
        evaluation=evaluation,
        labels=payload["labels"],
        dataset_indices=payload["dataset_indices"],
        defense=spec.defense,
        condition="saved_candidate_reeval",
        k=EXPECTED_K,
        data_seed=EXPECTED_DATA_SEED,
        attack_seed=spec.attack_seed,
        reeval_seed=args.reeval_seed,
        run_tag=f"common_mc{args.eval_draws}_{mode}",
        shard_count=1,
        shard_index=0,
        attack=spec.attack,
        attack_goal="exact_set",
        epsilon_px=EXPECTED_EPSILON_PX,
        requested_budget=EXPECTED_BUDGET,
        logical_budget="source_attack",
        attack_eot="source_attack",
        attack_api_queries=EXPECTED_BUDGET,
        candidate_file=str(spec.source_row["candidate_file"]),
    )
    enriched = [
        {
            "source_id": spec.source_id,
            "mode": mode,
            "source_version": spec.source_version,
            "source_protocol": spec.source_protocol,
            "source_attack": spec.attack,
            "source_defense": spec.defense,
            "source_attack_seed": spec.attack_seed,
            "source_candidate_sha256": candidate_hash,
            "common_random_numbers": True,
            "reeval_draws_counted_in_attack_budget": False,
            **row,
        }
        for row in rows
    ]
    summary_path, examples_path = partial_paths(output_dir, spec.source_id)
    atomic_write_text(summary_path, json.dumps(summary, indent=2, sort_keys=True))
    atomic_write_csv(examples_path, enriched)
    return summary, enriched


def build_comparisons(
    summaries: Sequence[Mapping[str, Any]],
    examples: Sequence[Mapping[str, str]],
    *,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    summary_by_id = {str(row["source_id"]): row for row in summaries}
    by_source: dict[str, dict[str, float]] = {}
    for row in examples:
        if row["source_defense"] != "proposed":
            continue
        by_source.setdefault(row["source_id"], {})[row["sample_id"]] = float(
            row["exact_set_probability"]
        )
    baseline_id = "v5_proposed_strongest"
    baseline = by_source[baseline_id]
    sample_ids = sorted(baseline)
    comparisons: list[dict[str, Any]] = []
    v6_arrays = []
    for offset, seed in enumerate(V6_SEEDS):
        source_id = f"v6_proposed_seed_{seed}"
        current = by_source[source_id]
        if sorted(current) != sample_ids:
            raise RuntimeError(f"Sample IDs differ for {source_id}")
        delta = np.array([current[item] - baseline[item] for item in sample_ids])
        mean, low, high = paired_bootstrap(
            delta,
            resamples=bootstrap_resamples,
            seed=bootstrap_seed + offset,
        )
        comparisons.append(
            {
                "comparison": f"{source_id}_minus_{baseline_id}",
                "v6_source_id": source_id,
                "attack_seed": seed,
                "v5_exact_set_pct": float(summary_by_id[baseline_id]["exact_set"]),
                "v6_exact_set_pct": float(summary_by_id[source_id]["exact_set"]),
                "v6_minus_v5_pp": 100.0 * mean,
                "ci95_low_pp": 100.0 * low,
                "ci95_high_pp": 100.0 * high,
                "positive_delta_means": "V6 attack is weaker (higher survival)",
                "verdict": classify_attack_difference(low, high),
                "inference_scope": (
                    "paired bootstrap over the fixed 100 images; attack seeds "
                    "are treated as observed runs"
                ),
                "bootstrap_resamples": bootstrap_resamples,
                "bootstrap_seed": bootstrap_seed + offset,
            }
        )
        v6_arrays.append(np.array([current[item] for item in sample_ids]))
    v6_mean = np.stack(v6_arrays).mean(axis=0)
    baseline_array = np.array([baseline[item] for item in sample_ids])
    mean, low, high = paired_bootstrap(
        v6_mean - baseline_array,
        resamples=bootstrap_resamples,
        seed=bootstrap_seed + 100,
    )
    comparisons.append(
        {
            "comparison": "v6_three_seed_mean_minus_v5_proposed_strongest",
            "v6_source_id": "three_seed_mean",
            "attack_seed": "4041,5051,6061",
            "v5_exact_set_pct": float(summary_by_id[baseline_id]["exact_set"]),
            "v6_exact_set_pct": 100.0 * float(v6_mean.mean()),
            "v6_seed_sd_pp": float(
                np.std(
                    [summary_by_id[f"v6_proposed_seed_{seed}"]["exact_set"] for seed in V6_SEEDS],
                    ddof=1,
                )
            ),
            "v6_minus_v5_pp": 100.0 * mean,
            "ci95_low_pp": 100.0 * low,
            "ci95_high_pp": 100.0 * high,
            "positive_delta_means": "V6 attack is weaker (higher survival)",
            "verdict": classify_attack_difference(low, high),
            "inference_scope": (
                "paired bootstrap over the fixed 100 images after averaging the "
                "three observed V6 seeds; V5 has one observed attack seed"
            ),
            "v6_seed_count": len(V6_SEEDS),
            "v5_seed_count": 1,
            "bootstrap_resamples": bootstrap_resamples,
            "bootstrap_seed": bootstrap_seed + 100,
        }
    )
    return comparisons


def normalize_config(args: argparse.Namespace, specs: Sequence[CandidateSpec]) -> dict[str, Any]:
    return {
        "protocol_version": REEVAL_PROTOCOL,
        "mode": args.mode,
        "v5_dir": str(args.v5_dir.expanduser().resolve()),
        "v6_root": str(args.v6_root.expanduser().resolve()),
        "output_dir": str(args.output_dir.expanduser().resolve()),
        "standard_weights": str(args.standard_weights.expanduser().resolve()),
        "dual_weights": str(args.dual_weights.expanduser().resolve()),
        "device": args.device,
        "batch_size": args.batch_size,
        "eval_draws": args.eval_draws,
        "reeval_seed": args.reeval_seed,
        "bootstrap_resamples": args.bootstrap_resamples,
        "bootstrap_seed": args.bootstrap_seed,
        "gpu_memory_fraction": args.gpu_memory_fraction,
        "rank_temperature": args.rank_temperature,
        "data_seed": EXPECTED_DATA_SEED,
        "k": EXPECTED_K,
        "epsilon_px": EXPECTED_EPSILON_PX,
        "source_attack_budget": EXPECTED_BUDGET,
        "candidate_source_ids": [spec.source_id for spec in specs],
        "canonical_clean_source_id": CANONICAL_CLEAN_SOURCE_ID,
        "common_random_numbers": True,
        "reeval_draws_counted_in_attack_budget": False,
        "script_sha256": sha256_file(Path(__file__).resolve()),
    }


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size < 1 or args.eval_draws < 2:
        raise ValueError("Batch size must be positive and eval_draws must be at least 2")
    if args.bootstrap_resamples < 1000:
        raise ValueError("Use at least 1000 bootstrap resamples")
    if args.progress_every_batches < 1:
        raise ValueError("progress_every_batches must be positive")
    if not 0 < args.gpu_memory_fraction <= 1:
        raise ValueError("gpu_memory_fraction must be in (0,1]")
    required = [
        args.v5_dir / "run_config_topk_v5.json",
        args.v5_dir / "attack_results_topk_v5.csv",
        args.standard_weights,
        args.dual_weights,
        PROJECT_DIR / "evaluate_general_topk_defense_v6_cifar10.py",
        PROJECT_DIR / "evaluate_final_top1_set2_defense_cifar10.py",
        PROJECT_DIR / "evaluate_top1_set2_gated_cifar10.py",
    ]
    for seed in V6_SEEDS:
        required.extend(
            [
                args.v6_root / f"seed_{seed}/run_config_topk_v6.json",
                args.v6_root / f"seed_{seed}/attack_results_topk_v6.csv",
            ]
        )
    missing = [str(path) for path in required if not path.expanduser().is_file()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n  " + "\n  ".join(missing))


def main() -> None:
    args = parse_args()
    validate_args(args)
    output_dir = args.output_dir.expanduser().resolve()
    lock = acquire_lock(output_dir)
    try:
        specs = build_candidate_specs(
            args.mode,
            args.v5_dir,
            args.v6_root,
            args.standard_weights,
            args.dual_weights,
        )
        # Lazy imports let the pure-data self-test run outside the GPU environment.
        import torch

        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no GPU is available")
        sys.path.insert(0, str(PROJECT_DIR))
        import evaluate_general_topk_defense_v6_cifar10 as evaluator

        device = torch.device(args.device)
        if device.type == "cuda":
            torch.cuda.set_device(device)
            index = device.index if device.index is not None else 0
            torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, index)
            torch.cuda.reset_peak_memory_stats(device)

        # V5 candidate payloads do not store clean images.  Establish the
        # canonical clean/label/index tensors from the first V6 proposed run,
        # then validate every V5/V6 payload against that exact subset.
        raw_payloads = {spec.source_id: load_payload(torch, spec) for spec in specs}
        spec_by_id = {spec.source_id: spec for spec in specs}
        canonical_spec = spec_by_id[CANONICAL_CLEAN_SOURCE_ID]
        canonical_raw = raw_payloads[CANONICAL_CLEAN_SOURCE_ID]
        if "clean" not in canonical_raw:
            raise ValueError(
                f"Canonical V6 payload has no clean tensor: "
                f"{canonical_spec.candidate_path}"
            )
        canonical_checked = validate_payload(
            torch, canonical_spec, canonical_raw, baseline=None
        )
        baseline = {
            "clean": canonical_checked["clean"],
            "labels": canonical_checked["labels"],
            "dataset_indices": canonical_checked["dataset_indices"],
        }
        payloads: dict[str, dict[str, Any]] = {}
        for spec in specs:
            if spec.source_id == CANONICAL_CLEAN_SOURCE_ID:
                checked = canonical_checked
            else:
                checked = validate_payload(
                    torch, spec, raw_payloads[spec.source_id], baseline
                )
            payloads[spec.source_id] = checked

        # Write provenance only after every source tensor has passed validation.
        config = normalize_config(args, specs)
        config_path = output_dir / "run_config_candidate_reeval.json"
        if config_path.is_file():
            existing = read_json(config_path)
            if existing != config:
                raise RuntimeError(
                    "Output directory contains a different re-evaluation config; "
                    "use a fresh directory."
                )
        else:
            atomic_write_text(config_path, json.dumps(config, indent=2, sort_keys=True))

        standard_model, standard_arch = evaluator.legacy.load_resnet20(
            args.standard_weights, device, evaluator.legacy.STANDARD_CIFAR10_STD
        )
        dual_model = evaluator.base.load_dual_defense(args.dual_weights, device)
        robust_model = evaluator.base.RobustPixelModel(dual_model).to(device).eval()
        standard_spec = evaluator.base.DefenseSpec(
            "standard", evaluator.base.make_actual(standard_model), standard_model, 0.0
        )
        clean_device = baseline["clean"].to(device)
        reference_logits = evaluator.base.compute_logits_batched(
            standard_spec, clean_device, args.batch_size, 1
        ).detach().cpu()
        strength_steps = 2 if args.mode == "full" else 0
        proposed_actual = evaluator.TopKTargetedGaussianDefense(
            standard_model=standard_model,
            gradient_model=robust_model,
            output_model=robust_model,
            top_k=EXPECTED_K,
            alpha=0.005,
            epsilon=0.03,
            steps=strength_steps,
            sigma=EXPECTED_SIGMA,
            top1_weight=1.0,
            set_weight=4.0,
        )
        defenses = {
            "proposed": evaluator.base.DefenseSpec(
                "proposed", proposed_actual, robust_model, EXPECTED_SIGMA
            ),
            "robust": evaluator.base.DefenseSpec(
                "robust", evaluator.base.make_actual(robust_model), robust_model, 0.0
            ),
        }
        print(f"Standard checkpoint architecture: {standard_arch}", flush=True)
        print(
            f"Common re-evaluation: mode={args.mode}, MC={args.eval_draws}, "
            f"seed={args.reeval_seed}, candidates={len(specs)}",
            flush=True,
        )

        summaries: list[dict[str, Any]] = []
        all_examples: list[dict[str, Any]] = []
        for position, spec in enumerate(specs, start=1):
            completed = read_completed_partial(output_dir, spec.source_id)
            if completed is not None:
                summary, examples = completed
                expected_partial = {
                    "mode": args.mode,
                    "source_version": spec.source_version,
                    "source_protocol": spec.source_protocol,
                    "source_attack": spec.attack,
                    "source_defense": spec.defense,
                    "source_attack_seed": spec.attack_seed,
                    "source_candidate_sha256": sha256_file(spec.candidate_path),
                    "clean_tensor_source": payloads[spec.source_id][
                        "clean_tensor_source"
                    ],
                    "reeval_seed": args.reeval_seed,
                    "reeval_draws_requested": args.eval_draws,
                    "batch_size": args.batch_size,
                }
                mismatches = {
                    key: (summary.get(key), value)
                    for key, value in expected_partial.items()
                    if summary.get(key) != value
                }
                if mismatches:
                    raise RuntimeError(
                        f"Partial provenance mismatch for {spec.source_id}: "
                        f"{mismatches}. Use a fresh output directory."
                    )
                summaries.append(summary)
                all_examples.extend(examples)
                print(f"PASS reusing completed partial {spec.source_id}", flush=True)
                continue
            print(
                f"[{position}/{len(specs)}] Evaluating {spec.source_id}: "
                f"{spec.attack}, defense={spec.defense}",
                flush=True,
            )
            summary, examples = evaluate_one(
                evaluator=evaluator,
                torch=torch,
                spec=spec,
                payload=payloads[spec.source_id],
                defense=defenses[spec.defense],
                reference_logits=reference_logits,
                mode=args.mode,
                output_dir=output_dir,
                args=args,
            )
            summaries.append(summary)
            all_examples.extend(examples)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(
                f"PASS {spec.source_id}: Exact-set={summary['exact_set']:.4f}%",
                flush=True,
            )

        order = {spec.source_id: index for index, spec in enumerate(specs)}
        summaries.sort(key=lambda row: order[str(row["source_id"])])
        all_examples.sort(
            key=lambda row: (order[str(row["source_id"])], int(row["position"]))
        )
        comparisons = build_comparisons(
            summaries,
            all_examples,
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_seed=args.bootstrap_seed,
        )
        atomic_write_csv(output_dir / "candidate_reeval_summary.csv", summaries)
        atomic_write_csv(output_dir / "candidate_reeval_per_example.csv", all_examples)
        atomic_write_csv(output_dir / "candidate_reeval_comparisons.csv", comparisons)
        primary = comparisons[-1]
        final_summary = {
            "protocol_version": config["protocol_version"],
            "mode": args.mode,
            "candidate_count": len(summaries),
            "per_example_rows": len(all_examples),
            "common_reeval_seed": args.reeval_seed,
            "requested_mc_draws": args.eval_draws,
            "comparison": primary,
            "interpretation_rule": (
                "Positive V6-minus-V5 Exact-set means the V6 candidate is weaker "
                "because it leaves higher survival."
            ),
            "source_candidate_sha256": {
                row["source_id"]: row["source_candidate_sha256"] for row in summaries
            },
        }
        atomic_write_text(
            output_dir / "candidate_reeval_summary.json",
            json.dumps(final_summary, indent=2, sort_keys=True),
        )
        print("\nCommon-MC candidate comparison", flush=True)
        for row in comparisons:
            print(
                f"{row['comparison']}: delta={float(row['v6_minus_v5_pp']):+.4f} pp "
                f"[{float(row['ci95_low_pp']):+.4f}, "
                f"{float(row['ci95_high_pp']):+.4f}] -> {row['verdict']}",
                flush=True,
            )
        print(f"Candidate re-evaluation complete: {output_dir}", flush=True)
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    main()
