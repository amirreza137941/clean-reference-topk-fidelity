#!/usr/bin/env python3
"""SaTML-oriented v4 evaluator for clean-reference Top-k fidelity on CIFAR-10.

The v4 protocol fixes four paper-critical issues in the earlier evaluator:

1. Black-box budgets count *returned API responses per example*.  EOT=m costs
   m queries.  Deterministic defenses automatically use EOT=1 so that they can
   spend the same budget on distinct candidates instead of repeated logits.
2. Stochastic fidelity is reported as a per-request probability over independent
   defense draws.  Metrics computed from mean logits are secondary diagnostics.
3. Adversarial candidates are re-evaluated with independent randomness and can
   be saved for later high-MC auditing.
4. CIFAR test indices, seeds, query counts, wall time, and shard identifiers are
   stored explicitly.  Two processes may safely split attack curves by using
   different output directories and ``--shard-count 2 --shard-index 0/1``.

Square, SignHunt, and NES in this file retain the project's local attack
implementations/semantics, but the curve runners below provide nested budgets
and exact API-query accounting.  They must be described as local Top-k
adaptations unless replaced by an official/reference implementation.

Scientifically important compatibility rule
---------------------------------------------
The existing ``fine_frontier_top1_set2`` dual checkpoint was trained for
exact Top-2 only.  Therefore ``dual`` and ``selector`` are valid paper
baselines only for k=2.  The script rejects their use for other k values unless
``--allow-legacy-dual-non2`` is explicitly supplied.  Such non-2 results are
diagnostic only and must not be presented as a trained general Top-k method.

Keep this script beside the original project dependencies:

* evaluate_final_top1_set2_defense_cifar10.py
* evaluate_top1_set2_gated_cifar10.py
* train_resnet20_dual_backbone_top1_set2_cifar10.py

The first two files are imported for the validated checkpoint loaders, CIFAR
sampling code, model wrappers, and black-box attack implementations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

try:
    import evaluate_final_top1_set2_defense_cifar10 as base
    import evaluate_top1_set2_gated_cifar10 as legacy
except ImportError as exc:
    raise SystemExit(
        "Place this script beside evaluate_final_top1_set2_defense_cifar10.py "
        "and evaluate_top1_set2_gated_cifar10.py (plus their dependencies)."
    ) from exc


METRIC_NAMES = (
    "top1",
    "coverage",
    "exact_set",
    "overlap",
    "jaccard",
    "weighted_membership",
    "rank_overlap",
    "margin",
    "hidden_topk_failure",
)
ATTACK_GOALS = ("top1", "coverage", "exact_set", "overlap", "joint")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "General Top-k clean and adaptive evaluation of the CIFAR-10 "
            "robust candidate-set defense"
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./datasets/cifar-10-batches-py"),
    )
    parser.add_argument(
        "--standard-weights",
        type=Path,
        default=Path(
            "./models/resnet20/cifar10/"
            "resnet20_cifar10.pth"
        ),
    )
    parser.add_argument(
        "--dual-weights",
        type=Path,
        default=Path(
            "./models/resnet20/"
            "fine_frontier_top1_set2/resnet20_staged_epsilon_final_weights.pt"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "./models/resnet20/"
            "paper_general_topk_v4_cifar10"
        ),
    )

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Attack/randomization seed. Does not choose the test subset.",
    )
    parser.add_argument(
        "--data-seed",
        type=int,
        default=3031,
        help="Seed used only for the stratified CIFAR-10 test subset.",
    )
    parser.add_argument(
        "--reeval-seed-base",
        type=int,
        default=9000000,
        help="Independent seed base for final candidate re-evaluation.",
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default="",
        help="Optional human-readable process/run identifier stored in every row.",
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help="Deterministically split attack curves across this many processes.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based attack-curve shard assigned to this process.",
    )
    parser.add_argument(
        "--gpu-memory-fraction",
        type=float,
        default=None,
        help=(
            "Optional CUDA allocator limit for this process. Use about 0.45 "
            "when two jobs share a 24-GB RTX 3090."
        ),
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--clean-examples", type=int, default=10000)
    parser.add_argument("--attack-examples", type=int, default=300)
    parser.add_argument(
        "--top-k",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="Candidate-set sizes. CIFAR-10 requires every k to be in [1,9].",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=METRIC_NAMES,
        default=list(METRIC_NAMES),
    )

    parser.add_argument(
        "--defenses",
        nargs="+",
        choices=["standard", "robust", "dual", "proposed", "selector"],
        default=["standard", "robust", "proposed"],
    )
    parser.add_argument(
        "--allow-legacy-dual-non2",
        action="store_true",
        help=(
            "Allow the Top-2-trained dual/selector checkpoint at k!=2. "
            "Diagnostic only; not a valid general Top-k paper result."
        ),
    )

    parser.add_argument(
        "--attacks",
        nargs="+",
        choices=["PGD", "Square", "SignHunt", "NES"],
        default=["PGD"],
    )
    parser.add_argument(
        "--attack-goals",
        nargs="+",
        choices=ATTACK_GOALS,
        default=["exact_set"],
    )
    parser.add_argument(
        "--attack-epsilons-px",
        type=float,
        nargs="+",
        default=[1.0],
        help="L-infinity radii in pixel units; every value is divided by 255.",
    )
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=[100, 500, 1000],
        help="Black-box budgets in returned API queries per example.",
    )
    parser.add_argument(
        "--budget-unit",
        choices=["api_queries", "logical_queries"],
        default="api_queries",
        help=(
            "api_queries is the paper protocol: EOT=m costs m queries. "
            "logical_queries is retained only for backward-compatible diagnostics."
        ),
    )
    parser.add_argument(
        "--attack-eot",
        type=int,
        default=5,
        help="EOT allocation for stochastic defenses; deterministic defenses use 1.",
    )
    parser.add_argument(
        "--force-deterministic-eot",
        action="store_true",
        help="Diagnostic only: also spend repeated EOT queries on deterministic defenses.",
    )
    parser.add_argument(
        "--eval-mc",
        "--final-eval-draws",
        dest="eval_draws",
        type=int,
        default=50,
        help="Independent defense draws used to estimate per-request fidelity.",
    )
    parser.add_argument(
        "--clean-eval-draws",
        type=int,
        default=10,
        help=(
            "Defense draws per clean example. Aggregate clean estimates usually "
            "need fewer draws than final adversarial-candidate re-evaluation."
        ),
    )
    parser.add_argument(
        "--report-mean-logits",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also report metrics of MC-mean logits as a secondary diagnostic.",
    )
    parser.add_argument(
        "--save-candidates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save adversarial tensors and original CIFAR indices for re-auditing.",
    )
    parser.add_argument(
        "--candidate-dtype",
        choices=["float16", "float32"],
        default="float16",
    )
    parser.add_argument(
        "--clean-alignment-threshold",
        type=float,
        default=0.5,
        help="Clean exact-set probability threshold for conditional retention.",
    )
    parser.add_argument(
        "--nes-samples",
        type=int,
        default=20,
        help="Even number of NES queries per gradient estimate before EOT expansion.",
    )
    parser.add_argument("--pgd-steps", type=int, default=20)
    parser.add_argument("--pgd-restarts", type=int, default=1)
    parser.add_argument("--pgd-alpha-cap-px", type=float, default=2.0)
    parser.add_argument("--joint-top1-weight", type=float, default=1.0)
    parser.add_argument("--joint-set-weight", type=float, default=1.0)

    # Generalized version of the already validated reinforcing path.
    parser.add_argument("--strength-alpha", type=float, default=0.005)
    parser.add_argument("--strength-epsilon", type=float, default=0.03)
    parser.add_argument(
        "--strength-steps",
        type=int,
        default=2,
        help="Targeted-strengthening steps; set to 0 for a noise-only ablation.",
    )
    parser.add_argument("--top1-weight", type=float, default=1.0)
    parser.add_argument("--set-weight", type=float, default=4.0)
    parser.add_argument("--fixed-sigma", type=float, default=0.02144277648925781)
    parser.add_argument(
        "--rank-temperature",
        type=float,
        default=0.5,
        help="Exponential rank decay used only by the rank-overlap metric.",
    )

    # Frozen legacy selector parameters.  No test-set calibration occurs here.
    parser.add_argument("--selector-threshold", type=float, default=1.0 / 3.0)
    parser.add_argument("--selector-gate-views", type=int, default=3)
    parser.add_argument("--selector-view-sigma", type=float, default=0.015)

    parser.add_argument("--save-per-example", action="store_true")
    parser.add_argument("--clean-only", action="store_true")
    parser.add_argument(
        "--reuse-clean-dir",
        type=Path,
        default=None,
        help=(
            "Reuse clean_results_topk_v4.csv and clean_per_example_topk_v4.csv "
            "from a completed clean-only run. This avoids duplicating clean MC "
            "evaluation across concurrent attack shards."
        ),
    )
    parser.add_argument("--resume-results", action="store_true")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Small smoke test. Never report quick results in a paper.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no GPU is available")
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("Batch size must be positive and workers non-negative")
    for name in ("clean_examples", "attack_examples"):
        value = int(getattr(args, name))
        if value <= 0 or value > 10000 or value % 10 != 0:
            raise ValueError(f"--{name.replace('_', '-')} must be divisible by 10 and <=10000")
    if not args.top_k or len(set(args.top_k)) != len(args.top_k):
        raise ValueError("--top-k must contain unique values")
    if any(k < 1 or k >= 10 for k in args.top_k):
        raise ValueError("Every CIFAR-10 k must be in [1,9]")
    if any(value <= 0 or value > 255 for value in args.attack_epsilons_px):
        raise ValueError("Every attack epsilon must be in (0,255]")
    if any(value < 2 for value in args.budgets):
        raise ValueError("Every black-box budget must be at least 2")
    if args.attack_eot < 1 or args.eval_draws < 1 or args.clean_eval_draws < 1:
        raise ValueError("EOT and MC counts must be positive")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("Require shard_count>=1 and 0<=shard_index<shard_count")
    if args.gpu_memory_fraction is not None and not 0 < args.gpu_memory_fraction <= 1:
        raise ValueError("--gpu-memory-fraction must be in (0,1]")
    if not 0 <= args.clean_alignment_threshold <= 1:
        raise ValueError("--clean-alignment-threshold must be in [0,1]")
    if args.nes_samples < 2 or args.nes_samples % 2:
        raise ValueError("--nes-samples must be an even integer >=2")
    if args.pgd_steps < 1 or args.pgd_restarts < 1 or args.pgd_alpha_cap_px <= 0:
        raise ValueError("PGD steps, restarts, and alpha cap must be positive")
    if args.strength_alpha <= 0 or args.strength_epsilon <= 0:
        raise ValueError("Strength alpha and epsilon must be positive")
    if args.strength_steps < 0:
        raise ValueError("Strength steps must be non-negative")
    if args.top1_weight < 0 or args.set_weight < 0:
        raise ValueError("Defense objective weights must be non-negative")
    if args.joint_top1_weight < 0 or args.joint_set_weight < 0:
        raise ValueError("Joint attack weights must be non-negative")
    if args.fixed_sigma < 0 or args.selector_view_sigma < 0:
        raise ValueError("Noise scales cannot be negative")
    if args.selector_gate_views < 1:
        raise ValueError("Selector gate views must be positive")
    if not 0 <= args.selector_threshold <= 1.000001:
        raise ValueError("Selector threshold must be in [0,1.000001]")
    legacy_requested = any(name in {"dual", "selector"} for name in args.defenses)
    if legacy_requested and any(k != 2 for k in args.top_k):
        if not args.allow_legacy_dual_non2:
            raise ValueError(
                "The available dual/selector checkpoint was trained only for k=2. "
                "Use only --top-k 2 with these defenses, or explicitly add "
                "--allow-legacy-dual-non2 for a diagnostic (non-paper) run."
            )


def stable_int(text: str, modulus: int = 2**31 - 1) -> int:
    """Stable cross-process integer; unlike Python hash(), survives restarts."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % modulus


def assigned_to_shard(key: Sequence[object], shard_count: int, shard_index: int) -> bool:
    token = "|".join(str(value) for value in key)
    return stable_int(token, shard_count) == shard_index


def load_balanced_subset_with_indices(
    data_dir: Path,
    n_examples: int,
    num_workers: int,
    device: torch.device,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reproduce the legacy stratified subset and retain original test indices."""
    if n_examples <= 0 or n_examples > 10000 or n_examples % 10 != 0:
        raise ValueError("n_examples must be divisible by 10 and in [10,10000]")
    root = legacy.resolve_torchvision_root(data_dir)
    dataset = torchvision.datasets.CIFAR10(
        root=str(root), train=False, download=False, transform=transforms.ToTensor()
    )
    rng = np.random.default_rng(seed)
    targets = np.asarray(dataset.targets)
    per_class = n_examples // 10
    indices: List[int] = []
    for label in range(10):
        class_indices = np.flatnonzero(targets == label)
        rng.shuffle(class_indices)
        indices.extend(class_indices[:per_class].tolist())
    rng.shuffle(indices)
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=n_examples,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    images, labels = next(iter(loader))
    return (
        images.to(device),
        labels.to(device),
        torch.as_tensor(indices, dtype=torch.long),
    )


def ranked_topk(logits: torch.Tensor, k: int) -> torch.Tensor:
    return logits.topk(k, dim=1).indices


def sorted_set(indices: torch.Tensor) -> torch.Tensor:
    return indices.sort(dim=1).values


def set_membership(reference: torch.Tensor, predicted: torch.Tensor) -> torch.Tensor:
    """Return [batch,k] membership of each ranked reference item in prediction."""
    return reference[:, :, None].eq(predicted[:, None, :]).any(dim=2)


def boundary_margin(logits: torch.Tensor, k: int) -> torch.Tensor:
    values = logits.topk(k + 1, dim=1).values
    return values[:, k - 1] - values[:, k]


def rank_weights(k: int, tau: float, device: torch.device) -> torch.Tensor:
    positions = torch.arange(k, device=device, dtype=torch.float32)
    weights = torch.exp(-float(tau) * positions)
    return weights / weights.sum()


@dataclass
class MetricBatch:
    summary: Dict[str, float]
    values: Dict[str, torch.Tensor]
    reference_set: torch.Tensor
    predicted_set: torch.Tensor


def compute_metric_batch(
    logits: torch.Tensor,
    labels: torch.Tensor,
    reference_logits: torch.Tensor,
    k: int,
    tau: float,
) -> MetricBatch:
    logits = logits.detach().cpu()
    labels = labels.detach().cpu()
    reference_logits = reference_logits.detach().cpu()
    reference = ranked_topk(reference_logits, k)
    predicted = ranked_topk(logits, k)
    membership = set_membership(reference, predicted)
    intersection = membership.float().sum(dim=1)
    exact = sorted_set(reference).eq(sorted_set(predicted)).all(dim=1).float()
    coverage = predicted.eq(labels[:, None]).any(dim=1).float()
    top1 = logits.argmax(dim=1).eq(labels).float()
    overlap = intersection / float(k)
    jaccard = intersection / (2.0 * float(k) - intersection).clamp_min(1.0)
    weights = rank_weights(k, tau, logits.device)
    rank_overlap_value = (membership.float() * weights[None, :]).sum(dim=1)
    margin = boundary_margin(logits, k)
    reference_margin = boundary_margin(reference_logits, k)
    reference_top1_match = logits.argmax(dim=1).eq(reference[:, 0]).float()
    hidden_topk_failure = reference_top1_match * (1.0 - exact)
    values = {
        "top1": top1,
        "coverage": coverage,
        "exact_set": exact,
        "overlap": overlap,
        "jaccard": jaccard,
        "weighted_membership": rank_overlap_value,
        # Deprecated compatibility alias. This is weighted set membership,
        # not preservation of the predicted rank order.
        "rank_overlap": rank_overlap_value,
        "margin": margin,
        "reference_margin": reference_margin,
        "reference_top1_match": reference_top1_match,
        "hidden_topk_failure": hidden_topk_failure,
    }
    summary = {
        "top1": 100.0 * float(top1.mean().item()),
        "coverage": 100.0 * float(coverage.mean().item()),
        "exact_set": 100.0 * float(exact.mean().item()),
        "overlap": 100.0 * float(overlap.mean().item()),
        "jaccard": 100.0 * float(jaccard.mean().item()),
        "weighted_membership": 100.0 * float(rank_overlap_value.mean().item()),
        "rank_overlap": 100.0 * float(rank_overlap_value.mean().item()),
        "margin": float(margin.mean().item()),
        "reference_margin": float(reference_margin.mean().item()),
        "reference_top1_match": 100.0 * float(reference_top1_match.mean().item()),
        "hidden_topk_failure": 100.0 * float(hidden_topk_failure.mean().item()),
    }
    return MetricBatch(summary, values, reference, predicted)


@dataclass
class EvaluationBundle:
    """Primary per-draw probabilities plus secondary mean-logit diagnostics."""

    summary: Dict[str, float]
    values: Dict[str, torch.Tensor]
    reference_set: torch.Tensor
    first_draw_set: torch.Tensor
    mean_logits_set: torch.Tensor
    mean_logits_summary: Dict[str, float]
    mean_logits_values: Dict[str, torch.Tensor]
    draws: int


def collect_actual_draws(
    defense: base.DefenseSpec,
    images: torch.Tensor,
    batch_size: int,
    draws: int,
) -> torch.Tensor:
    """Return CPU logits shaped [draw,batch,class] without averaging draws."""
    by_draw: List[List[torch.Tensor]] = [[] for _ in range(draws)]
    for low in range(0, len(images), batch_size):
        batch = images[low : low + batch_size]
        for draw in range(draws):
            by_draw[draw].append(defense.actual(batch).detach().cpu())
    return torch.stack(
        [torch.cat(parts, dim=0) for parts in by_draw], dim=0
    )


def evaluate_draws(
    draw_logits: torch.Tensor,
    labels: torch.Tensor,
    reference_logits: torch.Tensor,
    k: int,
    tau: float,
    report_mean_logits: bool,
) -> EvaluationBundle:
    if draw_logits.ndim != 3 or draw_logits.shape[0] < 1:
        raise ValueError("draw_logits must have shape [draw,batch,class]")
    draw_batches = [
        compute_metric_batch(logits, labels, reference_logits, k, tau)
        for logits in draw_logits
    ]
    value_names = list(draw_batches[0].values)
    values = {
        name: torch.stack([batch.values[name] for batch in draw_batches], dim=0).mean(0)
        for name in value_names
    }
    percentage_names = {
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
    summary = {
        name: (100.0 if name in percentage_names else 1.0)
        * float(tensor.mean().item())
        for name, tensor in values.items()
    }
    mean_logits = draw_logits.mean(dim=0)
    mean_batch = compute_metric_batch(
        mean_logits, labels, reference_logits, k, tau
    )
    mean_summary = mean_batch.summary if report_mean_logits else {}
    return EvaluationBundle(
        summary=summary,
        values=values,
        reference_set=draw_batches[0].reference_set,
        first_draw_set=draw_batches[0].predicted_set,
        mean_logits_set=mean_batch.predicted_set,
        mean_logits_summary=mean_summary,
        mean_logits_values=(mean_batch.values if report_mean_logits else {}),
        draws=int(draw_logits.shape[0]),
    )


def serialize_set(row: torch.Tensor) -> str:
    return "[" + ",".join(str(int(value)) for value in row.tolist()) + "]"


def per_example_rows(
    evaluation: EvaluationBundle,
    labels: torch.Tensor,
    dataset_indices: torch.Tensor,
    defense: str,
    condition: str,
    k: int,
    data_seed: int,
    attack_seed: int,
    reeval_seed: int,
    run_tag: str,
    shard_count: int,
    shard_index: int,
    attack: str = "",
    attack_goal: str = "",
    epsilon_px: object = "",
    requested_budget: object = "",
    logical_budget: object = "",
    attack_eot: object = "",
    attack_api_queries: object = "",
    candidate_file: str = "",
    clean_exact_probability: Optional[torch.Tensor] = None,
    clean_alignment_threshold: float = 0.5,
) -> List[Dict[str, object]]:
    labels_cpu = labels.detach().cpu()
    indices_cpu = dataset_indices.detach().cpu()
    if len(indices_cpu) != len(labels_cpu):
        raise ValueError("dataset_indices and labels must have equal length")
    if clean_exact_probability is not None:
        clean_exact_probability = clean_exact_probability.detach().cpu()
    rows: List[Dict[str, object]] = []
    for position in range(len(labels_cpu)):
        dataset_index = int(indices_cpu[position].item())
        clean_probability: object = ""
        clean_aligned: object = ""
        if clean_exact_probability is not None:
            clean_probability = float(clean_exact_probability[position].item())
            clean_aligned = int(clean_probability >= clean_alignment_threshold)
        mean_values = evaluation.mean_logits_values
        rows.append({
            "position": position,
            "sample_index": dataset_index,
            "sample_id": f"cifar10:test:index={dataset_index}",
            "data_seed": data_seed,
            "attack_seed": attack_seed,
            "reeval_seed": reeval_seed,
            "run_tag": run_tag,
            "shard_count": shard_count,
            "shard_index": shard_index,
            "condition": condition,
            "defense": defense,
            "attack": attack,
            "attack_goal": attack_goal,
            "k": k,
            "epsilon_px": epsilon_px,
            "requested_budget": requested_budget,
            "logical_budget": logical_budget,
            "attack_eot": attack_eot,
            "attack_api_queries": attack_api_queries,
            "eval_draws": evaluation.draws,
            "candidate_file": candidate_file,
            "true_label": int(labels_cpu[position].item()),
            "reference_set": serialize_set(evaluation.reference_set[position]),
            "first_draw_set": serialize_set(evaluation.first_draw_set[position]),
            "mean_logits_set": serialize_set(evaluation.mean_logits_set[position]),
            "top1_probability": float(evaluation.values["top1"][position].item()),
            "coverage_probability": float(
                evaluation.values["coverage"][position].item()
            ),
            "exact_set_probability": float(
                evaluation.values["exact_set"][position].item()
            ),
            "overlap": float(evaluation.values["overlap"][position].item()),
            "jaccard": float(evaluation.values["jaccard"][position].item()),
            "weighted_membership": float(
                evaluation.values["weighted_membership"][position].item()
            ),
            "hidden_topk_failure_probability": float(
                evaluation.values["hidden_topk_failure"][position].item()
            ),
            "boundary_margin": float(evaluation.values["margin"][position].item()),
            "reference_boundary_margin": float(
                evaluation.values["reference_margin"][position].item()
            ),
            "reference_top1_match_probability": float(
                evaluation.values["reference_top1_match"][position].item()
            ),
            "mean_logits_top1": (
                float(mean_values["top1"][position].item()) if mean_values else ""
            ),
            "mean_logits_exact_set": (
                float(mean_values["exact_set"][position].item()) if mean_values else ""
            ),
            "mean_logits_overlap": (
                float(mean_values["overlap"][position].item()) if mean_values else ""
            ),
            "clean_exact_probability": clean_probability,
            "clean_aligned": clean_aligned,
        })
    return rows


class TopKTargetedGaussianDefense:
    """Set-conditioned strengthening generalized from Top-2 to arbitrary k."""

    def __init__(
        self,
        standard_model: nn.Module,
        gradient_model: nn.Module,
        output_model: nn.Module,
        top_k: int,
        alpha: float,
        epsilon: float,
        steps: int,
        sigma: float,
        top1_weight: float,
        set_weight: float,
    ):
        self.standard_model = standard_model
        self.gradient_model = gradient_model
        self.output_model = output_model
        self.top_k = int(top_k)
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.steps = int(steps)
        self.sigma = float(sigma)
        self.top1_weight = float(top1_weight)
        self.set_weight = float(set_weight)

    def objective(
        self, logits: torch.Tensor, reference_topk: torch.Tensor
    ) -> torch.Tensor:
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

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            reference_topk = ranked_topk(self.standard_model(x), self.top_k)
        clean = x.detach()
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
        if self.sigma > 0:
            strengthened = (
                strengthened + self.sigma * torch.randn_like(strengthened)
            ).clamp(0.0, 1.0)
        with torch.no_grad():
            return self.output_model(strengthened)


class GeneralCorrectiveSelector:
    """Generalized gate; valid as a trained paper baseline only when k=2."""

    def __init__(
        self,
        standard_model: nn.Module,
        dual_model: nn.Module,
        proposed_actual: TopKTargetedGaussianDefense,
        top_k: int,
        threshold: float,
        gate_views: int,
        view_sigma: float,
    ):
        self.standard_model = standard_model
        self.dual_model = dual_model
        self.proposed_actual = proposed_actual
        self.top_k = int(top_k)
        self.threshold = float(threshold)
        self.gate_views = int(gate_views)
        self.view_sigma = float(view_sigma)

    def reference_stability(
        self, x: torch.Tensor, reference_set: torch.Tensor
    ) -> torch.Tensor:
        stable = torch.zeros(len(x), device=x.device, dtype=torch.float32)
        reference_sorted = sorted_set(reference_set)
        with torch.no_grad():
            for _ in range(self.gate_views):
                view = x
                if self.view_sigma > 0:
                    view = (x + self.view_sigma * torch.randn_like(x)).clamp(0, 1)
                view_set = sorted_set(ranked_topk(self.standard_model(view), self.top_k))
                stable += view_set.eq(reference_sorted).all(dim=1).float()
        return stable / float(self.gate_views)

    def forward_with_diagnostics(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        proposed_logits = self.proposed_actual(x)
        with torch.no_grad():
            reference = ranked_topk(self.standard_model(x), self.top_k)
            dual_logits = self.dual_model(x)
            reference_sorted = sorted_set(reference)
            proposed_match = sorted_set(
                ranked_topk(proposed_logits, self.top_k)
            ).eq(reference_sorted).all(dim=1)
            dual_match = sorted_set(
                ranked_topk(dual_logits, self.top_k)
            ).eq(reference_sorted).all(dim=1)
            correctable = dual_match & ~proposed_match
            stability = self.reference_stability(x, reference)
            use_dual = correctable & (stability >= self.threshold)
            selected = torch.where(use_dual[:, None], dual_logits, proposed_logits)
        return selected, {
            "use_dual": use_dual.float(),
            "correctable": correctable.float(),
            "stability": stability,
        }

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_diagnostics(x)[0]


def top1_attack_margin(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    row = torch.arange(len(labels), device=labels.device)
    true_logit = logits[row, labels]
    others = logits.clone()
    others[row, labels] = float("-inf")
    return true_logit - others.max(dim=1).values


def coverage_attack_margin(
    logits: torch.Tensor, labels: torch.Tensor, k: int
) -> torch.Tensor:
    """Positive exactly when the true label is in Top-k (ties aside)."""
    row = torch.arange(len(labels), device=labels.device)
    true_logit = logits[row, labels]
    others = logits.clone()
    others[row, labels] = float("-inf")
    kth_other = others.topk(k, dim=1).values[:, k - 1]
    return true_logit - kth_other


def exact_set_attack_margin(
    logits: torch.Tensor, reference_topk: torch.Tensor
) -> torch.Tensor:
    selected = logits.gather(1, reference_topk)
    outside = torch.ones_like(logits, dtype=torch.bool)
    outside.scatter_(1, reference_topk, False)
    strongest_outside = logits.masked_fill(
        ~outside, float("-inf")
    ).max(dim=1).values
    return selected.min(dim=1).values - strongest_outside


def overlap_attack_score(
    logits: torch.Tensor, reference_topk: torch.Tensor
) -> torch.Tensor:
    """Smooth positive objective that keeps optimizing after first set failure."""
    selected = logits.gather(1, reference_topk)
    outside = torch.ones_like(logits, dtype=torch.bool)
    outside.scatter_(1, reference_topk, False)
    outside_logits = logits.masked_fill(~outside, float("-inf"))
    # To reduce overlap rather than merely break exact equality, push every
    # reference item below several distinct outside candidates.  For k values
    # larger than half the label space, use all available outside classes.
    outside_count = logits.shape[1] - reference_topk.shape[1]
    competitors = min(reference_topk.shape[1], outside_count)
    strongest_outside = outside_logits.topk(competitors, dim=1).values
    return torch.nn.functional.softplus(
        selected[:, :, None] - strongest_outside[:, None, :]
    ).mean(dim=(1, 2))


def attack_score(
    logits: torch.Tensor,
    labels: torch.Tensor,
    reference_topk: torch.Tensor,
    k: int,
    goal: str,
    joint_top1_weight: float,
    joint_set_weight: float,
) -> torch.Tensor:
    if goal == "top1":
        return top1_attack_margin(logits, labels)
    if goal == "coverage":
        return coverage_attack_margin(logits, labels, k)
    if goal == "exact_set":
        return exact_set_attack_margin(logits, reference_topk)
    if goal == "overlap":
        return overlap_attack_score(logits, reference_topk)
    if goal == "joint":
        top1 = torch.nn.functional.softplus(top1_attack_margin(logits, labels))
        set_score = overlap_attack_score(logits, reference_topk)
        return joint_top1_weight * top1 + joint_set_weight * set_score
    raise ValueError(f"Unknown attack goal: {goal}")


def pgd_attack_batch(
    defense: base.DefenseSpec,
    images: torch.Tensor,
    labels: torch.Tensor,
    reference_topk: torch.Tensor,
    k: int,
    epsilon: float,
    alpha: float,
    steps: int,
    restarts: int,
    attack_eot: int,
    eval_mc: int,
    goal: str,
    joint_top1_weight: float,
    joint_set_weight: float,
) -> torch.Tensor:
    clean = images.detach()
    with torch.no_grad():
        best = clean.clone()
        best_score = attack_score(
            base.averaged_actual(defense, best, eval_mc),
            labels,
            reference_topk,
            k,
            goal,
            joint_top1_weight,
            joint_set_weight,
        )
    for _ in range(restarts):
        adversarial = (
            clean + torch.empty_like(clean).uniform_(-epsilon, epsilon)
        ).clamp(0.0, 1.0)
        for _ in range(steps):
            adversarial.requires_grad_(True)
            logits = base.bpda_eot_logits(defense, adversarial, attack_eot)
            score = attack_score(
                logits,
                labels,
                reference_topk,
                k,
                goal,
                joint_top1_weight,
                joint_set_weight,
            ).mean()
            gradient = torch.autograd.grad(score, adversarial, only_inputs=True)[0]
            adversarial = adversarial.detach() - alpha * gradient.sign()
            delta = (adversarial - clean).clamp(-epsilon, epsilon)
            adversarial = (clean + delta).clamp(0.0, 1.0).detach()
        with torch.no_grad():
            candidate_score = attack_score(
                base.averaged_actual(defense, adversarial, eval_mc),
                labels,
                reference_topk,
                k,
                goal,
                joint_top1_weight,
                joint_set_weight,
            )
            improved = candidate_score < best_score
            best[improved] = adversarial[improved]
            best_score[improved] = candidate_score[improved]
    return best.detach()


@dataclass
class QueryOracle:
    """Count returned defended-logit vectors, not internal component forwards."""

    defense: base.DefenseSpec
    eot: int
    logical_calls: int = 0

    @property
    def api_queries_per_example(self) -> int:
        return self.logical_calls * self.eot

    def __call__(self, candidate: torch.Tensor) -> torch.Tensor:
        self.logical_calls += 1
        return base.averaged_actual(self.defense, candidate, self.eot)


def requested_to_logical_budget(
    requested_budget: int, eot: int, budget_unit: str
) -> int:
    if budget_unit == "logical_queries":
        return int(requested_budget)
    return int(requested_budget) // int(eot)


def effective_attack_eot(
    defense: base.DefenseSpec,
    requested_eot: int,
    force_deterministic_eot: bool,
) -> int:
    if defense.sigma <= 0 and not force_deterministic_eot:
        return 1
    return int(requested_eot)


def _snapshot_at_checkpoints(
    snapshots: Dict[int, torch.Tensor],
    checkpoints: Sequence[int],
    query_index: int,
    candidate: torch.Tensor,
) -> None:
    if query_index in checkpoints:
        snapshots[query_index] = candidate.detach().clone()


def square_attack_curve(
    query_fn: QueryOracle,
    x: torch.Tensor,
    score_fn: Callable[[torch.Tensor], torch.Tensor],
    eps: float,
    checkpoints: Sequence[int],
) -> Dict[int, torch.Tensor]:
    """Nested local Square trajectory at the requested logical checkpoints."""
    checkpoints = sorted(set(int(value) for value in checkpoints))
    max_queries = max(checkpoints)
    batch, channels, height, width = x.shape
    snapshots: Dict[int, torch.Tensor] = {}
    x_best = x.detach().clone()
    best_margin = score_fn(query_fn(x_best))
    _snapshot_at_checkpoints(snapshots, checkpoints, 1, x_best)

    if max_queries >= 2:
        random_candidate = torch.clamp(
            x + torch.empty_like(x).uniform_(-eps, eps), 0.0, 1.0
        )
        random_margin = score_fn(query_fn(random_candidate))
        improved = random_margin < best_margin
        x_best[improved] = random_candidate[improved]
        best_margin[improved] = random_margin[improved]
        _snapshot_at_checkpoints(snapshots, checkpoints, 2, x_best)

    for query_index in range(3, max_queries + 1):
        ratio = max(1.0 - (query_index - 1) / max_queries, 0.05)
        patch_size = max(int(ratio * height), 1)
        rows = np.random.randint(0, max(height - patch_size + 1, 1), size=batch)
        cols = np.random.randint(0, max(width - patch_size + 1, 1), size=batch)
        candidate = x_best.clone()
        for sample in range(batch):
            if best_margin[sample] <= 0:
                continue
            new_value = (
                2
                * torch.randint(
                    0,
                    2,
                    (channels, patch_size, patch_size),
                    device=x.device,
                ).float()
                - 1
            ) * eps
            row, col = int(rows[sample]), int(cols[sample])
            patch = x[sample, :, row : row + patch_size, col : col + patch_size]
            candidate[sample, :, row : row + patch_size, col : col + patch_size] = (
                patch + new_value
            )
            candidate[sample] = torch.clamp(
                torch.clamp(candidate[sample], x[sample] - eps, x[sample] + eps),
                0.0,
                1.0,
            )
        candidate_margin = score_fn(query_fn(candidate))
        improved = candidate_margin < best_margin
        x_best[improved] = candidate[improved]
        best_margin[improved] = candidate_margin[improved]
        _snapshot_at_checkpoints(
            snapshots, checkpoints, query_index, x_best
        )
    return snapshots


def signhunt_attack_curve(
    query_fn: QueryOracle,
    x: torch.Tensor,
    score_fn: Callable[[torch.Tensor], torch.Tensor],
    eps: float,
    checkpoints: Sequence[int],
) -> Dict[int, torch.Tensor]:
    """Nested local SignHunt trajectory at logical-query checkpoints."""
    checkpoints = sorted(set(int(value) for value in checkpoints))
    max_queries = max(checkpoints)
    batch, channels, height, width = x.shape
    dimension = channels * height * width
    snapshots: Dict[int, torch.Tensor] = {}
    adversarial = x.detach().clone()
    best_margin = score_fn(query_fn(adversarial))
    _snapshot_at_checkpoints(snapshots, checkpoints, 1, adversarial)
    signs = 2 * torch.randint(0, 2, x.shape, device=x.device).float() - 1

    if max_queries >= 2:
        random_candidate = torch.clamp(x + eps * signs, 0.0, 1.0)
        random_candidate = torch.clamp(random_candidate, x - eps, x + eps)
        random_margin = score_fn(query_fn(random_candidate))
        improved = random_margin < best_margin
        adversarial[improved] = random_candidate[improved]
        best_margin[improved] = random_margin[improved]
        _snapshot_at_checkpoints(snapshots, checkpoints, 2, adversarial)

    for query_index in range(3, max_queries + 1):
        decay = max(1.0 - (query_index - 1) / max_queries, 0.02)
        number_to_flip = max(1, int(dimension * 0.1 * decay))
        new_signs = signs.clone()
        for sample in range(batch):
            if best_margin[sample] <= 0:
                continue
            indices = torch.randperm(dimension, device=x.device)[:number_to_flip]
            new_signs[sample].view(-1)[indices] *= -1
        candidate = torch.clamp(
            torch.clamp(x + eps * new_signs, 0.0, 1.0), x - eps, x + eps
        )
        candidate_margin = score_fn(query_fn(candidate))
        improved = candidate_margin < best_margin
        adversarial[improved] = candidate[improved]
        best_margin[improved] = candidate_margin[improved]
        signs[improved] = new_signs[improved]
        _snapshot_at_checkpoints(
            snapshots, checkpoints, query_index, adversarial
        )
    return snapshots


def nes_attack_curve(
    query_fn: QueryOracle,
    x: torch.Tensor,
    score_fn: Callable[[torch.Tensor], torch.Tensor],
    eps: float,
    checkpoints: Sequence[int],
    n_samples: int,
    sigma: float = 0.001,
    learning_rate: float = 0.01,
) -> Tuple[Dict[int, torch.Tensor], Dict[int, int]]:
    """NES trajectory; each checkpoint uses the latest update within budget."""
    checkpoints = sorted(set(int(value) for value in checkpoints))
    max_queries = max(checkpoints)
    adversarial = x.detach().clone()
    history: List[Tuple[int, torch.Tensor]] = [(0, adversarial.detach().clone())]
    queries_used = 0
    while queries_used + n_samples <= max_queries:
        gradient_estimate = torch.zeros_like(adversarial)
        for _ in range(n_samples // 2):
            noise = torch.randn_like(adversarial)
            loss_plus = -score_fn(
                query_fn(torch.clamp(adversarial + sigma * noise, 0.0, 1.0))
            )
            loss_minus = -score_fn(
                query_fn(torch.clamp(adversarial - sigma * noise, 0.0, 1.0))
            )
            gradient_estimate += (
                (loss_plus - loss_minus).view(-1, 1, 1, 1) * noise
            )
        queries_used += n_samples
        gradient_estimate /= float(n_samples) * sigma
        adversarial = torch.clamp(
            adversarial + learning_rate * gradient_estimate.sign(),
            x - eps,
            x + eps,
        ).clamp(0.0, 1.0)
        history.append((queries_used, adversarial.detach().clone()))

    snapshots: Dict[int, torch.Tensor] = {}
    used: Dict[int, int] = {}
    for checkpoint in checkpoints:
        eligible = [(q, candidate) for q, candidate in history if q <= checkpoint]
        query_count, candidate = eligible[-1]
        snapshots[checkpoint] = candidate.detach().clone()
        used[checkpoint] = query_count
    return snapshots, used


@dataclass
class AttackCurve:
    candidates: Dict[int, torch.Tensor]
    logical_budgets: Dict[int, int]
    api_queries: Dict[int, int]
    effective_eot: int
    wall_seconds: float


def evaluate_blackbox_curve(
    defense: base.DefenseSpec,
    attack_name: str,
    images: torch.Tensor,
    labels: torch.Tensor,
    reference_logits: torch.Tensor,
    k: int,
    epsilon: float,
    requested_budgets: Sequence[int],
    budget_unit: str,
    batch_size: int,
    requested_attack_eot: int,
    force_deterministic_eot: bool,
    goal: str,
    joint_top1_weight: float,
    joint_set_weight: float,
    nes_samples: int,
) -> AttackCurve:
    eot = effective_attack_eot(
        defense, requested_attack_eot, force_deterministic_eot
    )
    logical_budgets = {
        int(budget): requested_to_logical_budget(int(budget), eot, budget_unit)
        for budget in requested_budgets
    }
    if any(value < 1 for value in logical_budgets.values()):
        raise ValueError(
            f"At least one budget cannot pay for EOT={eot}: {logical_budgets}"
        )
    logical_checkpoints = sorted(set(logical_budgets.values()))
    by_budget_parts: Dict[int, List[torch.Tensor]] = {
        int(budget): [] for budget in requested_budgets
    }
    logical_used_by_budget: Dict[int, int] = {}
    batches = math.ceil(len(images) / batch_size)
    started = time.perf_counter()
    for batch_number, low in enumerate(range(0, len(images), batch_size), start=1):
        high = min(low + batch_size, len(images))
        x = images[low:high]
        y = labels[low:high]
        reference_topk = ranked_topk(reference_logits[low:high].to(x.device), k)

        def score_fn(candidate_logits: torch.Tensor) -> torch.Tensor:
            return attack_score(
                candidate_logits,
                y,
                reference_topk,
                k,
                goal,
                joint_top1_weight,
                joint_set_weight,
            )

        oracle = QueryOracle(defense=defense, eot=eot)
        if attack_name == "Square":
            snapshots = square_attack_curve(
                oracle, x, score_fn, epsilon, logical_checkpoints
            )
            used = {value: value for value in logical_checkpoints}
        elif attack_name == "SignHunt":
            snapshots = signhunt_attack_curve(
                oracle, x, score_fn, epsilon, logical_checkpoints
            )
            used = {value: value for value in logical_checkpoints}
        elif attack_name == "NES":
            samples = min(nes_samples, max(logical_checkpoints))
            samples -= samples % 2
            if samples < 2:
                raise ValueError("NES budget is too small for one antithetic pair")
            snapshots, used = nes_attack_curve(
                oracle,
                x,
                score_fn,
                epsilon,
                logical_checkpoints,
                n_samples=samples,
            )
        else:
            raise ValueError(f"Unsupported black-box attack: {attack_name}")

        for requested_budget, logical_budget in logical_budgets.items():
            by_budget_parts[requested_budget].append(
                snapshots[logical_budget].detach().cpu()
            )
            logical_used_by_budget[requested_budget] = used[logical_budget]
        print(
            f"    {attack_name} k={k} batch {batch_number}/{batches} "
            f"complete (EOT={eot})",
            flush=True,
        )

    candidates = {
        budget: torch.cat(parts, dim=0)
        for budget, parts in by_budget_parts.items()
    }
    api_queries = {
        budget: logical_used_by_budget[budget] * eot
        for budget in by_budget_parts
    }
    return AttackCurve(
        candidates=candidates,
        logical_budgets=logical_budgets,
        api_queries=api_queries,
        effective_eot=eot,
        wall_seconds=time.perf_counter() - started,
    )


def write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    exists = path.is_file() and path.stat().st_size > 0
    fieldnames = list(rows[0].keys())
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_reusable_clean_results(
    directory: Path,
    top_k: Sequence[int],
    defenses: Sequence[str],
    data_seed: int,
) -> Tuple[List[Dict[str, str]], Dict[Tuple[int, str], Dict[int, float]], List[Dict[str, str]]]:
    directory = directory.expanduser().resolve()
    summary_path = directory / "clean_results_topk_v4.csv"
    example_path = directory / "clean_per_example_topk_v4.csv"
    if not summary_path.is_file() or not example_path.is_file():
        raise FileNotFoundError(
            "--reuse-clean-dir must contain clean_results_topk_v4.csv and "
            "clean_per_example_topk_v4.csv"
        )
    allowed_k = {int(value) for value in top_k}
    allowed_defenses = set(defenses)
    summaries = [
        row
        for row in read_rows(summary_path)
        if int(row["k"]) in allowed_k and row["defense"] in allowed_defenses
    ]
    examples = [
        row
        for row in read_rows(example_path)
        if int(row["k"]) in allowed_k and row["defense"] in allowed_defenses
    ]
    if any(int(row["data_seed"]) != data_seed for row in examples):
        raise ValueError("Reused clean rows were generated with a different data seed")
    lookup: Dict[Tuple[int, str], Dict[int, float]] = {}
    for row in examples:
        key = (int(row["k"]), row["defense"])
        lookup.setdefault(key, {})[int(row["sample_index"])] = float(
            row["exact_set_probability"]
        )
    required = {(int(k), defense) for k in top_k for defense in defenses}
    missing = sorted(required - set(lookup))
    if missing:
        raise ValueError(f"Reused clean results miss requested k/defense pairs: {missing}")
    return summaries, lookup, examples


def attack_key(row: Mapping[str, object]) -> Tuple[str, ...]:
    return (
        str(row["defense"]),
        str(row["k"]),
        str(row["attack"]),
        str(row["attack_goal"]),
        str(row["epsilon_px"]),
        str(row["requested_budget"]),
    )


def summary_row(
    defense: str,
    condition: str,
    k: int,
    evaluation: EvaluationBundle,
    examples: int,
    sigma: float,
    requested_metrics: Sequence[str],
    data_seed: int,
    attack_seed: int,
    reeval_seed: int,
    run_tag: str,
    shard_count: int,
    shard_index: int,
    attack: str = "",
    attack_goal: str = "",
    epsilon_px: object = "",
    epsilon: object = "",
    requested_budget: object = "",
    logical_budget: object = "",
    pgd_restarts: object = "",
    attack_eot: object = "",
    attack_api_queries: object = "",
    wall_seconds: object = "",
    candidate_file: str = "",
    conditional_retention: object = "",
    clean_aligned_examples: object = "",
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "run_tag": run_tag,
        "data_seed": data_seed,
        "attack_seed": attack_seed,
        "reeval_seed": reeval_seed,
        "shard_count": shard_count,
        "shard_index": shard_index,
        "defense": defense,
        "condition": condition,
        "k": k,
        "attack": attack,
        "attack_goal": attack_goal,
        "epsilon_px": epsilon_px,
        "epsilon": epsilon,
        "requested_budget": requested_budget,
        "logical_budget": logical_budget,
        "pgd_restarts": pgd_restarts,
        "attack_eot": attack_eot,
        "attack_api_queries": attack_api_queries,
        "eval_draws": evaluation.draws,
        "examples": examples,
        "sigma": sigma,
        "wall_seconds": wall_seconds,
        "candidate_file": candidate_file,
        "conditional_retention": conditional_retention,
        "clean_aligned_examples": clean_aligned_examples,
        "attack_implementation": (
            "local_nested_v4" if attack in {"Square", "SignHunt", "NES"} else ""
        ),
    }
    for name in requested_metrics:
        row[name] = evaluation.summary[name]
        if evaluation.mean_logits_summary:
            row[f"mean_logits_{name}"] = evaluation.mean_logits_summary[name]
    row["reference_margin"] = evaluation.summary["reference_margin"]
    row["reference_top1_match"] = evaluation.summary["reference_top1_match"]
    return row


def comparison_table(
    clean_rows: Sequence[Mapping[str, object]],
    attack_rows: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    table: Dict[Tuple[str, str], Dict[str, object]] = {}
    order: List[Tuple[str, str]] = []
    for row in clean_rows:
        key = (str(row["defense"]), str(row["k"]))
        if key not in table:
            table[key] = {"defense": row["defense"], "k": row["k"]}
            order.append(key)
        for metric in METRIC_NAMES:
            if metric in row:
                table[key][f"clean_{metric}"] = row[metric]
    for row in attack_rows:
        key = (str(row["defense"]), str(row["k"]))
        if key not in table:
            table[key] = {"defense": row["defense"], "k": row["k"]}
            order.append(key)
        epsilon_tag = str(row["epsilon_px"]).replace(".", "p")
        attack_tag = str(row["attack"]).lower()
        goal_tag = str(row["attack_goal"])
        budget_tag = str(row["requested_budget"])
        prefix = f"{attack_tag}_eps{epsilon_tag}_{goal_tag}_b{budget_tag}"
        for metric in METRIC_NAMES:
            if metric in row:
                table[key][f"{prefix}_{metric}"] = row[metric]
    return [table[key] for key in order]


def make_defenses_for_k(
    args: argparse.Namespace,
    k: int,
    standard_model: nn.Module,
    robust_model: nn.Module,
    dual_model: nn.Module,
) -> Dict[str, base.DefenseSpec]:
    proposed_actual = TopKTargetedGaussianDefense(
        standard_model=standard_model,
        gradient_model=robust_model,
        output_model=robust_model,
        top_k=k,
        alpha=args.strength_alpha,
        epsilon=args.strength_epsilon,
        steps=args.strength_steps,
        sigma=args.fixed_sigma,
        top1_weight=args.top1_weight,
        set_weight=args.set_weight,
    )
    selector_actual = GeneralCorrectiveSelector(
        standard_model=standard_model,
        dual_model=dual_model,
        proposed_actual=proposed_actual,
        top_k=k,
        threshold=args.selector_threshold,
        gate_views=args.selector_gate_views,
        view_sigma=args.selector_view_sigma,
    )
    all_defenses = {
        "standard": base.DefenseSpec(
            "standard", base.make_actual(standard_model), standard_model, 0.0
        ),
        "robust": base.DefenseSpec(
            "robust", base.make_actual(robust_model), robust_model, 0.0
        ),
        "dual": base.DefenseSpec(
            "dual", base.make_actual(dual_model), dual_model, 0.0
        ),
        "proposed": base.DefenseSpec(
            "proposed", proposed_actual, robust_model, args.fixed_sigma
        ),
        "selector": base.DefenseSpec(
            "selector",
            selector_actual,
            robust_model,
            max(args.fixed_sigma, args.selector_view_sigma),
        ),
    }
    return {name: all_defenses[name] for name in args.defenses}


def evaluate_actual_defense(
    defense: base.DefenseSpec,
    images: torch.Tensor,
    labels: torch.Tensor,
    reference_logits: torch.Tensor,
    k: int,
    batch_size: int,
    requested_draws: int,
    tau: float,
    report_mean_logits: bool,
    seed: int,
) -> EvaluationBundle:
    base.seed_everything(seed)
    draws = requested_draws if defense.sigma > 0 else 1
    draw_logits = collect_actual_draws(defense, images, batch_size, draws)
    return evaluate_draws(
        draw_logits,
        labels.detach().cpu(),
        reference_logits.detach().cpu(),
        k,
        tau,
        report_mean_logits,
    )


def select_clean_probabilities(
    clean_indices: torch.Tensor,
    clean_probabilities: torch.Tensor,
    requested_indices: torch.Tensor,
) -> torch.Tensor:
    lookup = {
        int(index): float(probability)
        for index, probability in zip(
            clean_indices.tolist(), clean_probabilities.tolist()
        )
    }
    missing = [int(index) for index in requested_indices.tolist() if int(index) not in lookup]
    if missing:
        raise RuntimeError(
            "Attack subset is not contained in clean subset. Missing test indices: "
            + ",".join(str(value) for value in missing[:10])
        )
    return torch.tensor(
        [lookup[int(index)] for index in requested_indices.tolist()],
        dtype=torch.float32,
    )


def select_clean_probabilities_from_lookup(
    lookup: Mapping[int, float], requested_indices: torch.Tensor
) -> torch.Tensor:
    missing = [
        int(index) for index in requested_indices.tolist() if int(index) not in lookup
    ]
    if missing:
        raise RuntimeError(
            "Attack subset is absent from reused clean results. Missing test indices: "
            + ",".join(str(value) for value in missing[:10])
        )
    return torch.tensor(
        [float(lookup[int(index)]) for index in requested_indices.tolist()],
        dtype=torch.float32,
    )


def safe_token(value: object) -> str:
    text = str(value).replace("/", "_").replace(".", "p")
    return "".join(character for character in text if character.isalnum() or character in "-_")


def save_candidate_tensor(
    candidate_dir: Path,
    attacked: torch.Tensor,
    labels: torch.Tensor,
    dataset_indices: torch.Tensor,
    metadata: Mapping[str, object],
    dtype_name: str,
) -> Path:
    candidate_dir.mkdir(parents=True, exist_ok=True)
    ordered = [
        metadata["defense"],
        f"k{metadata['k']}",
        metadata["attack"],
        metadata["attack_goal"],
        f"eps{metadata['epsilon_px']}",
        f"q{metadata['requested_budget']}",
        f"seed{metadata['attack_seed']}",
    ]
    filename = "_".join(safe_token(value) for value in ordered) + ".pt"
    path = candidate_dir / filename
    storage_dtype = torch.float16 if dtype_name == "float16" else torch.float32
    torch.save(
        {
            "adversarial": attacked.detach().cpu().to(storage_dtype),
            "labels": labels.detach().cpu(),
            "dataset_indices": dataset_indices.detach().cpu(),
            "metadata": dict(metadata),
        },
        path,
    )
    return path


def evaluate_pgd_candidate(
    defense: base.DefenseSpec,
    images: torch.Tensor,
    labels: torch.Tensor,
    reference_logits: torch.Tensor,
    k: int,
    epsilon: float,
    alpha: float,
    steps: int,
    restarts: int,
    batch_size: int,
    attack_eot: int,
    selection_mc: int,
    goal: str,
    joint_top1_weight: float,
    joint_set_weight: float,
) -> Tuple[torch.Tensor, float]:
    parts: List[torch.Tensor] = []
    started = time.perf_counter()
    batches = math.ceil(len(images) / batch_size)
    for batch_number, low in enumerate(range(0, len(images), batch_size), start=1):
        high = min(low + batch_size, len(images))
        x = images[low:high]
        y = labels[low:high]
        reference_topk = ranked_topk(reference_logits[low:high].to(x.device), k)
        attacked = pgd_attack_batch(
            defense,
            x,
            y,
            reference_topk,
            k,
            epsilon,
            alpha,
            steps,
            restarts,
            attack_eot,
            selection_mc,
            goal,
            joint_top1_weight,
            joint_set_weight,
        )
        parts.append(attacked.detach().cpu())
        print(
            f"    PGD k={k} batch {batch_number}/{batches} complete",
            flush=True,
        )
    return torch.cat(parts, dim=0), time.perf_counter() - started


def main() -> None:
    args = parse_args()
    if args.quick:
        args.clean_examples = min(args.clean_examples, 1000)
        args.attack_examples = min(args.attack_examples, 100)
        args.attacks = ["Square"]
        args.attack_goals = ["exact_set"]
        args.attack_epsilons_px = [1.0]
        args.budgets = [10, 20]
        args.pgd_steps = min(args.pgd_steps, 3)
        args.pgd_restarts = 1
        args.attack_eot = min(args.attack_eot, 2)
        args.eval_draws = min(args.eval_draws, 2)
        args.clean_eval_draws = min(args.clean_eval_draws, 2)
        args.batch_size = min(args.batch_size, 10)
    validate_args(args)
    if args.attack_examples > args.clean_examples:
        raise ValueError(
            "--attack-examples cannot exceed --clean-examples because conditional "
            "retention requires the attacked subset to be clean-evaluated."
        )

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        if args.gpu_memory_fraction is not None:
            device_index = device.index if device.index is not None else 0
            torch.cuda.set_per_process_memory_fraction(
                args.gpu_memory_fraction, device_index
            )
        torch.cuda.reset_peak_memory_stats(device)
    base.seed_everything(args.seed)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_path = output_dir / "clean_results_topk_v4.csv"
    attack_path = output_dir / "attack_results_topk_v4.csv"
    clean_per_example_path = output_dir / "clean_per_example_topk_v4.csv"
    attack_per_example_path = output_dir / "attack_per_example_topk_v4.csv"
    candidate_dir = output_dir / "candidates_v4"
    if attack_path.exists() and not args.resume_results:
        raise FileExistsError(
            "attack_results_topk_v4.csv already exists. Use a new output directory "
            "or add --resume-results."
        )
    if attack_per_example_path.exists() and not args.resume_results:
        raise FileExistsError(
            "attack_per_example_topk_v4.csv already exists. Use a new output "
            "directory or add --resume-results."
        )

    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config["protocol_version"] = "topk_v4_api_query_fairness"
    config["attack_implementation"] = "local_nested_v4"
    config["torch_version"] = torch.__version__
    config["torch_cuda_version"] = torch.version.cuda
    config["gpu_name"] = (
        torch.cuda.get_device_name(device) if device.type == "cuda" else None
    )
    config["evaluator_sha256"] = hashlib.sha256(
        Path(__file__).resolve().read_bytes()
    ).hexdigest()
    with (output_dir / "run_config_topk_v4.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(config, handle, indent=2)

    standard_model, standard_arch = legacy.load_resnet20(
        args.standard_weights, device, legacy.STANDARD_CIFAR10_STD
    )
    dual_model = base.load_dual_defense(args.dual_weights, device)
    robust_model = base.RobustPixelModel(dual_model).to(device).eval()
    print(f"Standard checkpoint: {args.standard_weights} ({standard_arch})")
    print(f"Dual checkpoint:     {args.dual_weights}")
    if any(k != 2 for k in args.top_k) and any(
        name in {"dual", "selector"} for name in args.defenses
    ):
        print(
            "WARNING: Top-2-trained dual/selector is being used at k!=2. "
            "These rows are diagnostic only.",
            flush=True,
        )

    attack_images, attack_labels, attack_indices = load_balanced_subset_with_indices(
        args.data_dir,
        args.attack_examples,
        args.num_workers,
        device,
        args.data_seed,
    )
    standard_spec = base.DefenseSpec(
        "standard", base.make_actual(standard_model), standard_model, 0.0
    )
    attack_reference_logits = base.compute_logits_batched(
        standard_spec, attack_images, args.batch_size, 1
    )

    clean_rows: List[Dict[str, object]] = []
    clean_example_rows: List[Dict[str, object]] = []
    clean_probability_lookup: Dict[Tuple[int, str], Dict[int, float]] = {}
    if args.reuse_clean_dir is not None:
        reused_rows, clean_probability_lookup, reused_examples = (
            load_reusable_clean_results(
                args.reuse_clean_dir,
                args.top_k,
                args.defenses,
                args.data_seed,
            )
        )
        clean_rows = [dict(row) for row in reused_rows]
        clean_example_rows = [dict(row) for row in reused_examples]
        print(
            f"Reused clean evaluation from {args.reuse_clean_dir}",
            flush=True,
        )
    else:
        clean_images, clean_labels, clean_indices = load_balanced_subset_with_indices(
            args.data_dir,
            args.clean_examples,
            args.num_workers,
            device,
            args.data_seed,
        )
        clean_reference_logits = base.compute_logits_batched(
            standard_spec, clean_images, args.batch_size, 1
        )
        for k in args.top_k:
            defenses = make_defenses_for_k(
                args, k, standard_model, robust_model, dual_model
            )
            for name, defense in defenses.items():
                clean_seed = args.reeval_seed_base + stable_int(
                    f"clean|{args.data_seed}|k={k}"
                )
                evaluation = evaluate_actual_defense(
                    defense,
                    clean_images,
                    clean_labels,
                    clean_reference_logits,
                    k,
                    args.batch_size,
                    args.clean_eval_draws,
                    args.rank_temperature,
                    args.report_mean_logits,
                    clean_seed,
                )
                clean_probability_lookup[(k, name)] = {
                    int(index): float(probability)
                    for index, probability in zip(
                        clean_indices.tolist(),
                        evaluation.values["exact_set"].tolist(),
                    )
                }
                row = summary_row(
                    defense=name,
                    condition="clean",
                    k=k,
                    evaluation=evaluation,
                    examples=len(clean_images),
                    sigma=defense.sigma,
                    requested_metrics=args.metrics,
                    data_seed=args.data_seed,
                    attack_seed=args.seed,
                    reeval_seed=clean_seed,
                    run_tag=args.run_tag,
                    shard_count=args.shard_count,
                    shard_index=args.shard_index,
                )
                clean_rows.append(row)
                print(
                    f"Clean {name:>8s} k={k:2d} | "
                    f"T1(draw)={evaluation.summary['top1']:.2f}% "
                    f"Exact(draw)={evaluation.summary['exact_set']:.2f}% "
                    f"HiddenFail={evaluation.summary['hidden_topk_failure']:.2f}%",
                    flush=True,
                )
                if args.save_per_example:
                    clean_example_rows.extend(
                        per_example_rows(
                            evaluation=evaluation,
                            labels=clean_labels,
                            dataset_indices=clean_indices,
                            defense=name,
                            condition="clean",
                            k=k,
                            data_seed=args.data_seed,
                            attack_seed=args.seed,
                            reeval_seed=clean_seed,
                            run_tag=args.run_tag,
                            shard_count=args.shard_count,
                            shard_index=args.shard_index,
                        )
                    )
    write_rows(clean_path, clean_rows)
    if args.save_per_example or args.reuse_clean_dir is not None:
        write_rows(clean_per_example_path, clean_example_rows)

    if args.clean_only:
        write_rows(
            output_dir / "comparison_table_topk_v4.csv",
            comparison_table(clean_rows, []),
        )
        print("Clean-only Top-k v4 evaluation complete. Output:", output_dir)
        return

    attack_rows: List[Dict[str, object]] = list(read_rows(attack_path))
    completed = {attack_key(row) for row in attack_rows}
    for k in args.top_k:
        defenses = make_defenses_for_k(
            args, k, standard_model, robust_model, dual_model
        )
        for name, defense in defenses.items():
            clean_attack_probabilities = select_clean_probabilities_from_lookup(
                clean_probability_lookup[(k, name)],
                attack_indices,
            )
            for epsilon_px in args.attack_epsilons_px:
                epsilon = float(epsilon_px) / 255.0
                alpha_px = min(args.pgd_alpha_cap_px, float(epsilon_px) / 4.0)
                alpha = alpha_px / 255.0
                for attack_name in args.attacks:
                    for goal in args.attack_goals:
                        curve_key = (name, k, attack_name, goal, epsilon_px)
                        if not assigned_to_shard(
                            curve_key, args.shard_count, args.shard_index
                        ):
                            continue
                        budgets = (
                            [args.pgd_steps]
                            if attack_name == "PGD"
                            else sorted(set(int(value) for value in args.budgets))
                        )
                        missing_budgets = [
                            budget
                            for budget in budgets
                            if (
                                name,
                                str(k),
                                attack_name,
                                goal,
                                str(epsilon_px),
                                str(budget),
                            )
                            not in completed
                        ]
                        if not missing_budgets:
                            print("Skipping completed curve:", curve_key, flush=True)
                            continue

                        # Common attack randomness across defenses improves paired
                        # comparisons while each defense still sees its own logits.
                        combination_seed = args.seed + stable_int(
                            f"attack|k={k}|{attack_name}|{goal}|eps={epsilon_px}"
                        )
                        base.seed_everything(combination_seed)
                        print(
                            f"Attack curve defense={name} k={k} attack={attack_name} "
                            f"goal={goal} eps={epsilon_px:g}/255 budgets={budgets} "
                            f"shard={args.shard_index}/{args.shard_count}",
                            flush=True,
                        )

                        if attack_name == "PGD":
                            pgd_eot = effective_attack_eot(
                                defense,
                                args.attack_eot,
                                args.force_deterministic_eot,
                            )
                            attacked_cpu, curve_wall_seconds = evaluate_pgd_candidate(
                                defense=defense,
                                images=attack_images,
                                labels=attack_labels,
                                reference_logits=attack_reference_logits,
                                k=k,
                                epsilon=epsilon,
                                alpha=alpha,
                                steps=args.pgd_steps,
                                restarts=args.pgd_restarts,
                                batch_size=args.batch_size,
                                attack_eot=pgd_eot,
                                selection_mc=max(1, min(args.eval_draws, pgd_eot)),
                                goal=goal,
                                joint_top1_weight=args.joint_top1_weight,
                                joint_set_weight=args.joint_set_weight,
                            )
                            candidate_map = {args.pgd_steps: attacked_cpu}
                            logical_budget_map = {args.pgd_steps: args.pgd_steps}
                            api_query_map: Dict[int, object] = {args.pgd_steps: ""}
                            attack_eot_used = pgd_eot
                        else:
                            curve = evaluate_blackbox_curve(
                                defense=defense,
                                attack_name=attack_name,
                                images=attack_images,
                                labels=attack_labels,
                                reference_logits=attack_reference_logits,
                                k=k,
                                epsilon=epsilon,
                                requested_budgets=budgets,
                                budget_unit=args.budget_unit,
                                batch_size=args.batch_size,
                                requested_attack_eot=args.attack_eot,
                                force_deterministic_eot=args.force_deterministic_eot,
                                goal=goal,
                                joint_top1_weight=args.joint_top1_weight,
                                joint_set_weight=args.joint_set_weight,
                                nes_samples=args.nes_samples,
                            )
                            candidate_map = curve.candidates
                            logical_budget_map = curve.logical_budgets
                            api_query_map = curve.api_queries
                            attack_eot_used = curve.effective_eot
                            curve_wall_seconds = curve.wall_seconds

                        for budget in missing_budgets:
                            attacked_cpu = candidate_map[budget]
                            reeval_seed = args.reeval_seed_base + stable_int(
                                f"reeval|k={k}|{attack_name}|{goal}|"
                                f"eps={epsilon_px}|budget={budget}|data={args.data_seed}"
                            )
                            attacked = attacked_cpu.to(device=device, dtype=torch.float32)
                            evaluation = evaluate_actual_defense(
                                defense,
                                attacked,
                                attack_labels,
                                attack_reference_logits,
                                k,
                                args.batch_size,
                                args.eval_draws,
                                args.rank_temperature,
                                args.report_mean_logits,
                                reeval_seed,
                            )
                            eligible = clean_attack_probabilities >= float(
                                args.clean_alignment_threshold
                            )
                            conditional_retention: object = ""
                            if bool(eligible.any()):
                                conditional_retention = 100.0 * float(
                                    evaluation.values["exact_set"][eligible].mean().item()
                                )
                            candidate_file = ""
                            metadata = {
                                "protocol_version": "topk_v4_api_query_fairness",
                                "defense": name,
                                "k": k,
                                "attack": attack_name,
                                "attack_goal": goal,
                                "epsilon_px": epsilon_px,
                                "requested_budget": budget,
                                "logical_budget": logical_budget_map[budget],
                                "attack_eot": attack_eot_used,
                                "attack_api_queries": api_query_map[budget],
                                "attack_seed": combination_seed,
                                "reeval_seed": reeval_seed,
                                "data_seed": args.data_seed,
                                "run_tag": args.run_tag,
                                "shard_index": args.shard_index,
                                "shard_count": args.shard_count,
                            }
                            if args.save_candidates:
                                saved_path = save_candidate_tensor(
                                    candidate_dir,
                                    attacked_cpu,
                                    attack_labels,
                                    attack_indices,
                                    metadata,
                                    args.candidate_dtype,
                                )
                                candidate_file = str(saved_path.relative_to(output_dir))
                            row = summary_row(
                                defense=name,
                                condition="attack",
                                k=k,
                                evaluation=evaluation,
                                examples=len(attack_images),
                                sigma=defense.sigma,
                                requested_metrics=args.metrics,
                                data_seed=args.data_seed,
                                attack_seed=combination_seed,
                                reeval_seed=reeval_seed,
                                run_tag=args.run_tag,
                                shard_count=args.shard_count,
                                shard_index=args.shard_index,
                                attack=attack_name,
                                attack_goal=goal,
                                epsilon_px=epsilon_px,
                                epsilon=epsilon,
                                requested_budget=budget,
                                logical_budget=logical_budget_map[budget],
                                pgd_restarts=(
                                    args.pgd_restarts if attack_name == "PGD" else ""
                                ),
                                attack_eot=attack_eot_used,
                                attack_api_queries=api_query_map[budget],
                                wall_seconds=curve_wall_seconds,
                                candidate_file=candidate_file,
                                conditional_retention=conditional_retention,
                                clean_aligned_examples=int(eligible.sum().item()),
                            )
                            append_rows(attack_path, [row])
                            attack_rows.append(row)
                            completed.add(
                                (
                                    name,
                                    str(k),
                                    attack_name,
                                    goal,
                                    str(epsilon_px),
                                    str(budget),
                                )
                            )
                            if args.save_per_example:
                                append_rows(
                                    attack_per_example_path,
                                    per_example_rows(
                                        evaluation=evaluation,
                                        labels=attack_labels,
                                        dataset_indices=attack_indices,
                                        defense=name,
                                        condition="attack",
                                        k=k,
                                        data_seed=args.data_seed,
                                        attack_seed=combination_seed,
                                        reeval_seed=reeval_seed,
                                        run_tag=args.run_tag,
                                        shard_count=args.shard_count,
                                        shard_index=args.shard_index,
                                        attack=attack_name,
                                        attack_goal=goal,
                                        epsilon_px=epsilon_px,
                                        requested_budget=budget,
                                        logical_budget=logical_budget_map[budget],
                                        attack_eot=attack_eot_used,
                                        attack_api_queries=api_query_map[budget],
                                        candidate_file=candidate_file,
                                        clean_exact_probability=clean_attack_probabilities,
                                        clean_alignment_threshold=(
                                            args.clean_alignment_threshold
                                        ),
                                    ),
                                )
                            print(
                                f"  q={api_query_map[budget]} "
                                f"T1(draw)={evaluation.summary['top1']:.2f}% "
                                f"Exact(draw)={evaluation.summary['exact_set']:.2f}% "
                                f"Cond={conditional_retention}",
                                flush=True,
                            )
                            del attacked, evaluation
                        del candidate_map
                        if device.type == "cuda":
                            torch.cuda.empty_cache()

    write_rows(
        output_dir / "comparison_table_topk_v4.csv",
        comparison_table(clean_rows, attack_rows),
    )
    summary = {
        "protocol_version": "topk_v4_api_query_fairness",
        "research_scope": "clean-reference Top-k set fidelity",
        "standard_weights": str(args.standard_weights),
        "dual_weights": str(args.dual_weights),
        "data_seed": args.data_seed,
        "attack_seed": args.seed,
        "run_tag": args.run_tag,
        "shard": {"index": args.shard_index, "count": args.shard_count},
        "top_k": args.top_k,
        "metrics": args.metrics,
        "strength": {
            "alpha": args.strength_alpha,
            "epsilon": args.strength_epsilon,
            "steps": args.strength_steps,
            "top1_weight": args.top1_weight,
            "set_weight": args.set_weight,
        },
        "sigma": args.fixed_sigma,
        "budget_unit": args.budget_unit,
        "requested_attack_eot": args.attack_eot,
        "deterministic_eot_is_one": not args.force_deterministic_eot,
        "primary_stochastic_metric": "per-request probability over independent draws",
        "secondary_metric": "metrics of mean logits" if args.report_mean_logits else None,
        "eval_draws": args.eval_draws,
        "clean_eval_draws": args.clean_eval_draws,
        "clean_alignment_threshold": args.clean_alignment_threshold,
        "attack_implementation": "local nested Top-k adaptations; not official attacks",
        "legacy_dual_warning": (
            "The supplied dual/selector checkpoint was trained for k=2 only."
        ),
        "attack_note": (
            "Square, SignHunt, and NES use nested budget curves and query the "
            "actual defense. PGD is a diagnostic BPDA+EOT path."
        ),
        "per_example_saved": bool(args.save_per_example),
        "candidates_saved": bool(args.save_candidates),
        "gpu_peak_memory_gib": (
            float(torch.cuda.max_memory_allocated(device)) / (1024.0**3)
            if device.type == "cuda"
            else None
        ),
    }
    with (output_dir / "summary_topk_v4.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)
    print("General Top-k v4 evaluation complete. Output directory:", output_dir)


if __name__ == "__main__":
    main()
