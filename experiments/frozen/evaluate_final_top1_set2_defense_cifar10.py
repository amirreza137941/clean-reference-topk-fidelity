#!/usr/bin/env python3
"""Final adaptive evaluation for the CIFAR-10 Top1/exact-Set2 project.

Compared defenses
-----------------
standard
    Clean ResNet-20 reference (sanity/upper-bound row).
robust
    Frozen robust branch stored inside the fine-frontier checkpoint.
dual
    Fine-frontier DualBackboneDefense checkpoint.
proposed
    The thesis idea: standard Top-2 selection, targeted strengthening computed
    from the robust model, Gaussian noise, then robust-model output.
hybrid
    The same targeted+Gaussian preprocessor, but the final output is produced
    by the trained DualBackboneDefense.  The strengthening gradient still comes
    from the adversarially trained robust branch, so the original idea is kept.

Attacks
-------
* PGD-BPDA-EOT: exact gradients for standard/robust/dual and an identity-BPDA
  surrogate through the targeted+Gaussian preprocessor.  Stochastic outputs
  are averaged with EOT.
* Square and SignHunt: score-based attacks that query the actual defense.

Every attack is run separately for true-label Top-1 and exact unordered Set2.
Set2 is always compared with the clean standard model's Top-2 for the original
unperturbed image.  Strength and sigma are calibrated only on a held-out slice
of the CIFAR-10 training split, never on the reported test examples.

Dependencies: keep this file beside
``evaluate_top1_set2_gated_cifar10.py`` and
``train_resnet20_dual_backbone_top1_set2_cifar10.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    import evaluate_top1_set2_gated_cifar10 as legacy
    import train_resnet20_dual_backbone_top1_set2_cifar10 as dual_base
except ImportError as exc:
    raise SystemExit(
        "Place this script beside evaluate_top1_set2_gated_cifar10.py and "
        "train_resnet20_dual_backbone_top1_set2_cifar10.py"
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Final clean, PGD-BPDA-EOT, Square and SignHunt evaluation"
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
            "final_top1_set2_adaptive_evaluation"
        ),
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--clean-examples", type=int, default=10000)
    parser.add_argument("--attack-examples", type=int, default=300)
    parser.add_argument("--calibration-examples", type=int, default=500)
    parser.add_argument("--training-val-size", type=int, default=5000)
    parser.add_argument("--training-split-seed", type=int, default=42)

    parser.add_argument(
        "--defenses",
        nargs="+",
        default=["standard", "robust", "dual", "proposed", "hybrid"],
        choices=["standard", "robust", "dual", "proposed", "hybrid"],
    )
    parser.add_argument(
        "--attacks",
        nargs="+",
        default=["PGD", "Square", "SignHunt"],
        choices=["PGD", "Square", "SignHunt"],
    )
    parser.add_argument(
        "--attack-goals",
        nargs="+",
        default=["top1", "set2"],
        choices=["top1", "set2"],
    )
    parser.add_argument(
        "--attack-epsilons-px",
        type=float,
        nargs="+",
        default=[0.9],
        help="L-infinity radii in pixel units; each value is divided by 255",
    )
    parser.add_argument("--budgets", type=int, nargs="+", default=[500])
    parser.add_argument("--attack-eot", type=int, default=5)
    parser.add_argument("--eval-mc", type=int, default=10)

    parser.add_argument("--pgd-steps", type=int, default=20)
    parser.add_argument("--pgd-restarts", type=int, default=1)
    parser.add_argument(
        "--pgd-alpha-cap-px",
        type=float,
        default=2.0,
        help="PGD alpha is min(cap, epsilon/4), in pixel units",
    )

    # Targeted Top2 strengthening.  Default explicit values are the best point
    # found in the previous validation calibration; --tune-strength reruns a
    # compact validation search for the robust branch in the new checkpoint.
    parser.add_argument("--strength-alpha", type=float, default=0.005)
    parser.add_argument("--strength-epsilon", type=float, default=0.03)
    parser.add_argument("--strength-steps", type=int, default=2)
    parser.add_argument("--top1-weight", type=float, default=1.0)
    parser.add_argument("--set2-weight", type=float, default=4.0)
    parser.add_argument("--tune-strength", action="store_true")
    parser.add_argument(
        "--strength-alpha-grid", type=float, nargs="+", default=[0.003, 0.005]
    )
    parser.add_argument(
        "--strength-step-grid", type=int, nargs="+", default=[1, 2]
    )
    parser.add_argument(
        "--top1-weight-grid", type=float, nargs="+", default=[1.0, 1.5, 2.0]
    )
    parser.add_argument(
        "--set2-weight-grid", type=float, nargs="+", default=[2.5, 4.0]
    )
    parser.add_argument("--tune-max-top1-drop", type=float, default=0.5)

    parser.add_argument(
        "--sigma-drop",
        type=float,
        default=0.5,
        help="Allowed validation Set2 drop versus targeted sigma=0",
    )
    parser.add_argument(
        "--fixed-sigma",
        type=float,
        default=None,
        help="Skip sigma calibration and use this input-space Gaussian std",
    )
    parser.add_argument("--sigma-min", type=float, default=1e-4)
    parser.add_argument("--sigma-max", type=float, default=0.5)
    parser.add_argument("--sigma-search-iters", type=int, default=15)
    parser.add_argument("--sigma-trials", type=int, default=3)
    parser.add_argument("--sigma-tolerance", type=float, default=0.3)

    parser.add_argument("--clean-only", action="store_true")
    parser.add_argument(
        "--resume-results",
        action="store_true",
        help="Append only missing attack combinations in an existing output dir",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="100-image smoke test with tiny attack settings; not a thesis result",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA requested but no GPU is available")
    for name in ("clean_examples", "attack_examples", "calibration_examples"):
        value = int(getattr(args, name))
        upper = 10000 if name != "calibration_examples" else args.training_val_size
        if value <= 0 or value % 10 != 0 or value > upper:
            raise ValueError(f"--{name.replace('_', '-')} must be divisible by 10 and <= {upper}")
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("Batch size must be positive and workers non-negative")
    if args.attack_eot < 1 or args.eval_mc < 1:
        raise ValueError("--attack-eot and --eval-mc must be >= 1")
    if args.pgd_steps < 1 or args.pgd_restarts < 1 or args.pgd_alpha_cap_px <= 0:
        raise ValueError("PGD steps/restarts/cap must be positive")
    if any(value <= 0 or value > 255 for value in args.attack_epsilons_px):
        raise ValueError("Every --attack-epsilons-px value must be in (0,255]")
    if any(value < 2 for value in args.budgets):
        raise ValueError("Every black-box budget must be >= 2")
    if args.strength_alpha <= 0 or args.strength_epsilon <= 0:
        raise ValueError("Strength alpha and epsilon must be positive")
    if args.strength_steps < 1 or args.top1_weight < 0 or args.set2_weight < 0:
        raise ValueError("Strength steps must be positive and weights non-negative")
    if args.fixed_sigma is not None and args.fixed_sigma < 0:
        raise ValueError("--fixed-sigma cannot be negative")
    if args.sigma_drop < 0 or args.sigma_trials < 1:
        raise ValueError("Sigma drop must be non-negative and trials >= 1")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def torch_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def extract_state_dict(loaded: object) -> Mapping[str, torch.Tensor]:
    if not isinstance(loaded, Mapping):
        raise TypeError("Dual checkpoint is not a mapping/state_dict")
    for key in ("model_state_dict", "state_dict", "model"):
        candidate = loaded.get(key)
        if isinstance(candidate, Mapping) and candidate:
            loaded = candidate
            break
    state = dict(loaded)
    if state and all(key.startswith("module.") for key in state):
        state = {key[len("module."):]: value for key, value in state.items()}
    return state


def load_dual_defense(path: Path, device: torch.device) -> dual_base.DualBackboneDefense:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Dual checkpoint not found: {path}")
    model = dual_base.DualBackboneDefense(
        selector_temperature=1.0,
        primary_margin=1.0,
    ).to(device)
    state = extract_state_dict(torch_load(path, device))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Dual checkpoint/model mismatch. Missing={list(missing)}; "
            f"unexpected={list(unexpected)}"
        )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


class RobustPixelModel(nn.Module):
    """Expose the dual checkpoint's normalized robust branch on pixel inputs."""

    def __init__(self, dual: dual_base.DualBackboneDefense):
        super().__init__()
        self.dual = dual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dual.robust_branch(self.dual.normalize(x))


class TargetedGaussianDefense:
    """Original Top2 targeted-noise idea with configurable final output model."""

    def __init__(
        self,
        standard_model: nn.Module,
        gradient_model: nn.Module,
        output_model: nn.Module,
        alpha: float,
        epsilon: float,
        steps: int,
        sigma: float,
        top1_weight: float,
        set2_weight: float,
    ):
        self.standard_model = standard_model
        self.gradient_model = gradient_model
        self.output_model = output_model
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.steps = int(steps)
        self.sigma = float(sigma)
        self.top1_weight = float(top1_weight)
        self.set2_weight = float(set2_weight)

    def objective(
        self, logits: torch.Tensor, reference_top2: torch.Tensor
    ) -> torch.Tensor:
        selected = logits.gather(1, reference_top2)
        c1_logit = selected[:, 0]

        not_c1 = torch.ones_like(logits, dtype=torch.bool)
        not_c1.scatter_(1, reference_top2[:, :1], False)
        strongest_not_c1 = logits.masked_fill(
            ~not_c1, float("-inf")
        ).max(dim=1).values

        outside = torch.ones_like(logits, dtype=torch.bool)
        outside.scatter_(1, reference_top2, False)
        weakest_selected = selected.min(dim=1).values
        strongest_outside = logits.masked_fill(
            ~outside, float("-inf")
        ).max(dim=1).values

        top1_margin = c1_logit - strongest_not_c1
        set2_margin = weakest_selected - strongest_outside
        return self.top1_weight * top1_margin + self.set2_weight * set2_margin

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            reference_top2 = self.standard_model(x).topk(2, dim=1).indices
        base = x.detach()
        strengthened = base.clone()
        for _ in range(self.steps):
            with torch.enable_grad():
                variable = strengthened.detach().requires_grad_(True)
                objective = self.objective(
                    self.gradient_model(variable), reference_top2
                )
                gradient = torch.autograd.grad(
                    objective.sum(), variable, only_inputs=True
                )[0]
            strengthened = strengthened + self.alpha * gradient.sign()
            delta = (strengthened - base).clamp(-self.epsilon, self.epsilon)
            strengthened = (base + delta).clamp(0.0, 1.0).detach()
        if self.sigma > 0:
            strengthened = (
                strengthened + self.sigma * torch.randn_like(strengthened)
            ).clamp(0.0, 1.0)
        with torch.no_grad():
            return self.output_model(strengthened)


@dataclass
class DefenseSpec:
    name: str
    actual: Callable[[torch.Tensor], torch.Tensor]
    surrogate: nn.Module
    sigma: float = 0.0


def averaged_actual(
    defense: DefenseSpec, x: torch.Tensor, trials: int
) -> torch.Tensor:
    total: Optional[torch.Tensor] = None
    for _ in range(trials):
        logits = defense.actual(x).detach()
        total = logits if total is None else total + logits
    if total is None:
        raise RuntimeError("No Monte Carlo samples were evaluated")
    return total / float(trials)


def compute_logits_batched(
    defense: DefenseSpec,
    images: torch.Tensor,
    batch_size: int,
    trials: int,
) -> torch.Tensor:
    parts: List[torch.Tensor] = []
    for low in range(0, len(images), batch_size):
        batch = images[low : low + batch_size]
        parts.append(averaged_actual(defense, batch, trials).cpu())
    return torch.cat(parts, dim=0)


def bpda_eot_logits(
    defense: DefenseSpec,
    x: torch.Tensor,
    trials: int,
) -> torch.Tensor:
    """Actual forward values with an identity-preprocessor BPDA gradient."""
    total: Optional[torch.Tensor] = None
    for _ in range(trials):
        actual_logits = defense.actual(x).detach()
        surrogate_input = x
        if defense.sigma > 0:
            surrogate_input = (
                x + defense.sigma * torch.randn_like(x)
            ).clamp(0.0, 1.0)
        surrogate_logits = defense.surrogate(surrogate_input)
        bpda_logits = (
            actual_logits + surrogate_logits - surrogate_logits.detach()
        )
        total = bpda_logits if total is None else total + bpda_logits
    if total is None:
        raise RuntimeError("No EOT samples were evaluated")
    return total / float(trials)


def attack_margin(
    logits: torch.Tensor,
    labels: torch.Tensor,
    reference_top2: torch.Tensor,
    goal: str,
) -> torch.Tensor:
    if goal == "top1":
        return legacy.top1_attack_margin(logits, labels)
    if goal == "set2":
        return legacy.set2_attack_margin(logits, reference_top2)
    raise ValueError(f"Unknown attack goal: {goal}")


def pgd_bpda_eot_attack(
    defense: DefenseSpec,
    images: torch.Tensor,
    labels: torch.Tensor,
    reference_top2: torch.Tensor,
    epsilon: float,
    alpha: float,
    steps: int,
    restarts: int,
    attack_eot: int,
    eval_mc: int,
    goal: str,
) -> torch.Tensor:
    clean = images.detach()
    with torch.no_grad():
        best = clean.clone()
        best_margin = attack_margin(
            averaged_actual(defense, best, eval_mc),
            labels,
            reference_top2,
            goal,
        )

    for _ in range(restarts):
        adversarial = (
            clean + torch.empty_like(clean).uniform_(-epsilon, epsilon)
        ).clamp(0.0, 1.0)
        for _ in range(steps):
            adversarial.requires_grad_(True)
            logits = bpda_eot_logits(defense, adversarial, attack_eot)
            margin = attack_margin(
                logits, labels, reference_top2, goal
            ).mean()
            gradient = torch.autograd.grad(margin, adversarial, only_inputs=True)[0]
            adversarial = adversarial.detach() - alpha * gradient.sign()
            delta = (adversarial - clean).clamp(-epsilon, epsilon)
            adversarial = (clean + delta).clamp(0.0, 1.0).detach()

        with torch.no_grad():
            candidate_margin = attack_margin(
                averaged_actual(defense, adversarial, eval_mc),
                labels,
                reference_top2,
                goal,
            )
            improved = candidate_margin < best_margin
            best[improved] = adversarial[improved]
            best_margin[improved] = candidate_margin[improved]
    return best.detach()


def evaluate_pgd(
    defense: DefenseSpec,
    images: torch.Tensor,
    labels: torch.Tensor,
    reference_logits: torch.Tensor,
    epsilon: float,
    alpha: float,
    steps: int,
    restarts: int,
    batch_size: int,
    attack_eot: int,
    eval_mc: int,
    goal: str,
) -> Dict[str, float]:
    result_parts: List[torch.Tensor] = []
    batches = math.ceil(len(images) / batch_size)
    for batch_number, low in enumerate(range(0, len(images), batch_size), start=1):
        high = min(low + batch_size, len(images))
        x = images[low:high]
        y = labels[low:high]
        reference = reference_logits[low:high].to(x.device)
        reference_top2 = reference.topk(2, dim=1).indices
        attacked = pgd_bpda_eot_attack(
            defense,
            x,
            y,
            reference_top2,
            epsilon,
            alpha,
            steps,
            restarts,
            attack_eot,
            eval_mc,
            goal,
        )
        result_parts.append(averaged_actual(defense, attacked, eval_mc).cpu())
        print(f"    PGD batch {batch_number}/{batches} complete", flush=True)
    logits = torch.cat(result_parts, dim=0)
    return legacy.all_metrics(logits, labels.cpu(), reference_logits.cpu())


BLACKBOX_ATTACKS: Dict[str, Callable] = {
    "Square": legacy.square_attack,
    "SignHunt": legacy.signhunt_attack,
}


def evaluate_blackbox(
    defense: DefenseSpec,
    attack_name: str,
    images: torch.Tensor,
    labels: torch.Tensor,
    reference_logits: torch.Tensor,
    epsilon: float,
    budget: int,
    batch_size: int,
    attack_eot: int,
    eval_mc: int,
    goal: str,
) -> Dict[str, float]:
    attack = BLACKBOX_ATTACKS[attack_name]
    result_parts: List[torch.Tensor] = []
    batches = math.ceil(len(images) / batch_size)
    for batch_number, low in enumerate(range(0, len(images), batch_size), start=1):
        high = min(low + batch_size, len(images))
        x = images[low:high]
        y = labels[low:high]
        reference = reference_logits[low:high].to(x.device)
        reference_top2 = reference.topk(2, dim=1).indices

        def query_fn(candidate: torch.Tensor) -> torch.Tensor:
            return averaged_actual(defense, candidate, attack_eot)

        def score_fn(logits: torch.Tensor) -> torch.Tensor:
            return attack_margin(logits, y, reference_top2, goal)

        attacked = attack(
            query_fn,
            x,
            score_fn,
            eps=epsilon,
            n_queries=budget,
        )
        result_parts.append(averaged_actual(defense, attacked, eval_mc).cpu())
        print(
            f"    {attack_name} batch {batch_number}/{batches} complete",
            flush=True,
        )
    logits = torch.cat(result_parts, dim=0)
    return legacy.all_metrics(logits, labels.cpu(), reference_logits.cpu())


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


def append_row(path: Path, row: Mapping[str, object]) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def read_rows(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def attack_key(row: Mapping[str, object]) -> Tuple[str, str, str, str, str]:
    return (
        str(row["defense"]),
        str(row["attack"]),
        str(row["attack_goal"]),
        str(row["epsilon_px"]),
        str(row["budget_or_steps"]),
    )


def comparison_table(
    clean_rows: Sequence[Mapping[str, object]],
    attack_rows: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    names = [str(row["defense"]) for row in clean_rows]
    table: Dict[str, Dict[str, object]] = {
        str(row["defense"]): {
            "defense": row["defense"],
            "clean_top1": row["top1"],
            "clean_set2": row["set2"],
        }
        for row in clean_rows
    }
    for row in attack_rows:
        name = str(row["defense"])
        if name not in table:
            table[name] = {"defense": name}
            names.append(name)
        epsilon_tag = str(row["epsilon_px"]).replace(".", "p")
        attack_tag = str(row["attack"]).replace("-", "_").lower()
        goal = str(row["attack_goal"])
        budget = str(row["budget_or_steps"])
        prefix = f"{attack_tag}_eps{epsilon_tag}_{goal}_b{budget}"
        table[name][f"{prefix}_top1"] = row["top1"]
        table[name][f"{prefix}_set2"] = row["set2"]
    return [table[name] for name in names]


def make_actual(model: nn.Module) -> Callable[[torch.Tensor], torch.Tensor]:
    def call(x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return model(x)
    return call


def main() -> None:
    args = parse_args()
    if args.quick:
        args.clean_examples = min(args.clean_examples, 1000)
        args.attack_examples = min(args.attack_examples, 100)
        args.calibration_examples = min(args.calibration_examples, 100)
        args.attacks = ["PGD", "Square", "SignHunt"]
        args.attack_epsilons_px = [0.9]
        args.budgets = [50]
        args.pgd_steps = min(args.pgd_steps, 3)
        args.pgd_restarts = 1
        args.attack_eot = min(args.attack_eot, 2)
        args.eval_mc = min(args.eval_mc, 2)
        args.sigma_search_iters = min(args.sigma_search_iters, 6)
        args.sigma_trials = min(args.sigma_trials, 2)
        args.tune_strength = False
        args.batch_size = min(args.batch_size, 25)
    validate_args(args)

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    seed_everything(args.seed)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    attack_path = output_dir / "attack_results.csv"
    if attack_path.exists() and not args.resume_results:
        raise FileExistsError(
            "attack_results.csv already exists. Use a new --output-dir or "
            "add --resume-results to continue missing combinations."
        )
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    standard_model, standard_arch = legacy.load_resnet20(
        args.standard_weights,
        device,
        legacy.STANDARD_CIFAR10_STD,
    )
    dual_model = load_dual_defense(args.dual_weights, device)
    robust_model = RobustPixelModel(dual_model).to(device).eval()
    print(f"Standard checkpoint: {args.standard_weights} ({standard_arch})")
    print(f"Dual checkpoint:     {args.dual_weights}")

    clean_images, clean_labels = legacy.load_balanced_subset(
        args.data_dir,
        args.clean_examples,
        args.num_workers,
        device,
        args.seed,
    )
    attack_images, attack_labels = legacy.load_balanced_subset(
        args.data_dir,
        args.attack_examples,
        args.num_workers,
        device,
        args.seed,
    )
    calibration_images, calibration_labels = legacy.load_calibration_subset(
        data_dir=args.data_dir,
        n_examples=args.calibration_examples,
        training_val_size=args.training_val_size,
        split_seed=args.training_split_seed,
        num_workers=args.num_workers,
        device=device,
    )

    standard_spec = DefenseSpec(
        "standard", make_actual(standard_model), standard_model, 0.0
    )
    clean_reference_logits = compute_logits_batched(
        standard_spec, clean_images, args.batch_size, 1
    )
    attack_reference_logits = compute_logits_batched(
        standard_spec, attack_images, args.batch_size, 1
    )
    calibration_reference_logits = compute_logits_batched(
        standard_spec, calibration_images, args.batch_size, 1
    )

    calibration_rows: List[Dict[str, object]] = []
    if args.tune_strength:
        selected, calibration_rows, _ = legacy.tune_strength_parameters(
            standard_model=standard_model,
            robust_model=robust_model,
            images=calibration_images,
            labels=calibration_labels,
            reference_logits=calibration_reference_logits,
            batch_size=args.batch_size,
            epsilon=args.strength_epsilon,
            alpha_values=args.strength_alpha_grid,
            step_values=args.strength_step_grid,
            top1_weight_values=args.top1_weight_grid,
            set2_weight_values=args.set2_weight_grid,
            max_top1_drop=args.tune_max_top1_drop,
        )
        args.strength_alpha = float(selected["alpha"])
        args.strength_steps = int(selected["steps"])
        args.top1_weight = float(selected["top1_weight"])
        args.set2_weight = float(selected["set2_weight"])
        write_rows(output_dir / "strength_calibration.csv", calibration_rows)

    if args.fixed_sigma is None:
        sigma, achieved_set2, sigma_zero_set2 = legacy.calibrate_proposed_sigma(
            standard_model=standard_model,
            robust_model=robust_model,
            images=calibration_images,
            reference_logits=calibration_reference_logits,
            alpha=args.strength_alpha,
            targeted_epsilon=args.strength_epsilon,
            targeted_steps=args.strength_steps,
            top1_weight=args.top1_weight,
            set2_weight=args.set2_weight,
            target_drop=args.sigma_drop,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            iterations=args.sigma_search_iters,
            trials=args.sigma_trials,
            tolerance=args.sigma_tolerance,
            batch_size=args.batch_size,
        )
    else:
        sigma = float(args.fixed_sigma)
        sigma_zero_set2 = float("nan")
        achieved_set2 = float("nan")
    write_rows(
        output_dir / "sigma_calibration.csv",
        [{
            "sigma": sigma,
            "sigma_drop_target": args.sigma_drop,
            "sigma_zero_validation_set2": sigma_zero_set2,
            "selected_sigma_validation_set2": achieved_set2,
            "fixed_sigma": int(args.fixed_sigma is not None),
        }],
    )
    print(
        "Selected preprocessing: "
        f"alpha={args.strength_alpha:g}, steps={args.strength_steps}, "
        f"top1_weight={args.top1_weight:g}, set2_weight={args.set2_weight:g}, "
        f"sigma={sigma:.8f}"
    )

    proposed_actual = TargetedGaussianDefense(
        standard_model,
        robust_model,
        robust_model,
        args.strength_alpha,
        args.strength_epsilon,
        args.strength_steps,
        sigma,
        args.top1_weight,
        args.set2_weight,
    )
    hybrid_actual = TargetedGaussianDefense(
        standard_model,
        robust_model,
        dual_model,
        args.strength_alpha,
        args.strength_epsilon,
        args.strength_steps,
        sigma,
        args.top1_weight,
        args.set2_weight,
    )
    all_defenses: Dict[str, DefenseSpec] = {
        "standard": standard_spec,
        "robust": DefenseSpec(
            "robust", make_actual(robust_model), robust_model, 0.0
        ),
        "dual": DefenseSpec(
            "dual", make_actual(dual_model), dual_model, 0.0
        ),
        "proposed": DefenseSpec(
            "proposed", proposed_actual, robust_model, sigma
        ),
        "hybrid": DefenseSpec(
            "hybrid", hybrid_actual, dual_model, sigma
        ),
    }
    defenses = {name: all_defenses[name] for name in args.defenses}

    clean_rows: List[Dict[str, object]] = []
    for name, defense in defenses.items():
        trials = args.eval_mc if defense.sigma > 0 else 1
        logits = compute_logits_batched(
            defense, clean_images, args.batch_size, trials
        )
        metrics = legacy.all_metrics(
            logits, clean_labels.cpu(), clean_reference_logits
        )
        row = {
            "defense": name,
            "top1": metrics["top1"],
            "set2": metrics["set2"],
            "examples": len(clean_images),
            "mc_trials": trials,
            "sigma": defense.sigma,
        }
        clean_rows.append(row)
        print(
            f"Clean {name:>8s} | T1={metrics['top1']:.2f}% "
            f"S2={metrics['set2']:.2f}%"
        )
    write_rows(output_dir / "clean_results.csv", clean_rows)

    if args.clean_only:
        write_rows(output_dir / "comparison_table.csv", comparison_table(clean_rows, []))
        print("Clean-only evaluation complete. Output:", output_dir)
        return

    existing_rows: List[Dict[str, object]] = list(read_rows(attack_path))
    completed = {attack_key(row) for row in existing_rows}
    for name, defense in defenses.items():
        for epsilon_px in args.attack_epsilons_px:
            epsilon = epsilon_px / 255.0
            alpha_px = min(args.pgd_alpha_cap_px, epsilon_px / 4.0)
            alpha = alpha_px / 255.0
            for attack_name in args.attacks:
                budgets: Iterable[int]
                budgets = [args.pgd_steps] if attack_name == "PGD" else args.budgets
                for goal in args.attack_goals:
                    for budget in budgets:
                        key = (name, attack_name, goal, str(epsilon_px), str(budget))
                        if key in completed:
                            print("Skipping completed:", key)
                            continue
                        print(
                            f"Attack defense={name} attack={attack_name} "
                            f"goal={goal} eps={epsilon_px:g}/255 "
                            f"budget/steps={budget}",
                            flush=True,
                        )
                        if attack_name == "PGD":
                            metrics = evaluate_pgd(
                                defense,
                                attack_images,
                                attack_labels,
                                attack_reference_logits,
                                epsilon,
                                alpha,
                                args.pgd_steps,
                                args.pgd_restarts,
                                args.batch_size,
                                args.attack_eot,
                                args.eval_mc,
                                goal,
                            )
                        else:
                            metrics = evaluate_blackbox(
                                defense,
                                attack_name,
                                attack_images,
                                attack_labels,
                                attack_reference_logits,
                                epsilon,
                                budget,
                                args.batch_size,
                                args.attack_eot,
                                args.eval_mc,
                                goal,
                            )
                        row = {
                            "defense": name,
                            "attack": attack_name,
                            "attack_goal": goal,
                            "epsilon_px": epsilon_px,
                            "epsilon": epsilon,
                            "budget_or_steps": budget,
                            "pgd_restarts": args.pgd_restarts if attack_name == "PGD" else "",
                            "pgd_alpha_px": alpha_px if attack_name == "PGD" else "",
                            "attack_eot": args.attack_eot,
                            "eval_mc": args.eval_mc,
                            "effective_queries": (
                                budget * args.attack_eot
                                if attack_name != "PGD" else ""
                            ),
                            "examples": len(attack_images),
                            "top1": metrics["top1"],
                            "set2": metrics["set2"],
                        }
                        append_row(attack_path, row)
                        existing_rows.append(row)
                        completed.add(key)
                        print(
                            f"  result T1={metrics['top1']:.2f}% "
                            f"S2={metrics['set2']:.2f}%",
                            flush=True,
                        )

    write_rows(
        output_dir / "comparison_table.csv",
        comparison_table(clean_rows, existing_rows),
    )
    summary = {
        "standard_weights": str(args.standard_weights),
        "dual_weights": str(args.dual_weights),
        "strength": {
            "alpha": args.strength_alpha,
            "epsilon": args.strength_epsilon,
            "steps": args.strength_steps,
            "top1_weight": args.top1_weight,
            "set2_weight": args.set2_weight,
        },
        "sigma": sigma,
        "clean_examples": len(clean_images),
        "attack_examples": len(attack_images),
        "attack_note": (
            "PGD uses BPDA+EOT for targeted/Gaussian preprocessing; Square and "
            "SignHunt query the actual stochastic defense. Top1 and exact Set2 "
            "are attacked separately."
        ),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print("Evaluation complete. Output directory:", output_dir)


if __name__ == "__main__":
    main()
