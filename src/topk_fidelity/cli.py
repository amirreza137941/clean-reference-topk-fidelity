"""CLI for configuration inspection, CPU examples and existing CIFAR weights."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch

from .attacks import run_square_batch
from .config import ExperimentConfig
from .defense import CleanReferenceTopKDefense
from .demo import run_demo
from .evaluation import evaluate_responses
from .metrics import ranked_topk
from .models import load_cifar10_models, load_cifar100_models
from .randomness import seed_everything, stable_int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser(
        "show-config", help="validate and print experiment JSON"
    )
    inspect.add_argument("--config", type=Path, required=True)
    demo = commands.add_parser(
        "demo", help="run the real algorithms on tiny CPU models"
    )
    demo.add_argument(
        "--attack", choices=("SquareAdaptive", "SquareRacing"), default="SquareAdaptive"
    )
    demo.add_argument("--seed", type=int, default=7)
    evaluate = commands.add_parser(
        "evaluate-tensors",
        help="evaluate supplied CIFAR tensors and original checkpoints",
    )
    evaluate.add_argument("--config", type=Path, required=True)
    evaluate.add_argument(
        "--inputs",
        type=Path,
        required=True,
        help="PT mapping with images, labels, image_ids",
    )
    evaluate.add_argument("--reference-checkpoint", type=Path, required=True)
    evaluate.add_argument(
        "--anchor-checkpoint",
        type=Path,
        required=True,
        help="dual checkpoint for CIFAR-10; teacher-top2 anchor for CIFAR-100",
    )
    evaluate.add_argument(
        "--method",
        choices=("robust", "full_sigma0", "noise_only", "full"),
        default="full",
    )
    evaluate.add_argument(
        "--attack", choices=("SquareAdaptive", "SquareRacing"), required=True
    )
    evaluate.add_argument("--seed", type=int, required=True)
    evaluate.add_argument("--batch-size", type=int, default=4)
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--output-dir", type=Path, required=True)
    return parser


def _evaluate(args: argparse.Namespace) -> None:
    config = ExperimentConfig.from_file(args.config)
    if config.dataset not in {"cifar10", "cifar100"}:
        raise ValueError(
            "evaluate-tensors supports CIFAR checkpoints; use the model API or original ImageNet stages"
        )
    if args.seed not in config.search_seeds:
        raise ValueError("seed must be one of the configured search_seeds")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            "output directory must be empty; previous run files will not be overwritten"
        )
    payload = torch.load(args.inputs, map_location="cpu", weights_only=True)
    images, labels, image_ids = (
        payload["images"],
        payload["labels"],
        payload["image_ids"],
    )
    if (
        images.ndim != 4
        or tuple(images.shape[1:]) != (3, 32, 32)
        or images.dtype != torch.float32
        or not torch.isfinite(images).all()
        or torch.any(images < 0)
        or torch.any(images > 1)
        or len(images) == 0
    ):
        raise ValueError(
            "images must be nonempty finite float32 [N,3,32,32] pixels in [0,1]"
        )
    classes = 10 if config.dataset == "cifar10" else 100
    if (
        labels.shape != (len(images),)
        or labels.dtype != torch.long
        or torch.any(labels < 0)
        or torch.any(labels >= classes)
    ):
        raise ValueError("labels must be valid int64 class indices shaped [N]")
    image_ids = [str(value) for value in image_ids]
    if len(image_ids) != len(images) or len(set(image_ids)) != len(image_ids):
        raise ValueError("image_ids must be unique and match the image count")
    loader = (
        load_cifar10_models if config.dataset == "cifar10" else load_cifar100_models
    )
    reference, anchor = loader(
        args.reference_checkpoint, args.anchor_checkpoint, args.device
    )
    defense_config = config.defense.for_method(args.method)
    defense = CleanReferenceTopKDefense(reference, anchor, anchor, defense_config)
    adaptive = config.adaptive_for(defense_config.sigma > 0)
    rows: list[dict[str, object]] = []
    batch_details: list[dict[str, object]] = []

    for batch_number, start in enumerate(
        range(0, len(images), args.batch_size), start=1
    ):
        end = min(start + args.batch_size, len(images))
        batch = images[start:end].to(args.device)
        targets = labels[start:end].to(args.device)
        with torch.no_grad():
            clean_reference = reference(batch)
        candidate, details = run_square_batch(
            attack=args.attack,
            defense=defense,
            clean=batch,
            reference_topk=ranked_topk(clean_reference, config.defense.top_k),
            eps=config.attack_epsilon,
            api_budget=config.api_budget,
            p_init=config.p_init,
            adaptive_config=adaptive.as_dict(),
            racing_config=config.racing.as_dict(),
            proposal_seed=stable_int(
                f"{args.seed}|{args.attack}|proposal|batch={batch_number}|q={config.api_budget}"
            ),
            oracle_seed=stable_int(
                f"{args.seed}|{args.attack}|oracle|batch={batch_number}|q={config.api_budget}"
            ),
            mean_logit_reduction=config.mean_logit_reduction,
        )
        # Separate evaluation RNG; endpoint values never select a candidate.
        seed_everything(stable_int(f"{args.seed}|endpoint|batch={batch_number}"))
        draws = config.reevaluation_draws if defense_config.sigma > 0 else 1
        values = evaluate_responses(
            defense,
            candidate,
            targets,
            clean_reference,
            k=config.defense.top_k,
            draws=draws,
        )
        for offset in range(len(batch)):
            rows.append(
                {
                    "image_id": image_ids[start + offset],
                    "method": args.method,
                    "attack": args.attack,
                    "seed": args.seed,
                    "api_queries": details["api_queries"],
                    "endpoint_draws": draws,
                    **{name: float(value[offset]) for name, value in values.items()},
                }
            )
        batch_details.append(details)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_image.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "purpose": "user-supplied tensor evaluation; original split/audit not inferred",
        "package_version": "0.2.0",
        "config": config.as_dict(),
        "effective_defense": vars(defense_config),
        "effective_adaptive": adaptive.as_dict(),
        "method": args.method,
        "attack": args.attack,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "examples": len(images),
        "inputs_sha256": _sha256(args.inputs),
        "reference_checkpoint_sha256": _sha256(args.reference_checkpoint),
        "anchor_checkpoint_sha256": _sha256(args.anchor_checkpoint),
        "endpoint_evaluation_outside_attack_budget": True,
        "batches": batch_details,
    }
    (args.output_dir / "run.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output_dir), "examples": len(rows)}, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "show-config":
            print(
                json.dumps(ExperimentConfig.from_file(args.config).as_dict(), indent=2)
            )
        elif args.command == "demo":
            print(json.dumps(run_demo(args.attack, args.seed), indent=2))
        else:
            _evaluate(args)
    except (
        ValueError,
        TypeError,
        KeyError,
        FileNotFoundError,
        FileExistsError,
        RuntimeError,
    ) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0
