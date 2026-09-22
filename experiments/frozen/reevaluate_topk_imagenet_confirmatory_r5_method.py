#!/usr/bin/env python3
"""Independent common-MC=1000 reevaluation for one ImageNet R5 method."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

import calibrate_topk_imagenet_defense_r2 as r2
import reevaluate_topk_imagenet_attack_pilot_r4_mc1000 as r4_eval
import topk_imagenet_confirmatory_r5_common as common
import topk_imagenet_square_attack_core_r4 as core
from prepare_topk_imagenet_confirmatory_r5 import validate_input_payload
from search_topk_imagenet_attack_pilot_r4 import FrozenDefense
from search_topk_imagenet_confirmatory_r5 import (
    atomic_torch_save,
    candidate_identity,
    validate_candidate_payload,
)


METRICS = r4_eval.METRICS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-json", type=Path, required=True)
    parser.add_argument("--method", choices=common.METHODS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=common.REEVAL_BATCH_SIZE)
    parser.add_argument("--mc-chunk", type=int, default=common.REEVAL_MC_CHUNK)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.45)
    return parser.parse_args()


def load_clean_images(
    protocol: Mapping[str, Any],
    manifest_rows: Sequence[Mapping[str, Any]],
    input_payload: Mapping[str, Any],
    device: torch.device,
    num_workers: int,
) -> torch.Tensor:
    loader = r2.make_loader(
        Path(protocol["paths"]["data_root"]),
        manifest_rows,
        common.REFERENCE_BATCH_SIZE,
        num_workers,
        device,
    )
    images_all: list[torch.Tensor] = []
    labels_all: list[torch.Tensor] = []
    paths_all: list[str] = []
    for images, labels, paths in loader:
        images_all.append(images.detach().cpu().contiguous())
        labels_all.append(labels.detach().cpu().contiguous())
        paths_all.extend(str(path) for path in paths)
    images = torch.cat(images_all)
    labels = torch.cat(labels_all)
    if images.shape != (common.CONFIRMATORY_EXAMPLES, 3, 224, 224):
        raise AssertionError("ImageNet R5 clean image shape changed")
    if paths_all != list(input_payload["relative_paths"]):
        raise AssertionError("ImageNet R5 reevaluation path order changed")
    if not torch.equal(labels, input_payload["labels"]):
        raise AssertionError("ImageNet R5 reevaluation labels changed")
    if not torch.isfinite(images).all() or float(images.min()) < 0 or float(images.max()) > 1:
        raise AssertionError("ImageNet R5 clean images are invalid")
    if common.tensor_sha256(images) != input_payload["clean_tensor_sha256"]:
        raise AssertionError("ImageNet R5 clean image tensor changed before reevaluation")
    return images


def candidate_records(
    protocol: Mapping[str, Any],
    input_payload: Mapping[str, Any],
    method: str,
) -> list[dict[str, Any]]:
    root = Path(protocol["paths"]["output_root"])
    positions = list(range(common.CONFIRMATORY_EXAMPLES))
    paths = list(input_payload["relative_paths"])
    records: list[dict[str, Any]] = []
    shared: dict[str, str] | None = None
    for seed in common.SEARCH_SEEDS:
        directory = common.cell_dir(root, method, seed)
        candidate_path = directory / "candidate.pt"
        summary_path = directory / "search_summary.json"
        if not candidate_path.is_file() or not summary_path.is_file():
            raise FileNotFoundError(f"Incomplete ImageNet R5 search cell: {directory}")
        payload = torch.load(candidate_path, map_location="cpu", weights_only=False)
        expected = candidate_identity(
            protocol,
            method,
            seed,
            str(input_payload["payload_hash"]),
            str(input_payload["clean_tensor_sha256"]),
        )
        validate_candidate_payload(
            payload,
            expected,
            positions,
            paths,
            input_payload["labels"],
            input_payload["reference_top5"],
        )
        summary = common.read_json(summary_path)
        file_hash = common.file_sha256(candidate_path)
        if summary.get("status") != "PASS":
            raise AssertionError(f"ImageNet R5 search did not pass: {directory}")
        if summary.get("candidate_file_sha256") != file_hash:
            raise AssertionError(f"ImageNet R5 candidate file hash mismatch: {directory}")
        if summary.get("candidate_payload_hash") != payload["payload_hash"]:
            raise AssertionError(f"ImageNet R5 candidate payload hash mismatch: {directory}")
        anchors = {
            key: payload["tensor_sha256"][key]
            for key in ("labels", "confirmatory_positions", "reference_top5")
        }
        anchors["relative_paths_sha256"] = common.object_sha256(payload["relative_paths"])
        if shared is None:
            shared = anchors
        elif anchors != shared:
            raise AssertionError(f"ImageNet R5 {method} seeds do not share anchors")
        records.append(
            {
                "method": method,
                "attack": common.METHOD_ATTACK[method],
                "search_seed": seed,
                "candidate_path": candidate_path,
                "candidate_file_sha256": file_hash,
                "payload": payload,
            }
        )
    return records


def cache_or_evaluate(
    *,
    path: Path,
    expected: Mapping[str, Any],
    defense: FrozenDefense,
    images: torch.Tensor,
    labels: torch.Tensor,
    reference: torch.Tensor,
    device: torch.device,
    args: argparse.Namespace,
    draws: int,
) -> dict[str, Any]:
    if path.is_file():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        r4_eval.validate_evaluation(payload, expected)
        if int(payload.get("reeval_seed", -1)) != common.REEVAL_SEED:
            raise AssertionError("ImageNet R5 cached reevaluation seed changed")
        print(f"    RESUME {path.name}", flush=True)
        return dict(payload)
    started = time.perf_counter()
    payload: dict[str, Any] = {
        **expected,
        "examples": len(images),
        "draws": int(draws),
        "reeval_seed": common.REEVAL_SEED,
        "values": r4_eval.evaluate_images(
            defense=defense,
            images_cpu=images,
            labels_cpu=labels,
            reference_cpu=reference,
            device=device,
            batch_size=args.batch_size,
            mc_chunk=args.mc_chunk,
            draws=draws,
            random_seed=common.REEVAL_SEED,
        ),
        "wall_seconds": time.perf_counter() - started,
    }
    r4_eval.add_evaluation_hash(payload)
    r4_eval.validate_evaluation(payload, expected)
    atomic_torch_save(path, payload)
    return payload


def summary_row(
    record: Mapping[str, Any],
    attacked: Mapping[str, Any],
    clean: Mapping[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "protocol_id": common.PROTOCOL_ID,
        "protocol_hash": attacked["protocol_hash"],
        "method": record["method"],
        "attack": record["attack"],
        "search_seed": record["search_seed"],
        "examples": attacked["examples"],
        "eval_draws": attacked["draws"],
        "reeval_seed": common.REEVAL_SEED,
        "api_budget": common.API_BUDGET,
        "epsilon_px": common.EPSILON_PX,
        "candidate_file": str(record["candidate_path"]),
        "candidate_file_sha256": record["candidate_file_sha256"],
        "candidate_tensor_sha256": record["payload"]["tensor_sha256"]["adversarial"],
    }
    for metric in METRICS:
        suffix = "" if metric.startswith("mean_") else "_percent"
        scale = 1.0 if metric == "mean_boundary_margin_k5" else 100.0
        row[f"attacked_{metric}{suffix}"] = scale * float(
            attacked["values"][metric].mean()
        )
        row[f"clean_{metric}{suffix}"] = scale * float(clean["values"][metric].mean())
    row["clean_to_attack_exact_set_k5_delta_pp"] = (
        row["attacked_exact_set_k5_probability_percent"]
        - row["clean_exact_set_k5_probability_percent"]
    )
    return row


def build_per_example_rows(
    records: Sequence[Mapping[str, Any]],
    evaluations: Mapping[int, Mapping[str, Any]],
    clean: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        evaluation = evaluations[int(record["search_seed"])]
        payload = record["payload"]
        for position in range(common.CONFIRMATORY_EXAMPLES):
            reference = [int(value) for value in payload["reference_top5"][position].tolist()]
            row: dict[str, Any] = {
                "protocol_id": common.PROTOCOL_ID,
                "protocol_hash": evaluation["protocol_hash"],
                "method": record["method"],
                "attack": record["attack"],
                "search_seed": record["search_seed"],
                "confirmatory_position": position,
                "relative_path": payload["relative_paths"][position],
                "label": int(payload["labels"][position]),
                "reference_top1": reference[0],
                "reference_set_k5": "|".join(str(value) for value in sorted(reference)),
                "eval_draws": evaluation["draws"],
                "reeval_seed": common.REEVAL_SEED,
                "api_budget": common.API_BUDGET,
                "epsilon_px": common.EPSILON_PX,
                "candidate_file_sha256": record["candidate_file_sha256"],
                "candidate_tensor_sha256": payload["tensor_sha256"]["adversarial"],
            }
            for metric in METRICS:
                row[f"attacked_{metric}"] = float(evaluation["values"][metric][position])
                row[f"clean_{metric}"] = float(clean["values"][metric][position])
            rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    if args.batch_size != common.REEVAL_BATCH_SIZE:
        raise ValueError(f"ImageNet R5 requires --batch-size {common.REEVAL_BATCH_SIZE}")
    if args.mc_chunk != common.REEVAL_MC_CHUNK:
        raise ValueError(f"ImageNet R5 requires --mc-chunk {common.REEVAL_MC_CHUNK}")
    if args.num_workers < 0 or not 0 < args.gpu_memory_fraction <= 1:
        raise ValueError("Invalid ImageNet R5 worker or GPU-memory setting")
    protocol = common.read_json(args.protocol_json.expanduser().resolve())
    protocol_hash = common.validate_protocol(protocol)
    output_root = Path(protocol["paths"]["output_root"]).expanduser().resolve()
    expected_output = common.method_reeval_dir(output_root, args.method).resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir != expected_output:
        raise AssertionError("ImageNet R5 method reevaluation path is not frozen")

    parent = common.load_parent(
        Path(protocol["paths"]["r0_dir"]),
        Path(protocol["paths"]["r2_dir"]),
        Path(protocol["paths"]["r3_dir"]),
    )
    r4_info = common.load_r4_freeze(Path(protocol["paths"]["r4_root"]), parent)
    if r4_info["selection_hash"] != protocol["r4_selection_hash"]:
        raise AssertionError("ImageNet R4 freeze changed before R5 reevaluation")
    manifest_rows = common.canonical_confirmatory_rows(parent)
    input_path = output_root / "confirmatory_inputs.pt"
    input_payload = torch.load(input_path, map_location="cpu", weights_only=False)
    validate_input_payload(input_payload, protocol, manifest_rows)
    records = candidate_records(protocol, input_payload, args.method)

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, device=device)
    core.seed_everything(common.REEVAL_SEED)
    clean_images = load_clean_images(
        protocol, manifest_rows, input_payload, device, args.num_workers
    )
    labels = input_payload["labels"].detach().cpu()
    reference = input_payload["reference_top5"].detach().cpu()
    standard, robust, _audit = r2.load_models(
        Path(protocol["checkpoints"]["standard"]["path"]),
        Path(protocol["checkpoints"]["robust"]["path"]),
        device,
    )
    spec = common.method_spec(args.method)
    defense = FrozenDefense(spec, standard, robust, device)
    draws = common.REEVAL_DRAWS if spec.stochastic else 1
    output_dir.mkdir(parents=True, exist_ok=True)

    clean_expected = {
        "protocol_id": common.PROTOCOL_ID,
        "protocol_hash": protocol_hash,
        "condition": "clean",
        "method": args.method,
        "clean_tensor_sha256": common.tensor_sha256(clean_images),
        "label_tensor_sha256": common.tensor_sha256(labels),
        "reference_tensor_sha256": common.tensor_sha256(reference),
        "source_confirmatory_payload_hash": input_payload["payload_hash"],
        "split": "attack_confirmatory",
    }
    print(f"ImageNet R5 clean reevaluation {args.method}, draws={draws}", flush=True)
    clean = cache_or_evaluate(
        path=output_dir / "cache" / "clean.pt",
        expected=clean_expected,
        defense=defense,
        images=clean_images,
        labels=labels,
        reference=reference,
        device=device,
        args=args,
        draws=draws,
    )

    evaluations: dict[int, dict[str, Any]] = {}
    summaries: list[dict[str, Any]] = []
    for number, record in enumerate(records, start=1):
        seed = int(record["search_seed"])
        expected = {
            "protocol_id": common.PROTOCOL_ID,
            "protocol_hash": protocol_hash,
            "condition": "attacked",
            "method": args.method,
            "attack": common.METHOD_ATTACK[args.method],
            "search_seed": seed,
            "candidate_file_sha256": record["candidate_file_sha256"],
            "candidate_tensor_sha256": record["payload"]["tensor_sha256"]["adversarial"],
            "source_confirmatory_payload_hash": input_payload["payload_hash"],
            "split": "attack_confirmatory",
        }
        print(
            f"[{number}/{len(records)}] ImageNet R5 {args.method}/"
            f"{record['attack']}/seed={seed}, draws={draws}",
            flush=True,
        )
        evaluation = cache_or_evaluate(
            path=output_dir / "cache" / f"attacked_seed_{seed}.pt",
            expected=expected,
            defense=defense,
            images=record["payload"]["adversarial"].detach().cpu(),
            labels=labels,
            reference=reference,
            device=device,
            args=args,
            draws=draws,
        )
        evaluations[seed] = evaluation
        summaries.append(summary_row(record, evaluation, clean))

    rows = build_per_example_rows(records, evaluations, clean)
    rows_path = output_dir / "candidate_reeval_per_example.csv"
    summaries_path = output_dir / "candidate_reeval_summary.csv"
    common.atomic_csv(rows_path, rows)
    common.atomic_csv(summaries_path, summaries)
    final: dict[str, Any] = {
        "protocol_id": common.PROTOCOL_ID,
        "protocol_hash": protocol_hash,
        "status": "PASS",
        "method": args.method,
        "attack": common.METHOD_ATTACK[args.method],
        "search_seeds": list(common.SEARCH_SEEDS),
        "examples": common.CONFIRMATORY_EXAMPLES,
        "per_example_rows": len(rows),
        "eval_draws": draws,
        "reeval_seed": common.REEVAL_SEED,
        "candidate_reeval_per_example_sha256": common.file_sha256(rows_path),
        "candidate_reeval_summary_sha256": common.file_sha256(summaries_path),
        "no_attack_family_selection": True,
        "no_seed_selection": True,
        "no_per_example_candidate_selection": True,
        "paper_claim_authorized_by_method_stage": False,
    }
    final["method_summary_hash"] = common.object_sha256(final)
    common.atomic_json(output_dir / "method_summary.json", final)
    exact_values = [
        float(row["attacked_exact_set_k5_probability_percent"]) for row in summaries
    ]
    print(
        f"PASS ImageNet R5 reevaluation {args.method}: "
        + ",".join(f"{value:.3f}%" for value in exact_values),
        flush=True,
    )


if __name__ == "__main__":
    main()
