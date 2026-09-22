#!/usr/bin/env python3
"""Independent common-random-number reevaluation for one D1 method."""

from __future__ import annotations

import argparse
import contextlib
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

import calibrate_topk_imagenet_defense_r2 as r2
import topk_cifar100_square_attack_core_r4 as core
import topk_imagenet_posthoc_q20k_eot_d1_common as common
from prepare_topk_imagenet_posthoc_q20k_eot_d1 import validate_input_payload
from search_topk_imagenet_posthoc_q20k_eot_d1 import (
    FrozenDefense,
    atomic_torch_save,
    candidate_identity,
    validate_candidate_payload,
)


METRICS = (
    "top1_accuracy_probability",
    "reference_top1_match_probability",
    "label_coverage_k5_probability",
    "exact_set_k5_probability",
    "exact_set_k2_probability",
    "mean_logits_top1_accuracy",
    "mean_logits_exact_set_k5",
    "mean_logits_exact_set_k2",
    "mean_boundary_margin_k5",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-json", type=Path, required=True)
    parser.add_argument("--method", choices=common.METHODS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=common.REEVAL_BATCH_SIZE)
    parser.add_argument("--mc-chunk", type=int, default=common.REEVAL_MC_CHUNK)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.45)
    return parser.parse_args()


def autocast_off(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", enabled=False)
    return contextlib.nullcontext()


def boundary_margin(logits: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    selected_min = logits.gather(1, reference).min(1).values
    outside = torch.ones_like(logits, dtype=torch.bool)
    outside.scatter_(1, reference, False)
    strongest_outside = logits.masked_fill(~outside, -torch.inf).max(1).values
    return selected_min - strongest_outside


def evaluation_hash_input(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        key: value
        for key, value in payload.items()
        if key not in {"values", "evaluation_hash"}
    }
    result["value_sha256"] = {
        key: common.tensor_sha256(payload["values"][key])
        for key in sorted(payload["values"])
    }
    return result


def add_evaluation_hash(payload: dict[str, Any]) -> None:
    payload["value_sha256"] = {
        key: common.tensor_sha256(payload["values"][key])
        for key in sorted(payload["values"])
    }
    payload["evaluation_hash"] = common.object_sha256(evaluation_hash_input(payload))


def validate_evaluation(
    payload: Mapping[str, Any], expected: Mapping[str, Any]
) -> None:
    for key, value in expected.items():
        if payload.get(key) != value:
            raise AssertionError(f"D1 evaluation identity mismatch: {key}")
    values = payload.get("values", {})
    if set(values) != set(METRICS):
        raise AssertionError("D1 evaluation metric set changed")
    examples = int(payload.get("examples", -1))
    if examples != common.DIAGNOSTIC_EXAMPLES:
        raise AssertionError("D1 evaluation example count changed")
    for key in METRICS:
        tensor = values[key].detach().cpu()
        if tensor.shape != (examples,) or not torch.isfinite(tensor).all():
            raise AssertionError(f"Invalid D1 evaluation tensor: {key}")
        if key != "mean_boundary_margin_k5" and (
            float(tensor.min()) < -1e-12 or float(tensor.max()) > 1.0 + 1e-12
        ):
            raise AssertionError(f"D1 probability outside [0,1]: {key}")
        if common.tensor_sha256(tensor) != payload.get("value_sha256", {}).get(key):
            raise AssertionError(f"D1 evaluation tensor hash mismatch: {key}")
    if common.object_sha256(evaluation_hash_input(payload)) != payload.get(
        "evaluation_hash"
    ):
        raise AssertionError("D1 evaluation payload hash mismatch")


def evaluate_images(
    *,
    defense: FrozenDefense,
    images_cpu: torch.Tensor,
    labels_cpu: torch.Tensor,
    reference_cpu: torch.Tensor,
    device: torch.device,
    batch_size: int,
    mc_chunk: int,
    draws: int,
    random_seed: int,
) -> dict[str, torch.Tensor]:
    examples = len(images_cpu)
    stores = {key: torch.zeros(examples, dtype=torch.float64) for key in METRICS}
    generator = torch.Generator(device=device)
    generator.manual_seed(int(random_seed))
    for low in range(0, examples, batch_size):
        high = min(low + batch_size, examples)
        images = images_cpu[low:high].to(device, non_blocking=True)
        labels = labels_cpu[low:high].to(device, non_blocking=True)
        reference = reference_cpu[low:high].to(device, non_blocking=True)
        ref5_set = reference.sort(1).values
        ref2_set = reference[:, : common.SECONDARY_K].sort(1).values
        base = defense.base(images)
        counts = {
            key: torch.zeros(len(images), dtype=torch.float64, device=device)
            for key in ("top1", "reference_top1", "coverage5", "exact5", "exact2")
        }
        logit_sum = torch.zeros(
            len(images), 1000, dtype=torch.float64, device=device
        )
        completed = 0
        while completed < draws:
            chunk = min(mc_chunk, draws - completed)
            if defense.sigma > 0:
                noise = torch.randn(
                    (chunk,) + tuple(base.shape),
                    device=device,
                    dtype=base.dtype,
                    generator=generator,
                )
                queried = (base.unsqueeze(0) + defense.sigma * noise).clamp(0.0, 1.0)
                with torch.no_grad(), autocast_off(device):
                    logits = (
                        defense.robust(queried.flatten(0, 1))
                        .float()
                        .view(chunk, len(images), 1000)
                    )
            else:
                if draws != 1:
                    raise AssertionError(
                        "Deterministic D1 methods must use one reevaluation draw"
                    )
                with torch.no_grad(), autocast_off(device):
                    logits = defense.robust(base).float().unsqueeze(0)
            predictions = logits.argmax(2)
            set5 = (
                logits.topk(common.PRIMARY_K, dim=2, sorted=False)
                .indices.sort(2)
                .values
            )
            set2 = (
                logits.topk(common.SECONDARY_K, dim=2, sorted=False)
                .indices.sort(2)
                .values
            )
            counts["top1"] += predictions.eq(labels[None]).sum(0).double()
            counts["reference_top1"] += (
                predictions.eq(reference[None, :, 0]).sum(0).double()
            )
            counts["coverage5"] += (
                set5.eq(labels[None, :, None]).any(2).sum(0).double()
            )
            counts["exact5"] += set5.eq(ref5_set[None]).all(2).sum(0).double()
            counts["exact2"] += set2.eq(ref2_set[None]).all(2).sum(0).double()
            logit_sum += logits.double().sum(0)
            completed += chunk
        mean_logits = (logit_sum / float(draws)).float()
        mean5 = (
            mean_logits.topk(common.PRIMARY_K, dim=1, sorted=False)
            .indices.sort(1)
            .values
        )
        mean2 = (
            mean_logits.topk(common.SECONDARY_K, dim=1, sorted=False)
            .indices.sort(1)
            .values
        )
        stores["top1_accuracy_probability"][low:high] = (
            counts["top1"] / draws
        ).cpu()
        stores["reference_top1_match_probability"][low:high] = (
            counts["reference_top1"] / draws
        ).cpu()
        stores["label_coverage_k5_probability"][low:high] = (
            counts["coverage5"] / draws
        ).cpu()
        stores["exact_set_k5_probability"][low:high] = (
            counts["exact5"] / draws
        ).cpu()
        stores["exact_set_k2_probability"][low:high] = (
            counts["exact2"] / draws
        ).cpu()
        stores["mean_logits_top1_accuracy"][low:high] = (
            mean_logits.argmax(1).eq(labels).double().cpu()
        )
        stores["mean_logits_exact_set_k5"][low:high] = (
            mean5.eq(ref5_set).all(1).double().cpu()
        )
        stores["mean_logits_exact_set_k2"][low:high] = (
            mean2.eq(ref2_set).all(1).double().cpu()
        )
        stores["mean_boundary_margin_k5"][low:high] = (
            boundary_margin(mean_logits, reference).double().cpu()
        )
        print(f"      reevaluation {high}/{examples}, draws={draws}", flush=True)
    return stores


def candidate_records(
    protocol: Mapping[str, Any],
    inputs: Mapping[str, Any],
    method: str,
) -> list[dict[str, Any]]:
    root = Path(protocol["paths"]["output_root"])
    records: list[dict[str, Any]] = []
    shared: dict[str, Any] | None = None
    for seed in common.SEARCH_SEEDS:
        directory = common.cell_dir(root, method, seed)
        candidate_path = directory / "candidate.pt"
        summary_path = directory / "search_summary.json"
        rows_path = directory / "search_per_example.csv"
        if not all(path.is_file() for path in (candidate_path, summary_path, rows_path)):
            raise FileNotFoundError(f"Incomplete D1 search cell: {directory}")
        payload = torch.load(candidate_path, map_location="cpu", weights_only=False)
        identity = candidate_identity(
            protocol,
            method,
            seed,
            str(inputs["payload_hash"]),
            str(inputs["tensor_sha256"]["clean"]),
        )
        validate_candidate_payload(
            payload,
            identity,
            list(range(common.DIAGNOSTIC_EXAMPLES)),
            inputs["source_confirmatory_positions"].tolist(),
            inputs["relative_paths"],
            inputs["labels"],
            inputs["reference_top5"],
            clean=inputs["clean"],
        )
        summary = common.read_json(summary_path)
        file_hash = common.file_sha256(candidate_path)
        if (
            summary.get("status") != "PASS"
            or summary.get("candidate_file_sha256") != file_hash
            or summary.get("candidate_payload_hash") != payload["payload_hash"]
            or summary.get("search_per_example_sha256")
            != common.file_sha256(rows_path)
        ):
            raise AssertionError(f"D1 search summary mismatch: {directory}")
        anchors = {
            key: payload["tensor_sha256"][key]
            for key in (
                "labels",
                "diagnostic_positions",
                "source_confirmatory_positions",
                "reference_top5",
            )
        }
        anchors["relative_paths_sha256"] = common.object_sha256(
            payload["relative_paths"]
        )
        if shared is None:
            shared = anchors
        elif anchors != shared:
            raise AssertionError(f"D1 {method} seeds do not share anchors")
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
        validate_evaluation(payload, expected)
        if int(payload.get("reeval_seed", -1)) != common.REEVAL_SEED:
            raise AssertionError("D1 cached reevaluation seed changed")
        print(f"    RESUME {path.name}", flush=True)
        return dict(payload)
    started = time.perf_counter()
    payload: dict[str, Any] = {
        **expected,
        "examples": len(images),
        "draws": int(draws),
        "reeval_seed": common.REEVAL_SEED,
        "common_random_numbers": True,
        "values": evaluate_images(
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
    add_evaluation_hash(payload)
    validate_evaluation(payload, expected)
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
        "search_eot": common.ADAPTIVE_CONFIG_BY_METHOD[record["method"]][
            "search_eot"
        ],
        "candidate_file": str(record["candidate_path"]),
        "candidate_file_sha256": record["candidate_file_sha256"],
        "candidate_tensor_sha256": record["payload"]["tensor_sha256"][
            "adversarial"
        ],
        "post_hoc": True,
        "paper_claim_authorized": False,
    }
    for metric in METRICS:
        suffix = "" if metric.startswith("mean_") else "_percent"
        scale = 1.0 if metric == "mean_boundary_margin_k5" else 100.0
        row[f"attacked_{metric}{suffix}"] = scale * float(
            attacked["values"][metric].mean()
        )
        row[f"clean_{metric}{suffix}"] = scale * float(
            clean["values"][metric].mean()
        )
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
        for position in range(common.DIAGNOSTIC_EXAMPLES):
            reference = [
                int(value) for value in payload["reference_top5"][position].tolist()
            ]
            row: dict[str, Any] = {
                "protocol_id": common.PROTOCOL_ID,
                "protocol_hash": evaluation["protocol_hash"],
                "method": record["method"],
                "attack": record["attack"],
                "search_seed": record["search_seed"],
                "diagnostic_position": position,
                "source_confirmatory_position": int(
                    payload["source_confirmatory_positions"][position]
                ),
                "relative_path": payload["relative_paths"][position],
                "label": int(payload["labels"][position]),
                "reference_top1": reference[0],
                "reference_set_k5": "|".join(
                    str(value) for value in sorted(reference)
                ),
                "eval_draws": evaluation["draws"],
                "reeval_seed": common.REEVAL_SEED,
                "api_budget": common.API_BUDGET,
                "epsilon_px": common.EPSILON_PX,
                "search_eot": common.ADAPTIVE_CONFIG_BY_METHOD[record["method"]][
                    "search_eot"
                ],
                "candidate_file_sha256": record["candidate_file_sha256"],
                "candidate_tensor_sha256": payload["tensor_sha256"]["adversarial"],
                "post_hoc": True,
                "paper_claim_authorized": False,
            }
            for metric in METRICS:
                row[f"attacked_{metric}"] = float(
                    evaluation["values"][metric][position]
                )
                row[f"clean_{metric}"] = float(clean["values"][metric][position])
            rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    if args.batch_size != common.REEVAL_BATCH_SIZE:
        raise ValueError(f"D1 requires --batch-size {common.REEVAL_BATCH_SIZE}")
    if args.mc_chunk != common.REEVAL_MC_CHUNK:
        raise ValueError(f"D1 requires --mc-chunk {common.REEVAL_MC_CHUNK}")
    if not 0 < args.gpu_memory_fraction <= 1:
        raise ValueError("--gpu-memory-fraction must be in (0,1]")
    protocol_path = args.protocol_json.expanduser().resolve()
    protocol = common.read_json(protocol_path)
    protocol_hash = common.validate_protocol(protocol)
    output_root = Path(protocol["paths"]["output_root"]).expanduser().resolve()
    if protocol_path != output_root / "diagnostic_protocol.json":
        raise AssertionError("D1 protocol path is not canonical")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir != common.method_reeval_dir(output_root, args.method).resolve():
        raise AssertionError("D1 method reevaluation path is not frozen")

    parent = common.load_parent_r5(Path(protocol["paths"]["r5_root"]))
    input_path = output_root / "diagnostic_inputs.pt"
    inputs = torch.load(input_path, map_location="cpu", weights_only=False)
    validate_input_payload(inputs, protocol, parent["subset"])
    records = candidate_records(protocol, inputs, args.method)

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(
            args.gpu_memory_fraction, device=device
        )
    core.seed_everything(common.REEVAL_SEED)
    labels = inputs["labels"].detach().cpu()
    reference = inputs["reference_top5"].detach().cpu()
    clean_images = inputs["clean"].detach().cpu()
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
        "source_diagnostic_payload_hash": inputs["payload_hash"],
        "split": "fixed_r5_posthoc_subset",
        "post_hoc": True,
        "paper_claim_authorized": False,
    }
    print(f"D1 clean reevaluation {args.method}, draws={draws}", flush=True)
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
            "candidate_tensor_sha256": record["payload"]["tensor_sha256"][
                "adversarial"
            ],
            "source_diagnostic_payload_hash": inputs["payload_hash"],
            "split": "fixed_r5_posthoc_subset",
            "post_hoc": True,
            "paper_claim_authorized": False,
        }
        print(
            f"[{number}/{len(records)}] D1 {args.method}/"
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
        "examples": common.DIAGNOSTIC_EXAMPLES,
        "per_example_rows": len(rows),
        "eval_draws": draws,
        "reeval_seed": common.REEVAL_SEED,
        "common_random_numbers": True,
        "candidate_reeval_per_example_sha256": common.file_sha256(rows_path),
        "candidate_reeval_summary_sha256": common.file_sha256(summaries_path),
        "post_hoc": True,
        "paper_claim_authorized": False,
        "parent_r5_modified": False,
        "no_attack_family_selection": True,
        "no_seed_selection": True,
        "no_per_example_candidate_selection": True,
    }
    final["method_summary_hash"] = common.object_sha256(final)
    common.atomic_json(output_dir / "method_summary.json", final)
    exact_values = [
        float(row["attacked_exact_set_k5_probability_percent"])
        for row in summaries
    ]
    print(
        f"PASS D1 reevaluation {args.method}: "
        + ",".join(f"{value:.3f}%" for value in exact_values),
        flush=True,
    )
    print("PAPER_CLAIM_AUTHORIZED=NO (post-hoc diagnostic)", flush=True)


if __name__ == "__main__":
    main()
