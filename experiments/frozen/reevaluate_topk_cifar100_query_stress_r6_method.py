#!/usr/bin/env python3
"""Common-MC=1000 reevaluation for one frozen CIFAR-100 R6 method."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

import reevaluate_topk_cifar100_attack_pilot_r4_mc1000 as r4_eval
import topk_cifar100_query_stress_r6_common as common
import topk_cifar100_square_attack_core_r4 as core
from search_topk_cifar100_attack_pilot_r4 import FrozenDefense, atomic_torch_save, load_models
from search_topk_cifar100_query_stress_r6 import payload_identity, validate_payload


METRICS = r4_eval.METRICS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reevaluate one CIFAR-100 R6 method")
    parser.add_argument("--protocol-json", type=Path, required=True)
    parser.add_argument("--method", choices=common.METHODS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=common.REEVAL_BATCH_SIZE)
    parser.add_argument("--mc-chunk", type=int, default=common.REEVAL_MC_CHUNK)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.45)
    return parser.parse_args()


def candidate_records(
    protocol: Mapping[str, Any], stress_indices: Sequence[int], method: str
) -> list[dict[str, Any]]:
    output_root = Path(protocol["output_root"])
    attack = common.METHOD_ATTACK[method]
    records: list[dict[str, Any]] = []
    shared: dict[str, str] | None = None
    for budget in common.BUDGETS:
        for seed in common.SEARCH_SEEDS:
            directory = common.cell_dir(output_root, method, budget, seed)
            candidate_path = directory / "candidate.pt"
            summary_path = directory / "search_summary.json"
            if not candidate_path.is_file() or not summary_path.is_file():
                raise FileNotFoundError(f"R6 search cell is incomplete: {directory}")
            payload = torch.load(candidate_path, map_location="cpu", weights_only=False)
            identity = payload_identity(
                protocol_hash=str(protocol["protocol_hash"]),
                method=method,
                budget=budget,
                search_seed=seed,
            )
            validate_payload(payload, identity, stress_indices)
            summary = common.read_json(summary_path)
            candidate_file_hash = common.file_sha256(candidate_path)
            if summary.get("status") != "PASS":
                raise AssertionError(f"R6 search cell did not pass: {directory}")
            if summary.get("candidate_file_sha256") != candidate_file_hash:
                raise AssertionError(f"R6 candidate file hash mismatch: {candidate_path}")
            if summary.get("r5_final_summary_hash") != protocol["r5_final_summary_hash"]:
                raise AssertionError("R6 search cell points to a different R5 result")
            hashes = payload["tensor_sha256"]
            anchors = {
                key: hashes[key]
                for key in ("clean", "labels", "sample_indices", "reference_top2")
            }
            if shared is None:
                shared = anchors
            elif anchors != shared:
                raise AssertionError(f"R6 {method} cells do not share clean anchors")
            records.append(
                {
                    "method": method,
                    "attack": attack,
                    "api_budget": budget,
                    "search_seed": seed,
                    "candidate_path": candidate_path,
                    "candidate_file_sha256": candidate_file_hash,
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
    reference_top2: torch.Tensor,
    device: torch.device,
    args: argparse.Namespace,
    draws: int,
) -> dict[str, Any]:
    if path.is_file():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        r4_eval.validate_evaluation_payload(payload, expected)
        print(f"    RESUME {path.name}", flush=True)
        return dict(payload)
    started = time.monotonic()
    values = r4_eval.evaluate_images(
        defense=defense,
        images_cpu=images,
        labels_cpu=labels,
        reference_top2_cpu=reference_top2,
        device=device,
        batch_size=args.batch_size,
        mc_chunk=args.mc_chunk,
        draws=draws,
        random_seed=common.REEVAL_SEED,
    )
    payload: dict[str, Any] = {
        **expected,
        "examples": len(images),
        "draws": int(draws),
        "reeval_seed": common.REEVAL_SEED,
        "values": values,
        "wall_seconds": float(time.monotonic() - started),
    }
    r4_eval.add_evaluation_hash(payload)
    r4_eval.validate_evaluation_payload(payload, expected)
    atomic_torch_save(path, payload)
    return payload


def summary_row(
    record: Mapping[str, Any], evaluation: Mapping[str, Any], clean: Mapping[str, Any]
) -> dict[str, Any]:
    values = evaluation["values"]
    clean_values_map = clean["values"]
    row: dict[str, Any] = {
        "protocol_id": common.PROTOCOL_ID,
        "protocol_hash": evaluation["protocol_hash"],
        "method": record["method"],
        "attack": record["attack"],
        "api_budget": record["api_budget"],
        "search_seed": record["search_seed"],
        "examples": int(evaluation["examples"]),
        "eval_draws": int(evaluation["draws"]),
        "reeval_seed": common.REEVAL_SEED,
        "epsilon_px": common.EPSILON_PX,
        "candidate_file": str(record["candidate_path"]),
        "candidate_file_sha256": record["candidate_file_sha256"],
        "candidate_tensor_sha256": record["payload"]["tensor_sha256"]["adversarial"],
    }
    for metric in METRICS:
        attacked_key = (
            f"attacked_{metric}_percent"
            if metric != "mean_boundary_margin"
            else "attacked_mean_boundary_margin"
        )
        clean_key = (
            f"clean_{metric}_percent"
            if metric != "mean_boundary_margin"
            else "clean_mean_boundary_margin"
        )
        row[attacked_key] = (
            float(values[metric].mean())
            if metric == "mean_boundary_margin"
            else 100.0 * float(values[metric].mean())
        )
        row[clean_key] = (
            float(clean_values_map[metric].mean())
            if metric == "mean_boundary_margin"
            else 100.0 * float(clean_values_map[metric].mean())
        )
    row["clean_to_attack_exact_set_delta_pp"] = (
        float(row["attacked_exact_set_probability_percent"])
        - float(row["clean_exact_set_probability_percent"])
    )
    return row


def build_per_example_rows(
    records: Sequence[Mapping[str, Any]],
    evaluations: Mapping[tuple[int, int], Mapping[str, Any]],
    clean: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        budget = int(record["api_budget"])
        seed = int(record["search_seed"])
        evaluation = evaluations[(budget, seed)]
        payload = record["payload"]
        for position in range(int(evaluation["examples"])):
            reference = [int(value) for value in payload["reference_top2"][position].tolist()]
            row: dict[str, Any] = {
                "protocol_id": common.PROTOCOL_ID,
                "protocol_hash": evaluation["protocol_hash"],
                "method": record["method"],
                "attack": record["attack"],
                "api_budget": budget,
                "search_seed": seed,
                "sample_index": int(payload["sample_indices"][position]),
                "label": int(payload["labels"][position]),
                "reference_top1": reference[0],
                "reference_set": f"{min(reference)}|{max(reference)}",
                "eval_draws": int(evaluation["draws"]),
                "reeval_seed": common.REEVAL_SEED,
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
        raise ValueError(f"R6 requires --batch-size {common.REEVAL_BATCH_SIZE}")
    if args.mc_chunk != common.REEVAL_MC_CHUNK:
        raise ValueError(f"R6 requires --mc-chunk {common.REEVAL_MC_CHUNK}")
    if not 0 < args.gpu_memory_fraction <= 1:
        raise ValueError("GPU memory fraction must be in (0,1]")
    protocol = common.read_json(args.protocol_json.expanduser().resolve())
    protocol_hash = common.validate_protocol(protocol)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, device=device)
    core.seed_everything(common.REEVAL_SEED)

    common.load_r5_completion(Path(protocol["r5_root"]))
    parent = common.r5.load_parent_authorization(Path(protocol["r3_parent_dir"]))
    stress_indices = [int(value) for value in protocol["stress_indices"]]
    if stress_indices != common.derive_stress_indices(parent):
        raise AssertionError("R6 reevaluation stress split changed")
    records = candidate_records(protocol, stress_indices, args.method)
    first = records[0]["payload"]
    images = first["clean"].detach().cpu()
    labels = first["labels"].detach().cpu()
    reference_top2 = first["reference_top2"].detach().cpu()
    common.validate_stress_indices(stress_indices, parent)
    standard, anchor = load_models(protocol, device)
    spec = common.method_spec(args.method)
    defense = FrozenDefense(spec, standard, anchor, device)
    draws = common.REEVAL_DRAWS if spec.stochastic else 1
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    clean_expected = {
        "protocol_id": common.PROTOCOL_ID,
        "protocol_hash": protocol_hash,
        "condition": "clean",
        "method": args.method,
        "clean_tensor_sha256": first["tensor_sha256"]["clean"],
        "split": "query_stress_200_disjoint",
    }
    print(f"R6 clean reevaluation {args.method}, draws={draws}", flush=True)
    clean = cache_or_evaluate(
        path=output_dir / "cache" / "clean.pt",
        expected=clean_expected,
        defense=defense,
        images=images,
        labels=labels,
        reference_top2=reference_top2,
        device=device,
        args=args,
        draws=draws,
    )

    evaluations: dict[tuple[int, int], dict[str, Any]] = {}
    summaries: list[dict[str, Any]] = []
    for position, record in enumerate(records, start=1):
        budget = int(record["api_budget"])
        seed = int(record["search_seed"])
        expected = {
            "protocol_id": common.PROTOCOL_ID,
            "protocol_hash": protocol_hash,
            "condition": "attacked",
            "method": args.method,
            "attack": common.METHOD_ATTACK[args.method],
            "api_budget": budget,
            "search_seed": seed,
            "candidate_file_sha256": record["candidate_file_sha256"],
            "candidate_tensor_sha256": record["payload"]["tensor_sha256"]["adversarial"],
            "split": "query_stress_200_disjoint",
        }
        print(
            f"[{position}/{len(records)}] R6 {args.method}/{record['attack']}/"
            f"Q={budget}/seed={seed}, draws={draws}",
            flush=True,
        )
        evaluation = cache_or_evaluate(
            path=output_dir / "cache" / f"attacked_q{budget}_seed_{seed}.pt",
            expected=expected,
            defense=defense,
            images=record["payload"]["adversarial"].detach().cpu(),
            labels=labels,
            reference_top2=reference_top2,
            device=device,
            args=args,
            draws=draws,
        )
        evaluations[(budget, seed)] = evaluation
        summaries.append(summary_row(record, evaluation, clean))

    rows = build_per_example_rows(records, evaluations, clean)
    rows_path = output_dir / "candidate_reeval_per_example.csv"
    summaries_path = output_dir / "candidate_reeval_summary.csv"
    common.atomic_csv(rows_path, rows)
    common.atomic_csv(summaries_path, summaries)
    final = {
        "protocol_id": common.PROTOCOL_ID,
        "protocol_hash": protocol_hash,
        "status": "PASS",
        "method": args.method,
        "attack": common.METHOD_ATTACK[args.method],
        "budgets": list(common.BUDGETS),
        "search_seeds": list(common.SEARCH_SEEDS),
        "examples": common.STRESS_EXAMPLES,
        "per_example_rows": len(rows),
        "eval_draws": draws,
        "reeval_seed": common.REEVAL_SEED,
        "candidate_reeval_per_example_sha256": common.file_sha256(rows_path),
        "candidate_reeval_summary_sha256": common.file_sha256(summaries_path),
        "no_attack_family_selection": True,
        "no_seed_selection": True,
        "no_per_example_budget_selection": True,
        "paper_claim_authorized_by_method_stage": False,
    }
    final["method_summary_hash"] = common.object_sha256(final)
    common.atomic_json(output_dir / "method_summary.json", final)
    print(
        f"PASS R6 reevaluation {args.method}: "
        + ", ".join(
            f"Q={row['api_budget']} seed={row['search_seed']} "
            f"Exact={float(row['attacked_exact_set_probability_percent']):.3f}%"
            for row in summaries
        ),
        flush=True,
    )
    print(f"Complete: {output_dir}", flush=True)


if __name__ == "__main__":
    main()

