#!/usr/bin/env python3
"""Search one frozen CIFAR-100 R6 method/budget/seed stress-test cell."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torchvision
import torchvision.transforms as transforms

import calibrate_topk_cifar100_actual_defense_r1 as r1
import topk_cifar100_query_stress_r6_common as common
import topk_cifar100_square_attack_core_r4 as core
from search_topk_cifar100_attack_pilot_r4 import (
    FrozenDefense,
    add_payload_hash,
    atomic_torch_save,
    load_batch,
    load_models,
    validate_payload as validate_base_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One CIFAR-100 R6 query-stress cell")
    parser.add_argument("--protocol-json", type=Path, required=True)
    parser.add_argument("--method", choices=common.METHODS, required=True)
    parser.add_argument("--api-budget", type=int, choices=common.BUDGETS, required=True)
    parser.add_argument("--search-seed", type=int, choices=common.SEARCH_SEEDS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.45)
    return parser.parse_args()


def payload_identity(
    *,
    protocol_hash: str,
    method: str,
    budget: int,
    search_seed: int,
    batch_number: int | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "protocol_id": common.PROTOCOL_ID,
        "protocol_hash": protocol_hash,
        "method": method,
        "attack": common.METHOD_ATTACK[method],
        "search_seed": int(search_seed),
        "top_k": common.TOP_K,
        "epsilon_px": common.EPSILON_PX,
        "api_budget": int(budget),
        "split": "query_stress_200_disjoint",
    }
    if batch_number is not None:
        result["batch_number"] = int(batch_number)
    return result


def validate_payload(
    payload: Mapping[str, Any],
    expected: Mapping[str, Any],
    expected_indices: Sequence[int],
) -> None:
    validate_base_payload(payload, expected, expected_indices)


def validate_args(args: argparse.Namespace) -> None:
    expected_batch = int(common.SEARCH_BATCH_SIZE_BY_METHOD[args.method])
    if args.batch_size != expected_batch:
        raise ValueError(f"Frozen R6 requires --batch-size {expected_batch} for {args.method}")
    if args.api_budget not in common.BUDGETS:
        raise ValueError("API budget is not part of the frozen R6 protocol")
    if args.search_seed not in common.SEARCH_SEEDS:
        raise ValueError("Search seed is not part of the frozen R6 protocol")
    if not 0 < args.gpu_memory_fraction <= 1:
        raise ValueError("--gpu-memory-fraction must be in (0,1]")


def validate_resume_candidate(
    output_dir: Path,
    identity: Mapping[str, Any],
    stress_indices: Sequence[int],
    r5_final_hash: str,
) -> bool:
    candidate_path = output_dir / "candidate.pt"
    summary_path = output_dir / "search_summary.json"
    if not candidate_path.exists() and not summary_path.exists():
        return False
    if summary_path.is_file() and not candidate_path.is_file():
        raise AssertionError("R6 cell summary exists but candidate.pt is missing")
    payload = torch.load(candidate_path, map_location="cpu", weights_only=False)
    validate_payload(payload, identity, stress_indices)
    if not summary_path.is_file():
        print("RECOVER candidate.pt without summary; rebuilding metadata", flush=True)
        return False
    summary = common.read_json(summary_path)
    if summary.get("candidate_file_sha256") != common.file_sha256(candidate_path):
        raise AssertionError("Completed R6 candidate file hash mismatch")
    if summary.get("candidate_payload_hash") != payload.get("payload_hash"):
        raise AssertionError("Completed R6 candidate payload hash mismatch")
    if summary.get("r5_final_summary_hash") != r5_final_hash:
        raise AssertionError("Completed R6 cell points to a different R5 result")
    if int(summary.get("api_budget", -1)) != int(identity["api_budget"]):
        raise AssertionError("Completed R6 cell budget mismatch")
    return True


def main() -> None:
    args = parse_args()
    validate_args(args)
    protocol_path = args.protocol_json.expanduser().resolve()
    protocol = common.read_json(protocol_path)
    protocol_hash = common.validate_protocol(protocol)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, device=device)

    r5_info = common.load_r5_completion(Path(protocol["r5_root"]))
    if r5_info["final_hash"] != protocol["r5_final_summary_hash"]:
        raise AssertionError("R5 completion changed after the R6 protocol freeze")
    parent = common.r5.load_parent_authorization(Path(protocol["r3_parent_dir"]))
    if parent["handoff_hash"] != protocol["r3_parent_handoff_hash"]:
        raise AssertionError("R3 parent changed after the R6 protocol freeze")

    stress_indices = [int(value) for value in protocol["stress_indices"]]
    derived = common.derive_stress_indices(parent)
    if stress_indices != derived:
        raise AssertionError("R6 stress split differs from the deterministic frozen selection")
    labels_frozen, references_frozen = common.load_stress_references(parent, stress_indices)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    identity = payload_identity(
        protocol_hash=protocol_hash,
        method=args.method,
        budget=args.api_budget,
        search_seed=args.search_seed,
    )
    if validate_resume_candidate(
        output_dir, identity, stress_indices, str(protocol["r5_final_summary_hash"])
    ):
        print(f"RESUME complete R6 cell: {output_dir}", flush=True)
        return

    dataset_root = r1.r2_anchor.resolve_torchvision_root(Path(protocol["data_dir"]))
    dataset = torchvision.datasets.CIFAR100(
        root=str(dataset_root), train=False, download=False, transform=transforms.ToTensor()
    )
    common.r5.validate_dataset_against_parent(dataset, parent)
    standard, anchor = load_models(protocol, device)
    defense = FrozenDefense(common.method_spec(args.method), standard, anchor, device)
    attack = common.METHOD_ATTACK[args.method]

    partial_dir = output_dir / "partials"
    batches = math.ceil(len(stress_indices) / args.batch_size)
    payloads: list[dict[str, Any]] = []
    started = time.monotonic()
    for batch_number, low in enumerate(range(0, len(stress_indices), args.batch_size)):
        high = min(low + args.batch_size, len(stress_indices))
        batch_indices = stress_indices[low:high]
        partial_path = partial_dir / f"batch_{batch_number:03d}.pt"
        partial_identity = payload_identity(
            protocol_hash=protocol_hash,
            method=args.method,
            budget=args.api_budget,
            search_seed=args.search_seed,
            batch_number=batch_number,
        )
        if partial_path.is_file():
            partial = torch.load(partial_path, map_location="cpu", weights_only=False)
            validate_payload(partial, partial_identity, batch_indices)
            payloads.append(dict(partial))
            print(f"  RESUME batch {batch_number + 1}/{batches}", flush=True)
            continue

        clean_cpu, labels_cpu = load_batch(dataset, batch_indices)
        expected_labels = torch.tensor(labels_frozen[low:high], dtype=torch.long)
        if not torch.equal(labels_cpu, expected_labels):
            raise AssertionError("Dataset labels differ from the audited R3 rows")
        clean = clean_cpu.to(device)
        reference_top2 = torch.tensor(
            references_frozen[low:high], dtype=torch.long, device=device
        )
        proposal_seed = core.stable_int(
            f"{protocol_hash}|{args.method}|{attack}|Q={args.api_budget}|"
            f"{args.search_seed}|batch={batch_number}|proposal"
        )
        oracle_seed = core.stable_int(
            f"{protocol_hash}|{args.method}|{attack}|Q={args.api_budget}|"
            f"{args.search_seed}|batch={batch_number}|oracle"
        )
        batch_started = time.monotonic()
        adversarial, details = core.run_square_batch(
            attack=attack,
            defense=defense,
            clean=clean,
            reference_topk=reference_top2,
            eps=common.EPSILON_PX / 255.0,
            api_budget=args.api_budget,
            p_init=common.SQUARE_P_INIT,
            adaptive_config=common.ADAPTIVE_CONFIG,
            racing_config=common.RACING_CONFIG,
            proposal_seed=proposal_seed,
            oracle_seed=oracle_seed,
        )
        if int(details["api_queries"]) != args.api_budget:
            raise AssertionError("Square core did not consume the exact R6 budget")
        partial: dict[str, Any] = {
            **partial_identity,
            "clean": clean_cpu.contiguous(),
            "adversarial": adversarial.detach().cpu().contiguous(),
            "labels": labels_cpu.contiguous(),
            "sample_indices": torch.tensor(batch_indices, dtype=torch.long),
            "reference_top2": reference_top2.detach().cpu().contiguous(),
            "query_details": details,
            "proposal_seed": int(proposal_seed),
            "oracle_seed": int(oracle_seed),
            "wall_seconds": float(time.monotonic() - batch_started),
            "r5_final_summary_hash": str(protocol["r5_final_summary_hash"]),
        }
        add_payload_hash(partial)
        validate_payload(partial, partial_identity, batch_indices)
        atomic_torch_save(partial_path, partial)
        payloads.append(partial)
        print(
            f"  PASS batch {batch_number + 1}/{batches}: "
            f"queries={details['api_queries']} steps={details['search_steps']}",
            flush=True,
        )

    tensor_keys = ("clean", "adversarial", "labels", "sample_indices", "reference_top2")
    details = payloads[0]["query_details"]
    if any(payload["query_details"] != details for payload in payloads):
        raise AssertionError("R6 query plan changed across batches")
    candidate: dict[str, Any] = {
        **identity,
        **{
            key: torch.cat([payload[key] for payload in payloads], dim=0)
            for key in tensor_keys
        },
        "query_details": details,
        "batches": batches,
        "batch_size": int(args.batch_size),
        "partial_payload_hashes": [payload["payload_hash"] for payload in payloads],
        "search_wall_seconds": float(time.monotonic() - started),
        "candidate_schema": "clean_and_adversarial_query_stress_v1",
        "r5_final_summary_hash": str(protocol["r5_final_summary_hash"]),
    }
    add_payload_hash(candidate)
    validate_payload(candidate, identity, stress_indices)
    candidate_path = output_dir / "candidate.pt"
    atomic_torch_save(candidate_path, candidate)

    linf = (
        (candidate["adversarial"] - candidate["clean"])
        .abs()
        .flatten(1)
        .max(1)
        .values
    )
    rows: list[dict[str, Any]] = []
    for position, index in enumerate(stress_indices):
        reference = [int(value) for value in candidate["reference_top2"][position].tolist()]
        rows.append(
            {
                "protocol_id": common.PROTOCOL_ID,
                "protocol_hash": protocol_hash,
                "method": args.method,
                "attack": attack,
                "api_budget": args.api_budget,
                "search_seed": args.search_seed,
                "sample_index": index,
                "label": int(candidate["labels"][position]),
                "reference_top1": reference[0],
                "reference_set": f"{min(reference)}|{max(reference)}",
                "epsilon_px": common.EPSILON_PX,
                "linf": float(linf[position]),
                "candidate_tensor_sha256": candidate["tensor_sha256"]["adversarial"],
            }
        )
    common.atomic_csv(output_dir / "search_per_example.csv", rows)
    summary = {
        **identity,
        "status": "PASS",
        "examples": len(stress_indices),
        "candidate_schema": candidate["candidate_schema"],
        "candidate_path": str(candidate_path),
        "candidate_file_sha256": common.file_sha256(candidate_path),
        "candidate_payload_hash": candidate["payload_hash"],
        "tensor_sha256": candidate["tensor_sha256"],
        "query_details": details,
        "maximum_linf": float(linf.max()),
        "search_wall_seconds": candidate["search_wall_seconds"],
        "independent_mc1000_complete": False,
        "r4_r5_partitions_read": False,
        "query_stress_partition_read": True,
        "r5_final_summary_hash": str(protocol["r5_final_summary_hash"]),
        "attack_family_frozen_from_r4": True,
    }
    common.atomic_json(output_dir / "search_summary.json", summary)
    print(
        f"PASS R6 {args.method}/{attack}/Q={args.api_budget}/seed={args.search_seed}: "
        f"{len(stress_indices)} candidates, exact budget, max Linf={float(linf.max()):.8f}",
        flush=True,
    )
    print(f"Complete: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
