#!/usr/bin/env python3
"""Search one frozen method/seed cell on the 1,000-example R5 holdout."""

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
import topk_cifar100_confirmatory_r5_common as common
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
    parser = argparse.ArgumentParser(
        description="One frozen CIFAR-100 confirmatory R5 method/seed cell"
    )
    parser.add_argument("--protocol-json", type=Path, required=True)
    parser.add_argument("--method", choices=common.METHODS, required=True)
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
        "api_budget": common.API_BUDGET,
        "split": "confirmatory_1000",
    }
    if batch_number is not None:
        result["batch_number"] = int(batch_number)
    return result


def validate_payload(
    payload: Mapping[str, Any],
    expected: Mapping[str, Any],
    expected_indices: Sequence[int],
) -> None:
    # R4's schema validator is intentionally reused: it verifies tensor
    # shapes, hashes, [0,1] bounds, L-infinity feasibility and payload hash.
    validate_base_payload(payload, expected, expected_indices)


def validate_args(args: argparse.Namespace) -> None:
    expected_batch = int(common.SEARCH_BATCH_SIZE_BY_METHOD[args.method])
    if args.batch_size != expected_batch:
        raise ValueError(
            f"Frozen R5 requires --batch-size {expected_batch} for {args.method}"
        )
    if args.search_seed not in common.SEARCH_SEEDS:
        raise ValueError("Search seed is not part of the frozen R5 protocol")
    if not 0 < args.gpu_memory_fraction <= 1:
        raise ValueError("--gpu-memory-fraction must be in (0,1]")


def validate_resume_candidate(
    output_dir: Path,
    identity: Mapping[str, Any],
    confirmatory_indices: Sequence[int],
) -> bool:
    candidate_path = output_dir / "candidate.pt"
    summary_path = output_dir / "search_summary.json"
    if not candidate_path.exists() and not summary_path.exists():
        return False
    if summary_path.is_file() and not candidate_path.is_file():
        raise AssertionError("Cell summary exists but candidate.pt is missing")
    payload = torch.load(candidate_path, map_location="cpu", weights_only=False)
    validate_payload(payload, identity, confirmatory_indices)
    if not summary_path.is_file():
        print("RECOVER candidate.pt without summary; rebuilding metadata", flush=True)
        return False
    summary = common.read_json(summary_path)
    if summary.get("candidate_file_sha256") != common.file_sha256(candidate_path):
        raise AssertionError("Completed R5 candidate file hash mismatch")
    if summary.get("candidate_payload_hash") != payload.get("payload_hash"):
        raise AssertionError("Completed R5 candidate payload hash mismatch")
    if summary.get("r4_selection_hash") != common.EXPECTED_R4_SELECTION_HASH:
        raise AssertionError("Completed R5 cell used a different R4 selection")
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

    parent = common.load_parent_authorization(Path(protocol["parent_dir"]))
    if parent["handoff_hash"] != protocol["parent_handoff_hash"]:
        raise AssertionError("R3 parent handoff changed after R5 protocol freeze")
    r4_info = common.load_r4_freeze(Path(protocol["r4_root"]), parent)
    if r4_info["selection_hash"] != protocol["r4_selection_hash"]:
        raise AssertionError("R4 attack-family freeze changed after R5 protocol creation")

    confirmatory_indices = [int(value) for value in parent["confirmatory_indices"]]
    if common.sequence_sha256(confirmatory_indices) != protocol["indices_sha256"]:
        raise AssertionError("Confirmatory index hash differs from the frozen R5 protocol")
    if set(confirmatory_indices) & set(int(value) for value in parent["pilot_indices"]):
        raise AssertionError("R5 confirmatory split overlaps the R4 pilot")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    identity = payload_identity(
        protocol_hash=protocol_hash,
        method=args.method,
        search_seed=args.search_seed,
    )
    if validate_resume_candidate(output_dir, identity, confirmatory_indices):
        print(f"RESUME complete R5 cell: {output_dir}", flush=True)
        return

    dataset_root = r1.r2_anchor.resolve_torchvision_root(Path(protocol["data_dir"]))
    dataset = torchvision.datasets.CIFAR100(
        root=str(dataset_root),
        train=False,
        download=False,
        transform=transforms.ToTensor(),
    )
    common.validate_dataset_against_parent(dataset, parent)
    frozen_labels, frozen_references = common.load_frozen_confirmatory_references(
        parent["parent_dir"], confirmatory_indices
    )
    common.validate_confirmatory_labels(frozen_labels)
    standard, anchor = load_models(protocol, device)
    defense = FrozenDefense(common.method_spec(args.method), standard, anchor, device)
    attack = common.METHOD_ATTACK[args.method]

    partial_dir = output_dir / "partials"
    batches = math.ceil(len(confirmatory_indices) / args.batch_size)
    payloads: list[dict[str, Any]] = []
    started = time.monotonic()
    for batch_number, low in enumerate(
        range(0, len(confirmatory_indices), args.batch_size)
    ):
        high = min(low + args.batch_size, len(confirmatory_indices))
        batch_indices = confirmatory_indices[low:high]
        partial_path = partial_dir / f"batch_{batch_number:03d}.pt"
        partial_identity = payload_identity(
            protocol_hash=protocol_hash,
            method=args.method,
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
        expected_labels = torch.tensor(frozen_labels[low:high], dtype=torch.long)
        if not torch.equal(labels_cpu, expected_labels):
            raise AssertionError("Dataset labels differ from the audited R3 confirmatory rows")
        clean = clean_cpu.to(device)
        reference_top2 = torch.tensor(
            frozen_references[low:high], dtype=torch.long, device=device
        )
        proposal_seed = core.stable_int(
            f"{protocol_hash}|{args.method}|{attack}|{args.search_seed}|"
            f"batch={batch_number}|proposal"
        )
        oracle_seed = core.stable_int(
            f"{protocol_hash}|{args.method}|{attack}|{args.search_seed}|"
            f"batch={batch_number}|oracle"
        )
        batch_started = time.monotonic()
        adversarial, details = core.run_square_batch(
            attack=attack,
            defense=defense,
            clean=clean,
            reference_topk=reference_top2,
            eps=common.EPSILON_PX / 255.0,
            api_budget=common.API_BUDGET,
            p_init=common.SQUARE_P_INIT,
            adaptive_config=common.ADAPTIVE_CONFIG,
            racing_config=common.RACING_CONFIG,
            proposal_seed=proposal_seed,
            oracle_seed=oracle_seed,
        )
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
        raise AssertionError("Query plan changed across R5 batches")
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
        "candidate_schema": "clean_and_adversarial_confirmatory_v1",
        "r4_selection_hash": common.EXPECTED_R4_SELECTION_HASH,
    }
    add_payload_hash(candidate)
    validate_payload(candidate, identity, confirmatory_indices)
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
    for position, index in enumerate(confirmatory_indices):
        reference = [int(value) for value in candidate["reference_top2"][position].tolist()]
        rows.append(
            {
                "protocol_id": common.PROTOCOL_ID,
                "protocol_hash": protocol_hash,
                "method": args.method,
                "attack": attack,
                "search_seed": args.search_seed,
                "sample_index": index,
                "label": int(candidate["labels"][position]),
                "reference_top1": reference[0],
                "reference_set": f"{min(reference)}|{max(reference)}",
                "epsilon_px": common.EPSILON_PX,
                "linf": float(linf[position]),
                "api_queries": common.API_BUDGET,
                "candidate_tensor_sha256": candidate["tensor_sha256"]["adversarial"],
            }
        )
    common.atomic_csv(output_dir / "search_per_example.csv", rows)
    summary = {
        **identity,
        "status": "PASS",
        "examples": len(confirmatory_indices),
        "candidate_schema": candidate["candidate_schema"],
        "candidate_path": str(candidate_path),
        "candidate_file_sha256": common.file_sha256(candidate_path),
        "candidate_payload_hash": candidate["payload_hash"],
        "tensor_sha256": candidate["tensor_sha256"],
        "query_details": details,
        "maximum_linf": float(linf.max()),
        "search_wall_seconds": candidate["search_wall_seconds"],
        "independent_mc1000_complete": False,
        "confirmatory_partition_read": True,
        "pilot_partition_read": False,
        "r4_selection_hash": common.EXPECTED_R4_SELECTION_HASH,
        "attack_was_preselected_on_disjoint_pilot": True,
    }
    common.atomic_json(output_dir / "search_summary.json", summary)
    print(
        f"PASS R5 {args.method}/{attack}/seed={args.search_seed}: "
        f"1,000 candidates, exact Q={common.API_BUDGET}, "
        f"max Linf={float(linf.max()):.8f}",
        flush=True,
    )
    print(f"Complete: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
