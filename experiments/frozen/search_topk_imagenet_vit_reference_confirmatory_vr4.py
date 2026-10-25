#!/usr/bin/env python3
"""Run one frozen method/seed search on the 1,000-image ImageNet ViT-reference VR4 split."""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import calibrate_topk_imagenet_vit_reference_vr1 as vr1
import topk_imagenet_vit_reference_confirmatory_vr4_common as common
import topk_imagenet_vit_reference_square_attack_core_vr3 as core
from prepare_topk_imagenet_vit_reference_confirmatory_vr4 import validate_input_payload
from search_topk_imagenet_vit_reference_attack_pilot_vr3 import FrozenDefense


TENSOR_KEYS = ("adversarial", "labels", "confirmatory_positions", "reference_top5")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-json", type=Path, required=True)
    parser.add_argument("--method", choices=common.METHODS, required=True)
    parser.add_argument("--search-seed", type=int, choices=common.SEARCH_SEEDS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.45)
    return parser.parse_args()


def atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def add_payload_hash(payload: dict[str, Any]) -> None:
    payload["tensor_sha256"] = {
        key: common.tensor_sha256(payload[key]) for key in TENSOR_KEYS
    }
    hash_input = {
        key: value
        for key, value in payload.items()
        if key not in set(TENSOR_KEYS) | {"payload_hash"}
    }
    payload["payload_hash"] = common.object_sha256(hash_input)


def candidate_identity(
    protocol: Mapping[str, Any], method: str, search_seed: int, source_hash: str,
    source_clean_hash: str,
) -> dict[str, Any]:
    return {
        "protocol_id": common.PROTOCOL_ID,
        "protocol_hash": protocol["protocol_hash"],
        "method": method,
        "attack": common.METHOD_ATTACK[method],
        "search_seed": int(search_seed),
        "primary_k": common.PRIMARY_K,
        "secondary_k": common.SECONDARY_K,
        "epsilon_px": common.EPSILON_PX,
        "api_budget": common.API_BUDGET,
        "split": "attack_confirmatory",
        "source_confirmatory_payload_hash": str(source_hash),
        "source_clean_tensor_sha256": str(source_clean_hash),
        "vr3_selection_hash": common.EXPECTED_VR3_SELECTION_HASH,
    }


def validate_candidate_payload(
    payload: Mapping[str, Any],
    expected: Mapping[str, Any],
    expected_positions: Sequence[int],
    expected_paths: Sequence[str],
    expected_labels: torch.Tensor,
    expected_reference: torch.Tensor,
    *,
    clean: torch.Tensor | None = None,
) -> None:
    for key, value in expected.items():
        if payload.get(key) != value:
            raise AssertionError(f"ImageNet ViT-reference VR4 candidate identity mismatch: {key}")
    for key in TENSOR_KEYS:
        if key not in payload:
            raise AssertionError(f"ImageNet ViT-reference VR4 candidate misses {key}")
    adversarial = payload["adversarial"].detach().cpu()
    labels = payload["labels"].detach().cpu()
    positions = payload["confirmatory_positions"].detach().cpu()
    reference = payload["reference_top5"].detach().cpu()
    count = len(expected_positions)
    if adversarial.shape != (count, 3, 224, 224):
        raise AssertionError(f"ImageNet ViT-reference VR4 candidate shape changed: {tuple(adversarial.shape)}")
    if labels.shape != (count,) or positions.shape != (count,):
        raise AssertionError("ImageNet ViT-reference VR4 candidate metadata shape changed")
    if reference.shape != (count, common.PRIMARY_K):
        raise AssertionError("ImageNet ViT-reference VR4 candidate reference shape changed")
    if positions.tolist() != [int(value) for value in expected_positions]:
        raise AssertionError("ImageNet ViT-reference VR4 candidate position order changed")
    if list(payload.get("relative_paths", [])) != list(expected_paths):
        raise AssertionError("ImageNet ViT-reference VR4 candidate path order changed")
    if not torch.equal(labels, expected_labels.detach().cpu()):
        raise AssertionError("ImageNet ViT-reference VR4 candidate labels changed")
    if not torch.equal(reference, expected_reference.detach().cpu()):
        raise AssertionError("ImageNet ViT-reference VR4 candidate references changed")
    if int(reference.min()) < 0 or int(reference.max()) >= 1000:
        raise AssertionError("ImageNet ViT-reference VR4 candidate reference has an invalid class")
    if not reference.sort(1).values.diff(dim=1).ne(0).all():
        raise AssertionError("ImageNet ViT-reference VR4 candidate reference has duplicate members")
    if not torch.isfinite(adversarial).all():
        raise AssertionError("ImageNet ViT-reference VR4 candidate contains non-finite pixels")
    if float(adversarial.min()) < -1e-7 or float(adversarial.max()) > 1 + 1e-7:
        raise AssertionError("ImageNet ViT-reference VR4 candidate pixels are outside [0,1]")
    if clean is not None:
        clean_cpu = clean.detach().cpu()
        if clean_cpu.shape != adversarial.shape:
            raise AssertionError("ImageNet ViT-reference VR4 clean/candidate shape mismatch")
        linf = (adversarial - clean_cpu).abs().flatten(1).max(1).values
        if float(linf.max()) > common.EPSILON_PX / 255.0 + 1e-6:
            raise AssertionError("ImageNet ViT-reference VR4 candidate violates the L-infinity radius")
        expected_clean_hash = common.tensor_sha256(clean_cpu)
        if "batch_number" in payload and payload.get("clean_tensor_sha256") != expected_clean_hash:
            raise AssertionError("ImageNet ViT-reference VR4 candidate clean-batch hash mismatch")
    for key in TENSOR_KEYS:
        if common.tensor_sha256(payload[key]) != payload.get("tensor_sha256", {}).get(key):
            raise AssertionError(f"ImageNet ViT-reference VR4 candidate tensor hash mismatch: {key}")
    hash_input = {
        key: value
        for key, value in payload.items()
        if key not in set(TENSOR_KEYS) | {"payload_hash"}
    }
    if common.object_sha256(hash_input) != payload.get("payload_hash"):
        raise AssertionError("ImageNet ViT-reference VR4 candidate payload hash mismatch")


def validate_resume(
    output_dir: Path,
    identity: Mapping[str, Any],
    input_payload: Mapping[str, Any],
) -> bool:
    candidate_path = output_dir / "candidate.pt"
    summary_path = output_dir / "search_summary.json"
    rows_path = output_dir / "search_per_example.csv"
    if not candidate_path.exists() and not summary_path.exists() and not rows_path.exists():
        return False
    if not (candidate_path.is_file() and summary_path.is_file() and rows_path.is_file()):
        print("RECOVER incomplete final cell from validated partial batches", flush=True)
        return False
    payload = torch.load(candidate_path, map_location="cpu", weights_only=False)
    validate_candidate_payload(
        payload,
        identity,
        list(range(common.CONFIRMATORY_EXAMPLES)),
        list(input_payload["relative_paths"]),
        input_payload["labels"],
        input_payload["reference_top5"],
    )
    summary = common.read_json(summary_path)
    if summary.get("status") != "PASS":
        raise AssertionError("ImageNet ViT-reference VR4 resume summary did not pass")
    if summary.get("candidate_file_sha256") != common.file_sha256(candidate_path):
        raise AssertionError("ImageNet ViT-reference VR4 resume candidate file hash changed")
    if summary.get("candidate_payload_hash") != payload.get("payload_hash"):
        raise AssertionError("ImageNet ViT-reference VR4 resume candidate payload hash changed")
    if summary.get("search_per_example_sha256") != common.file_sha256(rows_path):
        raise AssertionError("ImageNet ViT-reference VR4 resume search-row bytes changed")
    if summary.get("query_details") != payload.get("query_details"):
        raise AssertionError("ImageNet ViT-reference VR4 resume query details changed")
    if int(payload.get("query_details", {}).get("api_queries", -1)) != common.API_BUDGET:
        raise AssertionError("ImageNet ViT-reference VR4 resume candidate did not use exact Q=5000")
    rows = common.read_csv(rows_path)
    if len(rows) != common.CONFIRMATORY_EXAMPLES:
        raise AssertionError("ImageNet ViT-reference VR4 resume search-row count changed")
    tolerance = common.EPSILON_PX / 255.0 + 1e-6
    for position, row in enumerate(rows):
        if (
            row.get("protocol_hash") != identity["protocol_hash"]
            or row.get("method") != identity["method"]
            or row.get("attack") != identity["attack"]
            or int(row["search_seed"]) != int(identity["search_seed"])
            or int(row["confirmatory_position"]) != position
            or row.get("relative_path") != input_payload["relative_paths"][position]
            or int(row["api_queries"]) != common.API_BUDGET
            or row.get("candidate_tensor_sha256")
            != payload["tensor_sha256"]["adversarial"]
        ):
            raise AssertionError("ImageNet ViT-reference VR4 resume search-row identity changed")
        linf = float(row["linf"])
        if not math.isfinite(linf) or linf < 0 or linf > tolerance:
            raise AssertionError("ImageNet ViT-reference VR4 resume rows report an infeasible candidate")
    return True


def main() -> None:
    args = parse_args()
    if args.batch_size != common.SEARCH_BATCH_SIZE_BY_METHOD[args.method]:
        raise ValueError(
            f"Frozen ImageNet ViT-reference VR4 batch size for {args.method} is "
            f"{common.SEARCH_BATCH_SIZE_BY_METHOD[args.method]}"
        )
    if args.num_workers < 0 or not 0 < args.gpu_memory_fraction <= 1:
        raise ValueError("Invalid ImageNet ViT-reference VR4 worker or GPU-memory setting")
    protocol_path = args.protocol_json.expanduser().resolve()
    protocol = common.read_json(protocol_path)
    protocol_hash = common.validate_protocol(protocol)
    output_root = Path(protocol["paths"]["output_root"]).expanduser().resolve()
    expected_dir = common.cell_dir(output_root, args.method, args.search_seed).resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir != expected_dir:
        raise AssertionError(f"ImageNet ViT-reference VR4 cell path is not frozen: {output_dir} != {expected_dir}")

    parent = common.load_parent(
        Path(protocol["paths"]["r0_dir"]),
        Path(protocol["paths"]["calibration_dir"]),
        Path(protocol["paths"]["clean_dir"]),
    )
    vr3_info = common.load_vr3_freeze(Path(protocol["paths"]["vr3_root"]), parent)
    if vr3_info["selection_hash"] != protocol["vr3_selection_hash"]:
        raise AssertionError("ImageNet VR3 freeze changed after VR4 protocol creation")
    manifest_rows = common.canonical_confirmatory_rows(parent)
    pilot_paths = {str(row["relative_path"]) for row in parent["pilot_pool"]}
    if any(str(row["relative_path"]) in pilot_paths for row in manifest_rows):
        raise AssertionError("ImageNet ViT-reference VR4 confirmatory split overlaps the VR3 pilot pool")

    input_path = output_root / "confirmatory_inputs.pt"
    input_payload = torch.load(input_path, map_location="cpu", weights_only=False)
    validate_input_payload(input_payload, protocol, manifest_rows)
    prepare_summary = common.read_json(output_root / "prepare_summary.json")
    if prepare_summary.get("status") != "PASS":
        raise PermissionError("ImageNet ViT-reference VR4 preparation did not pass")
    if prepare_summary.get("confirmatory_inputs_payload_hash") != input_payload["payload_hash"]:
        raise AssertionError("ImageNet ViT-reference VR4 prepared input identity changed")
    if prepare_summary.get("confirmatory_inputs_file_sha256") != common.file_sha256(input_path):
        raise AssertionError("ImageNet ViT-reference VR4 prepared input bytes changed")

    identity = candidate_identity(
        protocol,
        args.method,
        args.search_seed,
        str(input_payload["payload_hash"]),
        str(input_payload["clean_tensor_sha256"]),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if validate_resume(output_dir, identity, input_payload):
        print(
            f"RESUME PASS ImageNet ViT-reference VR4 {args.method}/{common.METHOD_ATTACK[args.method]}/"
            f"seed={args.search_seed}",
            flush=True,
        )
        return

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, device=device)
    standard, robust, _audit = vr1.load_models(
        Path(protocol["checkpoints"]["standard"]["path"]),
        Path(protocol["checkpoints"]["robust"]["path"]),
        device,
    )
    defense = FrozenDefense(common.method_spec(args.method), standard, robust, device)
    attack = common.METHOD_ATTACK[args.method]
    loader = vr1.make_loader(
        Path(protocol["paths"]["data_root"]),
        manifest_rows,
        args.batch_size,
        args.num_workers,
        device,
    )
    partial_dir = output_dir / "partials"
    partial_dir.mkdir(parents=True, exist_ok=True)
    batches = math.ceil(common.CONFIRMATORY_EXAMPLES / args.batch_size)
    partials: list[dict[str, Any]] = []
    linf_parts: list[torch.Tensor] = []
    clean_digest = hashlib.sha256()
    clean_digest.update(b"float32")
    clean_digest.update(
        np.asarray(
            (common.CONFIRMATORY_EXAMPLES, 3, 224, 224), dtype="<i8"
        ).tobytes()
    )
    wall_start = time.perf_counter()
    for batch_number, (images, labels, paths) in enumerate(loader):
        low = batch_number * args.batch_size
        high = min(low + len(images), common.CONFIRMATORY_EXAMPLES)
        positions = list(range(low, high))
        expected_paths = list(input_payload["relative_paths"])[low:high]
        if [str(path) for path in paths] != expected_paths:
            raise AssertionError("ImageNet ViT-reference VR4 DataLoader path order changed")
        expected_labels = input_payload["labels"][low:high]
        expected_reference = input_payload["reference_top5"][low:high]
        if not torch.equal(labels.detach().cpu(), expected_labels):
            raise AssertionError("ImageNet ViT-reference VR4 DataLoader labels changed")

        partial_path = partial_dir / f"batch_{batch_number:03d}.pt"
        partial_identity = {
            **identity,
            "batch_number": batch_number,
            "batch_start": low,
            "batch_stop": high,
        }
        clean_cpu = images.detach().cpu().contiguous()
        if clean_cpu.dtype != torch.float32:
            raise AssertionError("ImageNet ViT-reference VR4 preprocessing no longer returns float32")
        clean_digest.update(clean_cpu.numpy().tobytes(order="C"))
        if partial_path.is_file():
            partial = torch.load(partial_path, map_location="cpu", weights_only=False)
            validate_candidate_payload(
                partial,
                partial_identity,
                positions,
                expected_paths,
                expected_labels,
                expected_reference,
                clean=clean_cpu,
            )
            print(f"  RESUME batch {batch_number + 1}/{batches}", flush=True)
        else:
            clean = images.to(device, non_blocking=True)
            reference = expected_reference.to(device, non_blocking=True)
            proposal_seed = core.stable_int(
                f"{protocol_hash}|{args.method}|{attack}|{args.search_seed}|"
                f"batch={batch_number}|proposal"
            )
            oracle_seed = core.stable_int(
                f"{protocol_hash}|{args.method}|{attack}|{args.search_seed}|"
                f"batch={batch_number}|oracle"
            )
            started = time.perf_counter()
            adversarial, details = core.run_square_batch(
                attack=attack,
                defense=defense,
                clean=clean,
                reference_topk=reference,
                eps=common.EPSILON_PX / 255.0,
                api_budget=common.API_BUDGET,
                p_init=common.SQUARE_P_INIT,
                adaptive_config=common.ADAPTIVE_CONFIG,
                racing_config=common.RACING_CONFIG,
                proposal_seed=proposal_seed,
                oracle_seed=oracle_seed,
            )
            if int(details["api_queries"]) != common.API_BUDGET:
                raise AssertionError("ImageNet ViT-reference VR4 attack did not use the exact API budget")
            partial = {
                **partial_identity,
                "relative_paths": expected_paths,
                "adversarial": adversarial.detach().cpu().contiguous(),
                "labels": expected_labels.detach().cpu().contiguous(),
                "confirmatory_positions": torch.tensor(positions, dtype=torch.long),
                "reference_top5": expected_reference.detach().cpu().contiguous(),
                "clean_tensor_sha256": common.tensor_sha256(clean_cpu),
                "query_details": details,
                "proposal_seed": int(proposal_seed),
                "oracle_seed": int(oracle_seed),
                "wall_seconds": time.perf_counter() - started,
            }
            add_payload_hash(partial)
            validate_candidate_payload(
                partial,
                partial_identity,
                positions,
                expected_paths,
                expected_labels,
                expected_reference,
                clean=clean_cpu,
            )
            atomic_torch_save(partial_path, partial)
            print(
                f"  PASS batch {batch_number + 1}/{batches}: "
                f"Q={details['api_queries']}",
                flush=True,
            )
        partials.append(dict(partial))
        linf_parts.append(
            (partial["adversarial"] - clean_cpu).abs().flatten(1).max(1).values
        )
    if len(partials) != batches:
        raise AssertionError("ImageNet ViT-reference VR4 DataLoader produced an unexpected batch count")
    if clean_digest.hexdigest() != input_payload["clean_tensor_sha256"]:
        raise AssertionError("ImageNet ViT-reference VR4 clean image tensor changed after preparation")
    query_details = partials[0]["query_details"]
    if any(partial["query_details"] != query_details for partial in partials):
        raise AssertionError("ImageNet ViT-reference VR4 query plan changed across batches")

    candidate: dict[str, Any] = {
        **identity,
        "relative_paths": list(input_payload["relative_paths"]),
        "adversarial": torch.cat([partial["adversarial"] for partial in partials]),
        "labels": torch.cat([partial["labels"] for partial in partials]),
        "confirmatory_positions": torch.cat(
            [partial["confirmatory_positions"] for partial in partials]
        ),
        "reference_top5": torch.cat([partial["reference_top5"] for partial in partials]),
        "query_details": query_details,
        "partial_payload_hashes": [partial["payload_hash"] for partial in partials],
        "source_clean_batch_sha256": [
            partial["clean_tensor_sha256"] for partial in partials
        ],
        "batches": batches,
        "batch_size": args.batch_size,
        "candidate_schema": "shared_clean_manifest_plus_adversarial_tensor_v1",
        "search_wall_seconds": time.perf_counter() - wall_start,
    }
    add_payload_hash(candidate)
    validate_candidate_payload(
        candidate,
        identity,
        list(range(common.CONFIRMATORY_EXAMPLES)),
        list(input_payload["relative_paths"]),
        input_payload["labels"],
        input_payload["reference_top5"],
    )
    candidate_path = output_dir / "candidate.pt"
    atomic_torch_save(candidate_path, candidate)

    linf = torch.cat(linf_parts)
    search_rows = []
    for position in range(common.CONFIRMATORY_EXAMPLES):
        reference = [int(value) for value in candidate["reference_top5"][position].tolist()]
        search_rows.append(
            {
                "protocol_id": common.PROTOCOL_ID,
                "protocol_hash": protocol_hash,
                "method": args.method,
                "attack": attack,
                "search_seed": args.search_seed,
                "confirmatory_position": position,
                "relative_path": candidate["relative_paths"][position],
                "label": int(candidate["labels"][position]),
                "reference_top1": reference[0],
                "reference_set_k5": "|".join(str(value) for value in sorted(reference)),
                "epsilon_px": common.EPSILON_PX,
                "linf": float(linf[position]),
                "api_queries": common.API_BUDGET,
                "candidate_tensor_sha256": candidate["tensor_sha256"]["adversarial"],
            }
        )
    rows_path = output_dir / "search_per_example.csv"
    common.atomic_csv(rows_path, search_rows)
    summary = {
        **identity,
        "status": "PASS",
        "examples": common.CONFIRMATORY_EXAMPLES,
        "batches": batches,
        "candidate_schema": candidate["candidate_schema"],
        "candidate_path": str(candidate_path),
        "candidate_file_sha256": common.file_sha256(candidate_path),
        "candidate_payload_hash": candidate["payload_hash"],
        "candidate_tensor_sha256": candidate["tensor_sha256"]["adversarial"],
        "search_per_example_sha256": common.file_sha256(rows_path),
        "query_details": query_details,
        "maximum_linf": float(linf.max()),
        "maximum_linf_px": 255.0 * float(linf.max()),
        "search_wall_seconds": candidate["search_wall_seconds"],
        "independent_mc1000_complete": False,
        "confirmatory_partition_read": True,
        "pilot_partition_read": False,
        "attack_was_preselected_on_disjoint_pilot": True,
        "attack_family_selection_performed": False,
        "seed_selection_performed": False,
        "paper_claim_authorized_by_search": False,
    }
    common.atomic_json(output_dir / "search_summary.json", summary)
    print(
        f"PASS ImageNet ViT-reference VR4 {args.method}/{attack}/seed={args.search_seed}: "
        f"1,000 candidates, exact Q={common.API_BUDGET}, "
        f"max Linf={summary['maximum_linf_px']:.5f}/255",
        flush=True,
    )


if __name__ == "__main__":
    main()
