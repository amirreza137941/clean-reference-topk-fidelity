#!/usr/bin/env python3
"""Search one frozen method/attack/seed cell of CIFAR-100 pilot R4."""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import calibrate_topk_cifar100_actual_defense_r1 as r1
import topk_cifar100_attack_pilot_r4_common as common
import topk_cifar100_square_attack_core_r4 as core


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One CIFAR-100 adaptive attack pilot R4 cell")
    parser.add_argument("--protocol-json", type=Path, required=True)
    parser.add_argument("--method", choices=common.METHODS, required=True)
    parser.add_argument("--attack", choices=common.ATTACKS, required=True)
    parser.add_argument("--search-seed", type=int, choices=common.SEARCH_SEEDS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.45)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if not 0 < args.gpu_memory_fraction <= 1:
        raise ValueError("--gpu-memory-fraction must be in (0,1]")
    if args.search_seed not in common.SEARCH_SEEDS:
        raise ValueError("Search seed is not part of the frozen pilot")
    expected_batch = int(common.SEARCH_BATCH_SIZE_BY_METHOD[args.method])
    if args.batch_size != expected_batch:
        raise ValueError(
            f"Frozen R4 requires --batch-size {expected_batch} for {args.method}"
        )


def autocast_off(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", enabled=False)
    return contextlib.nullcontext()


class FrozenDefense:
    """Exactly the R2-selected inference defense queried as one black-box API."""

    def __init__(
        self,
        spec: common.MethodSpec,
        standard: nn.Module,
        anchor: nn.Module,
        device: torch.device,
    ) -> None:
        self.spec = spec
        self.standard = standard
        self.anchor = anchor
        self.device = device
        self.sigma = float(spec.sigma)

    def base(self, x: torch.Tensor) -> torch.Tensor:
        if self.spec.strength_steps <= 0:
            return x.detach()
        # no_grad (not inference_mode) produces a reference tensor usable by
        # the input-gradient strengthening graph.
        with torch.no_grad(), autocast_off(self.device):
            reference_top2 = r1.ranked_top2(self.standard(x).float()).clone()
        return r1.strengthen_inputs(
            x,
            reference_top2,
            self.anchor,
            self.spec.strength_alpha_px,
            self.spec.strength_epsilon_px,
            self.spec.strength_steps,
            self.spec.top1_weight,
            self.spec.set_weight,
        )

    def actual(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base(x)
        if self.sigma > 0:
            base = (base + self.sigma * torch.randn_like(base)).clamp(0.0, 1.0)
        with torch.no_grad(), autocast_off(self.device):
            return self.anchor(base).float()

    def draw_logits(self, x: torch.Tensor, draws: int, chunk_size: int = 20) -> torch.Tensor:
        """Return independent API responses while caching deterministic preprocessing.

        Strengthening is deterministic for a fixed request.  Caching it inside
        an EOT/racing score changes neither the response distribution nor API
        accounting, and avoids repeating two input-gradient steps per draw.
        """

        draws = int(draws)
        if draws < 1 or chunk_size < 1:
            raise ValueError("draws and chunk_size must be positive")
        base = self.base(x)
        if self.sigma <= 0:
            with torch.no_grad(), autocast_off(self.device):
                logits = self.anchor(base).float()
            return logits.unsqueeze(0).expand(draws, -1, -1).clone()
        parts: list[torch.Tensor] = []
        completed = 0
        while completed < draws:
            chunk = min(int(chunk_size), draws - completed)
            noise = torch.randn(
                (chunk,) + tuple(base.shape), device=base.device, dtype=base.dtype
            )
            queried = (base.unsqueeze(0) + self.sigma * noise).clamp(0.0, 1.0)
            with torch.no_grad(), autocast_off(self.device):
                logits = self.anchor(queried.flatten(0, 1)).float().view(
                    chunk, len(base), 100
                )
            parts.append(logits)
            completed += chunk
        return torch.cat(parts, dim=0)


def atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def payload_identity(
    *,
    protocol_hash: str,
    method: str,
    attack: str,
    search_seed: int,
    batch_number: int | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "protocol_id": common.PROTOCOL_ID,
        "protocol_hash": protocol_hash,
        "method": method,
        "attack": attack,
        "search_seed": int(search_seed),
        "top_k": common.TOP_K,
        "epsilon_px": common.EPSILON_PX,
        "api_budget": common.API_BUDGET,
    }
    if batch_number is not None:
        result["batch_number"] = int(batch_number)
    return result


def add_payload_hash(payload: dict[str, Any]) -> None:
    tensor_keys = ("clean", "adversarial", "labels", "sample_indices", "reference_top2")
    payload["tensor_sha256"] = {
        key: common.tensor_sha256(payload[key]) for key in tensor_keys
    }
    hash_input = {
        key: value
        for key, value in payload.items()
        if key not in set(tensor_keys) | {"payload_hash"}
    }
    payload["payload_hash"] = common.object_sha256(hash_input)


def validate_payload(
    payload: Mapping[str, Any],
    expected: Mapping[str, Any],
    expected_indices: Sequence[int],
) -> None:
    for key, value in expected.items():
        if payload.get(key) != value:
            raise AssertionError(f"Candidate payload identity mismatch: {key}")
    required = {"clean", "adversarial", "labels", "sample_indices", "reference_top2"}
    missing = required - set(payload)
    if missing:
        raise AssertionError(f"Candidate payload misses {sorted(missing)}")
    tensors = {key: payload[key].detach().cpu() for key in required}
    if tensors["clean"].shape != tensors["adversarial"].shape:
        raise AssertionError("Clean/adversarial tensor shape mismatch")
    if tensors["clean"].ndim != 4 or tensors["clean"].shape[1:] != (3, 32, 32):
        raise AssertionError("Candidate image tensor shape mismatch")
    examples = len(tensors["clean"])
    if tensors["labels"].shape != (examples,):
        raise AssertionError("Candidate label shape mismatch")
    if tensors["sample_indices"].shape != (examples,):
        raise AssertionError("Candidate index shape mismatch")
    if tensors["reference_top2"].shape != (examples, 2):
        raise AssertionError("Candidate reference shape mismatch")
    indices = [int(value) for value in tensors["sample_indices"].tolist()]
    if indices != [int(value) for value in expected_indices]:
        raise AssertionError("Candidate payload uses the wrong pilot order")
    if not torch.isfinite(tensors["clean"]).all() or not torch.isfinite(tensors["adversarial"]).all():
        raise AssertionError("Candidate payload contains non-finite pixels")
    if float(tensors["adversarial"].min()) < -1e-7 or float(tensors["adversarial"].max()) > 1 + 1e-7:
        raise AssertionError("Candidate pixels are outside [0,1]")
    linf = (tensors["adversarial"] - tensors["clean"]).abs().flatten(1).max(1).values
    if float(linf.max()) > common.EPSILON_PX / 255.0 + 1e-6:
        raise AssertionError("Candidate violates the frozen L-infinity radius")
    saved_hashes = payload.get("tensor_sha256", {})
    for key in required:
        if common.tensor_sha256(tensors[key]) != saved_hashes.get(key):
            raise AssertionError(f"Candidate tensor hash mismatch: {key}")
    hash_input = {
        key: value
        for key, value in payload.items()
        if key not in required | {"payload_hash"}
    }
    if common.object_sha256(hash_input) != payload.get("payload_hash"):
        raise AssertionError("Candidate payload hash mismatch")


def load_batch(dataset: Any, indices: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
    images: list[torch.Tensor] = []
    labels: list[int] = []
    for index in indices:
        image, label = dataset[int(index)]
        images.append(image)
        labels.append(int(label))
    return torch.stack(images, dim=0), torch.tensor(labels, dtype=torch.long)


def load_models(protocol: Mapping[str, Any], device: torch.device) -> tuple[nn.Module, nn.Module]:
    checkpoints = protocol["checkpoints"]
    standard_path = Path(checkpoints["standard"]["path"])
    anchor_path = Path(checkpoints["anchor"]["path"])
    return r1.load_models(standard_path, anchor_path, device)


def validate_resume_candidate(
    output_dir: Path,
    identity: Mapping[str, Any],
    pilot_indices: Sequence[int],
) -> bool:
    candidate_path = output_dir / "candidate.pt"
    summary_path = output_dir / "search_summary.json"
    if not candidate_path.exists() and not summary_path.exists():
        return False
    if summary_path.is_file() and not candidate_path.is_file():
        raise AssertionError("Cell summary exists but candidate.pt is missing")
    if candidate_path.is_file() and not summary_path.is_file():
        # A process may have stopped in the tiny window after the atomic
        # candidate write and before the JSON summary write.  Validate the
        # candidate, then rebuild it from already validated batch partials.
        payload = torch.load(candidate_path, map_location="cpu", weights_only=False)
        validate_payload(payload, identity, pilot_indices)
        print("RECOVER candidate.pt without summary; rebuilding final metadata", flush=True)
        return False
    payload = torch.load(candidate_path, map_location="cpu", weights_only=False)
    validate_payload(payload, identity, pilot_indices)
    summary = common.read_json(summary_path)
    if summary.get("candidate_file_sha256") != common.file_sha256(candidate_path):
        raise AssertionError("Completed cell candidate file hash mismatch")
    if summary.get("candidate_payload_hash") != payload.get("payload_hash"):
        raise AssertionError("Completed cell payload hash mismatch")
    return True


def main() -> None:
    args = parse_args()
    validate_args(args)
    protocol = common.read_json(args.protocol_json.expanduser().resolve())
    protocol_hash = common.validate_protocol(protocol)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, device=device)

    parent = common.load_parent_authorization(Path(protocol["parent_dir"]))
    if parent["handoff_hash"] != protocol["parent_handoff_hash"]:
        raise AssertionError("Parent handoff changed after protocol freeze")
    pilot_indices = [int(value) for value in parent["pilot_indices"]]
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    identity = payload_identity(
        protocol_hash=protocol_hash,
        method=args.method,
        attack=args.attack,
        search_seed=args.search_seed,
    )
    if validate_resume_candidate(output_dir, identity, pilot_indices):
        print(f"RESUME complete cell: {output_dir}", flush=True)
        return

    dataset_root = r1.r2_anchor.resolve_torchvision_root(Path(protocol["data_dir"]))
    dataset = torchvision.datasets.CIFAR100(
        root=str(dataset_root), train=False, download=False, transform=transforms.ToTensor()
    )
    common.validate_dataset_against_parent(dataset, parent)
    frozen_labels, frozen_references = common.load_frozen_pilot_references(
        parent["parent_dir"], pilot_indices
    )
    standard, anchor = load_models(protocol, device)
    defense = FrozenDefense(common.method_spec(args.method), standard, anchor, device)

    partial_dir = output_dir / "partials"
    batches = math.ceil(len(pilot_indices) / args.batch_size)
    payloads: list[dict[str, Any]] = []
    started = time.monotonic()
    for batch_number, low in enumerate(range(0, len(pilot_indices), args.batch_size)):
        high = min(low + args.batch_size, len(pilot_indices))
        batch_indices = pilot_indices[low:high]
        partial_path = partial_dir / f"batch_{batch_number:03d}.pt"
        partial_identity = payload_identity(
            protocol_hash=protocol_hash,
            method=args.method,
            attack=args.attack,
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
            raise AssertionError("Local dataset labels differ from frozen R3 pilot rows")
        clean = clean_cpu.to(device)
        labels = labels_cpu.to(device)
        reference_top2 = torch.tensor(
            frozen_references[low:high], dtype=torch.long, device=device
        )
        proposal_seed = core.stable_int(
            f"{protocol_hash}|{args.method}|{args.attack}|{args.search_seed}|batch={batch_number}|proposal"
        )
        oracle_seed = core.stable_int(
            f"{protocol_hash}|{args.method}|{args.attack}|{args.search_seed}|batch={batch_number}|oracle"
        )
        batch_started = time.monotonic()
        adversarial, details = core.run_square_batch(
            attack=args.attack,
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
        partial = {
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

    keys = ("clean", "adversarial", "labels", "sample_indices", "reference_top2")
    details = payloads[0]["query_details"]
    if any(payload["query_details"] != details for payload in payloads):
        raise AssertionError("Query plan changed across batches")
    candidate: dict[str, Any] = {
        **identity,
        **{key: torch.cat([payload[key] for payload in payloads], dim=0) for key in keys},
        "query_details": details,
        "batches": batches,
        "batch_size": int(args.batch_size),
        "partial_payload_hashes": [payload["payload_hash"] for payload in payloads],
        "search_wall_seconds": float(time.monotonic() - started),
        "candidate_schema": "clean_and_adversarial_v1",
    }
    add_payload_hash(candidate)
    validate_payload(candidate, identity, pilot_indices)
    candidate_path = output_dir / "candidate.pt"
    atomic_torch_save(candidate_path, candidate)
    search_rows = []
    linf = (candidate["adversarial"] - candidate["clean"]).abs().flatten(1).max(1).values
    for position, index in enumerate(pilot_indices):
        reference = candidate["reference_top2"][position].tolist()
        search_rows.append(
            {
                "protocol_id": common.PROTOCOL_ID,
                "protocol_hash": protocol_hash,
                "method": args.method,
                "attack": args.attack,
                "search_seed": args.search_seed,
                "sample_index": index,
                "label": int(candidate["labels"][position]),
                "reference_top1": int(reference[0]),
                "reference_set": f"{min(reference)}|{max(reference)}",
                "epsilon_px": common.EPSILON_PX,
                "linf": float(linf[position]),
                "api_queries": common.API_BUDGET,
                "candidate_tensor_sha256": candidate["tensor_sha256"]["adversarial"],
            }
        )
    common.atomic_csv(output_dir / "search_per_example.csv", search_rows)
    summary = {
        **identity,
        "status": "PASS",
        "examples": len(pilot_indices),
        "candidate_schema": candidate["candidate_schema"],
        "candidate_path": str(candidate_path),
        "candidate_file_sha256": common.file_sha256(candidate_path),
        "candidate_payload_hash": candidate["payload_hash"],
        "tensor_sha256": candidate["tensor_sha256"],
        "query_details": details,
        "maximum_linf": float(linf.max()),
        "search_wall_seconds": candidate["search_wall_seconds"],
        "independent_mc1000_complete": False,
        "confirmatory_partition_read": False,
    }
    common.atomic_json(output_dir / "search_summary.json", summary)
    print(
        f"PASS {args.method}/{args.attack}/seed={args.search_seed}: "
        f"200 candidates, exact Q={common.API_BUDGET}, max Linf={float(linf.max()):.8f}",
        flush=True,
    )
    print(f"Complete: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
