#!/usr/bin/env python3
"""Run one frozen ImageNet R4 method/attack/seed search cell."""

from __future__ import annotations

import argparse
import contextlib
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

import calibrate_topk_imagenet_defense_r2 as r2
import topk_imagenet_attack_pilot_r4_common as common
import topk_imagenet_square_attack_core_r4 as core
from prepare_topk_imagenet_attack_pilot_r4 import validate_input_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-json", type=Path, required=True)
    parser.add_argument("--method", choices=common.METHODS, required=True)
    parser.add_argument("--attack", choices=common.ATTACKS, required=True)
    parser.add_argument("--search-seed", type=int, choices=common.SEARCH_SEEDS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.45)
    return parser.parse_args()


def autocast_off(device: torch.device):
    return torch.autocast(device_type="cuda", enabled=False) if device.type == "cuda" else contextlib.nullcontext()


class FrozenDefense:
    """The R2-selected defense exposed as one black-box logit API."""

    def __init__(self, spec: common.MethodSpec, standard: nn.Module, robust: nn.Module, device: torch.device):
        self.spec = spec
        self.standard = standard
        self.robust = robust
        self.device = device
        self.sigma = float(spec.sigma)

    def base(self, x: torch.Tensor) -> torch.Tensor:
        if self.spec.strength_steps <= 0:
            return x.detach()
        # no_grad, rather than inference_mode, keeps this reference compatible
        # with the subsequent input-gradient graph.
        with torch.no_grad(), autocast_off(self.device):
            reference = self.standard(x).float().topk(common.PRIMARY_K, dim=1, sorted=True).indices.clone()
        trajectory = r2.strengthen_trajectory(
            x,
            reference,
            self.robust,
            self.spec.strength_alpha_px,
            self.spec.strength_epsilon_px,
            [self.spec.strength_steps],
            self.spec.top1_weight,
            self.spec.set_weight,
            False,
        )
        return trajectory[self.spec.strength_steps]

    def draw_logits(self, x: torch.Tensor, draws: int, chunk_size: int = 10) -> torch.Tensor:
        draws, chunk_size = int(draws), int(chunk_size)
        if draws < 1 or chunk_size < 1:
            raise ValueError("draws and chunk_size must be positive")
        base = self.base(x)
        if self.sigma <= 0:
            with torch.no_grad(), autocast_off(self.device):
                logits = self.robust(base).float()
            return logits.unsqueeze(0).expand(draws, -1, -1).clone()
        outputs: list[torch.Tensor] = []
        completed = 0
        while completed < draws:
            chunk = min(chunk_size, draws - completed)
            noise = torch.randn((chunk,) + tuple(base.shape), device=base.device, dtype=base.dtype)
            queried = (base.unsqueeze(0) + self.sigma * noise).clamp(0.0, 1.0)
            with torch.no_grad(), autocast_off(self.device):
                logits = self.robust(queried.flatten(0, 1)).float().view(chunk, len(base), 1000)
            outputs.append(logits)
            completed += chunk
        return torch.cat(outputs, dim=0)


def atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


TENSOR_KEYS = ("clean", "adversarial", "labels", "pilot_positions", "reference_top5")


def add_payload_hash(payload: dict[str, Any]) -> None:
    payload["tensor_sha256"] = {key: common.tensor_sha256(payload[key]) for key in TENSOR_KEYS}
    hash_input = {key: value for key, value in payload.items() if key not in set(TENSOR_KEYS) | {"payload_hash"}}
    payload["payload_hash"] = common.object_sha256(hash_input)


def validate_candidate_payload(
    payload: Mapping[str, Any], identity: Mapping[str, Any], expected_positions: Sequence[int],
    expected_paths: Sequence[str],
) -> None:
    for key, value in identity.items():
        if payload.get(key) != value:
            raise AssertionError(f"Candidate identity mismatch: {key}")
    for key in TENSOR_KEYS:
        if key not in payload:
            raise AssertionError(f"Candidate misses {key}")
    clean = payload["clean"].detach().cpu()
    adversarial = payload["adversarial"].detach().cpu()
    labels = payload["labels"].detach().cpu()
    positions = payload["pilot_positions"].detach().cpu()
    reference = payload["reference_top5"].detach().cpu()
    count = len(expected_positions)
    if clean.shape != (count, 3, 224, 224) or adversarial.shape != clean.shape:
        raise AssertionError("Candidate image shape mismatch")
    if labels.shape != (count,) or positions.shape != (count,) or reference.shape != (count, 5):
        raise AssertionError("Candidate metadata tensor shape mismatch")
    if int(reference.min()) < 0 or int(reference.max()) >= 1000:
        raise AssertionError("Candidate reference contains an invalid class id")
    if not reference.sort(1).values.diff(dim=1).ne(0).all():
        raise AssertionError("Candidate reference contains duplicate Top-5 members")
    if positions.tolist() != [int(value) for value in expected_positions]:
        raise AssertionError("Candidate pilot order mismatch")
    if list(payload.get("relative_paths", [])) != list(expected_paths):
        raise AssertionError("Candidate path order mismatch")
    if not torch.isfinite(clean).all() or not torch.isfinite(adversarial).all():
        raise AssertionError("Candidate contains non-finite pixels")
    if float(adversarial.min()) < -1e-7 or float(adversarial.max()) > 1 + 1e-7:
        raise AssertionError("Candidate pixels are outside [0,1]")
    linf = (adversarial - clean).abs().flatten(1).max(1).values
    if float(linf.max()) > common.EPSILON_PX / 255.0 + 1e-6:
        raise AssertionError("Candidate violates the frozen L-infinity radius")
    for key in TENSOR_KEYS:
        if common.tensor_sha256(payload[key]) != payload.get("tensor_sha256", {}).get(key):
            raise AssertionError(f"Candidate tensor hash mismatch: {key}")
    hash_input = {key: value for key, value in payload.items() if key not in set(TENSOR_KEYS) | {"payload_hash"}}
    if common.object_sha256(hash_input) != payload.get("payload_hash"):
        raise AssertionError("Candidate payload hash mismatch")


def identity(
    protocol: Mapping[str, Any], method: str, attack: str, seed: int,
    pilot_payload_hash: str,
) -> dict[str, Any]:
    return {
        "protocol_id": common.PROTOCOL_ID,
        "protocol_hash": protocol["protocol_hash"],
        "method": method,
        "attack": attack,
        "search_seed": int(seed),
        "primary_k": common.PRIMARY_K,
        "epsilon_px": common.EPSILON_PX,
        "api_budget": common.API_BUDGET,
        "source_pilot_payload_hash": str(pilot_payload_hash),
    }


def main() -> None:
    args = parse_args()
    protocol = common.read_json(args.protocol_json)
    common.validate_protocol(protocol)
    expected_batch = common.SEARCH_BATCH_SIZE_BY_METHOD[args.method]
    if args.batch_size != expected_batch:
        raise ValueError(f"Frozen {args.method} batch size is {expected_batch}")
    if not 0 < args.gpu_memory_fraction <= 1:
        raise ValueError("--gpu-memory-fraction must be in (0,1]")
    output_dir = args.output_dir.expanduser().resolve()
    expected_dir = common.cell_dir(Path(protocol["paths"]["output_root"]), args.method, args.attack, args.search_seed).resolve()
    if output_dir != expected_dir:
        raise AssertionError(f"Cell output path is not frozen: {output_dir} != {expected_dir}")
    input_path = Path(protocol["paths"]["output_root"]) / "pilot_inputs.pt"
    pilot = torch.load(input_path, map_location="cpu", weights_only=False)
    validate_input_payload(pilot, protocol)
    prepare_summary = common.read_json(Path(protocol["paths"]["output_root"]) / "prepare_summary.json")
    if prepare_summary.get("status") != "PASS" or prepare_summary.get("protocol_hash") != protocol["protocol_hash"]:
        raise AssertionError("Pilot preparation is not complete")
    if prepare_summary.get("pilot_inputs_payload_hash") != pilot.get("payload_hash"):
        raise AssertionError("Pilot payload differs from preparation summary")
    if common.file_sha256(input_path) != prepare_summary.get("pilot_inputs_file_sha256"):
        raise AssertionError("pilot_inputs.pt bytes changed after protocol finalization")
    base_identity = identity(
        protocol, args.method, args.attack, args.search_seed, str(pilot["payload_hash"])
    )
    all_positions = [int(value) for value in pilot["pilot_positions"]]
    all_paths = list(pilot["relative_paths"])
    candidate_path = output_dir / "candidate.pt"
    summary_path = output_dir / "search_summary.json"
    if candidate_path.is_file() and summary_path.is_file():
        payload = torch.load(candidate_path, map_location="cpu", weights_only=False)
        validate_candidate_payload(payload, base_identity, all_positions, all_paths)
        summary = common.read_json(summary_path)
        if summary.get("candidate_payload_hash") != payload["payload_hash"] or summary.get("status") != "PASS":
            raise AssertionError("Completed cell summary/candidate mismatch")
        print(f"RESUME PASS {args.method}/{args.attack}/seed={args.search_seed}", flush=True)
        return

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, device=device)
    standard_path = Path(protocol["checkpoints"]["standard"]["path"])
    robust_path = Path(protocol["checkpoints"]["robust"]["path"])
    standard, robust, _audit = r2.load_models(standard_path, robust_path, device)
    defense = FrozenDefense(common.method_spec(args.method), standard, robust, device)
    partial_dir = output_dir / "partials"
    partial_dir.mkdir(parents=True, exist_ok=True)
    starts = list(range(0, common.PILOT_EXAMPLES, args.batch_size))
    wall_start = time.perf_counter()
    partial_payloads: list[Mapping[str, Any]] = []
    for batch_number, start in enumerate(starts):
        stop = min(start + args.batch_size, common.PILOT_EXAMPLES)
        batch_positions = all_positions[start:stop]
        batch_paths = all_paths[start:stop]
        partial_path = partial_dir / f"batch_{batch_number:03d}.pt"
        partial_identity = {**base_identity, "batch_number": batch_number, "batch_start": start, "batch_stop": stop}
        if partial_path.is_file():
            partial = torch.load(partial_path, map_location="cpu", weights_only=False)
            validate_candidate_payload(partial, partial_identity, batch_positions, batch_paths)
            print(f"RESUME batch {batch_number + 1}/{len(starts)}", flush=True)
        else:
            clean = pilot["clean"][start:stop].to(device)
            labels = pilot["labels"][start:stop].to(device)
            reference = pilot["reference_top5"][start:stop].to(device)
            proposal_seed = common.stable_seed(args.search_seed, f"{args.method}|{args.attack}|batch={batch_number}|proposal")
            oracle_seed = common.stable_seed(args.search_seed, f"{args.method}|{args.attack}|batch={batch_number}|oracle")
            batch_start = time.perf_counter()
            adversarial, details = core.run_square_batch(
                attack=args.attack,
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
                raise AssertionError("Attack failed exact API-query accounting")
            partial = {
                **partial_identity,
                "relative_paths": batch_paths,
                "clean": clean.detach().cpu(),
                "adversarial": adversarial.detach().cpu(),
                "labels": labels.detach().cpu(),
                "pilot_positions": torch.tensor(batch_positions, dtype=torch.long),
                "reference_top5": reference.detach().cpu(),
                "query_details": details,
                "proposal_seed": proposal_seed,
                "oracle_seed": oracle_seed,
                "wall_seconds": time.perf_counter() - batch_start,
            }
            add_payload_hash(partial)
            validate_candidate_payload(partial, partial_identity, batch_positions, batch_paths)
            atomic_torch_save(partial_path, partial)
            print(f"PASS batch {batch_number + 1}/{len(starts)}: Q={details['api_queries']}", flush=True)
        partial_payloads.append(partial)
    for key in ("clean", "adversarial", "labels", "pilot_positions", "reference_top5"):
        values = [part[key] for part in partial_payloads]
        combined = torch.cat(values, dim=0)
        if key == "clean": clean_all = combined
        elif key == "adversarial": adversarial_all = combined
        elif key == "labels": labels_all = combined
        elif key == "pilot_positions": positions_all = combined
        else: reference_all = combined
    candidate: dict[str, Any] = {
        **base_identity,
        "relative_paths": all_paths,
        "clean": clean_all,
        "adversarial": adversarial_all,
        "labels": labels_all,
        "pilot_positions": positions_all,
        "reference_top5": reference_all,
        "query_details": partial_payloads[0]["query_details"],
        "partial_payload_hashes": [part["payload_hash"] for part in partial_payloads],
        "partial_files": len(partial_payloads),
        "wall_seconds": time.perf_counter() - wall_start,
    }
    if any(part["query_details"] != candidate["query_details"] for part in partial_payloads):
        raise AssertionError("Query plan differs across batches")
    add_payload_hash(candidate)
    validate_candidate_payload(candidate, base_identity, all_positions, all_paths)
    atomic_torch_save(candidate_path, candidate)
    summary = {
        **base_identity,
        "candidate_payload_hash": candidate["payload_hash"],
        "candidate_file_sha256": common.file_sha256(candidate_path),
        "candidate_tensor_sha256": candidate["tensor_sha256"]["adversarial"],
        "examples": common.PILOT_EXAMPLES,
        "batches": len(partial_payloads),
        "query_details": candidate["query_details"],
        "max_linf_px": 255.0 * float((adversarial_all - clean_all).abs().flatten(1).max()),
        "wall_seconds": candidate["wall_seconds"],
        "paper_claim_authorized": False,
        "status": "PASS",
    }
    common.atomic_json(summary_path, summary)
    print(f"PASS {args.method}/{args.attack}/seed={args.search_seed}: "
          f"Q={common.API_BUDGET}, max Linf={summary['max_linf_px']:.5f}/255", flush=True)


if __name__ == "__main__":
    main()
