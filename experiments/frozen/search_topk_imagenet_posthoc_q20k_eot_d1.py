#!/usr/bin/env python3
"""Run one resumable method/seed cell for the ImageNet D1 Q=20k stress test."""

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
import topk_cifar100_square_attack_core_r4 as core
import topk_imagenet_posthoc_q20k_eot_d1_common as common
from prepare_topk_imagenet_posthoc_q20k_eot_d1 import validate_input_payload


TENSOR_KEYS = (
    "adversarial",
    "labels",
    "diagnostic_positions",
    "source_confirmatory_positions",
    "reference_top5",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-json", type=Path, required=True)
    parser.add_argument("--method", choices=common.METHODS, required=True)
    parser.add_argument("--search-seed", type=int, choices=common.SEARCH_SEEDS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.45)
    return parser.parse_args()


def autocast_off(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", enabled=False)
    return contextlib.nullcontext()


class FrozenDefense:
    """The unchanged R5 defense exposed as a black-box full-logit API."""

    def __init__(
        self,
        spec: Any,
        standard: nn.Module,
        robust: nn.Module,
        device: torch.device,
    ) -> None:
        self.spec = spec
        self.standard = standard
        self.robust = robust
        self.device = device
        self.sigma = float(spec.sigma)

    def base(self, x: torch.Tensor) -> torch.Tensor:
        if self.spec.strength_steps <= 0:
            return x.detach()
        with torch.no_grad(), autocast_off(self.device):
            reference = (
                self.standard(x)
                .float()
                .topk(common.PRIMARY_K, dim=1, sorted=True)
                .indices.clone()
            )
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

    def draw_logits(
        self, x: torch.Tensor, draws: int, chunk_size: int = 10
    ) -> torch.Tensor:
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
            noise = torch.randn(
                (chunk,) + tuple(base.shape),
                device=base.device,
                dtype=base.dtype,
            )
            queried = (base.unsqueeze(0) + self.sigma * noise).clamp(0.0, 1.0)
            with torch.no_grad(), autocast_off(self.device):
                logits = (
                    self.robust(queried.flatten(0, 1))
                    .float()
                    .view(chunk, len(base), 1000)
                )
            outputs.append(logits)
            completed += chunk
        return torch.cat(outputs, dim=0)


def atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def payload_hash_input(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if key not in set(TENSOR_KEYS) | {"payload_hash"}
    }


def add_payload_hash(payload: dict[str, Any]) -> None:
    payload["tensor_sha256"] = {
        key: common.tensor_sha256(payload[key]) for key in TENSOR_KEYS
    }
    payload["payload_hash"] = common.object_sha256(payload_hash_input(payload))


def candidate_identity(
    protocol: Mapping[str, Any],
    method: str,
    search_seed: int,
    input_payload_hash: str,
    clean_hash: str,
) -> dict[str, Any]:
    config = common.ADAPTIVE_CONFIG_BY_METHOD[method]
    return {
        "protocol_id": common.PROTOCOL_ID,
        "protocol_hash": protocol["protocol_hash"],
        "parent_r5_protocol_hash": common.EXPECTED_R5_PROTOCOL_HASH,
        "method": method,
        "attack": common.METHOD_ATTACK[method],
        "search_seed": int(search_seed),
        "primary_k": common.PRIMARY_K,
        "secondary_k": common.SECONDARY_K,
        "epsilon_px": common.EPSILON_PX,
        "api_budget": common.API_BUDGET,
        "search_eot": int(config["search_eot"]),
        "refresh_eot": int(config["refresh_eot"]),
        "source_diagnostic_payload_hash": str(input_payload_hash),
        "source_clean_tensor_sha256": str(clean_hash),
        "post_hoc": True,
        "paper_claim_authorized": False,
    }


def validate_candidate_payload(
    payload: Mapping[str, Any],
    expected: Mapping[str, Any],
    expected_positions: Sequence[int],
    expected_source_positions: Sequence[int],
    expected_paths: Sequence[str],
    expected_labels: torch.Tensor,
    expected_reference: torch.Tensor,
    *,
    clean: torch.Tensor | None = None,
) -> None:
    for key, value in expected.items():
        if payload.get(key) != value:
            raise AssertionError(f"D1 candidate identity mismatch: {key}")
    for key in TENSOR_KEYS:
        if key not in payload:
            raise AssertionError(f"D1 candidate misses {key}")
    adversarial = payload["adversarial"].detach().cpu()
    labels = payload["labels"].detach().cpu()
    positions = payload["diagnostic_positions"].detach().cpu()
    source_positions = payload["source_confirmatory_positions"].detach().cpu()
    reference = payload["reference_top5"].detach().cpu()
    count = len(expected_positions)
    if adversarial.shape != (count, 3, 224, 224):
        raise AssertionError("D1 candidate image shape changed")
    if labels.shape != (count,) or reference.shape != (count, common.PRIMARY_K):
        raise AssertionError("D1 candidate metadata shape changed")
    if positions.tolist() != [int(value) for value in expected_positions]:
        raise AssertionError("D1 candidate positions changed")
    if source_positions.tolist() != [int(value) for value in expected_source_positions]:
        raise AssertionError("D1 source positions changed")
    if list(payload.get("relative_paths", [])) != list(expected_paths):
        raise AssertionError("D1 candidate path order changed")
    if not torch.equal(labels, expected_labels.detach().cpu()):
        raise AssertionError("D1 candidate labels changed")
    if not torch.equal(reference, expected_reference.detach().cpu()):
        raise AssertionError("D1 candidate references changed")
    if not torch.isfinite(adversarial).all():
        raise AssertionError("D1 candidate contains non-finite pixels")
    if float(adversarial.min()) < -1e-7 or float(adversarial.max()) > 1 + 1e-7:
        raise AssertionError("D1 candidate pixels are outside [0,1]")
    if clean is not None:
        clean_cpu = clean.detach().cpu()
        if clean_cpu.shape != adversarial.shape:
            raise AssertionError("D1 clean/candidate shape mismatch")
        linf = (adversarial - clean_cpu).abs().flatten(1).max(1).values
        if float(linf.max()) > common.EPSILON_PX / 255.0 + 1e-6:
            raise AssertionError("D1 candidate violates the L-infinity radius")
        if payload.get("clean_tensor_sha256") != common.tensor_sha256(clean_cpu):
            raise AssertionError("D1 candidate clean-batch hash changed")
    for key in TENSOR_KEYS:
        if common.tensor_sha256(payload[key]) != payload.get("tensor_sha256", {}).get(key):
            raise AssertionError(f"D1 candidate tensor hash mismatch: {key}")
    if common.object_sha256(payload_hash_input(payload)) != payload.get("payload_hash"):
        raise AssertionError("D1 candidate payload hash mismatch")


def validate_resume(
    output_dir: Path,
    identity: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> bool:
    candidate_path = output_dir / "candidate.pt"
    summary_path = output_dir / "search_summary.json"
    rows_path = output_dir / "search_per_example.csv"
    existing = [path.exists() for path in (candidate_path, summary_path, rows_path)]
    if not any(existing):
        return False
    if not all(existing):
        print("RECOVER incomplete D1 final cell from validated partials", flush=True)
        return False
    payload = torch.load(candidate_path, map_location="cpu", weights_only=False)
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
    if summary.get("status") != "PASS":
        raise AssertionError("D1 resume summary did not pass")
    if summary.get("candidate_file_sha256") != common.file_sha256(candidate_path):
        raise AssertionError("D1 resume candidate file hash changed")
    if summary.get("candidate_payload_hash") != payload.get("payload_hash"):
        raise AssertionError("D1 resume candidate payload hash changed")
    if summary.get("search_per_example_sha256") != common.file_sha256(rows_path):
        raise AssertionError("D1 resume search-row bytes changed")
    details = summary.get("query_details", {})
    if int(details.get("api_queries", -1)) != common.API_BUDGET:
        raise AssertionError("D1 resume did not use exact Q=20000")
    if int(details.get("search_eot", -1)) != int(identity["search_eot"]):
        raise AssertionError("D1 resume search EOT changed")
    rows = common.read_csv(rows_path)
    if len(rows) != common.DIAGNOSTIC_EXAMPLES:
        raise AssertionError("D1 resume row count changed")
    return True


def main() -> None:
    args = parse_args()
    expected_batch = common.SEARCH_BATCH_SIZE_BY_METHOD[args.method]
    if args.batch_size != expected_batch:
        raise ValueError(f"Frozen D1 batch size for {args.method} is {expected_batch}")
    if not 0 < args.gpu_memory_fraction <= 1:
        raise ValueError("--gpu-memory-fraction must be in (0,1]")
    protocol = common.read_json(args.protocol_json.expanduser().resolve())
    protocol_hash = common.validate_protocol(protocol)
    output_root = Path(protocol["paths"]["output_root"]).expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    expected_dir = common.cell_dir(output_root, args.method, args.search_seed).resolve()
    if output_dir != expected_dir:
        raise AssertionError("D1 cell output path differs from the frozen protocol")
    parent = common.load_parent_r5(Path(protocol["paths"]["r5_root"]))
    subset = parent["subset"]
    input_path = output_root / "diagnostic_inputs.pt"
    inputs = torch.load(input_path, map_location="cpu", weights_only=False)
    validate_input_payload(inputs, protocol, subset)
    prepare_summary = common.read_json(output_root / "prepare_summary.json")
    if (
        prepare_summary.get("status") != "PASS"
        or prepare_summary.get("diagnostic_inputs_payload_hash") != inputs["payload_hash"]
        or prepare_summary.get("diagnostic_inputs_file_sha256")
        != common.file_sha256(input_path)
    ):
        raise PermissionError("D1 preparation is incomplete or changed")

    identity = candidate_identity(
        protocol,
        args.method,
        args.search_seed,
        str(inputs["payload_hash"]),
        str(inputs["tensor_sha256"]["clean"]),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if validate_resume(output_dir, identity, inputs):
        print(
            f"RESUME PASS D1 {args.method}/seed={args.search_seed}", flush=True
        )
        return

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(
            args.gpu_memory_fraction, device=device
        )
    standard, robust, _audit = r2.load_models(
        Path(protocol["checkpoints"]["standard"]["path"]),
        Path(protocol["checkpoints"]["robust"]["path"]),
        device,
    )
    defense = FrozenDefense(common.method_spec(args.method), standard, robust, device)
    attack = common.METHOD_ATTACK[args.method]
    adaptive_config = common.ADAPTIVE_CONFIG_BY_METHOD[args.method]
    partial_dir = output_dir / "partials"
    partial_dir.mkdir(parents=True, exist_ok=True)
    starts = list(range(0, common.DIAGNOSTIC_EXAMPLES, args.batch_size))
    partials: list[dict[str, Any]] = []
    wall_start = time.perf_counter()

    for batch_number, low in enumerate(starts):
        high = min(low + args.batch_size, common.DIAGNOSTIC_EXAMPLES)
        positions = list(range(low, high))
        source_positions = inputs["source_confirmatory_positions"][low:high]
        expected_paths = list(inputs["relative_paths"])[low:high]
        expected_labels = inputs["labels"][low:high]
        expected_reference = inputs["reference_top5"][low:high]
        clean_cpu = inputs["clean"][low:high].detach().cpu().contiguous()
        partial_path = partial_dir / f"batch_{batch_number:03d}.pt"
        partial_identity = {
            **identity,
            "batch_number": batch_number,
            "batch_start": low,
            "batch_stop": high,
        }
        if partial_path.is_file():
            partial = torch.load(partial_path, map_location="cpu", weights_only=False)
            validate_candidate_payload(
                partial,
                partial_identity,
                positions,
                source_positions.tolist(),
                expected_paths,
                expected_labels,
                expected_reference,
                clean=clean_cpu,
            )
            print(f"  RESUME batch {batch_number + 1}/{len(starts)}", flush=True)
        else:
            clean = clean_cpu.to(device, non_blocking=True)
            reference = expected_reference.to(device, non_blocking=True)
            proposal_seed = common.stable_seed(
                args.search_seed,
                f"{protocol_hash}|{args.method}|batch={batch_number}|proposal",
            )
            oracle_seed = common.stable_seed(
                args.search_seed,
                f"{protocol_hash}|{args.method}|batch={batch_number}|oracle",
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
                adaptive_config=adaptive_config,
                racing_config=common.RACING_CONFIG,
                proposal_seed=proposal_seed,
                oracle_seed=oracle_seed,
            )
            if int(details["api_queries"]) != common.API_BUDGET:
                raise AssertionError("D1 attack failed exact Q=20000 accounting")
            if int(details["search_eot"]) != int(adaptive_config["search_eot"]):
                raise AssertionError("D1 attack used the wrong search EOT")
            partial = {
                **partial_identity,
                "relative_paths": expected_paths,
                "adversarial": adversarial.detach().cpu().contiguous(),
                "labels": expected_labels.detach().cpu().contiguous(),
                "diagnostic_positions": torch.tensor(positions, dtype=torch.long),
                "source_confirmatory_positions": source_positions.detach()
                .cpu()
                .contiguous(),
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
                source_positions.tolist(),
                expected_paths,
                expected_labels,
                expected_reference,
                clean=clean_cpu,
            )
            atomic_torch_save(partial_path, partial)
            print(
                f"  PASS batch {batch_number + 1}/{len(starts)}: "
                f"Q={details['api_queries']} EOT={details['search_eot']}",
                flush=True,
            )
        partials.append(dict(partial))

    query_details = partials[0]["query_details"]
    if any(partial["query_details"] != query_details for partial in partials):
        raise AssertionError("D1 query plan changed across batches")
    candidate: dict[str, Any] = {
        **identity,
        "relative_paths": list(inputs["relative_paths"]),
        "adversarial": torch.cat([part["adversarial"] for part in partials]),
        "labels": torch.cat([part["labels"] for part in partials]),
        "diagnostic_positions": torch.cat(
            [part["diagnostic_positions"] for part in partials]
        ),
        "source_confirmatory_positions": torch.cat(
            [part["source_confirmatory_positions"] for part in partials]
        ),
        "reference_top5": torch.cat([part["reference_top5"] for part in partials]),
        "clean_tensor_sha256": inputs["tensor_sha256"]["clean"],
        "query_details": query_details,
        "partial_payload_hashes": [part["payload_hash"] for part in partials],
        "batches": len(partials),
        "batch_size": args.batch_size,
        "candidate_schema": "posthoc_shared_clean_plus_adversarial_v1",
        "search_wall_seconds": time.perf_counter() - wall_start,
    }
    add_payload_hash(candidate)
    validate_candidate_payload(
        candidate,
        identity,
        list(range(common.DIAGNOSTIC_EXAMPLES)),
        inputs["source_confirmatory_positions"].tolist(),
        inputs["relative_paths"],
        inputs["labels"],
        inputs["reference_top5"],
        clean=inputs["clean"],
    )
    candidate_path = output_dir / "candidate.pt"
    atomic_torch_save(candidate_path, candidate)

    linf = (
        candidate["adversarial"] - inputs["clean"]
    ).abs().flatten(1).max(1).values
    search_rows: list[dict[str, Any]] = []
    for position in range(common.DIAGNOSTIC_EXAMPLES):
        reference = [int(value) for value in candidate["reference_top5"][position]]
        search_rows.append(
            {
                "protocol_id": common.PROTOCOL_ID,
                "protocol_hash": protocol_hash,
                "method": args.method,
                "attack": attack,
                "search_seed": args.search_seed,
                "diagnostic_position": position,
                "source_confirmatory_position": int(
                    candidate["source_confirmatory_positions"][position]
                ),
                "relative_path": candidate["relative_paths"][position],
                "label": int(candidate["labels"][position]),
                "reference_top1": reference[0],
                "reference_set_k5": "|".join(
                    str(value) for value in sorted(reference)
                ),
                "epsilon_px": common.EPSILON_PX,
                "linf": float(linf[position]),
                "api_queries": common.API_BUDGET,
                "search_eot": identity["search_eot"],
                "refresh_eot": identity["refresh_eot"],
                "candidate_tensor_sha256": candidate["tensor_sha256"][
                    "adversarial"
                ],
            }
        )
    rows_path = output_dir / "search_per_example.csv"
    common.atomic_csv(rows_path, search_rows)
    summary = {
        **identity,
        "status": "PASS",
        "examples": common.DIAGNOSTIC_EXAMPLES,
        "batches": len(partials),
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
        "parent_r5_modified": False,
        "seed_selection_performed": False,
        "attack_family_selection_performed": False,
        "independent_mc1000_complete": False,
    }
    common.atomic_json(output_dir / "search_summary.json", summary)
    print(
        f"PASS D1 {args.method}/seed={args.search_seed}: "
        f"{common.DIAGNOSTIC_EXAMPLES} candidates, Q={common.API_BUDGET}, "
        f"EOT={identity['search_eot']}, max Linf={summary['maximum_linf_px']:.5f}/255",
        flush=True,
    )


if __name__ == "__main__":
    main()
