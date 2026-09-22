#!/usr/bin/env python3
"""Freeze the outcome-blind ImageNet R4 pilot and its clean Top-5 targets."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Mapping

import torch

import calibrate_topk_imagenet_defense_r2 as r2
import topk_imagenet_attack_pilot_r4_common as common


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=common.DEFAULT_DATA_ROOT)
    parser.add_argument("--r0-dir", type=Path, default=common.DEFAULT_R0_DIR)
    parser.add_argument("--r2-dir", type=Path, default=common.DEFAULT_R2_DIR)
    parser.add_argument("--r3-dir", type=Path, default=common.DEFAULT_R3_DIR)
    parser.add_argument("--output-root", type=Path, default=common.DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.45)
    return parser.parse_args()


def atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def validate_input_payload(payload: Mapping[str, Any], protocol: Mapping[str, Any]) -> None:
    for key in ("protocol_id", "protocol_hash"):
        if payload.get(key) != protocol.get(key):
            raise AssertionError(f"Pilot-input identity mismatch: {key}")
    required = ("clean", "labels", "reference_top5")
    for key in required:
        if key not in payload:
            raise AssertionError(f"Pilot-input payload misses {key}")
    clean = payload["clean"].detach().cpu()
    labels = payload["labels"].detach().cpu()
    reference = payload["reference_top5"].detach().cpu()
    paths = list(payload.get("relative_paths", []))
    if clean.shape != (common.PILOT_EXAMPLES, 3, 224, 224):
        raise AssertionError(f"Unexpected clean tensor shape: {tuple(clean.shape)}")
    if labels.shape != (common.PILOT_EXAMPLES,) or reference.shape != (common.PILOT_EXAMPLES, 5):
        raise AssertionError("Unexpected pilot label/reference shape")
    if len(paths) != common.PILOT_EXAMPLES or len(set(paths)) != len(paths):
        raise AssertionError("Pilot paths are missing or duplicated")
    if not torch.isfinite(clean).all() or float(clean.min()) < 0 or float(clean.max()) > 1:
        raise AssertionError("Pilot pixels are invalid")
    if not torch.equal(reference[:, 0], payload["standard_prediction"].detach().cpu()):
        raise AssertionError("Reference Top-1 differs from standard prediction")
    saved = payload.get("tensor_sha256", {})
    for key in required + ("standard_prediction",):
        if common.tensor_sha256(payload[key]) != saved.get(key):
            raise AssertionError(f"Pilot tensor hash mismatch: {key}")
    hash_input = {
        key: value for key, value in payload.items()
        if key not in set(required) | {"standard_prediction", "payload_hash"}
    }
    if common.object_sha256(hash_input) != payload.get("payload_hash"):
        raise AssertionError("Pilot-input payload hash mismatch")


def prepare_summary_payload(
    payload: Mapping[str, Any], payload_path: Path, manifest_path: Path
) -> dict[str, Any]:
    return {
        "protocol_id": common.PROTOCOL_ID,
        "protocol_hash": payload["protocol_hash"],
        "pilot_inputs_payload_hash": payload["payload_hash"],
        "pilot_inputs_file_sha256": common.file_sha256(payload_path),
        "pilot_manifest_file_sha256": common.file_sha256(manifest_path),
        "pilot_examples": common.PILOT_EXAMPLES,
        "pilot_distinct_classes": len(set(payload["class_names"])),
        "standard_clean_top1_percent": payload["standard_clean_top1_percent"],
        "standard_clean_top5_percent": payload["standard_clean_top5_percent"],
        "model_load_audit": payload["model_load_audit"],
        "confirmatory_outcomes_read": False,
        "paper_claim_authorized": False,
        "status": "PASS",
    }


def main() -> None:
    args = parse_args()
    if args.num_workers < 0 or not 0 < args.gpu_memory_fraction <= 1:
        raise ValueError("Invalid worker or GPU-memory setting")
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    parent = common.validate_parent(args.r0_dir, args.r2_dir, args.r3_dir)
    pilot_rows = common.select_pilot_rows(parent["split_rows"])
    protocol = common.build_protocol_payload(
        parent=parent,
        data_root=args.data_root,
        output_root=output_root,
        core_path=common.DEFAULT_CIFAR_CORE,
    )
    common.validate_protocol(protocol)
    common.write_or_validate_json(output_root / "attack_pilot_protocol.json", protocol)

    manifest_rows = [
        {
            **row,
            "protocol_id": common.PROTOCOL_ID,
            "protocol_hash": protocol["protocol_hash"],
        }
        for row in pilot_rows
    ]
    manifest_path = output_root / "attack_pilot_manifest.csv"
    if manifest_path.is_file():
        observed = common.read_csv(manifest_path)
        canonical = [
            {key: str(value) for key, value in row.items()} for row in manifest_rows
        ]
        if observed != canonical:
            raise AssertionError("Frozen pilot manifest differs from this invocation")
    else:
        common.atomic_csv(manifest_path, manifest_rows)

    payload_path = output_root / "pilot_inputs.pt"
    if payload_path.is_file():
        payload = torch.load(payload_path, map_location="cpu", weights_only=False)
        validate_input_payload(payload, protocol)
        if list(payload["relative_paths"]) != [str(row["relative_path"]) for row in pilot_rows]:
            raise AssertionError("Stored pilot paths differ from the frozen manifest")
        if payload["labels"].tolist() != [int(row["target"]) for row in pilot_rows]:
            raise AssertionError("Stored pilot labels differ from the frozen manifest")
        common.write_or_validate_json(
            output_root / "prepare_summary.json",
            prepare_summary_payload(payload, payload_path, manifest_path),
        )
        print(f"RESUME validated {payload_path}", flush=True)
        print("PILOT_SEARCH_READY=YES", flush=True)
        print("CONFIRMATORY_OUTCOMES_READ=NO", flush=True)
        return

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, device=device)
    standard_path = Path(protocol["checkpoints"]["standard"]["path"])
    robust_path = Path(protocol["checkpoints"]["robust"]["path"])
    standard, _robust, model_audit = r2.load_models(standard_path, robust_path, device)
    loader = r2.make_loader(
        args.data_root.expanduser().resolve(), pilot_rows, common.REFERENCE_BATCH_SIZE,
        args.num_workers, device,
    )
    images_all: list[torch.Tensor] = []
    labels_all: list[torch.Tensor] = []
    reference_all: list[torch.Tensor] = []
    predictions_all: list[torch.Tensor] = []
    paths_all: list[str] = []
    # Keep stored clean tensors ordinary (non-inference) tensors because R4's
    # full defense later differentiates with respect to its input.
    with torch.no_grad():
        for images, labels, paths in loader:
            images = images.to(device, non_blocking=True)
            logits = standard(images).float()
            reference = logits.topk(common.PRIMARY_K, dim=1, sorted=True).indices
            images_all.append(images.detach().cpu().clone())
            labels_all.append(labels.detach().cpu().clone())
            reference_all.append(reference.detach().cpu().clone())
            predictions_all.append(logits.argmax(1).detach().cpu().clone())
            paths_all.extend(str(path) for path in paths)
    clean = torch.cat(images_all)
    labels = torch.cat(labels_all).to(torch.long)
    reference = torch.cat(reference_all).to(torch.long)
    predictions = torch.cat(predictions_all).to(torch.long)
    expected_paths = [str(row["relative_path"]) for row in pilot_rows]
    if paths_all != expected_paths:
        raise AssertionError("DataLoader changed the frozen pilot order")
    tensors = {
        "clean": clean,
        "labels": labels,
        "reference_top5": reference,
        "standard_prediction": predictions,
    }
    payload: dict[str, Any] = {
        "protocol_id": common.PROTOCOL_ID,
        "protocol_hash": protocol["protocol_hash"],
        "partition": "attack_pilot",
        "examples": common.PILOT_EXAMPLES,
        "relative_paths": paths_all,
        "class_names": [str(row["class_name"]) for row in pilot_rows],
        "pilot_positions": list(range(common.PILOT_EXAMPLES)),
        **tensors,
        "tensor_sha256": {key: common.tensor_sha256(value) for key, value in tensors.items()},
        "standard_clean_top1_percent": 100.0 * float(predictions.eq(labels).float().mean()),
        "standard_clean_top5_percent": 100.0 * float(reference.eq(labels[:, None]).any(1).float().mean()),
        "model_load_audit": model_audit,
    }
    hash_input = {
        key: value for key, value in payload.items()
        if key not in set(tensors) | {"payload_hash"}
    }
    payload["payload_hash"] = common.object_sha256(hash_input)
    validate_input_payload(payload, protocol)
    atomic_torch_save(payload_path, payload)
    common.atomic_json(
        output_root / "prepare_summary.json",
        prepare_summary_payload(payload, payload_path, manifest_path),
    )
    print(f"PASS froze {common.PILOT_EXAMPLES} outcome-blind pilot images", flush=True)
    print(f"  standard Top1={payload['standard_clean_top1_percent']:.3f}% "
          f"Top5={payload['standard_clean_top5_percent']:.3f}%", flush=True)
    print("PILOT_SEARCH_READY=YES", flush=True)
    print("CONFIRMATORY_OUTCOMES_READ=NO", flush=True)


if __name__ == "__main__":
    main()
