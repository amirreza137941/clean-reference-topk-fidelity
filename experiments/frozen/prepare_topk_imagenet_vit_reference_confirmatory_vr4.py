#!/usr/bin/env python3
"""Open the frozen ImageNet ViT-reference VR4 split and freeze its clean Top-5 references."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

import calibrate_topk_imagenet_vit_reference_vr1 as vr1
import topk_imagenet_vit_reference_confirmatory_vr4_common as common


TENSOR_KEYS = (
    "labels",
    "confirmatory_positions",
    "reference_top5",
    "standard_prediction",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-json", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
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


def validate_input_payload(
    payload: Mapping[str, Any],
    protocol: Mapping[str, Any],
    expected_rows: list[dict[str, Any]],
) -> None:
    for key in ("protocol_id", "protocol_hash"):
        if payload.get(key) != protocol.get(key):
            raise AssertionError(f"ImageNet ViT-reference VR4 input identity mismatch: {key}")
    if payload.get("split") != "attack_confirmatory":
        raise AssertionError("ImageNet ViT-reference VR4 input split changed")
    if int(payload.get("examples", -1)) != common.CONFIRMATORY_EXAMPLES:
        raise AssertionError("ImageNet ViT-reference VR4 input example count changed")
    if "clean" in payload:
        raise AssertionError("VR4 shared input payload should not duplicate clean image tensors")
    for key in TENSOR_KEYS:
        if key not in payload:
            raise AssertionError(f"ImageNet ViT-reference VR4 input payload misses {key}")

    labels = payload["labels"].detach().cpu()
    positions = payload["confirmatory_positions"].detach().cpu()
    reference = payload["reference_top5"].detach().cpu()
    prediction = payload["standard_prediction"].detach().cpu()
    if labels.shape != (common.CONFIRMATORY_EXAMPLES,):
        raise AssertionError("ImageNet ViT-reference VR4 label tensor shape changed")
    if positions.shape != labels.shape:
        raise AssertionError("ImageNet ViT-reference VR4 position tensor shape changed")
    if reference.shape != (common.CONFIRMATORY_EXAMPLES, common.PRIMARY_K):
        raise AssertionError("ImageNet ViT-reference VR4 reference tensor shape changed")
    if prediction.shape != labels.shape:
        raise AssertionError("ImageNet ViT-reference VR4 standard-prediction shape changed")
    if positions.tolist() != list(range(common.CONFIRMATORY_EXAMPLES)):
        raise AssertionError("ImageNet ViT-reference VR4 confirmatory positions are not canonical")
    common.validate_confirmatory_labels([int(value) for value in labels.tolist()])
    if int(reference.min()) < 0 or int(reference.max()) >= 1000:
        raise AssertionError("ImageNet ViT-reference VR4 reference contains an invalid class")
    if not reference.sort(1).values.diff(dim=1).ne(0).all():
        raise AssertionError("ImageNet ViT-reference VR4 Top-5 reference contains duplicates")
    if not torch.equal(reference[:, 0], prediction):
        raise AssertionError("ImageNet ViT-reference VR4 reference Top-1 differs from standard prediction")

    expected_paths = [str(row["relative_path"]) for row in expected_rows]
    expected_classes = [str(row["class_name"]) for row in expected_rows]
    expected_labels = [int(row["target"]) for row in expected_rows]
    if list(payload.get("relative_paths", [])) != expected_paths:
        raise AssertionError("ImageNet ViT-reference VR4 input paths differ from the frozen manifest")
    if list(payload.get("class_names", [])) != expected_classes:
        raise AssertionError("ImageNet ViT-reference VR4 input class names differ from the frozen manifest")
    if labels.tolist() != expected_labels:
        raise AssertionError("ImageNet ViT-reference VR4 input labels differ from the frozen manifest")
    if payload.get("confirmatory_rows_sha256") != common.EXPECTED_CONFIRMATORY_ROWS_SHA256:
        raise AssertionError("ImageNet ViT-reference VR4 input row hash changed")
    clean_hash = str(payload.get("clean_tensor_sha256", ""))
    if len(clean_hash) != 64 or any(char not in "0123456789abcdef" for char in clean_hash):
        raise AssertionError("ImageNet ViT-reference VR4 clean-tensor hash is invalid")

    for key in TENSOR_KEYS:
        if common.tensor_sha256(payload[key]) != payload.get("tensor_sha256", {}).get(key):
            raise AssertionError(f"ImageNet ViT-reference VR4 input tensor hash mismatch: {key}")
    hash_input = {
        key: value
        for key, value in payload.items()
        if key not in set(TENSOR_KEYS) | {"payload_hash"}
    }
    if common.object_sha256(hash_input) != payload.get("payload_hash"):
        raise AssertionError("ImageNet ViT-reference VR4 input payload hash mismatch")


def summary_payload(
    payload: Mapping[str, Any], payload_path: Path, manifest_path: Path
) -> dict[str, Any]:
    return {
        "protocol_id": common.PROTOCOL_ID,
        "protocol_hash": payload["protocol_hash"],
        "status": "PASS",
        "split": "attack_confirmatory",
        "examples": common.CONFIRMATORY_EXAMPLES,
        "examples_per_class": 1,
        "confirmatory_rows_sha256": common.EXPECTED_CONFIRMATORY_ROWS_SHA256,
        "confirmatory_inputs_payload_hash": payload["payload_hash"],
        "confirmatory_inputs_file_sha256": common.file_sha256(payload_path),
        "confirmatory_manifest_file_sha256": common.file_sha256(manifest_path),
        "clean_tensor_sha256": payload["clean_tensor_sha256"],
        "standard_clean_top1_percent": payload["standard_clean_top1_percent"],
        "standard_clean_top5_percent": payload["standard_clean_top5_percent"],
        "confirmatory_outcomes_read": True,
        "attack_family_selection_performed": False,
        "defense_retuning_performed": False,
        "paper_claim_authorized_by_preparation": False,
        "model_load_audit": payload["model_load_audit"],
    }


def main() -> None:
    args = parse_args()
    if args.num_workers < 0 or not 0 < args.gpu_memory_fraction <= 1:
        raise ValueError("Invalid ImageNet ViT-reference VR4 worker or GPU-memory setting")
    protocol_path = args.protocol_json.expanduser().resolve()
    protocol = common.read_json(protocol_path)
    protocol_hash = common.validate_protocol(protocol)
    output_root = Path(protocol["paths"]["output_root"]).expanduser().resolve()
    if protocol_path != output_root / "confirmatory_protocol.json":
        raise AssertionError("ImageNet ViT-reference VR4 protocol path is not canonical")

    parent = common.load_parent(
        Path(protocol["paths"]["r0_dir"]),
        Path(protocol["paths"]["calibration_dir"]),
        Path(protocol["paths"]["clean_dir"]),
    )
    vr3_info = common.load_vr3_freeze(Path(protocol["paths"]["vr3_root"]), parent)
    if vr3_info["selection_hash"] != protocol["vr3_selection_hash"]:
        raise AssertionError("ImageNet VR3 freeze changed after the VR4 protocol was written")
    rows = common.canonical_confirmatory_rows(parent)

    manifest_rows = [
        {
            **row,
            "protocol_id": common.PROTOCOL_ID,
            "protocol_hash": protocol_hash,
        }
        for row in rows
    ]
    manifest_path = output_root / "confirmatory_manifest.csv"
    if manifest_path.is_file():
        observed = common.read_csv(manifest_path)
        expected = [{key: str(value) for key, value in row.items()} for row in manifest_rows]
        if observed != expected:
            raise AssertionError("Frozen ImageNet ViT-reference VR4 manifest differs from this invocation")
    else:
        common.atomic_csv(manifest_path, manifest_rows)

    payload_path = output_root / "confirmatory_inputs.pt"
    summary_path = output_root / "prepare_summary.json"
    if payload_path.is_file():
        payload = torch.load(payload_path, map_location="cpu", weights_only=False)
        validate_input_payload(payload, protocol, rows)
        common.write_or_validate_json(
            summary_path, summary_payload(payload, payload_path, manifest_path)
        )
        print(f"RESUME validated {payload_path}", flush=True)
        print("CONFIRMATORY_SEARCH_READY=YES", flush=True)
        return

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, device=device)
    standard, _robust, model_audit = vr1.load_models(
        Path(protocol["checkpoints"]["standard"]["path"]),
        Path(protocol["checkpoints"]["robust"]["path"]),
        device,
    )
    loader = vr1.make_loader(
        Path(protocol["paths"]["data_root"]),
        rows,
        common.REFERENCE_BATCH_SIZE,
        args.num_workers,
        device,
    )
    labels_all: list[torch.Tensor] = []
    references_all: list[torch.Tensor] = []
    predictions_all: list[torch.Tensor] = []
    paths_all: list[str] = []
    clean_digest = hashlib.sha256()
    clean_digest.update(b"float32")
    clean_digest.update(
        np.asarray(
            (common.CONFIRMATORY_EXAMPLES, 3, 224, 224), dtype="<i8"
        ).tobytes()
    )
    with torch.no_grad():
        for images, labels, paths in loader:
            images_cpu = images.detach().cpu().contiguous()
            if images_cpu.dtype != torch.float32:
                raise AssertionError("ImageNet ViT-reference VR4 preprocessing no longer returns float32")
            clean_digest.update(images_cpu.numpy().tobytes(order="C"))
            logits = standard(images.to(device, non_blocking=True)).float()
            reference = logits.topk(common.PRIMARY_K, dim=1, sorted=True).indices
            labels_all.append(labels.detach().cpu().clone())
            references_all.append(reference.detach().cpu().clone())
            predictions_all.append(logits.argmax(1).detach().cpu().clone())
            paths_all.extend(str(path) for path in paths)
    labels = torch.cat(labels_all).to(torch.long)
    references = torch.cat(references_all).to(torch.long)
    predictions = torch.cat(predictions_all).to(torch.long)
    if paths_all != [str(row["relative_path"]) for row in rows]:
        raise AssertionError("ImageNet DataLoader changed the frozen VR4 order")

    payload: dict[str, Any] = {
        "protocol_id": common.PROTOCOL_ID,
        "protocol_hash": protocol_hash,
        "split": "attack_confirmatory",
        "examples": common.CONFIRMATORY_EXAMPLES,
        "confirmatory_rows_sha256": common.EXPECTED_CONFIRMATORY_ROWS_SHA256,
        "relative_paths": paths_all,
        "class_names": [str(row["class_name"]) for row in rows],
        "labels": labels,
        "confirmatory_positions": torch.arange(common.CONFIRMATORY_EXAMPLES, dtype=torch.long),
        "reference_top5": references,
        "standard_prediction": predictions,
        "clean_tensor_sha256": clean_digest.hexdigest(),
        "standard_clean_top1_percent": 100.0 * float(predictions.eq(labels).float().mean()),
        "standard_clean_top5_percent": 100.0
        * float(references.eq(labels[:, None]).any(1).float().mean()),
        "model_load_audit": model_audit,
        "clean_images_intentionally_not_stored": True,
    }
    add_payload_hash(payload)
    validate_input_payload(payload, protocol, rows)
    atomic_torch_save(payload_path, payload)
    common.atomic_json(summary_path, summary_payload(payload, payload_path, manifest_path))
    print(
        f"PASS froze {common.CONFIRMATORY_EXAMPLES} ImageNet ViT-reference VR4 references: "
        f"standard Top1={payload['standard_clean_top1_percent']:.3f}% "
        f"Top5={payload['standard_clean_top5_percent']:.3f}%",
        flush=True,
    )
    print("CONFIRMATORY_SEARCH_READY=YES", flush=True)
    print("ATTACK_FAMILY_SELECTION_PERFORMED=NO", flush=True)


if __name__ == "__main__":
    main()
