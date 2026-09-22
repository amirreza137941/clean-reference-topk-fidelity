#!/usr/bin/env python3
"""Freeze the path-only 100-image subset and its clean tensors for D1."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

import calibrate_topk_imagenet_defense_r2 as r2
import topk_imagenet_posthoc_q20k_eot_d1_common as common


TENSOR_KEYS = (
    "clean",
    "labels",
    "diagnostic_positions",
    "source_confirmatory_positions",
    "reference_top5",
    "standard_prediction",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-json", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


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


def validate_parent_inputs(
    payload: Mapping[str, Any], parent: Mapping[str, Any]
) -> None:
    if (
        payload.get("protocol_id") != common.PARENT_PROTOCOL_ID
        or payload.get("protocol_hash") != common.EXPECTED_R5_PROTOCOL_HASH
        or payload.get("payload_hash") != common.EXPECTED_R5_INPUTS_PAYLOAD_HASH
        or payload.get("clean_tensor_sha256")
        != common.EXPECTED_R5_CLEAN_TENSOR_SHA256
        or payload.get("split") != "attack_confirmatory"
        or int(payload.get("examples", -1)) != 1000
    ):
        raise AssertionError("The source R5 input identity changed")
    positions = payload["confirmatory_positions"].detach().cpu()
    labels = payload["labels"].detach().cpu()
    reference = payload["reference_top5"].detach().cpu()
    prediction = payload["standard_prediction"].detach().cpu()
    if positions.tolist() != list(range(1000)):
        raise AssertionError("R5 source positions are not canonical")
    if labels.shape != (1000,) or reference.shape != (1000, 5):
        raise AssertionError("R5 source metadata shapes changed")
    if prediction.shape != labels.shape or not torch.equal(reference[:, 0], prediction):
        raise AssertionError("R5 source Top-1/reference alignment changed")
    if list(payload.get("relative_paths", [])) != [
        row["relative_path"] for row in parent["manifest"]
    ]:
        raise AssertionError("R5 source paths differ from its frozen manifest")
    if labels.tolist() != [int(row["target"]) for row in parent["manifest"]]:
        raise AssertionError("R5 source labels differ from its frozen manifest")
    for key in ("labels", "confirmatory_positions", "reference_top5", "standard_prediction"):
        if common.tensor_sha256(payload[key]) != payload.get("tensor_sha256", {}).get(key):
            raise AssertionError(f"R5 source tensor hash changed: {key}")
    hash_input = {
        key: value
        for key, value in payload.items()
        if key
        not in {
            "labels",
            "confirmatory_positions",
            "reference_top5",
            "standard_prediction",
            "payload_hash",
        }
    }
    if common.object_sha256(hash_input) != payload.get("payload_hash"):
        raise AssertionError("R5 source payload hash changed")


def validate_input_payload(
    payload: Mapping[str, Any],
    protocol: Mapping[str, Any],
    subset: list[dict[str, Any]],
) -> None:
    expected_identity = {
        "protocol_id": common.PROTOCOL_ID,
        "protocol_hash": protocol["protocol_hash"],
        "parent_r5_protocol_hash": common.EXPECTED_R5_PROTOCOL_HASH,
        "condition": "posthoc_q20k_eot_stress",
        "examples": common.DIAGNOSTIC_EXAMPLES,
        "subset_rows_sha256": common.subset_rows_sha256(subset),
        "parent_r5_inputs_payload_hash": common.EXPECTED_R5_INPUTS_PAYLOAD_HASH,
        "parent_r5_clean_tensor_sha256": common.EXPECTED_R5_CLEAN_TENSOR_SHA256,
        "parent_r5_full_clean_stream_verified": True,
    }
    for key, value in expected_identity.items():
        if payload.get(key) != value:
            raise AssertionError(f"D1 input identity mismatch: {key}")
    for key in TENSOR_KEYS:
        if key not in payload:
            raise AssertionError(f"D1 input payload misses {key}")
    clean = payload["clean"].detach().cpu()
    labels = payload["labels"].detach().cpu()
    diagnostic_positions = payload["diagnostic_positions"].detach().cpu()
    source_positions = payload["source_confirmatory_positions"].detach().cpu()
    reference = payload["reference_top5"].detach().cpu()
    prediction = payload["standard_prediction"].detach().cpu()
    count = common.DIAGNOSTIC_EXAMPLES
    if clean.shape != (count, 3, 224, 224) or clean.dtype != torch.float32:
        raise AssertionError("D1 clean tensor shape/dtype changed")
    if labels.shape != (count,) or reference.shape != (count, common.PRIMARY_K):
        raise AssertionError("D1 metadata tensor shape changed")
    if diagnostic_positions.tolist() != list(range(count)):
        raise AssertionError("D1 positions are not canonical")
    expected_source = [int(row["source_confirmatory_position"]) for row in subset]
    if source_positions.tolist() != expected_source:
        raise AssertionError("D1 source positions changed")
    if list(payload.get("relative_paths", [])) != [row["relative_path"] for row in subset]:
        raise AssertionError("D1 path order changed")
    if list(payload.get("class_names", [])) != [row["class_name"] for row in subset]:
        raise AssertionError("D1 class-name order changed")
    if labels.tolist() != [int(row["target"]) for row in subset]:
        raise AssertionError("D1 labels changed")
    if not torch.equal(reference[:, 0], prediction):
        raise AssertionError("D1 standard Top-1/reference alignment changed")
    if not torch.isfinite(clean).all() or float(clean.min()) < 0 or float(clean.max()) > 1:
        raise AssertionError("D1 clean images are invalid")
    if int(reference.min()) < 0 or int(reference.max()) >= 1000:
        raise AssertionError("D1 reference contains an invalid class")
    if not reference.sort(1).values.diff(dim=1).ne(0).all():
        raise AssertionError("D1 reference contains duplicate Top-5 members")
    for key in TENSOR_KEYS:
        if common.tensor_sha256(payload[key]) != payload.get("tensor_sha256", {}).get(key):
            raise AssertionError(f"D1 tensor hash mismatch: {key}")
    if common.object_sha256(payload_hash_input(payload)) != payload.get("payload_hash"):
        raise AssertionError("D1 payload hash mismatch")


def summary_payload(
    payload: Mapping[str, Any], payload_path: Path, manifest_path: Path
) -> dict[str, Any]:
    return {
        "protocol_id": common.PROTOCOL_ID,
        "protocol_hash": payload["protocol_hash"],
        "status": "PASS",
        "post_hoc": True,
        "paper_claim_authorized": False,
        "examples": common.DIAGNOSTIC_EXAMPLES,
        "subset_rows_sha256": payload["subset_rows_sha256"],
        "diagnostic_inputs_payload_hash": payload["payload_hash"],
        "diagnostic_inputs_file_sha256": common.file_sha256(payload_path),
        "diagnostic_manifest_file_sha256": common.file_sha256(manifest_path),
        "clean_tensor_sha256": payload["tensor_sha256"]["clean"],
        "parent_r5_inputs_file_sha256": common.EXPECTED_R5_INPUTS_FILE_SHA256,
        "parent_r5_clean_tensor_sha256": payload[
            "parent_r5_clean_tensor_sha256"
        ],
        "parent_r5_full_clean_stream_verified": True,
        "parent_r5_unchanged": True,
    }


def main() -> None:
    args = parse_args()
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    protocol_path = args.protocol_json.expanduser().resolve()
    protocol = common.read_json(protocol_path)
    common.validate_protocol(protocol)
    output_root = Path(protocol["paths"]["output_root"]).expanduser().resolve()
    if protocol_path != output_root / "diagnostic_protocol.json":
        raise AssertionError("D1 protocol path is not canonical")

    parent = common.load_parent_r5(Path(protocol["paths"]["r5_root"]))
    subset = parent["subset"]
    if common.subset_rows_sha256(subset) != protocol["subset_rows_sha256"]:
        raise AssertionError("D1 subset changed after protocol freeze")

    manifest_rows = [
        {
            **row,
            "protocol_id": common.PROTOCOL_ID,
            "protocol_hash": protocol["protocol_hash"],
        }
        for row in subset
    ]
    manifest_path = output_root / "diagnostic_manifest.csv"
    if manifest_path.is_file():
        observed = common.read_csv(manifest_path)
        expected = [{key: str(value) for key, value in row.items()} for row in manifest_rows]
        if observed != expected:
            raise AssertionError("D1 manifest differs from the frozen subset")
    else:
        common.atomic_csv(manifest_path, manifest_rows)

    parent_inputs = torch.load(parent["paths"]["inputs"], map_location="cpu", weights_only=False)
    validate_parent_inputs(parent_inputs, parent)
    payload_path = output_root / "diagnostic_inputs.pt"
    summary_path = output_root / "prepare_summary.json"
    if payload_path.is_file():
        payload = torch.load(payload_path, map_location="cpu", weights_only=False)
        validate_input_payload(payload, protocol, subset)
        common.write_or_validate_json(
            summary_path, summary_payload(payload, payload_path, manifest_path)
        )
        print(f"RESUME validated {payload_path}", flush=True)
        print("POSTHOC_Q20K_SEARCH_READY=YES", flush=True)
        return

    device = torch.device(args.device)
    source_positions = [int(row["source_confirmatory_position"]) for row in subset]
    source_set = set(source_positions)
    loader_rows = [
        {
            "target": int(row["target"]),
            "class_name": str(row["class_name"]),
            "relative_path": str(row["relative_path"]),
        }
        for row in parent["manifest"]
    ]
    loader = r2.make_loader(
        Path(protocol["paths"]["data_root"]),
        loader_rows,
        common.REFERENCE_BATCH_SIZE,
        args.num_workers,
        device,
    )
    selected_images: dict[int, torch.Tensor] = {}
    selected_labels: dict[int, torch.Tensor] = {}
    observed_paths: list[str] = []
    observed_labels: list[torch.Tensor] = []
    clean_digest = hashlib.sha256()
    clean_digest.update(b"float32")
    clean_digest.update(
        np.asarray((1000, 3, 224, 224), dtype="<i8").tobytes()
    )
    cursor = 0
    for images, labels, paths in loader:
        images_cpu = images.detach().cpu().contiguous()
        labels_cpu = labels.detach().cpu().contiguous()
        if images_cpu.dtype != torch.float32:
            raise AssertionError("ImageNet D1 preprocessing no longer returns float32")
        clean_digest.update(images_cpu.numpy().tobytes(order="C"))
        observed_labels.append(labels_cpu)
        observed_paths.extend(str(path) for path in paths)
        for offset in range(len(images_cpu)):
            source_position = cursor + offset
            if source_position in source_set:
                selected_images[source_position] = images_cpu[offset].clone()
                selected_labels[source_position] = labels_cpu[offset].clone()
        cursor += len(images_cpu)
    expected_parent_paths = [row["relative_path"] for row in parent["manifest"]]
    if observed_paths != expected_parent_paths or cursor != 1000:
        raise AssertionError("ImageNet loader changed the frozen R5 path order")
    all_labels = torch.cat(observed_labels).to(torch.long)
    if not torch.equal(all_labels, parent_inputs["labels"].detach().cpu()):
        raise AssertionError("ImageNet loader labels differ from frozen R5 labels")
    if clean_digest.hexdigest() != parent_inputs["clean_tensor_sha256"]:
        raise AssertionError("ImageNet clean bytes changed since the audited R5 run")
    if set(selected_images) != source_set or set(selected_labels) != source_set:
        raise AssertionError("ImageNet loader did not recover the complete D1 subset")

    clean = torch.stack([selected_images[position] for position in source_positions])
    labels = torch.stack([selected_labels[position] for position in source_positions]).to(
        torch.long
    )
    expected_paths = [row["relative_path"] for row in subset]
    parent_labels = parent_inputs["labels"][source_positions].detach().cpu().to(
        torch.long
    )
    if not torch.equal(labels, parent_labels):
        raise AssertionError("D1 subset labels differ from frozen R5 labels")

    payload: dict[str, Any] = {
        "protocol_id": common.PROTOCOL_ID,
        "protocol_hash": protocol["protocol_hash"],
        "parent_r5_protocol_hash": common.EXPECTED_R5_PROTOCOL_HASH,
        "condition": "posthoc_q20k_eot_stress",
        "examples": common.DIAGNOSTIC_EXAMPLES,
        "subset_rows_sha256": common.subset_rows_sha256(subset),
        "relative_paths": expected_paths,
        "class_names": [row["class_name"] for row in subset],
        "clean": clean,
        "labels": labels,
        "diagnostic_positions": torch.arange(
            common.DIAGNOSTIC_EXAMPLES, dtype=torch.long
        ),
        "source_confirmatory_positions": torch.tensor(
            source_positions, dtype=torch.long
        ),
        "reference_top5": parent_inputs["reference_top5"][source_positions]
        .detach()
        .cpu()
        .to(torch.long),
        "standard_prediction": parent_inputs["standard_prediction"][source_positions]
        .detach()
        .cpu()
        .to(torch.long),
        "parent_r5_inputs_payload_hash": parent_inputs["payload_hash"],
        "parent_r5_clean_tensor_sha256": parent_inputs["clean_tensor_sha256"],
        "parent_r5_full_clean_stream_verified": True,
        "clean_storage_dtype": "float32",
    }
    add_payload_hash(payload)
    validate_input_payload(payload, protocol, subset)
    atomic_torch_save(payload_path, payload)
    common.atomic_json(
        summary_path, summary_payload(payload, payload_path, manifest_path)
    )
    print(
        f"PASS froze {common.DIAGNOSTIC_EXAMPLES} path-only-selected R5 images",
        flush=True,
    )
    print("POSTHOC_Q20K_SEARCH_READY=YES", flush=True)
    print("PAPER_CLAIM_AUTHORIZED=NO (post-hoc diagnostic)", flush=True)


if __name__ == "__main__":
    main()
