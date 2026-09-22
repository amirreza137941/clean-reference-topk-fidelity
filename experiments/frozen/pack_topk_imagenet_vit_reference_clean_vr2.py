#!/usr/bin/env python3
"""Create a small transfer archive from a completed VR2 clean confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import zipfile
from pathlib import Path

import topk_imagenet_vit_reference_clean_vr2_common as common


DEFAULT_ROOT = Path(".")
DEFAULT_OUTPUT_DIR = (
    DEFAULT_ROOT
    / "models/imagenet_topk/paper_topk_imagenet_vit_reference_crossarch_clean_confirmation_vr2"
)

OUTPUT_FILES = (
    "frozen_protocol_vr2.json",
    "run_config.json",
    "model_load_audit.json",
    "clean_confirmation_manifest.json",
    "official_clean_summary.csv",
    "paired_class_bootstrap.csv",
    "preregistered_clean_gates.csv",
    "classwise_summary.csv",
    "reference_quality_decomposition.csv",
    "partial_runtime_summary.csv",
    "official_clean_gate.json",
    "final_summary.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pack small VR2 transfer artifacts")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--archive", type=Path)
    return parser.parse_args()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    archive = (
        args.archive.expanduser().resolve()
        if args.archive is not None
        else output_dir.parent / f"{output_dir.name}_transfer.zip"
    )
    final = common.read_json(output_dir / "final_summary.json")
    if final.get("protocol_id") != common.PROTOCOL_ID or final.get("status") != "PASS":
        raise AssertionError("VR2 final_summary.json is not a completed protocol output")
    if final.get("official_clean_confirmation_completed") is not True:
        raise AssertionError("VR2 clean confirmation is not complete")
    config = common.read_json(output_dir / "run_config.json")
    if config.get("config_hash") != final.get("config_hash"):
        raise AssertionError("VR2 config/final hash mismatch")
    protocol = common.read_json(output_dir / "frozen_protocol_vr2.json")
    if protocol != common.frozen_protocol_payload():
        raise AssertionError("VR2 frozen protocol differs before transfer packaging")
    if protocol.get("protocol_hash") != final.get("protocol_hash"):
        raise AssertionError("VR2 protocol/final hash mismatch")

    sources: list[tuple[str, Path]] = []
    for filename in OUTPUT_FILES:
        path = output_dir / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        sources.append((filename, path))
    parent_selection = (
        Path(config["paths"]["calibration_dir"]) / "defense_selection_vr1.json"
    )
    if not parent_selection.is_file():
        raise FileNotFoundError(parent_selection)
    if common.file_sha256(parent_selection) != common.EXPECTED_SELECTION_FILE_SHA256:
        raise AssertionError("Parent selection changed before transfer packaging")
    sources.append(("parent/defense_selection_vr1.json", parent_selection))

    payloads = [(arcname, path.read_bytes()) for arcname, path in sources]
    checksums = "".join(
        f"{sha256_bytes(payload)}  {arcname}\n" for arcname, payload in payloads
    ).encode("utf-8")
    transfer_manifest = {
        "protocol_id": common.PROTOCOL_ID,
        "protocol_hash": final["protocol_hash"],
        "config_hash": final["config_hash"],
        "selection_hash": common.EXPECTED_SELECTION_HASH,
        "official_clean_confirmation_passed": bool(
            final["official_clean_confirmation_passed"]
        ),
        "attack_stage_ready": bool(final["attack_stage_ready"]),
        "paper_claim_authorized": False,
        "excluded_large_artifacts": [
            "clean_confirmation_manifest.csv",
            "official_clean_per_example.csv.gz",
            "partials/",
        ],
        "files": [arcname for arcname, _payload in payloads],
    }
    manifest_bytes = (
        json.dumps(transfer_manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")

    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_name(archive.name + ".tmp")
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as handle:
        for arcname, payload in payloads:
            handle.writestr(arcname, payload)
        handle.writestr("SHA256SUMS.txt", checksums)
        handle.writestr("transfer_manifest.json", manifest_bytes)
    os.replace(temporary, archive)
    with zipfile.ZipFile(archive, "r") as handle:
        bad = handle.testzip()
        if bad is not None:
            raise AssertionError(f"Transfer ZIP failed CRC validation: {bad}")
    print(f"Transfer bundle: {archive}")
    print(f"Size: {archive.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
