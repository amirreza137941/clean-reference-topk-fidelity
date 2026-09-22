#!/usr/bin/env python3
"""Audit VR3 and create a small transfer ZIP without candidate tensors."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

import topk_imagenet_vit_reference_attack_pilot_vr3_common as common


PROJECT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=common.DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--zip-path", type=Path)
    return parser.parse_args()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def required_files(root: Path) -> list[Path]:
    fixed = [
        root / "attack_pilot_protocol.json",
        root / "attack_pilot_manifest.csv",
        root / "prepare_summary.json",
        root / "common_mc1000" / "candidate_reeval_summary.csv",
        root / "common_mc1000" / "candidate_reeval_per_example.csv",
        root / "common_mc1000" / "attack_family_three_seed_summary.csv",
        root / "common_mc1000" / "attack_family_paired_bootstrap.csv",
        root / "common_mc1000" / "selected_attack_envelope.csv",
        root / "common_mc1000" / "pilot_exploratory_comparisons.csv",
        root / "common_mc1000" / "confirmatory_attack_freeze.json",
        root / "common_mc1000" / "final_summary.json",
    ]
    summaries = [
        common.cell_dir(root, method, attack, seed) / "search_summary.json"
        for method, attack, seed in common.expected_cells()
    ]
    files = fixed + summaries
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError("Cannot pack incomplete VR3 output:\n  " + "\n  ".join(missing))
    return files


def run_audit(root: Path) -> None:
    command = [
        sys.executable,
        "-u",
        str(PROJECT_DIR / "audit_topk_imagenet_vit_reference_attack_pilot_vr3.py"),
        "--output-root",
        str(root),
    ]
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_DIR, check=True)


def main() -> None:
    args = parse_args()
    root = args.output_root.expanduser().resolve()
    zip_path = (
        args.zip_path.expanduser().resolve()
        if args.zip_path is not None
        else root.with_name(root.name + "_transfer.zip")
    )
    run_audit(root)
    protocol = common.read_json(root / "attack_pilot_protocol.json")
    final = common.read_json(root / "common_mc1000" / "final_summary.json")
    freeze = common.read_json(root / "common_mc1000" / "confirmatory_attack_freeze.json")
    if final.get("status") != "PASS" or final.get("confirmatory_attack_stage_ready") is not True:
        raise AssertionError("VR3 did not finish with confirmatory stage ready")
    if final.get("adversarial_paper_claim_authorized") is not False:
        raise AssertionError("Pilot unexpectedly authorized an adversarial paper claim")
    if freeze.get("confirmatory_partition", {}).get("outcomes_read") is not False:
        raise AssertionError("Confirmatory outcomes were read")

    files = required_files(root)
    archive_root = "paper_topk_imagenet_vit_reference_crossarch_attack_pilot_vr3_transfer"
    checksums: list[str] = []
    encoded: list[tuple[str, bytes]] = []
    for path in files:
        relative = path.relative_to(root).as_posix()
        data = path.read_bytes()
        encoded.append((relative, data))
        checksums.append(f"{sha256_bytes(data)}  {relative}")
    checksum_bytes = ("\n".join(checksums) + "\n").encode("utf-8")
    manifest: dict[str, Any] = {
        "protocol_id": common.PROTOCOL_ID,
        "protocol_hash": protocol["protocol_hash"],
        "status": final["status"],
        "confirmatory_attack_stage_ready": True,
        "confirmatory_partition_outcomes_read": False,
        "adversarial_paper_claim_authorized": False,
        "robust_vit_claim_authorized": False,
        "confirmatory_attack_freeze_hash": final["confirmatory_attack_freeze_hash"],
        "files": [relative for relative, _data in encoded],
        "excluded_large_artifacts": [
            "pilot_inputs.pt",
            "search/**/candidate.pt",
            "search/**/partials/",
            "common_mc1000/cache/",
        ],
    }
    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = zip_path.with_name(zip_path.name + ".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for relative, data in encoded:
            archive.writestr(f"{archive_root}/{relative}", data)
        archive.writestr(f"{archive_root}/SHA256SUMS.txt", checksum_bytes)
        archive.writestr(f"{archive_root}/transfer_manifest.json", manifest_bytes)
    temporary.replace(zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"ZIP CRC validation failed at {bad}")
    print(f"PASS transfer ZIP: {zip_path}", flush=True)
    print(f"  size={zip_path.stat().st_size} bytes, files={len(encoded) + 2}", flush=True)
    print("  candidate tensors and reevaluation caches excluded", flush=True)


if __name__ == "__main__":
    main()
