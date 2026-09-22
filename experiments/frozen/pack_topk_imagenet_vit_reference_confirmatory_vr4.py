#!/usr/bin/env python3
"""Audit VR4 and create a transfer ZIP without candidates or reevaluation caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

import topk_imagenet_vit_reference_confirmatory_vr4_common as common


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
        root / "confirmatory_protocol.json",
        root / "confirmatory_manifest.csv",
        root / "confirmatory_inputs.pt",
        root / "prepare_summary.json",
        root / "vr3_confirmatory_attack_freeze_snapshot.json",
    ]
    search_files: list[Path] = []
    for method, _attack, seed in common.expected_cells():
        directory = common.cell_dir(root, method, seed)
        search_files.extend(
            [
                directory / "search_summary.json",
                directory / "search_per_example.csv",
            ]
        )
    method_files: list[Path] = []
    for method in common.METHODS:
        directory = common.method_reeval_dir(root, method)
        method_files.extend(
            [
                directory / "candidate_reeval_summary.csv",
                directory / "method_summary.json",
            ]
        )
    analysis = root / "confirmatory_analysis"
    analysis_files = [
        analysis / "confirmatory_candidate_summary.csv",
        analysis / "confirmatory_method_summary.csv",
        analysis / "confirmatory_paired_bootstrap.csv",
        analysis / "confirmatory_classwise.csv",
        analysis / "confirmatory_per_example.csv",
        analysis / "paper_claim_handoff.json",
        analysis / "final_summary.json",
    ]
    files = fixed + search_files + method_files + analysis_files
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Cannot pack incomplete ImageNet ViT-reference VR4 output:\n  "
            + "\n  ".join(missing)
        )
    return files


def run_audit(root: Path) -> None:
    command = [
        sys.executable,
        "-u",
        str(PROJECT_DIR / "audit_topk_imagenet_vit_reference_confirmatory_vr4.py"),
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

    protocol = common.read_json(root / "confirmatory_protocol.json")
    protocol_hash = common.validate_protocol(protocol)
    final = common.read_json(root / "confirmatory_analysis" / "final_summary.json")
    handoff = common.read_json(
        root / "confirmatory_analysis" / "paper_claim_handoff.json"
    )
    if final.get("status") != "PASS":
        raise AssertionError("ImageNet ViT-reference VR4 did not finish with PASS")
    if final.get("protocol_hash") != protocol_hash:
        raise AssertionError("ImageNet ViT-reference VR4 final/protocol identity mismatch")
    if final.get("paper_claim_authorized") is not handoff.get("authorized"):
        raise AssertionError("ImageNet ViT-reference VR4 claim decisions differ")
    if final.get("robust_vit_claim_authorized") is not False:
        raise AssertionError("ImageNet ViT-reference VR4 cannot authorize robust-ViT claims")
    if handoff.get("robust_vit_claim_authorized") is not False:
        raise AssertionError("ImageNet ViT-reference VR4 handoff permits a robust-ViT claim")

    files = required_files(root)
    archive_root = (
        "paper_topk_imagenet_vit_reference_crossarch_confirmatory_vr4_transfer"
    )
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
        "protocol_hash": protocol_hash,
        "status": final["status"],
        "claim_scope": "cross_architecture_vit_reference_transfer",
        "architecture_roles": protocol["architecture_roles"],
        "paper_claim_authorized": bool(final["paper_claim_authorized"]),
        "robust_vit_claim_authorized": False,
        "paper_claim_handoff_hash": final["paper_claim_handoff_hash"],
        "final_summary_hash": final["final_summary_hash"],
        "files": [relative for relative, _data in encoded],
        "excluded_large_or_duplicate_artifacts": [
            "search/**/candidate.pt",
            "search/**/partials/",
            "reevaluation/**/cache/",
            "reevaluation/**/candidate_reeval_per_example.csv",
            ".*.lock",
        ],
    }
    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = zip_path.with_name(zip_path.name + ".tmp")
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
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
    print(
        f"  size={zip_path.stat().st_size} bytes, files={len(encoded) + 2}",
        flush=True,
    )
    print("  candidate tensors and reevaluation caches excluded", flush=True)
    print(
        f"  PAPER_CLAIM_AUTHORIZED={'YES' if final['paper_claim_authorized'] else 'NO'}",
        flush=True,
    )
    print("  ROBUST_VIT_CLAIM_AUTHORIZED=NO", flush=True)


if __name__ == "__main__":
    main()
