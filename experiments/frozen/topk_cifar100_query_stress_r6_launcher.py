#!/usr/bin/env python3
"""Launch/resume helpers for CIFAR-100 R6 query-budget stress testing."""

from __future__ import annotations

import fcntl
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import topk_cifar100_query_stress_r6_common as common


SEARCH = common.PROJECT_DIR / "search_topk_cifar100_query_stress_r6.py"
REEVALUATE = common.PROJECT_DIR / "reevaluate_topk_cifar100_query_stress_r6_method.py"
FINALIZE = common.PROJECT_DIR / "finalize_topk_cifar100_query_stress_r6.py"
AUDIT = common.PROJECT_DIR / "audit_topk_cifar100_query_stress_r6.py"


def acquire_lock(path: Path, label: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.seek(0)
        owner = handle.read().strip() or "unknown"
        raise SystemExit(f"{label} is already running (recorded PID: {owner})") from exc
    handle.seek(0)
    handle.truncate()
    handle.write(str(os.getpid()))
    handle.flush()
    return handle


def ensure_protocol(
    *,
    output_root: Path = common.DEFAULT_OUTPUT_ROOT,
    r5_root: Path = common.DEFAULT_R5_ROOT,
    r3_parent_dir: Path = common.DEFAULT_R3_PARENT_DIR,
    data_dir: Path = common.DEFAULT_DATA_DIR,
    r2_root: Path = common.DEFAULT_R2_ROOT,
    dev_dir: Path = common.DEFAULT_DEV_DIR,
) -> Path:
    common.validate_script_inventory()
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / ".protocol.lock"
    lock = lock_path.open("a+", encoding="utf-8")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
    lock.seek(0)
    lock.truncate()
    lock.write(str(os.getpid()))
    lock.flush()
    try:
        parent = common.r5.load_parent_authorization(r3_parent_dir)
        r5_info = common.load_r5_completion(r5_root)
        if r5_info["protocol"]["parent_handoff_hash"] != parent["handoff_hash"]:
            raise AssertionError("R5 and R6 do not share the same audited R3 parent")
        stress_indices = common.derive_stress_indices(parent)
        payload = common.build_protocol_payload(
            parent=parent,
            r5_info=r5_info,
            stress_indices=stress_indices,
            data_dir=data_dir,
            r2_root=r2_root,
            dev_dir=dev_dir,
            output_root=output_root,
        )
        protocol_path = output_root / "query_stress_protocol.json"
        common.write_or_validate_json(protocol_path, payload)
        common.validate_protocol(common.read_json(protocol_path))

        split_manifest = {
            "protocol_id": common.PROTOCOL_ID,
            "protocol_hash": payload["protocol_hash"],
            "selection_seed": common.STRESS_SELECTION_SEED,
            "selection_uses_labels_only": True,
            "examples": common.STRESS_EXAMPLES,
            "examples_per_class": common.STRESS_PER_CLASS,
            "indices": stress_indices,
            "indices_sha256": common.sequence_sha256(stress_indices),
            "excluded_r4_indices_sha256": payload["excluded_r4_indices_sha256"],
            "excluded_r5_indices_sha256": payload["excluded_r5_indices_sha256"],
            "disjoint_from_r4_r5": True,
            "attack_outcomes_read_for_selection": False,
        }
        split_manifest["manifest_hash"] = common.object_sha256(split_manifest)
        common.write_or_validate_json(output_root / "stress_split_manifest.json", split_manifest)

        r5_snapshot = {
            "r5_protocol_id": r5_info["protocol"]["protocol_id"],
            "r5_protocol_hash": r5_info["protocol_hash"],
            "r5_final_summary_hash": r5_info["final_hash"],
            "r5_paper_handoff_hash": r5_info["handoff_hash"],
            "r5_primary_result": r5_info["handoff"]["primary_comparison"],
            "r6_use": "motivation_and_parent_identity_only_no_retuning",
        }
        r5_snapshot["snapshot_hash"] = common.object_sha256(r5_snapshot)
        common.write_or_validate_json(output_root / "r5_completion_snapshot.json", r5_snapshot)
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()
    return output_root / "query_stress_protocol.json"


def run_command(command: Sequence[str]) -> None:
    print("+ " + " ".join(str(value) for value in command), flush=True)
    subprocess.run(list(command), cwd=common.PROJECT_DIR, check=True)


def run_method(
    *,
    method: str,
    output_root: Path = common.DEFAULT_OUTPUT_ROOT,
    device: str = "cuda:0",
    gpu_memory_fraction: float = 0.45,
) -> None:
    if method not in common.METHODS:
        raise ValueError(f"Unknown R6 method: {method}")
    output_root = output_root.expanduser().resolve()
    protocol_path = ensure_protocol(output_root=output_root)
    lock = acquire_lock(output_root / f".{method}.lock", f"R6 {method} stage")
    try:
        attack = common.METHOD_ATTACK[method]
        batch_size = common.SEARCH_BATCH_SIZE_BY_METHOD[method]
        total = len(common.BUDGETS) * len(common.SEARCH_SEEDS)
        position = 0
        for budget in common.BUDGETS:
            for seed in common.SEARCH_SEEDS:
                position += 1
                cell = common.cell_dir(output_root, method, budget, seed)
                print(
                    f"[{position}/{total}] R6 {method}: {attack}/Q={budget}/seed={seed}",
                    flush=True,
                )
                run_command(
                    [
                        sys.executable,
                        "-u",
                        str(SEARCH),
                        "--protocol-json",
                        str(protocol_path),
                        "--method",
                        method,
                        "--api-budget",
                        str(budget),
                        "--search-seed",
                        str(seed),
                        "--output-dir",
                        str(cell),
                        "--device",
                        device,
                        "--batch-size",
                        str(batch_size),
                        "--gpu-memory-fraction",
                        str(gpu_memory_fraction),
                    ]
                )
        run_command(
            [
                sys.executable,
                "-u",
                str(REEVALUATE),
                "--protocol-json",
                str(protocol_path),
                "--method",
                method,
                "--output-dir",
                str(common.method_reeval_dir(output_root, method)),
                "--device",
                device,
                "--batch-size",
                str(common.REEVAL_BATCH_SIZE),
                "--mc-chunk",
                str(common.REEVAL_MC_CHUNK),
                "--gpu-memory-fraction",
                str(gpu_memory_fraction),
            ]
        )
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()
    print(f"R6 method complete: {method} -> {output_root}", flush=True)


def run_group(
    methods: Sequence[str],
    *,
    label: str,
    output_root: Path = common.DEFAULT_OUTPUT_ROOT,
    device: str = "cuda:0",
    gpu_memory_fraction: float = 0.45,
) -> None:
    output_root = output_root.expanduser().resolve()
    ensure_protocol(output_root=output_root)
    group_lock = acquire_lock(output_root / f".group_{label}.lock", f"R6 {label} group")
    try:
        for method in methods:
            run_method(
                method=method,
                output_root=output_root,
                device=device,
                gpu_memory_fraction=gpu_memory_fraction,
            )
    finally:
        fcntl.flock(group_lock.fileno(), fcntl.LOCK_UN)
        group_lock.close()


def run_finalize(*, output_root: Path = common.DEFAULT_OUTPUT_ROOT) -> None:
    output_root = output_root.expanduser().resolve()
    protocol_path = ensure_protocol(output_root=output_root)
    lock = acquire_lock(output_root / ".finalize.lock", "R6 final analysis")
    try:
        run_command(
            [
                sys.executable,
                "-u",
                str(FINALIZE),
                "--protocol-json",
                str(protocol_path),
                "--output-dir",
                str(output_root / "query_stress_analysis"),
            ]
        )
        run_command(
            [
                sys.executable,
                "-u",
                str(AUDIT),
                "--output-root",
                str(output_root),
                "--r5-root",
                str(common.DEFAULT_R5_ROOT),
                "--r3-parent-dir",
                str(common.DEFAULT_R3_PARENT_DIR),
            ]
        )
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()

