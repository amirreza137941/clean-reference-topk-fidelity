#!/usr/bin/env python3
"""Launch/resume helpers for CIFAR-100 confirmatory R5."""

from __future__ import annotations

import fcntl
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import topk_cifar100_confirmatory_r5_common as common


SEARCH = common.PROJECT_DIR / "search_topk_cifar100_confirmatory_r5.py"
REEVALUATE = common.PROJECT_DIR / "reevaluate_topk_cifar100_confirmatory_r5_method.py"
FINALIZE = common.PROJECT_DIR / "finalize_topk_cifar100_confirmatory_r5.py"
AUDIT = common.PROJECT_DIR / "audit_topk_cifar100_confirmatory_r5.py"


def required_inputs() -> list[Path]:
    return [
        *(common.PROJECT_DIR / name for name in common.SCRIPT_NAMES),
        *(common.PROJECT_DIR / name for name in common.DEPENDENCY_NAMES),
    ]


def validate_project_files() -> None:
    missing = [str(path) for path in required_inputs() if not path.is_file()]
    if missing:
        raise SystemExit(
            "Missing required existing/new CIFAR-100 files:\n  " + "\n  ".join(missing)
        )


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
    parent_dir: Path = common.DEFAULT_PARENT_DIR,
    r4_root: Path = common.DEFAULT_R4_ROOT,
    data_dir: Path = common.DEFAULT_DATA_DIR,
    r2_root: Path = common.DEFAULT_R2_ROOT,
    dev_dir: Path = common.DEFAULT_DEV_DIR,
) -> Path:
    validate_project_files()
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
        parent = common.load_parent_authorization(parent_dir)
        r4_info = common.load_r4_freeze(r4_root, parent)
        payload = common.build_protocol_payload(
            parent=parent,
            r4_info=r4_info,
            data_dir=data_dir,
            r2_root=r2_root,
            dev_dir=dev_dir,
            output_root=output_root,
        )
        protocol_path = output_root / "confirmatory_protocol.json"
        common.write_or_validate_json(protocol_path, payload)
        common.validate_protocol(common.read_json(protocol_path))
        common.write_or_validate_json(
            output_root / "r4_confirmatory_attack_freeze_snapshot.json",
            r4_info["freeze"],
        )
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()
    return output_root / "confirmatory_protocol.json"


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
        raise ValueError(f"Unknown R5 method: {method}")
    output_root = output_root.expanduser().resolve()
    protocol_path = ensure_protocol(output_root=output_root)
    lock = acquire_lock(output_root / f".{method}.lock", f"R5 {method} method stage")
    try:
        attack = common.METHOD_ATTACK[method]
        batch_size = common.SEARCH_BATCH_SIZE_BY_METHOD[method]
        for position, seed in enumerate(common.SEARCH_SEEDS, start=1):
            cell = common.cell_dir(output_root, method, seed)
            print(
                f"[{position}/{len(common.SEARCH_SEEDS)}] R5 {method}: "
                f"{attack}/seed={seed}",
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
    print(f"R5 method complete: {method} -> {output_root}", flush=True)


def run_finalize(
    *,
    output_root: Path = common.DEFAULT_OUTPUT_ROOT,
) -> None:
    output_root = output_root.expanduser().resolve()
    protocol_path = ensure_protocol(output_root=output_root)
    lock = acquire_lock(output_root / ".finalize.lock", "R5 final analysis")
    try:
        run_command(
            [
                sys.executable,
                "-u",
                str(FINALIZE),
                "--protocol-json",
                str(protocol_path),
                "--output-dir",
                str(output_root / "confirmatory_analysis"),
            ]
        )
        run_command(
            [
                sys.executable,
                "-u",
                str(AUDIT),
                "--output-root",
                str(output_root),
                "--parent-dir",
                str(common.DEFAULT_PARENT_DIR),
                "--r4-root",
                str(common.DEFAULT_R4_ROOT),
            ]
        )
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()

