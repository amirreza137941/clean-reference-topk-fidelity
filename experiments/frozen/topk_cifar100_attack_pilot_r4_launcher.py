#!/usr/bin/env python3
"""Launch/resume helpers for CIFAR-100 adaptive attack pilot R4."""

from __future__ import annotations

import fcntl
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import topk_cifar100_attack_pilot_r4_common as common


SEARCH = common.PROJECT_DIR / "search_topk_cifar100_attack_pilot_r4.py"
REEVALUATE = common.PROJECT_DIR / "reevaluate_topk_cifar100_attack_pilot_r4_mc1000.py"
AUDIT = common.PROJECT_DIR / "audit_topk_cifar100_attack_pilot_r4.py"


def required_inputs() -> list[Path]:
    return [
        *(common.PROJECT_DIR / name for name in common.SCRIPT_NAMES),
        common.PROJECT_DIR / "calibrate_topk_cifar100_actual_defense_r1.py",
        common.PROJECT_DIR / "topk_cifar100_defense_calibration_common.py",
        common.PROJECT_DIR / "topk_cifar100_r2_common.py",
        common.PROJECT_DIR / "train_resnet20_teacher_top2_adversarial_cifar100_r2.py",
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
    data_dir: Path = common.DEFAULT_DATA_DIR,
    r2_root: Path = common.DEFAULT_R2_ROOT,
    dev_dir: Path = common.DEFAULT_DEV_DIR,
) -> Path:
    validate_project_files()
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    # Protocol creation is very short and may be reached concurrently by the
    # light/full launchers.  Wait for this specific lock instead of making the
    # second tmux fail spuriously.
    lock_path = output_root / ".protocol.lock"
    lock = lock_path.open("a+", encoding="utf-8")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
    lock.seek(0)
    lock.truncate()
    lock.write(str(os.getpid()))
    lock.flush()
    try:
        parent = common.load_parent_authorization(parent_dir)
        payload = common.build_protocol_payload(
            parent=parent,
            data_dir=data_dir,
            r2_root=r2_root,
            dev_dir=dev_dir,
            output_root=output_root,
        )
        protocol_path = output_root / "pilot_protocol.json"
        common.write_or_validate_json(protocol_path, payload)
        common.validate_protocol(common.read_json(protocol_path))
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()
    return output_root / "pilot_protocol.json"


def run_command(command: Sequence[str]) -> None:
    print("+ " + " ".join(str(value) for value in command), flush=True)
    subprocess.run(list(command), cwd=common.PROJECT_DIR, check=True)


def run_group(
    *,
    group: str,
    methods: Sequence[str],
    batch_size: int,
    output_root: Path = common.DEFAULT_OUTPUT_ROOT,
    device: str = "cuda:0",
    gpu_memory_fraction: float = 0.45,
) -> None:
    if set(methods) not in (set(common.LIGHT_METHODS), set(common.FULL_METHODS)):
        raise ValueError("Launcher group must be exactly light or full")
    output_root = output_root.expanduser().resolve()
    protocol_path = ensure_protocol(output_root=output_root)
    lock = acquire_lock(output_root / f".{group}.lock", f"R4 {group} search group")
    try:
        total = len(methods) * len(common.ATTACKS) * len(common.SEARCH_SEEDS)
        position = 0
        for method in methods:
            for attack in common.ATTACKS:
                for seed in common.SEARCH_SEEDS:
                    position += 1
                    cell = common.cell_dir(output_root, method, attack, seed)
                    print(
                        f"[{position}/{total}] R4 {group}: {method}/{attack}/seed={seed}",
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
                            "--attack",
                            attack,
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
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()
    print(f"R4 {group} search group complete: {output_root}", flush=True)


def run_reevaluation(
    *,
    output_root: Path = common.DEFAULT_OUTPUT_ROOT,
    device: str = "cuda:0",
    batch_size: int = common.REEVAL_BATCH_SIZE,
    mc_chunk: int = common.REEVAL_MC_CHUNK,
    gpu_memory_fraction: float = 0.45,
) -> None:
    output_root = output_root.expanduser().resolve()
    protocol_path = ensure_protocol(output_root=output_root)
    lock = acquire_lock(output_root / ".reeval.lock", "R4 common MC=1000 reevaluation")
    try:
        run_command(
            [
                sys.executable,
                "-u",
                str(REEVALUATE),
                "--protocol-json",
                str(protocol_path),
                "--output-dir",
                str(output_root / "common_mc1000"),
                "--device",
                device,
                "--batch-size",
                str(batch_size),
                "--mc-chunk",
                str(mc_chunk),
                "--gpu-memory-fraction",
                str(gpu_memory_fraction),
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
            ]
        )
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()
