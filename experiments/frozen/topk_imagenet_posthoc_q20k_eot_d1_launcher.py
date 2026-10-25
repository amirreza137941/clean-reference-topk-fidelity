#!/usr/bin/env python3
"""Resumable launch helpers for the ImageNet ResNet-50 D1 diagnostic."""

from __future__ import annotations

import fcntl
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import topk_imagenet_posthoc_q20k_eot_d1_common as common


PREPARE = common.PROJECT_DIR / "prepare_topk_imagenet_posthoc_q20k_eot_d1.py"
SEARCH = common.PROJECT_DIR / "search_topk_imagenet_posthoc_q20k_eot_d1.py"
REEVALUATE = (
    common.PROJECT_DIR / "reevaluate_topk_imagenet_posthoc_q20k_eot_d1.py"
)
FINALIZE = common.PROJECT_DIR / "finalize_topk_imagenet_posthoc_q20k_eot_d1.py"
AUDIT = common.PROJECT_DIR / "audit_topk_imagenet_posthoc_q20k_eot_d1.py"


def required_inputs() -> list[Path]:
    return [
        *(common.PROJECT_DIR / name for name in common.SCRIPT_NAMES),
        *(common.PROJECT_DIR / name for name in common.DEPENDENCY_NAMES),
    ]


def validate_project_files() -> None:
    missing = [str(path) for path in required_inputs() if not path.is_file()]
    if missing:
        raise SystemExit("Missing D1/R5 project files:\n  " + "\n  ".join(missing))


def acquire_lock(path: Path, label: str, *, blocking: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    mode = fcntl.LOCK_EX if blocking else fcntl.LOCK_EX | fcntl.LOCK_NB
    try:
        fcntl.flock(handle.fileno(), mode)
    except BlockingIOError as exc:
        handle.seek(0)
        owner = handle.read().strip() or "unknown"
        raise SystemExit(f"{label} is already running (recorded PID: {owner})") from exc
    handle.seek(0)
    handle.truncate()
    handle.write(str(os.getpid()))
    handle.flush()
    return handle


def release_lock(handle) -> None:
    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    handle.close()


def run_command(command: Sequence[str]) -> None:
    print("+ " + " ".join(str(value) for value in command), flush=True)
    subprocess.run(list(command), cwd=common.PROJECT_DIR, check=True)


def ensure_protocol_and_inputs(
    *,
    output_root: Path = common.DEFAULT_OUTPUT_ROOT,
    r5_root: Path = common.DEFAULT_R5_ROOT,
    device: str = "cuda:0",
    num_workers: int = 2,
) -> Path:
    validate_project_files()
    output_root = output_root.expanduser().resolve()
    r5_root = r5_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    lock = acquire_lock(
        output_root / ".protocol.lock", "ImageNet D1 protocol", blocking=True
    )
    try:
        parent = common.load_parent_r5(r5_root)
        payload = common.build_protocol_payload(parent=parent, output_root=output_root)
        protocol_path = output_root / "diagnostic_protocol.json"
        common.write_or_validate_json(protocol_path, payload)
        common.validate_protocol(common.read_json(protocol_path))
        run_command(
            [
                sys.executable,
                "-u",
                str(PREPARE),
                "--protocol-json",
                str(protocol_path),
                "--device",
                device,
                "--num-workers",
                str(num_workers),
            ]
        )
    finally:
        release_lock(lock)
    return output_root / "diagnostic_protocol.json"


def run_prepare(
    *,
    output_root: Path = common.DEFAULT_OUTPUT_ROOT,
    r5_root: Path = common.DEFAULT_R5_ROOT,
    device: str = "cuda:0",
    num_workers: int = 2,
) -> None:
    protocol = ensure_protocol_and_inputs(
        output_root=output_root,
        r5_root=r5_root,
        device=device,
        num_workers=num_workers,
    )
    print(f"ImageNet D1 preparation complete: {protocol}", flush=True)


def run_method(
    *,
    method: str,
    output_root: Path = common.DEFAULT_OUTPUT_ROOT,
    r5_root: Path = common.DEFAULT_R5_ROOT,
    device: str = "cuda:0",
    num_workers: int = 2,
    gpu_memory_fraction: float = 0.45,
) -> None:
    if method not in common.METHODS:
        raise ValueError(f"Unknown D1 method: {method}")
    output_root = output_root.expanduser().resolve()
    protocol_path = ensure_protocol_and_inputs(
        output_root=output_root,
        r5_root=r5_root,
        device=device,
        num_workers=num_workers,
    )
    lock = acquire_lock(output_root / f".{method}.lock", f"ImageNet D1 {method}")
    try:
        attack = common.METHOD_ATTACK[method]
        batch_size = common.SEARCH_BATCH_SIZE_BY_METHOD[method]
        eot = common.ADAPTIVE_CONFIG_BY_METHOD[method]["search_eot"]
        for number, seed in enumerate(common.SEARCH_SEEDS, start=1):
            cell = common.cell_dir(output_root, method, seed)
            print(
                f"[{number}/{len(common.SEARCH_SEEDS)}] ImageNet D1 {method}: "
                f"{attack}/seed={seed}/Q={common.API_BUDGET}/EOT={eot}",
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
        release_lock(lock)
    print(f"ImageNet D1 method complete: {method} -> {output_root}", flush=True)


def run_finalize(
    *,
    output_root: Path = common.DEFAULT_OUTPUT_ROOT,
    r5_root: Path = common.DEFAULT_R5_ROOT,
    device: str = "cuda:0",
    num_workers: int = 2,
) -> None:
    output_root = output_root.expanduser().resolve()
    protocol_path = ensure_protocol_and_inputs(
        output_root=output_root,
        r5_root=r5_root,
        device=device,
        num_workers=num_workers,
    )
    lock = acquire_lock(
        output_root / ".finalize.lock", "ImageNet D1 final analysis"
    )
    try:
        run_command(
            [
                sys.executable,
                "-u",
                str(FINALIZE),
                "--protocol-json",
                str(protocol_path),
                "--output-dir",
                str(output_root / "diagnostic_analysis"),
            ]
        )
        run_command(
            [
                sys.executable,
                "-u",
                str(AUDIT),
                "--output-root",
                str(output_root),
            ]
        )
    finally:
        release_lock(lock)

