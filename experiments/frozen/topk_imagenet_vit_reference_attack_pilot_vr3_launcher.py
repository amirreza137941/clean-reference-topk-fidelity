#!/usr/bin/env python3
"""Shared resumable launch logic for the ImageNet ViT-reference VR3 pilot."""

from __future__ import annotations

import fcntl
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence

import topk_imagenet_vit_reference_attack_pilot_vr3_common as common


PROJECT_DIR = Path(__file__).resolve().parent


def run(command: Sequence[str]) -> None:
    print("+ " + " ".join(str(value) for value in command), flush=True)
    subprocess.run(list(command), cwd=PROJECT_DIR, check=True)


@contextmanager
def lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield


def self_test() -> None:
    run([sys.executable, "-u", str(PROJECT_DIR / "test_topk_imagenet_vit_reference_attack_pilot_vr3.py")])


def prepare(output_root: Path = common.DEFAULT_OUTPUT_ROOT) -> Path:
    output_root = output_root.expanduser().resolve()
    with lock(output_root / ".prepare.lock"):
        run([
            sys.executable, "-u", str(PROJECT_DIR / "prepare_topk_imagenet_vit_reference_attack_pilot_vr3.py"),
            "--output-root", str(output_root),
        ])
    return output_root / "attack_pilot_protocol.json"


def run_search_group(methods: Sequence[str], group_name: str, *, run_tests: bool = True) -> None:
    if run_tests:
        self_test()
    protocol_path = prepare()
    root = common.DEFAULT_OUTPUT_ROOT.expanduser().resolve()
    with lock(root / f".{group_name}.lock"):
        for method in methods:
            for attack in common.ATTACKS:
                for seed in common.SEARCH_SEEDS:
                    directory = common.cell_dir(root, method, attack, seed)
                    run([
                        sys.executable, "-u", str(PROJECT_DIR / "search_topk_imagenet_vit_reference_attack_pilot_vr3.py"),
                        "--protocol-json", str(protocol_path),
                        "--method", method,
                        "--attack", attack,
                        "--search-seed", str(seed),
                        "--batch-size", str(common.SEARCH_BATCH_SIZE_BY_METHOD[method]),
                        "--output-dir", str(directory),
                    ])
    print(f"SEARCH_GROUP_COMPLETE={group_name}", flush=True)


def run_reevaluation(*, run_tests: bool = True) -> None:
    if run_tests:
        self_test()
    protocol_path = prepare()
    root = common.DEFAULT_OUTPUT_ROOT.expanduser().resolve()
    missing = [
        str(common.cell_dir(root, method, attack, seed) / "candidate.pt")
        for method, attack, seed in common.expected_cells()
        if not (common.cell_dir(root, method, attack, seed) / "candidate.pt").is_file()
    ]
    if missing:
        raise FileNotFoundError("Search is incomplete; missing candidates:\n  " + "\n  ".join(missing))
    output_dir = root / "common_mc1000"
    with lock(root / ".reeval.lock"):
        run([
            sys.executable, "-u", str(PROJECT_DIR / "reevaluate_topk_imagenet_vit_reference_attack_pilot_vr3_mc1000.py"),
            "--protocol-json", str(protocol_path),
            "--output-dir", str(output_dir),
        ])
        run([
            sys.executable, "-u", str(PROJECT_DIR / "audit_topk_imagenet_vit_reference_attack_pilot_vr3.py"),
            "--output-root", str(root),
        ])

