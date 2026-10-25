#!/usr/bin/env python3
"""Fail if a release contains identifying paths, likely secrets, or large files."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_TRACKED_BYTES = 10 * 1024 * 1024
TEXT_SUFFIXES = {
    ".cff",
    ".csv",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".svg",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
SKIP_PARTS = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "build",
    "dist",
}
REQUIRED = {
    ".github/workflows/ci.yml",
    ".gitignore",
    "LICENSE",
    "MANIFEST.sha256",
    "README.md",
    "THIRD_PARTY_NOTICES.md",
    "docs/ANONYMOUS_ARTIFACT.md",
    "docs/REPRODUCIBILITY.md",
    "src/topk_fidelity/defense.py",
    "src/topk_fidelity/metrics.py",
    "src/topk_fidelity/attacks/adaptive.py",
    "src/topk_fidelity/attacks/racing.py",
    "src/topk_fidelity/models/checkpoints.py",
    "configs/cifar10.json",
    "configs/cifar100.json",
    "docs/MIGRATION.md",
    "docs/VALIDATION.md",
}

FORBIDDEN = {
    "absolute Unix home path": re.compile(r"/(?:home|Users)/[A-Za-z0-9._-]+/"),
    "private SSH key": re.compile(r"BEGIN (?:OPENSSH|RSA|EC|DSA) PRIVATE KEY"),
    "GitHub token": re.compile(r"\b(?:ghp|github_pat)_[A-Za-z0-9_]{20,}\b"),
    "OpenAI-style secret": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "AWS access key": re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
}


def files() -> list[Path]:
    return [
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and not any(
            part in SKIP_PARTS or part.endswith(".egg-info") for part in path.parts
        )
    ]


def main() -> int:
    errors: list[str] = []
    existing = {str(path.relative_to(ROOT)) for path in files()}
    for required in sorted(REQUIRED - existing):
        errors.append(f"missing required file: {required}")

    for path in files():
        relative = path.relative_to(ROOT)
        size = path.stat().st_size
        if size > MAX_TRACKED_BYTES:
            errors.append(f"file exceeds 10 MiB: {relative} ({size} bytes)")
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {
            ".gitattributes",
            ".gitignore",
        }:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            errors.append(f"expected text but could not decode UTF-8: {relative}")
            continue
        for label, pattern in FORBIDDEN.items():
            if pattern.search(text):
                errors.append(f"{label} found in {relative}")

    if errors:
        print("Release audit FAILED:")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"Release audit passed ({len(existing)} files checked).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
