#!/usr/bin/env python3
"""Shared contracts for the ImageNet robust-anchor acquisition stage (R1).

This module deliberately contains no torch/torchvision import.  The self-test
can therefore validate protocol constants and download guards before CUDA/model
initialization.  R1 downloads one published model checkpoint only; it never
downloads or modifies ImageNet.
"""

from __future__ import annotations

import csv
import hashlib
import html.parser
import json
import os
import shutil
import time
import urllib.parse
import urllib.request
from http.cookiejar import CookieJar
from pathlib import Path
from typing import Any, Mapping, Sequence


PROTOCOL_ID = "imagenet_topk_robust_anchor_r1_20260910"
PARENT_PROTOCOL_ID = "imagenet_topk_preflight_r0_20260910"
FROZEN_SPLIT_SHA256 = (
    "829b84ffac9f2fe4d97f812b42bf91d8d6254662a8dcfdee99ec85345ca88619"
)
STANDARD_CHECKPOINT_SHA256 = (
    "c377ee6a665d9cb868b071b2623dff729084f6b3fe0faad6b79c0ab51b99a33f"
)

ROBUSTBENCH_MODEL_ID = "Salman2020Do_R50"
ROBUSTBENCH_DATASET = "imagenet"
ROBUSTBENCH_THREAT_MODEL = "Linf"
ROBUSTBENCH_EPSILON = 4.0 / 255.0
ROBUSTBENCH_GDRIVE_ID = "1TmT5oGa1UvVjM3d-XeSj_XmKqBNRUg8r"
ROBUSTBENCH_REGISTRY_URL = (
    "https://github.com/RobustBench/robustbench/blob/master/"
    "robustbench/model_zoo/imagenet.py"
)
ROBUSTBENCH_LEADERBOARD_URL = "https://robustbench.github.io/"
PAPER_URL = "https://arxiv.org/abs/2007.08489"
OFFICIAL_CLEAN_ACCURACY_PERCENT = 64.02
OFFICIAL_ROBUST_ACCURACY_PERCENT = 34.96
PREPROCESSING_ID = "Res256Crop224"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

EXPECTED_ARCHITECTURE_ID = "torchvision_resnet50"
EXPECTED_FAMILY = "resnet50"
EXPECTED_NUM_CLASSES = 1000
EXPECTED_SANITY_EXAMPLES = 1000
MINIMUM_RAW_CHECKPOINT_BYTES = 50_000_000
MAXIMUM_RAW_CHECKPOINT_BYTES = 250_000_000

SCRIPT_NAMES = [
    "topk_imagenet_robust_anchor_common_r1.py",
    "prepare_topk_imagenet_robust_anchor_r1.py",
    "audit_topk_imagenet_robust_anchor_r1.py",
    "test_topk_imagenet_robust_anchor_r1.py",
    "run_topk_imagenet_robust_anchor_r1.py",
]
REQUIRED_R0_SOURCES = [
    "topk_imagenet_preflight_common.py",
    "preflight_topk_imagenet_r0.py",
]


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def stable_hash(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def atomic_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    fieldnames: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def human_bytes(value: int) -> str:
    number = float(value)
    for suffix in ("B", "KiB", "MiB", "GiB"):
        if number < 1024.0 or suffix == "GiB":
            return f"{number:.2f} {suffix}"
        number /= 1024.0
    raise AssertionError("unreachable")


def validate_download_candidate(path: Path) -> dict[str, object]:
    """Reject HTML/error pages and implausible checkpoint sizes before torch.load."""
    if not path.is_file():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    if not MINIMUM_RAW_CHECKPOINT_BYTES <= size <= MAXIMUM_RAW_CHECKPOINT_BYTES:
        raise ValueError(
            f"Downloaded file size {size} is outside the frozen "
            f"[{MINIMUM_RAW_CHECKPOINT_BYTES}, {MAXIMUM_RAW_CHECKPOINT_BYTES}] byte guard"
        )
    with path.open("rb") as handle:
        prefix = handle.read(512).lstrip().lower()
    if prefix.startswith((b"<!doctype html", b"<html", b"<?xml")):
        raise ValueError("Download is an HTML/XML response, not a PyTorch checkpoint")
    return {
        "path": str(path),
        "bytes": size,
        "human_size": human_bytes(size),
        "sha256": sha256_file(path),
    }


class _DriveFormParser(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.action: str | None = None
        self.inputs: dict[str, str] = {}
        self._in_download_form = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {key: value for key, value in attrs}
        if tag == "form" and values.get("id") == "download-form":
            self._in_download_form = True
            self.action = values.get("action")
        elif tag == "input" and self._in_download_form:
            name = values.get("name")
            value = values.get("value")
            if name is not None and value is not None:
                self.inputs[name] = value

    def handle_endtag(self, tag: str) -> None:
        if tag == "form":
            self._in_download_form = False


def _stream_urllib_response(response: Any, destination: Path) -> None:
    total = int(response.headers.get("Content-Length", "0") or 0)
    written = 0
    last_report = time.monotonic()
    with destination.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            written += len(chunk)
            if time.monotonic() - last_report >= 5.0:
                suffix = f"/{human_bytes(total)}" if total else ""
                print(f"    downloaded {human_bytes(written)}{suffix}", flush=True)
                last_report = time.monotonic()


def download_with_urllib(gdrive_id: str, destination: Path) -> str:
    """Dependency-free Google Drive download with current and legacy flows."""
    user_agent = (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
    opener = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(CookieJar())
    )
    initial_url = "https://drive.usercontent.google.com/download?" + urllib.parse.urlencode(
        {"id": gdrive_id, "export": "download", "confirm": "t"}
    )
    request = urllib.request.Request(initial_url, headers={"User-Agent": user_agent})
    response = opener.open(request, timeout=120)
    content_type = response.headers.get("Content-Type", "").lower()
    disposition = response.headers.get("Content-Disposition", "").lower()
    if "text/html" in content_type and "attachment" not in disposition:
        body = response.read(2 * 1024 * 1024).decode("utf-8", errors="replace")
        response.close()
        parser = _DriveFormParser()
        parser.feed(body)
        if not parser.action:
            raise RuntimeError(
                "Google Drive returned a confirmation/access page without a download form"
            )
        follow_url = urllib.parse.urljoin(initial_url, parser.action)
        if parser.inputs:
            follow_url += ("&" if "?" in follow_url else "?") + urllib.parse.urlencode(
                parser.inputs
            )
        response = opener.open(
            urllib.request.Request(follow_url, headers={"User-Agent": user_agent}),
            timeout=120,
        )
    try:
        _stream_urllib_response(response, destination)
    finally:
        response.close()
    return "urllib_google_drive"


def download_with_gdown(gdrive_id: str, destination: Path) -> str:
    try:
        import gdown  # type: ignore
    except ImportError as exc:
        raise RuntimeError("gdown is not installed") from exc
    result = gdown.download(
        id=gdrive_id,
        output=str(destination),
        quiet=False,
        resume=False,
    )
    if result is None or not destination.is_file():
        raise RuntimeError("gdown did not create the requested checkpoint")
    return "gdown_google_drive"


def acquire_official_checkpoint(
    destination: Path,
    backend: str,
    allow_download: bool,
) -> dict[str, object]:
    """Acquire exactly one frozen RobustBench registry object, atomically.

    Existing valid files are reused without network access.  Existing invalid
    files are never overwritten automatically; the user must inspect/move them.
    """
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        report = validate_download_candidate(destination)
        return {**report, "acquisition": "reused_existing", "network_used": False}
    if not allow_download:
        raise FileNotFoundError(
            f"Official raw checkpoint is absent and --no-download was set: {destination}"
        )
    if backend not in {"auto", "gdown", "urllib"}:
        raise ValueError(f"Unknown download backend: {backend}")

    methods = [backend] if backend != "auto" else ["gdown", "urllib"]
    failures: list[str] = []
    for method in methods:
        temporary = destination.with_name(
            destination.name + f".part.{os.getpid()}.{method}"
        )
        if temporary.exists():
            temporary.unlink()
        try:
            print(
                f"Downloading only {ROBUSTBENCH_MODEL_ID} via {method}; "
                "ImageNet is not downloaded.",
                flush=True,
            )
            used = (
                download_with_gdown(ROBUSTBENCH_GDRIVE_ID, temporary)
                if method == "gdown"
                else download_with_urllib(ROBUSTBENCH_GDRIVE_ID, temporary)
            )
            report = validate_download_candidate(temporary)
            os.replace(temporary, destination)
            return {
                **report,
                "path": str(destination),
                "acquisition": used,
                "network_used": True,
            }
        except Exception as exc:
            failures.append(f"{method}: {type(exc).__name__}: {exc}")
            if temporary.exists():
                failed = temporary.with_name(temporary.name + ".failed")
                if failed.exists():
                    failed = temporary.with_name(
                        temporary.name + f".failed.{int(time.time())}"
                    )
                shutil.move(str(temporary), str(failed))
                failures[-1] += f" (response preserved at {failed})"
    raise RuntimeError(
        "All official-checkpoint download backends failed. "
        + " | ".join(failures)
        + f". You may use `python -m gdown {ROBUSTBENCH_GDRIVE_ID} -O {destination}` "
        "and rerun; do not download or rearrange ImageNet."
    )


def source_provenance() -> dict[str, object]:
    return {
        "provider": "RobustBench official model zoo",
        "model_id": ROBUSTBENCH_MODEL_ID,
        "dataset": ROBUSTBENCH_DATASET,
        "threat_model": ROBUSTBENCH_THREAT_MODEL,
        "epsilon_linf_pixel": ROBUSTBENCH_EPSILON,
        "epsilon_linf_255": 4.0,
        "architecture": "ResNet-50",
        "preprocessing": PREPROCESSING_ID,
        "normalization_mean": list(IMAGENET_MEAN),
        "normalization_std": list(IMAGENET_STD),
        "google_drive_id": ROBUSTBENCH_GDRIVE_ID,
        "registry_url": ROBUSTBENCH_REGISTRY_URL,
        "leaderboard_url": ROBUSTBENCH_LEADERBOARD_URL,
        "paper_url": PAPER_URL,
        "robustbench_reported_clean_accuracy_percent": OFFICIAL_CLEAN_ACCURACY_PERCENT,
        "robustbench_reported_robust_accuracy_percent": OFFICIAL_ROBUST_ACCURACY_PERCENT,
        "paper_claim_scope": (
            "Published adversarially trained ImageNet ResNet-50 anchor. The R1 PGD "
            "check is identity/provenance sanity only, not a reproduction of RobustBench."
        ),
    }


def script_hashes(project_dir: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for name in SCRIPT_NAMES + REQUIRED_R0_SOURCES:
        path = project_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Required source is missing: {path}")
        result[name] = sha256_file(path)
    return result
