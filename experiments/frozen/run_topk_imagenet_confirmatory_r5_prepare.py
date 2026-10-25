#!/usr/bin/env python3
"""Freeze the ImageNet R5 protocol, manifest, and clean Top-5 references."""

from topk_imagenet_confirmatory_r5_launcher import run_prepare


if __name__ == "__main__":
    run_prepare()

