"""Original stable seed derivation and experiment RNG initialization."""

from __future__ import annotations

import hashlib
import random

import numpy as np
import torch


def stable_int(text: str, modulus: int = 2**31 - 1) -> int:
    return int(hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:16], 16) % modulus


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
