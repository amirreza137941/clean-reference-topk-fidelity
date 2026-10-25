"""Original pixel-space L-infinity Square proposals.

Proposal RNG order, 16 sign attempts, clipping and patch bounds are preserved.
The upstream BSD notice is reproduced in THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import math

import numpy as np
import torch


def square_vertical_stripe_initialization(
    x: torch.Tensor, eps: float, rng: np.random.Generator
) -> torch.Tensor:
    batch, channels, _, width = x.shape
    signs = rng.choice(
        np.asarray([-1.0, 1.0], dtype=np.float32),
        size=(batch, channels, 1, width),
    )
    delta = torch.from_numpy(signs).to(device=x.device, dtype=x.dtype) * float(eps)
    return (x + delta).clamp(0.0, 1.0)


def square_reference_linf_proposal(
    x: torch.Tensor,
    incumbent: torch.Tensor,
    eps: float,
    p: float,
    rng: np.random.Generator,
) -> torch.Tensor:
    batch, channels, height, width = x.shape
    if height != width:
        raise ValueError("Square proposal expects square images")
    side = int(round(math.sqrt(float(p) * height * width)))
    side = min(max(side, 1), height - 1)
    candidate = incumbent.detach().clone()
    for sample in range(batch):
        row = int(rng.integers(0, height - side))
        col = int(rng.integers(0, width - side))
        clean_window = x[sample, :, row : row + side, col : col + side]
        old_window = incumbent[sample, :, row : row + side, col : col + side]
        signs_np = rng.choice(
            np.asarray([-1.0, 1.0], dtype=np.float32),
            size=(16, channels, 1, 1),
        )
        signs = torch.from_numpy(signs_np).to(device=x.device, dtype=x.dtype)
        windows = (clean_window.unsqueeze(0) + float(eps) * signs).clamp(0.0, 1.0)
        changed = (windows - old_window.unsqueeze(0)).abs().gt(1e-7).flatten(1).any(1)
        first_changed = changed.to(torch.int64).argmax()
        fallback = (clean_window - float(eps) * signs[0]).clamp(0.0, 1.0)
        new_window = torch.where(changed.any(), windows[first_changed], fallback)
        candidate[sample, :, row : row + side, col : col + side] = new_window
    return candidate
