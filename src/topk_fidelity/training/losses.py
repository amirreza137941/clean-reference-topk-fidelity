"""Original teacher Top-2 ranking and knowledge-distillation losses.

teacher_top2 is fixed from the CLEAN teacher input during training, unlike the
online request reference used by the inference defense. Optimizer schedules,
PGD generation and checkpoint selection remain in the archived train programs.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F


def set2_margin_loss(
    logits: torch.Tensor, teacher_top2: torch.Tensor, margin: float
) -> torch.Tensor:
    """Require both teacher-selected logits to outrank every other class."""
    selected = logits.gather(1, teacher_top2)
    weakest_selected = selected.min(dim=1).values
    outside_mask = torch.ones_like(logits, dtype=torch.bool)
    outside_mask.scatter_(1, teacher_top2, False)
    strongest_outside = logits.masked_fill(~outside_mask, float("-inf")).max(1).values
    return F.softplus(strongest_outside - weakest_selected + margin).mean()


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    return F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction="batchmean",
    ) * (temperature * temperature)
