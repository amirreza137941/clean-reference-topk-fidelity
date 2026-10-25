"""Run the clean-reference wrapper on a tiny model without external data."""

from __future__ import annotations

import torch
from torch import nn

from topk_fidelity import (
    CleanReferenceTopKDefense,
    DefenseConfig,
    compute_metric_batch,
)


class TinyClassifier(nn.Module):
    def __init__(self, classes: int = 6) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def main() -> None:
    torch.manual_seed(7)
    reference_model = TinyClassifier().eval()
    robust_model = TinyClassifier().eval()
    images = torch.rand(4, 3, 8, 8)
    labels = torch.tensor([0, 1, 2, 3])

    defense = CleanReferenceTopKDefense(
        standard_model=reference_model,
        gradient_model=robust_model,
        output_model=robust_model,
        config=DefenseConfig(
            top_k=2,
            alpha=0.005,
            epsilon=0.03,
            steps=2,
            sigma=0.01,
        ),
    )

    with torch.no_grad():
        reference_logits = reference_model(images)
    defended_logits = defense(images)
    metrics = compute_metric_batch(defended_logits, labels, reference_logits, k=2)
    print(metrics.summary)


if __name__ == "__main__":
    main()
