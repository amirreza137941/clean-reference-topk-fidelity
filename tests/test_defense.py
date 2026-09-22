import torch
from torch import nn

from topk_fidelity.defense import CleanReferenceTopKDefense, DefenseConfig


class FlatLinear(nn.Module):
    def __init__(self, in_features: int, classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, classes, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x.flatten(1))


def make_defense(**overrides: object) -> CleanReferenceTopKDefense:
    model = FlatLinear(4, 4)
    with torch.no_grad():
        model.linear.weight.copy_(torch.eye(4))
    params = dict(top_k=2, alpha=0.1, epsilon=0.2, steps=1, sigma=0.0)
    params.update(overrides)
    return CleanReferenceTopKDefense(model, model, model, DefenseConfig(**params))


def test_objective_combines_top1_and_set_margins() -> None:
    defense = make_defense(steps=0)
    logits = torch.tensor([[5.0, 4.0, 1.0, 0.0]])
    reference = torch.tensor([[0, 1]])
    # Top-1 margin = 1, set margin = 3, weights = 1 and 4.
    assert defense.objective(logits, reference).item() == 13.0


def test_strengthening_is_projected_and_clipped() -> None:
    defense = make_defense(alpha=1.0, epsilon=0.05, steps=3)
    x = torch.tensor([[0.01, 0.40, 0.60, 0.99]])
    reference = defense.reference_set(x)
    strengthened = defense.strengthen(x, reference)
    assert torch.max(torch.abs(strengthened - x)).item() <= 0.050001
    assert torch.all((0.0 <= strengthened) & (strengthened <= 1.0))


def test_zero_step_zero_noise_matches_output_model() -> None:
    defense = make_defense(steps=0, sigma=0.0)
    x = torch.tensor([[0.2, 0.7, 0.1, 0.0]])
    expected = defense.output_model(x)
    actual = defense(x)
    assert torch.equal(actual, expected)
