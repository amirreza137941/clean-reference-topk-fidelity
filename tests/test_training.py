"""Compare loss values and gradients to both original training scripts."""

import pytest
import torch
from conftest import original_symbols

from topk_fidelity.training import distillation_loss, set2_margin_loss


@pytest.mark.parametrize(
    "filename",
    [
        "train_resnet20_teacher_top2_adversarial_cifar10.py",
        "train_resnet20_teacher_top2_adversarial_cifar100_r2.py",
    ],
)
def test_original_training_losses_and_gradients(filename):
    original = original_symbols(filename, ["set2_margin_loss", "distillation_loss"])
    torch.manual_seed(17)
    student = torch.randn(5, 10, requires_grad=True)
    teacher = torch.randn(5, 10)
    fixed_clean_top2 = teacher.topk(2, dim=1).indices
    old = original.set2_margin_loss(
        student, fixed_clean_top2, 0.5
    ) + original.distillation_loss(student, teacher, 4.0)
    new = set2_margin_loss(student, fixed_clean_top2, 0.5) + distillation_loss(
        student, teacher, 4.0
    )
    torch.testing.assert_close(new, old, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.autograd.grad(new, student)[0],
        torch.autograd.grad(old, student)[0],
        rtol=0,
        atol=0,
    )
