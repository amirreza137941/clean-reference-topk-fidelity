import torch

from topk_fidelity.metrics import (
    compute_metric_batch,
    evaluate_draws,
    exact_set_match,
    ranked_topk,
)


def test_exact_set_is_unordered() -> None:
    reference = torch.tensor([[4, 2], [1, 3]])
    predicted = torch.tensor([[2, 4], [1, 0]])
    assert exact_set_match(reference, predicted).tolist() == [True, False]


def test_metric_batch_separates_top1_and_exact_set() -> None:
    reference_logits = torch.tensor([[5.0, 4.0, 1.0, 0.0]])
    defended_logits = torch.tensor([[5.0, 1.0, 4.0, 0.0]])
    labels = torch.tensor([0])
    batch = compute_metric_batch(defended_logits, labels, reference_logits, k=2)
    assert batch.summary["top1"] == 100.0
    assert batch.summary["reference_top1_match"] == 100.0
    assert batch.summary["exact_set"] == 0.0
    assert batch.summary["hidden_topk_failure"] == 100.0


def test_evaluate_draws_averages_metrics_not_logits() -> None:
    reference_logits = torch.tensor([[5.0, 4.0, 0.0]])
    labels = torch.tensor([0])
    draws = torch.tensor(
        [
            [[5.0, 4.0, 0.0]],
            [[5.0, 0.0, 4.0]],
        ]
    )
    values = evaluate_draws(draws, labels, reference_logits, k=2)
    assert values["exact_set"].item() == 0.5
    assert ranked_topk(reference_logits, 2).tolist() == [[0, 1]]
