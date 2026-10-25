#!/usr/bin/env python3
"""Unit tests for the Gaussian-ensemble screening script."""

from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]


@unittest.skipUnless(torch is not None, "PyTorch is unavailable in this runtime")
class GaussianEnsembleScreenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import evaluate_topk_cifar100_gaussian_ensemble_screen_r7 as subject

        cls.subject = subject

    def test_parse_int_tuple(self) -> None:
        self.assertEqual(self.subject.parse_int_tuple("8,1,4"), (1, 4, 8))
        with self.assertRaises(Exception):
            self.subject.parse_int_tuple("1,1")

    def test_balanced_subset_positions(self) -> None:
        labels = torch.arange(100).repeat_interleave(3)
        positions = self.subject.balanced_subset_positions(labels, 200, seed=7)
        selected = labels.index_select(0, positions)
        counts = torch.bincount(selected, minlength=100)
        self.assertTrue(torch.equal(counts, torch.full((100,), 2)))

    def test_prefix_aggregates(self) -> None:
        logits = torch.tensor(
            [[[[1.0, 0.0]], [[3.0, 2.0]], [[5.0, 4.0]], [[7.0, 6.0]]]]
        )
        mean_logits, mean_probabilities = self.subject.prefix_aggregates(
            logits, (1, 2, 4)
        )
        self.assertTrue(torch.allclose(mean_logits[1], logits[:, 0]))
        self.assertTrue(
            torch.allclose(mean_logits[2], (logits[:, 0] + logits[:, 1]) / 2.0)
        )
        self.assertTrue(torch.allclose(mean_logits[4], logits.mean(1)))
        self.assertEqual(mean_probabilities[4].shape, (1, 1, 2))
        self.assertTrue(
            torch.allclose(
                mean_probabilities[4].sum(-1), torch.ones((1, 1)), atol=1e-6
            )
        )

    def test_select_compute_matched_b(self) -> None:
        rows = [
            {"method": "full", "B": 1, "batch_size": 1, "latency_p50_ms": 50.0},
            {
                "method": "gaussian_mean_logits",
                "B": 4,
                "batch_size": 1,
                "latency_p50_ms": 20.0,
            },
            {
                "method": "gaussian_mean_logits",
                "B": 8,
                "batch_size": 1,
                "latency_p50_ms": 48.0,
            },
            {
                "method": "gaussian_mean_logits",
                "B": 16,
                "batch_size": 1,
                "latency_p50_ms": 60.0,
            },
        ]
        selected = self.subject.select_compute_matched_b(
            rows, selection_batch_size=1, tolerance=0.05
        )
        self.assertEqual(selected["selected_B"], 8)
        self.assertEqual(selected["closest_B"], 8)

    def test_sigma_zero_all_logit_prefixes_match_robust(self) -> None:
        class TinyAnchor(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("class_scale", torch.linspace(-1.0, 1.0, 100))

            def forward(self, images):
                mean = images.flatten(1).mean(1, keepdim=True)
                first = images[:, :1, :1, :1].flatten(1)
                return (
                    mean * self.class_scale[None]
                    + 0.01 * first * self.class_scale.flip(0)[None]
                )

        class IdentityFullDefense:
            def base(self, images):
                return images.detach()

        torch.manual_seed(11)
        images = torch.rand(4, 3, 32, 32)
        labels = torch.tensor([99, 99, 99, 99], dtype=torch.long)
        references = torch.tensor([[99, 98]] * 4, dtype=torch.long)
        sample_indices = torch.arange(4)
        result = self.subject.evaluate_condition(
            images_cpu=images,
            labels_cpu=labels,
            reference_top2_cpu=references,
            sample_indices_cpu=sample_indices,
            anchor=TinyAnchor().eval(),
            full_defense=IdentityFullDefense(),
            device=torch.device("cpu"),
            sigma=0.0,
            full_sigma=0.0,
            b_values=(1, 2, 4),
            outer_draws=3,
            batch_size=2,
            outer_chunk=2,
            forward_microbatch=8,
            random_seed=13,
        )
        robust = result["robust"]
        for method, metrics in result.items():
            for name in self.subject.METRIC_NAMES:
                if (
                    method.startswith("gaussian_mean_probabilities")
                    and name == "boundary_margin_mean"
                ):
                    continue
                self.assertTrue(
                    torch.allclose(metrics[name], robust[name], atol=1e-10),
                    msg=f"{method}/{name}",
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
