# Result summaries

`paper_summary.csv` transcribes the current audited ResNet-reference
confirmatory table into a machine-readable, small file. Percentages are
absolute exact clean-reference set fidelity, not relative changes.

The ImageNet/ViT row is intentionally absent because VR4 is still running. Add
it only after the frozen stage's finalizer and audit both succeed. Raw examples,
candidate tensors, and Monte Carlo draws do not belong in Git.
