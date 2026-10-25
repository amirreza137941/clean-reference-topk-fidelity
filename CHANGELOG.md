# Changelog

## 0.2.0

- Extract the actual original Square search loops, objectives, proposal and
  archive logic into separate modules.
- Extract original CIFAR model definitions and training losses; retain
  checkpoint key layouts, normalization and the robust-branch adapter.
- Add validated per-dataset JSON settings, explicit deterministic-refresh and
  mean-logit accumulation routes, and a bounded per-image query ledger.
- Add a CPU demo, configuration CLI and evaluation of supplied CIFAR tensors,
  recording input/checkpoint hashes and query counts.
- Compare original and refactored values, gradients, attack candidates,
  accounting, checkpoint routes and training losses on CPU.
- Add migration/protocol guides, Persian instructions and CI style/build checks.
- Update the existing manuscript summary to include ViT as the online reference.
- Anchor ignored dataset/model/output directories to the repository root, so
  source code in src/topk_fidelity/models remains trackable.
- Preserve archived experiment Python sources and the existing licenses.

No new GPU training, full dataset run, speedup measurement or robustness result
is claimed by this release.
