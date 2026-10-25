# Refactor validation

Validation date: 2026-10-08. Tests run locally on CPU.

## Environment

| Component | Tested version |
|---|---|
| Python | 3.12.14 |
| PyTorch | 2.14.1+cpu |
| NumPy | 2.3.5 |
| pytest | 9.1.1 |
| Ruff | 0.16.10 |
| CUDA availability | false |

These versions record the verification environment. They are not claimed to
be the original GPU experiment environment.

## Evidence

64 tests passed. The suite includes:

- exact CPU equality of original/refactored request logits and objective
  gradients, k=2 and k=5, deterministic/noisy and zero-step/strengthening modes;
- a separate comparison with the actual CIFAR-10 Top-2 wrapper;
- exact returned candidate and query-detail equality against the original
  generic R4 SquareAdaptive and SquareRacing core, batch sizes 1 and 3;
- direct comparisons with CIFAR-10 Adaptive and Racing, including zero-refresh
  deterministic search and sequential mean-logit accumulation;
- actual original ResNet-20 checkpoint layouts: projection and option-A
  reference blocks, bare/wrapped weights, DataParallel prefixes, CIFAR-100
  aliases and normalization, dual robust-branch output and input gradients;
- equality of teacher ranking/distillation loss values and gradients against
  both original training scripts;
- per-response ExactSet and joint hidden-failure scoring, chunk-invariant
  endpoint accumulation, request-level reference recomputation and hard query
  limit enforcement;
- configuration agreement with archived frozen settings, and CLI evaluation
  of synthetic CIFAR tensors/checkpoints through the real ResNet architectures.

All 167 archived Python programs are byte-identical to the supplied v0.1.0
repository ZIP. The release records moved symbols in
`refactor_provenance.json` and hashes distributable files in
`../MANIFEST.sha256`.

Style checks, byte compilation, release audit, configuration/demo entry
points and source/wheel builds are checked as part of package preparation.
The wheel retains both the original MIT license and the Square BSD notice.

## Scope

The checkpoint tests use generated weights in the original architectures and
container layouts; the actual research checkpoint bytes are not included.
GPU/AMP kernels, original datasets, full 5,000-query dataset sweeps, training,
historical parent-artifact authorization and runtime benchmarks were not run
again. Existing manuscript results remain tied to their original artifacts.

Exact equality in the listed CPU comparisons supports preservation of those
paths. It does not certify equality across hardware, all arbitrary oracles or
unseen serialized checkpoint formats. The archived scientific programs remain
authoritative for a full historical replay.
