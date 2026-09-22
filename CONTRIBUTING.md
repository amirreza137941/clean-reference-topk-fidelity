# Contributing

This is a research artifact under double-blind review. During review, please
use the anonymous issue channel designated by the artifact maintainers and do
not submit changes that reveal author identities.

For code changes:

1. create a focused branch;
2. add or update tests;
3. run `python scripts/audit_release.py` and `pytest -q`;
4. describe any change to the evaluation protocol explicitly.

Do not silently change frozen seeds, splits, budgets, checkpoint hashes, or
Monte Carlo counts. A protocol change must create a new stage/version rather
than overwrite the frozen one.
