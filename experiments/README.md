# Experiments

This directory separates standalone experiment code from the paper-facing runtime
modules.

## Subdirectories

- `ablations/`: experiments that support paper claims or appendix tables.
- `legacy_rebuttal/`: one-off scripts and notebooks created during rebuttal.
- `legacy_prototypes/`: older evaluator prototypes kept for traceability.
- `scratch/`: temporary work that is not part of the reproducible paper path.

The integrated `classifier_ablation/` package remains at the repository root for
now because many scripts import it by that package name.

