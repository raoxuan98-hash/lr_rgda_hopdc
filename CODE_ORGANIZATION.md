# LR-RGDA Code Organization

This note records the intended paper-aligned names and public interfaces for the
`lr_rgda` codebase.

## Paper-Aligned Method Names

| Paper term | Canonical code interface | Backward-compatible names |
| --- | --- | --- |
| LR-RGDA | `--classifier_types lr_rgda` | `qda`, `lrrgda`, `low_rank_rgda` |
| Full RGDA | `--classifier_types rgda_full` | `rgda`, `full_rgda` |
| HopDC | `--compensator_types "SeqFT + HopDC"` | `"SeqFT + Hopfield"` |
| No drift compensation | `--compensator_types SeqFT` | unchanged |

The old names remain accepted so older experiment scripts can still run, but new
scripts and logs should use `LR-RGDA` and `HopDC`.

## Main Runtime Path

1. `main.py` parses experiment arguments.
2. `trainer.py` creates the data manager and starts the learner.
3. `models/subspace_lora.py::SubspaceLoRA` runs task-wise backbone adaptation.
4. `compensator/distribution_compensator.py::DistributionCompensator` builds
   current and drift-compensated Gaussian statistics.
5. `classifier/classifier_builder.py::ClassifierReconstructor` constructs
   analytic classifiers from those statistics.
6. `classifier/gaussian_classifier.py::LowRankGaussianDA` implements LR-RGDA.
7. `compensator/hopdc.py::HopDC` exposes the Hopfield-based distribution
   compensator.

## Experiment Layout

Standalone experiment artifacts are now separated from the runtime root:

- `experiments/ablations/gaussianity/`: skewness/kurtosis and non-Gaussian
  stress-test scripts.
- `experiments/ablations/efficiency/`: standalone efficiency comparison.
- `experiments/legacy_rebuttal/`: rebuttal-era one-off scripts and notebooks.
- `experiments/legacy_prototypes/`: old evaluator prototypes.
- `scripts/maintenance/`: dataset preparation and feature-cache utilities.
- `sh/legacy_root/`: old root-level launchers kept for traceability.

The existing `classifier_ablation/` package stays at the root for now because it
has many internal imports and shell launchers that reference that package path.
Treat it as the active ablation package until a dedicated import-compatible
migration is done.

## Defaults

The default Stage-2 configuration now follows the paper method:

```bash
--classifier_types lr_rgda
--compensator_types SeqFT "SeqFT + HopDC"
```

This evaluates both the uncompensated baseline and the HopDC-compensated
variant with LR-RGDA.

## Cleanup Priorities

1. Keep paper-facing entry points in `main.py`, `trainer.py`, `models/`,
   `classifier/`, `compensator/`, and `utils/`.
2. Continue moving old rebuttal, plotting, and exploratory scripts under
   `experiments/legacy_*` only after confirming no paper table depends on them.
3. Split argument parsing from heavy training imports so `python main.py --help`
   works without importing `timm` and the full model stack.
4. Replace legacy log strings such as `QDA` and `Hopfield` in new scripts with
   `LR-RGDA` and `HopDC`.
