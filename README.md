# LR-RGDA / HopDC

This repository contains the code for the TPAMI submission on scalable analytic
classifiers and associative drift compensation for class-incremental learning.

## Paper-Facing Entry

The canonical runtime entry remains:

```bash
python main.py \
  --dataset cifar100_224 \
  --smart_defaults \
  --classifier_types lr_rgda \
  --compensator_types SeqFT "SeqFT + HopDC"
```

Canonical names:

- `lr_rgda`: Low-Rank Factorized RGDA.
- `rgda_full`: full-rank RGDA.
- `SeqFT + HopDC`: Hopfield-based Distribution Compensator.

Legacy aliases such as `qda` and `SeqFT + Hopfield` are still accepted, but new
experiments should use the paper names.

## Evaluation Datasets

The code now accepts the following PILOT/MOS processed ImageFolder datasets:

- `imagenet-a`: ImageNet-A, 200 classes, default protocol `20 + 20`.
- `vtab`: VTAB, 50 classes, default protocol `10 + 10`.

The expected server data root is:

```text
/data1/open_datasets/chinese-clip-eval/elevater/
```

After unpacking the official archives, the directories should be:

```text
/data1/open_datasets/chinese-clip-eval/elevater/imagenet-a/train
/data1/open_datasets/chinese-clip-eval/elevater/imagenet-a/test
/data1/open_datasets/chinese-clip-eval/elevater/vtab/train
/data1/open_datasets/chinese-clip-eval/elevater/vtab/test
```

Example:

```bash
python main.py --dataset imagenet-a --smart_defaults --classifier_types lr_rgda
python main.py --dataset vtab --smart_defaults --classifier_types lr_rgda
```

## Organization

- `classifier/`: LR-RGDA, full RGDA, LDA, SGD and other classifier builders.
- `compensator/`: HopDC and drift-compensation baselines.
- `models/`: backbone adaptation and LoRA/full-finetuning wrappers.
- `utils/`: dataset managers and shared utilities.
- `classifier_ablation/`: existing ablation package retained in place.
- `experiments/ablations/`: standalone paper-support experiments.
- `experiments/legacy_rebuttal/`: rebuttal-era one-off scripts and notebooks.
- `experiments/legacy_prototypes/`: old evaluator prototypes.
- `configs/`: scaffold for future config-driven experiment definitions.
- `sh/`: existing shell launchers, kept during transition.
- `scripts/maintenance/`: dataset and feature-cache utilities.

See `CODE_ORGANIZATION.md` and `MIGRATION_MANIFEST.md` for details.

## Maintenance Utilities

Dataset preparation and feature-cache utilities live under
`scripts/maintenance/`. Older root-level launchers were moved to
`sh/legacy_root/`.
