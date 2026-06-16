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

## Optional LR-RGDA-MC Classifier

The default `lr_rgda` classifier remains the analytic single-center LR-RGDA
baseline. The optional `lr_rgda_mc` classifier can be evaluated in the same
backbone-training run; it uses multiple per-class centers and affine-only
classifier fitting from compact diagonal-GMM replay statistics.

```bash
python main.py \
  --dataset cifar100_224 \
  --smart_defaults \
  --classifier_types lr_rgda lr_rgda_mc \
  --compensator_types SeqFT "SeqFT + HopDC"
```

- `lr_rgda`: single-center analytic LR-RGDA; no classifier fitting.
- `lr_rgda_mc`: default `--rgda_mc_num_centers 4`,
  `--rgda_mc_train_iter 200`, `--rgda_gmm_k 4`,
  `--rgda_mc_fit_samples_per_class 16`.
- `--rgda_gmm_sample_mode mean`: repeats fitted GMM component means for compact
  replay; this is the default because it has been more stable than stochastic
  GMM samples in recent classifier-replay experiments.
- `--rgda_gmm_sample_mode sample`: samples from the stored diagonal-Gaussian GMM
  components when stochastic replay is explicitly desired.

Report `lr_rgda_mc` separately from the current single-seed main table until
multi-seed confidence intervals are rerun.

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
