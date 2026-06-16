# Migration Manifest

This manifest tracks the first organization pass. The goal is to make the paper
pipeline visible while keeping old experiment code discoverable.

## Directory Roles

| Path | Role | Stability |
| --- | --- | --- |
| `main.py`, `trainer.py` | Primary runtime entry and training orchestration | Paper-facing |
| `classifier/` | Analytic and baseline classifier builders | Paper-facing |
| `compensator/` | HopDC and drift-compensation baselines | Paper-facing |
| `models/` | Backbone adaptation, LoRA/full fine-tuning, random projection baselines | Paper-facing |
| `utils/` | Dataset managers and shared helpers | Paper-facing |
| `classifier_ablation/` | Integrated ablation package used by many appendix/rebuttal scripts | Ablation, keep in place for now |
| `experiments/ablations/` | Standalone paper-support ablations moved out of the root | Paper-support |
| `experiments/legacy_rebuttal/` | Rebuttal-era one-off analysis and visualization scripts | Legacy |
| `experiments/legacy_prototypes/` | Old prototype evaluators and baseline tests | Legacy |
| `scripts/maintenance/` | Dataset and feature-cache maintenance utilities | Utility |
| `configs/` | Future config-driven experiment definitions | Scaffold |
| `sh/` | Existing shell launchers; still supported but should be gradually reduced | Transitional |

## Moved Files

| Old path | New path | Reason |
| --- | --- | --- |
| `feature_statistics.py` | `experiments/ablations/gaussianity/feature_statistics.py` | Gaussianity/skewness/kurtosis appendix support |
| `power_transform_experiment.py` | `experiments/ablations/gaussianity/power_transform_experiment.py` | Non-Gaussian stress-test appendix support |
| `run_stats_parallel.sh` | `experiments/ablations/gaussianity/run_stats_parallel.sh` | Launcher for Gaussianity statistics |
| `run_all_pt.sh` | `experiments/ablations/gaussianity/run_all_pt.sh` | Launcher for power-transform stress tests |
| `exp_efficiency_comparison.py` | `experiments/ablations/efficiency/exp_efficiency_comparison.py` | Standalone efficiency comparison |
| `rebuttal_tnse可视化.py` | `experiments/legacy_rebuttal/visualization/rebuttal_tnse可视化.py` | Rebuttal visualization |
| `rebuttal_tnse可视化.ipynb` | `experiments/legacy_rebuttal/visualization/rebuttal_tnse可视化.ipynb` | Rebuttal visualization notebook |
| `rebuttal_峰度偏度测试.py` | `experiments/legacy_rebuttal/gaussianity_old/rebuttal_峰度偏度测试.py` | Older Gaussianity test prototype |
| `classifier_evaluation.py` | `experiments/legacy_prototypes/classifier_eval/classifier_evaluation.py` | Old classifier evaluator prototype |
| `evaluate_baselines.py` | `experiments/legacy_prototypes/classifier_eval/evaluate_baselines.py` | Old baseline evaluator prototype |
| `dataset_resplitter.py` | `scripts/maintenance/dataset_resplitter.py` | Dataset preparation utility |
| `test_cross_domain_data.py` | `scripts/maintenance/test_cross_domain_data.py` | Dataset sanity-check utility |
| `extract_features.py` | `scripts/maintenance/feature_cache/extract_features.py` | Feature-cache extraction utility |
| `extract_seed_1993.sh` | `scripts/maintenance/feature_cache/extract_seed_1993.sh` | Feature-cache extraction launcher |
| `extract_seed_1993_parallel.sh` | `scripts/maintenance/feature_cache/extract_seed_1993_parallel.sh` | Parallel feature-cache extraction launcher |
| `eval_hopdc.sh` | `sh/legacy_root/eval_hopdc.sh` | Old root launcher |
| `run_lora_kd.sh` | `sh/legacy_root/run_lora_kd.sh` | Old root launcher |
| `run_stage1.sh` | `sh/legacy_root/run_stage1.sh` | Old root launcher |
| `run_stage1_cross.sh` | `sh/legacy_root/run_stage1_cross.sh` | Old root launcher |

## Keep-In-Place Decisions

`classifier_ablation/` is intentionally not moved in this pass. Its files import
each other through the package name `classifier_ablation`, and many shell scripts
call those paths directly. Moving it should be a separate migration with import
aliases or wrapper entry points.

The root runtime files are also intentionally kept in place to preserve current
server usage:

- `main.py`
- `trainer.py`
- `models/`
- `classifier/`
- `compensator/`
- `utils/`

## Next Migration

1. Add config files under `configs/` for the exact TPAMI table runs.
2. Replace broad shell launchers in `sh/` with small config-driven launchers.
3. Split `main.py` parser from heavy training imports.
4. Migrate `classifier_ablation/` only after adding compatibility wrappers.
