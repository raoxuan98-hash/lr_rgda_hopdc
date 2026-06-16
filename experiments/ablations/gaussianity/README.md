# Gaussianity Ablations

Purpose:

- Estimate skewness/kurtosis of cached class-wise features.
- Run power-transform stress tests for non-Gaussian features.

Main files:

- `feature_statistics.py`
- `power_transform_experiment.py`
- `run_stats_parallel.sh`
- `run_all_pt.sh`

Run from the `lr_rgda` root so imports such as `classifier.*` resolve correctly.

