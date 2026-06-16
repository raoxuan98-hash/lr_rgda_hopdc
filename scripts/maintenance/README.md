# Maintenance Scripts

Utilities that support data preparation, feature-cache extraction, and sanity
checks. These scripts are not paper-facing experiment entry points.

- `dataset_resplitter.py`: rebuild balanced datasets.
- `test_cross_domain_data.py`: sanity-check cross-domain data managers.
- `feature_cache/`: scripts for extracting cached features from trained runs.

Run from the `lr_rgda` root so imports such as `utils.*` and `models.*` resolve.

