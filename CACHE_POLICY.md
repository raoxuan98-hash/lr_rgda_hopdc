# Cache And Result Directory Policy

This project has large server-side caches and experiment outputs that should not
be overwritten or deleted during code sync.

## Protected Remote Directories

When syncing local code to `/home/raoxuan/projects/lr_rgda`, protect at least:

- `cached_data/`
- `logs/`
- `balanced_datasets/`
- `RGDA_*/`
- `sldc_*/`
- `sldc_logs_*/`
- `random_projector_logs_*/`
- `实验结果保存/`
- `paper_review_and_rebuttal_plans/`
- `paper_comment_and_rebuttal/`
- `.history/`
- `.git/`
- `~/`

Also protect generated result/checkpoint files:

- `*.pt`
- `*.pth`
- `*.csv`
- `nohup.out`

## Safe-To-Delete Caches

The following are reproducible and can be cleaned after validation:

- `__pycache__/`
- `*.pyc`
- `.ipynb_checkpoints/`

## Recommended Sync Command

Run from the workspace root:

```bash
rsync -avz --delete \
  --exclude='.git/' \
  --exclude='.history/' \
  --exclude='.codebuddy/' \
  --exclude='.trae/' \
  --exclude='~/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='cached_data/' \
  --exclude='logs/' \
  --exclude='balanced_datasets/' \
  --exclude='RGDA_*/' \
  --exclude='sldc_*/' \
  --exclude='sldc_logs_*/' \
  --exclude='random_projector_logs_*/' \
  --exclude='实验结果保存/' \
  --exclude='paper_review_and_rebuttal_plans/' \
  --exclude='paper_comment_and_rebuttal/' \
  --exclude='*.pth' \
  --exclude='*.pt' \
  --exclude='*.csv' \
  --exclude='nohup.out' \
  --exclude='wandb/' \
  lr_rgda/ raoxuan@10.20.34.30:/home/raoxuan/projects/lr_rgda/
```

Use `--dry-run --itemize-changes` first when changing exclude rules.

