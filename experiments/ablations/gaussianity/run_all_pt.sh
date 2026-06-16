#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Find all task_9_features.pt and task_19_features.pt
FEATURE_FILES=$(find . -name "task_9_features.pt" -o -name "task_19_features.pt")

echo "Starting Yeo-Johnson Transform Experiments..."
for file in $FEATURE_FILES; do
    echo "Processing $file..."
    python "$SCRIPT_DIR/power_transform_experiment.py" --features_path "$file" --mode all
done

echo "Done! Results saved to power_transform_results.csv"
