#!/bin/bash

# ==============================================================================
# run_stats_parallel.sh - Parallel Feature Statistics Analysis
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Find all cached_features directories that contain .pt files
# We look into the experiment directories we processed earlier
directories=$(find RGDA_WD_2025-12-19-within -name "cached_features" -type d)

mkdir -p logs/stats_parallel

echo "🚀 Starting feature statistics analysis for all extracted experiments..."

for features_dir in $directories; do
    # Check if there are any .pt files
    if ls "$features_dir"/*_features.pt >/dev/null 2>&1; then
        # Generate a clean output name based on the parent directory structure
        # e.g., cifar100_Basic, imagenet-r_Full, etc.
        ds_name=$(echo "$features_dir" | grep -o "DS_[^/]*" | cut -d'_' -f2-)
        method_type=$(echo "$features_dir" | grep -o "r4_[^/]*" | cut -d'_' -f2)
        output_name="stats_${ds_name}_${method_type}"
        
        echo "  [Analyze] $ds_name | $method_type -> $features_dir"
        
        # Run in background
        python "$SCRIPT_DIR/feature_statistics.py" \
            --features_dir "$features_dir" \
            --output_name "$output_name" > "logs/stats_parallel/${output_name}.log" 2>&1 &
    else
        echo "  [Skip] No features found in $features_dir"
    fi
done

wait
echo "✅ All feature statistics analysis completed!"
