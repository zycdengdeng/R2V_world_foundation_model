#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run all 7 ablation experiments in parallel on 8 GPUs (using 7)
# Each experiment trains single-view (Front Wide) with different control combinations
#
# Usage:
#   bash scripts/run_ablation_experiments.sh
#
# Or run individual experiment:
#   bash scripts/run_ablation_experiments.sh hdmap_only 0

set -e

# Environment setup
export IMAGINAIRE_OUTPUT_ROOT="/mnt/zihanw/Output_R2V_ablation"
export HF_HOME="/mnt/zihanw/.cache/huggingface"
export HF_HUB_CACHE="/mnt/zihanw/.cache/huggingface/hub"
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_ENABLE_HF_TRANSFER=0
export TRANSFORMERS_CACHE="/mnt/zihanw/.cache/huggingface/transformers"

# Create output directory
mkdir -p $IMAGINAIRE_OUTPUT_ROOT

# Config file path
CONFIG_FILE="cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/config.py"

# Function to run a single experiment
run_experiment() {
    local experiment_name=$1
    local gpu_id=$2
    local log_file="${IMAGINAIRE_OUTPUT_ROOT}/logs/${experiment_name}.log"

    mkdir -p "${IMAGINAIRE_OUTPUT_ROOT}/logs"

    echo "[GPU $gpu_id] Starting experiment: $experiment_name"
    echo "[GPU $gpu_id] Log file: $log_file"

    CUDA_VISIBLE_DEVICES=$gpu_id \
    python -m cosmos_transfer2 \
        --config $CONFIG_FILE \
        experiment=$experiment_name \
        > "$log_file" 2>&1 &

    echo "[GPU $gpu_id] PID: $!"
}

# Check if running single experiment or all
if [ $# -eq 2 ]; then
    # Run single experiment
    run_experiment $1 $2
    wait
    echo "Single experiment completed: $1"
    exit 0
fi

# Run all 7 experiments in parallel
echo "=========================================="
echo "Starting 7 Ablation Experiments"
echo "=========================================="
echo "Output: $IMAGINAIRE_OUTPUT_ROOT"
echo ""

# 1. HDMap only (GPU 0)
run_experiment "ablation_hdmap_only" 0

# 2. Blur only (GPU 1)
run_experiment "ablation_blur_only" 1

# 3. Depth only (GPU 2)
run_experiment "ablation_depth_only" 2

# 4. HDMap + Blur (GPU 3)
run_experiment "ablation_hdmap_blur" 3

# 5. HDMap + Depth (GPU 4)
run_experiment "ablation_hdmap_depth" 4

# 6. Blur + Depth (GPU 5)
run_experiment "ablation_blur_depth" 5

# 7. All controls (GPU 6)
run_experiment "ablation_all_controls" 6

echo ""
echo "=========================================="
echo "All 7 experiments started!"
echo "=========================================="
echo ""
echo "Monitor progress:"
echo "  tail -f ${IMAGINAIRE_OUTPUT_ROOT}/logs/ablation_hdmap_only.log"
echo "  tail -f ${IMAGINAIRE_OUTPUT_ROOT}/logs/ablation_blur_only.log"
echo "  ... etc"
echo ""
echo "Check GPU usage:"
echo "  watch -n 1 nvidia-smi"
echo ""

# Wait for all background processes
wait

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
