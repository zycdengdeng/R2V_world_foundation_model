#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run all 7 ablation experiments in parallel on 8 GPUs (using 7)
# Each experiment trains single-view (Front Wide) with different control combinations
#
# Usage:
#   # Run all 7 experiments in parallel
#   bash scripts/run_ablation_experiments.sh
#
#   # Run single experiment on specific GPU
#   bash scripts/run_ablation_experiments.sh ablation_hdmap_only 0

set -e

# ============================================================================
# Configuration
# ============================================================================

# Output directory (inside main output folder)
export IMAGINAIRE_OUTPUT_ROOT="/mnt/zihanw/Output_R2V_world_foundation_model_v1/ablation"

# HuggingFace configuration
export HF_HOME="/mnt/zihanw/.cache/huggingface"
export HF_HUB_CACHE="/mnt/zihanw/.cache/huggingface/hub"
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_ENABLE_HF_TRANSFER=0
export TRANSFORMERS_CACHE="/mnt/zihanw/.cache/huggingface/transformers"

# WandB mode: "disabled", "online", or "offline"
WANDB_MODE=${WANDB_MODE:-"online"}

# Config file path
CONFIG_FILE="cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/config.py"

# Create output directory
mkdir -p $IMAGINAIRE_OUTPUT_ROOT
mkdir -p "${IMAGINAIRE_OUTPUT_ROOT}/logs"

# ============================================================================
# Function to run a single experiment
# ============================================================================

run_experiment() {
    local experiment_name=$1
    local gpu_id=$2
    local master_port=$((12345 + gpu_id))
    local log_file="${IMAGINAIRE_OUTPUT_ROOT}/logs/${experiment_name}.log"

    echo "[GPU $gpu_id] Starting experiment: $experiment_name"
    echo "[GPU $gpu_id] Master port: $master_port"
    echo "[GPU $gpu_id] Log file: $log_file"

    # Single GPU training with torchrun (same as original script)
    CUDA_VISIBLE_DEVICES=$gpu_id \
    WORLD_SIZE=1 \
    WANDB_MODE=$WANDB_MODE \
    torchrun \
        --nproc_per_node=1 \
        --master_port=${master_port} \
        -m scripts.train \
        --config=$CONFIG_FILE \
        -- \
        experiment=${experiment_name} \
        job.wandb_mode=${WANDB_MODE} \
        > "$log_file" 2>&1 &

    echo "[GPU $gpu_id] PID: $!"
}

# ============================================================================
# Main
# ============================================================================

echo "=============================================="
echo "Ablation Experiments - Single View Training"
echo "=============================================="
echo "Output: $IMAGINAIRE_OUTPUT_ROOT"
echo "WandB Mode: $WANDB_MODE"
echo "=============================================="

# Check if running single experiment or all
if [ $# -eq 2 ]; then
    # Run single experiment
    echo "Running single experiment: $1 on GPU $2"
    run_experiment $1 $2
    wait
    echo "Experiment completed: $1"
    exit 0
fi

# Run all 7 experiments in parallel
echo ""
echo "Starting 7 experiments in parallel..."
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
echo "=============================================="
echo "All 7 experiments started!"
echo "=============================================="
echo ""
echo "Experiments:"
echo "  [GPU 0] ablation_hdmap_only"
echo "  [GPU 1] ablation_blur_only"
echo "  [GPU 2] ablation_depth_only"
echo "  [GPU 3] ablation_hdmap_blur"
echo "  [GPU 4] ablation_hdmap_depth"
echo "  [GPU 5] ablation_blur_depth"
echo "  [GPU 6] ablation_all_controls"
echo ""
echo "Monitor logs:"
echo "  tail -f ${IMAGINAIRE_OUTPUT_ROOT}/logs/ablation_hdmap_only.log"
echo ""
echo "Check GPU usage:"
echo "  watch -n 1 nvidia-smi"
echo ""

# Wait for all background processes
wait

echo ""
echo "=============================================="
echo "All experiments completed!"
echo "=============================================="
echo ""
echo "Results saved to: ${IMAGINAIRE_OUTPUT_ROOT}/cosmos_ablation/single_view/"
echo "(i.e., /mnt/zihanw/Output_R2V_world_foundation_model_v1/ablation/cosmos_ablation/single_view/)"
