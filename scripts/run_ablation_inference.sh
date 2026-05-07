#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run inference for all 7 ablation experiments
#
# Usage:
#   # Run all 7 experiments inference in parallel
#   bash scripts/run_ablation_inference.sh
#
#   # Run single experiment inference
#   bash scripts/run_ablation_inference.sh ablation_hdmap_only 0

set -e

# ============================================================================
# Configuration
# ============================================================================

# Base paths
ABLATION_BASE="/mnt/zihanw/Output_R2V_world_foundation_model_v1/ablation_all"
CKPT_BASE="${ABLATION_BASE}/cosmos_ablation/single_view"
OUTPUT_BASE="${ABLATION_BASE}/inference"

# Checkpoint iteration
CKPT_ITER="iter_000001000"

# Environment
export HF_HOME="/mnt/zihanw/.cache/huggingface"
export HF_HUB_CACHE="/mnt/zihanw/.cache/huggingface/hub"
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_ENABLE_HF_TRANSFER=0
export TRANSFORMERS_CACHE="/mnt/zihanw/.cache/huggingface/transformers"

# Test scene IDs
SCENE_IDS="031 033 053 056 076 077 088 089"

# Create output directory
mkdir -p "${OUTPUT_BASE}"
mkdir -p "${OUTPUT_BASE}/logs"

# ============================================================================
# Experiment configurations
# ============================================================================

# Format: experiment_name:checkpoint_folder
EXPERIMENTS=(
    "ablation_hdmap_only:ablation_hdmap_only_20260506_191239"
    "ablation_blur_only:ablation_blur_only_20260506_191239"
    "ablation_depth_only:ablation_depth_only_20260506_191240"
    "ablation_hdmap_blur:ablation_hdmap_blur_20260506_191240"
    "ablation_hdmap_depth:ablation_hdmap_depth_20260506_191240"
    "ablation_blur_depth:ablation_blur_depth_20260506_191240"
    "ablation_all_controls:ablation_all_controls_20260506_191239"
)

# ============================================================================
# Function to run inference for a single experiment
# ============================================================================

run_inference() {
    local experiment_name=$1
    local ckpt_folder=$2
    local gpu_id=$3

    local ckpt_path="${CKPT_BASE}/${ckpt_folder}/checkpoints/${CKPT_ITER}"
    local output_dir="${OUTPUT_BASE}/${experiment_name}"
    local log_file="${OUTPUT_BASE}/logs/${experiment_name}.log"

    echo "[GPU $gpu_id] Experiment: $experiment_name"
    echo "[GPU $gpu_id] Checkpoint: $ckpt_path"
    echo "[GPU $gpu_id] Output: $output_dir"
    echo "[GPU $gpu_id] Log: $log_file"

    CUDA_VISIBLE_DEVICES=$gpu_id \
    python -m cosmos_transfer2.experiments.custom.inference_ablation_single_view \
        --experiment $experiment_name \
        --ckpt_path $ckpt_path \
        --output_dir $output_dir \
        --scene_ids $SCENE_IDS \
        > "$log_file" 2>&1 &

    echo "[GPU $gpu_id] PID: $!"
}

# ============================================================================
# Main
# ============================================================================

echo "=============================================="
echo "Ablation Inference - Single View"
echo "=============================================="
echo "Checkpoint iteration: $CKPT_ITER"
echo "Output: $OUTPUT_BASE"
echo "=============================================="

# Check if running single experiment
if [ $# -eq 2 ]; then
    experiment_name=$1
    gpu_id=$2

    # Find the checkpoint folder for this experiment
    for exp_config in "${EXPERIMENTS[@]}"; do
        IFS=':' read -r exp_name ckpt_folder <<< "$exp_config"
        if [ "$exp_name" == "$experiment_name" ]; then
            echo "Running single experiment: $experiment_name on GPU $gpu_id"
            run_inference "$exp_name" "$ckpt_folder" "$gpu_id"
            wait
            echo "Inference completed: $experiment_name"
            exit 0
        fi
    done
    echo "Error: Unknown experiment $experiment_name"
    exit 1
fi

# Run all 7 experiments in parallel
echo ""
echo "Starting 7 inference experiments in parallel..."
echo ""

gpu_id=0
for exp_config in "${EXPERIMENTS[@]}"; do
    IFS=':' read -r exp_name ckpt_folder <<< "$exp_config"
    run_inference "$exp_name" "$ckpt_folder" "$gpu_id"
    gpu_id=$((gpu_id + 1))
done

echo ""
echo "=============================================="
echo "All 7 inference experiments started!"
echo "=============================================="
echo ""
echo "Monitor logs:"
echo "  tail -f ${OUTPUT_BASE}/logs/ablation_hdmap_only.log"
echo ""
echo "Check GPU usage:"
echo "  watch -n 1 nvidia-smi"
echo ""

# Wait for all
wait

echo ""
echo "=============================================="
echo "All inference completed!"
echo "=============================================="
echo "Results saved to: ${OUTPUT_BASE}/"
