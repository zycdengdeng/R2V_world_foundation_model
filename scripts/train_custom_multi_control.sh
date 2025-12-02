#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Custom Multi-Control Post Training Script
# Option B: Train 3 control heads (vis, depth, hdmap_bbox) from Predict2.5 base
#
# This script trains ControlNet heads from scratch using Predict2.5 multiview
# as the base model. All 3 control types will be learned from random initialization.

set -e

# ============================================================================
# Configuration - MODIFY THESE FOR YOUR SETUP
# ============================================================================

# Number of GPUs to use
NUM_GPUS=${NUM_GPUS:-8}

# Output directory for checkpoints and logs
export IMAGINAIRE_OUTPUT_ROOT="/mnt/zihanw/Output_R2V_world_foundation_model_v1"

# HuggingFace configuration (for model download)
export HF_HOME="/mnt/zihanw/.cache/huggingface"
export HF_HUB_CACHE="/mnt/zihanw/.cache/huggingface/hub"
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_ENABLE_HF_TRANSFER=0
export TRANSFORMERS_CACHE="/mnt/zihanw/.cache/huggingface/transformers"

# Master port for distributed training (change if port is in use)
MASTER_PORT=${MASTER_PORT:-12345}

# Experiment name (choose one):
#   - custom_multi_control_post_train: Full training (20000 iterations)
#   - custom_multi_control_post_train_small: Quick test (500 iterations)
EXPERIMENT=${EXPERIMENT:-"custom_multi_control_post_train"}

# Weights & Biases mode: "disabled", "online", or "offline"
WANDB_MODE=${WANDB_MODE:-"disabled"}

# ============================================================================
# Pre-flight Checks
# ============================================================================

echo "=============================================="
echo "Custom Multi-Control Post Training (Option B)"
echo "=============================================="
echo "Base Model: Predict2.5 Multiview (no ControlNet)"
echo "Control Heads: vis (blur), depth, hdmap_bbox"
echo "=============================================="
echo "NUM_GPUS: ${NUM_GPUS}"
echo "OUTPUT_ROOT: ${IMAGINAIRE_OUTPUT_ROOT}"
echo "HF_HOME: ${HF_HOME}"
echo "EXPERIMENT: ${EXPERIMENT}"
echo "WANDB_MODE: ${WANDB_MODE}"
echo "=============================================="

# Create output directory if it doesn't exist
mkdir -p "${IMAGINAIRE_OUTPUT_ROOT}"

# Check HuggingFace cache
echo ""
echo "Checking HuggingFace cache..."
if [ -d "${HF_HOME}/hub/models--nvidia--Cosmos-Predict2.5-2B" ]; then
    echo "  [OK] Cosmos-Predict2.5-2B found in cache"
else
    echo "  [INFO] Cosmos-Predict2.5-2B will be downloaded during training"
fi

# ============================================================================
# Training Data Information
# ============================================================================

echo ""
echo "Training Data Configuration:"
echo "  - Blur dataset: /mnt/zihanw/proj_utils_pro/transfer_video_maker/output/BlurProjection"
echo "  - Depth dataset: /mnt/zihanw/proj_utils_pro/transfer_video_maker/output/DepthSparse"
echo "  - HDMap dataset: /mnt/zihanw/proj_utils_pro/transfer_video_maker/output/HDMapBbox"
echo ""
echo "Data Split:"
echo "  - Training scenes (16): 017, 019, 020, 022, 045, 049, 051, 055, 057, 059, 065, 067, 069, 073, 075, 077"
echo "  - Test scenes (2): 047, 061"
echo ""

# ============================================================================
# Run Training
# ============================================================================

echo "Starting training..."
echo "  hint_keys: vis_depth_hdmap (3 control heads)"
echo "  max_iter: 20000 (full) / 500 (small)"
echo "  save_iter: 500 (full) / 100 (small)"
echo ""

# Set WORLD_SIZE for context parallel configuration
export WORLD_SIZE=${NUM_GPUS}

torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=${MASTER_PORT} \
    -m scripts.train \
    --config=cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/config.py \
    -- \
    experiment=${EXPERIMENT} \
    job.wandb_mode=${WANDB_MODE}

echo ""
echo "=============================================="
echo "Training completed!"
echo "=============================================="
echo "Checkpoints saved to: ${IMAGINAIRE_OUTPUT_ROOT}/cosmos_transfer_custom/multi_control/"
echo ""
echo "To convert checkpoint for inference, run:"
echo "  CHECKPOINTS_DIR=\${IMAGINAIRE_OUTPUT_ROOT}/cosmos_transfer_custom/multi_control/2b_custom_multi_control_post_train/checkpoints"
echo "  CHECKPOINT_ITER=\$(cat \$CHECKPOINTS_DIR/latest_checkpoint.txt)"
echo "  python scripts/convert_distcp_to_pt.py \$CHECKPOINTS_DIR/\$CHECKPOINT_ITER/model \$CHECKPOINTS_DIR/\$CHECKPOINT_ITER"
echo ""
