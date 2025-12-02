#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Custom Multi-Control Post Training Script
# Trains a model with 3 control types: blur, depth, hdmap_bbox

set -e

# ============================================================================
# Configuration - MODIFY THESE FOR YOUR SETUP
# ============================================================================

# Number of GPUs to use
NUM_GPUS=${NUM_GPUS:-8}

# Output directory for checkpoints and logs
export IMAGINAIRE_OUTPUT_ROOT="/mnt/zihanw/Output_R2V_world_foundation_model_v1"

# Master port for distributed training (change if port is in use)
MASTER_PORT=${MASTER_PORT:-12345}

# Experiment name (choose one):
#   - custom_multi_control_post_train: Full training (5000 iterations)
#   - custom_multi_control_post_train_small: Quick test (500 iterations)
EXPERIMENT=${EXPERIMENT:-"custom_multi_control_post_train"}

# Weights & Biases mode: "disabled", "online", or "offline"
WANDB_MODE=${WANDB_MODE:-"disabled"}

# ============================================================================
# Pre-flight Checks
# ============================================================================

echo "=============================================="
echo "Custom Multi-Control Post Training"
echo "=============================================="
echo "NUM_GPUS: ${NUM_GPUS}"
echo "OUTPUT_ROOT: ${IMAGINAIRE_OUTPUT_ROOT}"
echo "EXPERIMENT: ${EXPERIMENT}"
echo "WANDB_MODE: ${WANDB_MODE}"
echo "=============================================="

# Create output directory if it doesn't exist
mkdir -p "${IMAGINAIRE_OUTPUT_ROOT}"

# Check if checkpoint exists
CHECKPOINT_DIR="${IMAGINAIRE_OUTPUT_ROOT}/checkpoints"
if [ ! -d "${CHECKPOINT_DIR}" ]; then
    echo "WARNING: Checkpoint directory does not exist: ${CHECKPOINT_DIR}"
    echo "Please download the pre-trained checkpoint first."
    echo ""
    echo "Expected checkpoint file:"
    echo "  ${CHECKPOINT_DIR}/Cosmos-1.0-Transfer-7B-AV-MultiView_consolidated.pt"
    echo ""
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
echo "Checkpoints saved to: ${IMAGINAIRE_OUTPUT_ROOT}"
echo ""
echo "To convert checkpoint for inference, run:"
echo "  python scripts/convert_distcp_to_pt.py <checkpoint_dir>/model <output_dir>"
echo ""
