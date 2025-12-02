#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Custom Multi-Control Post Training Script
# Trains 3 control heads (blur, depth, hdmap) using Transfer2.5 base
#
# Usage:
#   # First time setup + training:
#   CUDA_VISIBLE_DEVICES=2,3,4,5 NUM_GPUS=4 WANDB_MODE=online bash scripts/train_custom_multi_control.sh
#
#   # Skip login (if already logged in):
#   CUDA_VISIBLE_DEVICES=2,3,4,5 NUM_GPUS=4 WANDB_MODE=online SKIP_LOGIN=1 bash scripts/train_custom_multi_control.sh

set -e

# ============================================================================
# Configuration - MODIFY THESE FOR YOUR SETUP
# ============================================================================

# Project root directory
PROJECT_ROOT="/mnt/zihanw/R2V_world_foundation_model_v1"

# Number of GPUs to use
NUM_GPUS=${NUM_GPUS:-8}

# Output directory for checkpoints and logs
export IMAGINAIRE_OUTPUT_ROOT="/mnt/zihanw/Output_R2V_world_foundation_model_v1"

# HuggingFace configuration
export HF_HOME="/mnt/zihanw/.cache/huggingface"
export HF_HUB_CACHE="/mnt/zihanw/.cache/huggingface/hub"
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_ENABLE_HF_TRANSFER=0
export TRANSFORMERS_CACHE="/mnt/zihanw/.cache/huggingface/transformers"

# HuggingFace token (set via environment variable or huggingface-cli login)
# To set: export HF_TOKEN=your_token_here
HF_TOKEN=${HF_TOKEN:-""}

# Master port for distributed training (change if port is in use)
MASTER_PORT=${MASTER_PORT:-12345}

# Experiment name (choose one):
#   - custom_multi_control_post_train: Full training (20000 iterations)
#   - custom_multi_control_post_train_small: Quick test (500 iterations)
EXPERIMENT=${EXPERIMENT:-"custom_multi_control_post_train"}

# Weights & Biases mode: "disabled", "online", or "offline"
WANDB_MODE=${WANDB_MODE:-"online"}

# Skip login steps (set to 1 if already logged in)
SKIP_LOGIN=${SKIP_LOGIN:-0}

# ============================================================================
# Environment Setup
# ============================================================================

echo "=============================================="
echo "Custom Multi-Control Post Training"
echo "=============================================="
echo "Training 4 views: front_wide, cross_left, cross_right, rear_right"
echo "Control Heads: blur (scratch), depth (scratch), hdmap (pre-trained)"
echo "=============================================="

# Change to project directory
cd "${PROJECT_ROOT}"

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
else
    echo "Warning: Virtual environment not found at ${PROJECT_ROOT}/.venv"
fi

# Create cache directories
mkdir -p "${HF_HOME}"
mkdir -p "${IMAGINAIRE_OUTPUT_ROOT}"

# ============================================================================
# Authentication (HuggingFace & WandB)
# ============================================================================

if [ "${SKIP_LOGIN}" != "1" ]; then
    echo ""
    echo "Setting up authentication..."

    # HuggingFace login
    if [ -n "${HF_TOKEN}" ]; then
        echo "  - Logging into HuggingFace with token..."
        huggingface-cli login --token "${HF_TOKEN}" 2>/dev/null || true
    else
        echo "  - HF_TOKEN not set, assuming already logged in via 'huggingface-cli login'"
    fi

    # WandB login (interactive if not already logged in)
    if [ "${WANDB_MODE}" == "online" ]; then
        echo "  - Checking WandB login status..."
        if ! wandb status 2>/dev/null | grep -q "Logged in"; then
            echo "  - Please login to WandB:"
            wandb login
        else
            echo "  - WandB already logged in"
        fi
    fi
else
    echo ""
    echo "Skipping login (SKIP_LOGIN=1)"
fi

# ============================================================================
# Pre-flight Checks
# ============================================================================

echo ""
echo "=============================================="
echo "Configuration"
echo "=============================================="
echo "NUM_GPUS: ${NUM_GPUS}"
echo "OUTPUT_ROOT: ${IMAGINAIRE_OUTPUT_ROOT}"
echo "HF_HOME: ${HF_HOME}"
echo "EXPERIMENT: ${EXPERIMENT}"
echo "WANDB_MODE: ${WANDB_MODE}"
echo "=============================================="

# Check HuggingFace cache
echo ""
echo "Checking HuggingFace cache..."
if [ -d "${HF_HOME}/hub/models--nvidia--Cosmos-Transfer2.5-2B" ]; then
    echo "  [OK] Cosmos-Transfer2.5-2B found in cache"
else
    echo "  [INFO] Cosmos-Transfer2.5-2B will be downloaded during training"
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
echo "Camera Views (4 fixed):"
echo "  - camera_front_wide_120fov"
echo "  - camera_cross_left_120fov"
echo "  - camera_cross_right_120fov"
echo "  - camera_rear_right_70fov"
echo ""
echo "Data Split:"
echo "  - Training scenes (16): 017, 019, 020, 022, 045, 049, 051, 055, 057, 059, 065, 067, 069, 073, 075, 077"
echo "  - Test scenes (2): 047, 061"
echo ""

# ============================================================================
# Run Training
# ============================================================================

echo "Starting training..."
echo "  hint_keys: blur_depth_hdmap (blur/depth from scratch, hdmap pre-trained)"
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
