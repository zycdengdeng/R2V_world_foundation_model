#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Single-view HDMap fine-tuning of Cosmos-Transfer2.5 on car-side data.
# Single camera (front_wide_120fov), single control (hdmap_bbox), 0 conditional frames.

set -e

# ============================================================================
# Configuration - MODIFY THESE FOR YOUR SETUP
# ============================================================================

# Number of GPUs (FSDP only; no context parallelism for single view)
NUM_GPUS=${NUM_GPUS:-4}

# Output directory for checkpoints and logs
export IMAGINAIRE_OUTPUT_ROOT="/mnt/zihanw/Output_R2V_world_foundation_model_v1"

# HuggingFace cache config
export HF_HOME="/mnt/zihanw/.cache/huggingface"
export HF_HUB_CACHE="/mnt/zihanw/.cache/huggingface/hub"
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_ENABLE_HF_TRANSFER=0
export TRANSFORMERS_CACHE="/mnt/zihanw/.cache/huggingface/transformers"

MASTER_PORT=${MASTER_PORT:-12346}

# Experiment name (choose one):
#   - single_view_hdmap_post_train       : Full training (5000 iterations)
#   - single_view_hdmap_post_train_smoke : Quick sanity check (50 iterations)
EXPERIMENT=${EXPERIMENT:-"single_view_hdmap_post_train"}

WANDB_MODE=${WANDB_MODE:-"disabled"}

# ============================================================================
# Pre-flight info
# ============================================================================

echo "=============================================="
echo "Single-view HDMap Fine-tuning (Cosmos-Transfer2.5)"
echo "=============================================="
echo "View:         camera_front_wide_120fov"
echo "Control:      hdmap_bbox (continues from pre-trained head)"
echo "NUM_GPUS:     ${NUM_GPUS}  (FSDP shard size = ${NUM_GPUS}, context_parallel = 1)"
echo "OUTPUT_ROOT:  ${IMAGINAIRE_OUTPUT_ROOT}"
echo "HF_HOME:      ${HF_HOME}"
echo "EXPERIMENT:   ${EXPERIMENT}"
echo "WANDB_MODE:   ${WANDB_MODE}"
echo "=============================================="

mkdir -p "${IMAGINAIRE_OUTPUT_ROOT}"

# ============================================================================
# Run training
# ============================================================================

# WORLD_SIZE drives fsdp_shard_size in the experiment config
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
echo "Checkpoints saved to: ${IMAGINAIRE_OUTPUT_ROOT}/cosmos_transfer_custom/single_view_hdmap/"
