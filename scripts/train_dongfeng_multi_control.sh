#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Dongfeng Multi-Control Post Training Script
#
# Fresh post-training from the official Transfer2.5 multiview checkpoint
# (pre-trained hdmap_bbox control head) on the Dongfeng ego-view data at
# /mnt2/dongfeng_ego_output/transfer2. See
# cosmos_transfer2/experiments/custom/dongfeng_multi_control_experiment.py

set -e

# ============================================================================
# Configuration
# ============================================================================

NUM_GPUS=${NUM_GPUS:-8}

# Output on /mnt2 (Dongfeng runs are kept separate from the /mnt output root)
export IMAGINAIRE_OUTPUT_ROOT=${IMAGINAIRE_OUTPUT_ROOT:-"/mnt2/Output_R2V_dongfeng"}

# HuggingFace configuration
export HF_HOME="/mnt/zihanw/.cache/huggingface"
export HF_HUB_CACHE="/mnt/zihanw/.cache/huggingface/hub"
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_ENABLE_HF_TRANSFER=0
export TRANSFORMERS_CACHE="/mnt/zihanw/.cache/huggingface/transformers"

MASTER_PORT=${MASTER_PORT:-12349}

EXPERIMENT=${EXPERIMENT:-"dongfeng_multi_control_post_train"}

WANDB_MODE=${WANDB_MODE:-"disabled"}

# ============================================================================
# Pre-flight
# ============================================================================

echo "=============================================="
echo "Dongfeng Multi-Control Post Training"
echo "=============================================="
echo "Base Model: official Transfer2.5 Multiview checkpoint"
echo "Control Heads: hdmap (pre-trained), blur (scratch), depth (scratch)"
echo "Data: /mnt2/dongfeng_ego_output/transfer2"
echo "=============================================="
echo "NUM_GPUS: ${NUM_GPUS}"
echo "OUTPUT_ROOT: ${IMAGINAIRE_OUTPUT_ROOT}"
echo "EXPERIMENT: ${EXPERIMENT}"
echo "WANDB_MODE: ${WANDB_MODE}"
if [ -n "${DONGFENG_TEST_SCENES}" ]; then
    echo "DONGFENG_TEST_SCENES: ${DONGFENG_TEST_SCENES}"
fi
echo "=============================================="

mkdir -p "${IMAGINAIRE_OUTPUT_ROOT}"

# ============================================================================
# Run Training
# ============================================================================

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
echo "Checkpoints: ${IMAGINAIRE_OUTPUT_ROOT}/cosmos_transfer_custom/dongfeng/"
echo "=============================================="
