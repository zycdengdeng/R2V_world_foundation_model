#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Multi-control training script with HF mirror support
# This script ensures the HF_ENDPOINT is set before Python imports huggingface_hub

# Set HuggingFace mirror for faster downloads in China
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_URL="${HF_ENDPOINT}"
export HUGGINGFACE_HUB_ENDPOINT="${HF_ENDPOINT}"

# Set cache directory
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"

# Optional: disable symlinks for network filesystems
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

echo "=================================="
echo "HuggingFace Configuration:"
echo "  HF_ENDPOINT: ${HF_ENDPOINT}"
echo "  HF_HOME: ${HF_HOME}"
echo "  HF_HUB_CACHE: ${HF_HUB_CACHE}"
echo "=================================="

# Get number of GPUs
# Default to 2 GPUs for 2-view training
# Constraints: state_t=6 AND num_heads=16 → valid cp_size = {1, 2}
NGPUS="${NGPUS:-2}"
MASTER_PORT="${MASTER_PORT:-12341}"

echo "Running with ${NGPUS} GPUs on port ${MASTER_PORT}"
echo "=================================="

# Run the training
exec torchrun \
    --nproc_per_node=${NGPUS} \
    --master_port=${MASTER_PORT} \
    -m scripts.train \
    --config=cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/config.py \
    -- experiment=zihanw_multicontrol_post_train "$@"
