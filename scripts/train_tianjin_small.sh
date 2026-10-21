#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Small-scale Tianjin (roadside) post-training: 20/75 train scenes, 3000 iters,
# fresh start from the official Transfer2.5 multiview checkpoint. Test scenes
# are identical to the full run for comparable metrics. See
# cosmos_transfer2/experiments/custom/tianjin_small_experiment.py

set -e

NUM_GPUS=${NUM_GPUS:-8}

export IMAGINAIRE_OUTPUT_ROOT=${IMAGINAIRE_OUTPUT_ROOT:-"/mnt/zihanw/Output_R2V_world_foundation_model_v1"}

export HF_HOME="/mnt/zihanw/.cache/huggingface"
export HF_HUB_CACHE="/mnt/zihanw/.cache/huggingface/hub"
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_ENABLE_HF_TRANSFER=0
export TRANSFORMERS_CACHE="/mnt/zihanw/.cache/huggingface/transformers"

MASTER_PORT=${MASTER_PORT:-12351}
EXPERIMENT=${EXPERIMENT:-"tianjin_multi_control_small"}
WANDB_MODE=${WANDB_MODE:-"disabled"}

echo "=============================================="
echo "Tianjin Small Multi-Control Post Training"
echo "=============================================="
echo "NUM_GPUS: ${NUM_GPUS}"
echo "OUTPUT_ROOT: ${IMAGINAIRE_OUTPUT_ROOT}"
echo "EXPERIMENT: ${EXPERIMENT}"
echo "WANDB_MODE: ${WANDB_MODE}"
if [ -n "${TIANJIN_TRAIN_SCENES}" ]; then
    echo "TIANJIN_TRAIN_SCENES: ${TIANJIN_TRAIN_SCENES}"
fi
echo "=============================================="

mkdir -p "${IMAGINAIRE_OUTPUT_ROOT}"

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
echo "Training completed!"
echo "Checkpoints: ${IMAGINAIRE_OUTPUT_ROOT}/cosmos_transfer_custom/tianjin_small/"
