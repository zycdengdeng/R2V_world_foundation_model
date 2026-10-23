#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Small-scale Tianjin (roadside) post-training: 10/75 train scenes, 400 iters,
# fresh start from the official Transfer2.5 multiview checkpoint; test scenes
# 053+077. After training it automatically picks the checkpoint with the
# lowest test loss (parsed from the training log) and runs inference on the
# test scenes. Disable the auto-inference stage with AUTO_INFER=0.
# See cosmos_transfer2/experiments/custom/tianjin_small_experiment.py and
# docs/tianjin_small_run.md

set -e
set -o pipefail

NUM_GPUS=${NUM_GPUS:-8}

export IMAGINAIRE_OUTPUT_ROOT=${IMAGINAIRE_OUTPUT_ROOT:-"/mnt/zihanw/Output_R2V_world_foundation_model_v1"}

export HF_HOME="/mnt/zihanw/.cache/huggingface"
export HF_HUB_CACHE="/mnt/zihanw/.cache/huggingface/hub"
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_ENABLE_HF_TRANSFER=0
export TRANSFORMERS_CACHE="/mnt/zihanw/.cache/huggingface/transformers"

# Mitigate CUDA memory fragmentation: allocation patterns spike during DCP
# checkpoint saves (allgather of sharded states) and a fragmented allocator
# can fail there even though steady-state training fits comfortably.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MASTER_PORT=${MASTER_PORT:-12351}
EXPERIMENT=${EXPERIMENT:-"tianjin_multi_control_small"}
WANDB_MODE=${WANDB_MODE:-"disabled"}
AUTO_INFER=${AUTO_INFER:-1}
INFER_SCENES=${INFER_SCENES:-"053 077"}

GROUP_DIR="${IMAGINAIRE_OUTPUT_ROOT}/cosmos_transfer_custom/tianjin_small"
TRAIN_LOG="${IMAGINAIRE_OUTPUT_ROOT}/tianjin_small_train_$(date +%Y%m%d_%H%M%S).log"

echo "=============================================="
echo "Tianjin Small Multi-Control Post Training"
echo "=============================================="
echo "NUM_GPUS: ${NUM_GPUS}"
echo "OUTPUT_ROOT: ${IMAGINAIRE_OUTPUT_ROOT}"
echo "EXPERIMENT: ${EXPERIMENT}"
echo "WANDB_MODE: ${WANDB_MODE}"
echo "AUTO_INFER: ${AUTO_INFER} (scenes: ${INFER_SCENES})"
echo "TRAIN_LOG: ${TRAIN_LOG}"
if [ -n "${TIANJIN_TRAIN_SCENES}" ]; then
    echo "TIANJIN_TRAIN_SCENES: ${TIANJIN_TRAIN_SCENES}"
fi
echo "=============================================="

mkdir -p "${IMAGINAIRE_OUTPUT_ROOT}"

export WORLD_SIZE=${NUM_GPUS}

# ============================================================================
# Stage 1: training (output tee'd to TRAIN_LOG for checkpoint selection)
# ============================================================================

torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=${MASTER_PORT} \
    -m scripts.train \
    --config=cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/config.py \
    -- \
    experiment=${EXPERIMENT} \
    job.wandb_mode=${WANDB_MODE} \
    2>&1 | tee "${TRAIN_LOG}"

echo ""
echo "Training completed!"

# ============================================================================
# Stage 2: pick the checkpoint with the lowest test loss
# ============================================================================

RUN_DIR=$(ls -td "${GROUP_DIR}"/2b_tianjin_small_*/ 2>/dev/null | head -1)
RUN_DIR=${RUN_DIR%/}
if [ -z "${RUN_DIR}" ]; then
    echo "ERROR: no run directory found under ${GROUP_DIR}"
    exit 1
fi
echo "Run directory: ${RUN_DIR}"

# Lines look like: [EveryNTestLoss] Iteration 100: test_loss = 0.046736
BEST_ITER=$(grep -oE 'Iteration [0-9]+: test_loss = [0-9.eE+-]+' "${TRAIN_LOG}" \
    | awk '{iter=$2; sub(":","",iter); val=$5+0; if (best=="" || val<bestval) {best=iter; bestval=val}}
           END {if (best!="") print best}')

if [ -n "${BEST_ITER}" ]; then
    CKPT_NAME=$(printf 'iter_%09d' "${BEST_ITER}")
    echo "Lowest test loss at iteration ${BEST_ITER} -> ${CKPT_NAME}"
else
    CKPT_NAME=""
fi

if [ -z "${CKPT_NAME}" ] || [ ! -d "${RUN_DIR}/checkpoints/${CKPT_NAME}" ]; then
    echo "Falling back to latest checkpoint (no usable test-loss record)"
    CKPT_NAME=$(cat "${RUN_DIR}/checkpoints/latest_checkpoint.txt")
fi

CKPT_PATH="${RUN_DIR}/checkpoints/${CKPT_NAME}"
echo "Selected checkpoint: ${CKPT_PATH}"

if [ "${AUTO_INFER}" != "1" ]; then
    echo "AUTO_INFER=0 -> skipping inference. Run it manually with the checkpoint above."
    exit 0
fi

# ============================================================================
# Stage 3: inference on the test scenes with the selected checkpoint
# ============================================================================

INFER_OUT="${RUN_DIR}/inference_${CKPT_NAME}"
echo "Inference output: ${INFER_OUT}"

torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=$((MASTER_PORT + 1)) \
    -m cosmos_transfer2.experiments.custom.inference \
    --experiment ${EXPERIMENT} \
    --ckpt_path "${CKPT_PATH}" \
    --output_dir "${INFER_OUT}" \
    --context_parallel_size ${NUM_GPUS} \
    --num_views 7 \
    --scene_ids ${INFER_SCENES}

echo ""
echo "=============================================="
echo "All done."
echo "  Best checkpoint : ${CKPT_PATH}"
echo "  Generated videos: ${INFER_OUT}"
echo "=============================================="
