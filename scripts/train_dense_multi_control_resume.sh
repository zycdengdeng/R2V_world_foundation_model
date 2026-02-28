#!/bin/bash
# Resume Training Script for Dense Multi-Control
# Resumes training from iter 2200 checkpoint
#
# Checkpoint: /mnt/zihanw/Output_R2V_world_foundation_model_v2/cosmos_transfer_dense/multi_control/
#             2b_dense_multi_control_20260223_171249/checkpoints/iter_000002200

set -e

# ============================================================================
# Configuration
# ============================================================================

NUM_GPUS=${NUM_GPUS:-8}

export IMAGINAIRE_OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-/mnt/zihanw/Output_R2V_world_foundation_model_v2}"

# HuggingFace configuration
export HF_HOME="/mnt/zihanw/.cache/huggingface"
export HF_HUB_CACHE="/mnt/zihanw/.cache/huggingface/hub"
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_ENABLE_HF_TRANSFER=0
export TRANSFORMERS_CACHE="/mnt/zihanw/.cache/huggingface/transformers"

MASTER_PORT=${MASTER_PORT:-12345}

# Resume experiment
EXPERIMENT="dense_multi_control_post_train_resume"

WANDB_MODE=${WANDB_MODE:-"disabled"}

# ============================================================================
# Pre-flight Checks
# ============================================================================

RESUME_CKPT="/mnt/zihanw/Output_R2V_world_foundation_model_v2/cosmos_transfer_dense/multi_control/2b_dense_multi_control_20260223_171249/checkpoints/iter_000002200"

echo "=============================================="
echo "Dense Multi-Control Resume Training"
echo "=============================================="
echo "Resuming from: iter 2200"
echo "Checkpoint: ${RESUME_CKPT}"
echo "NUM_GPUS: ${NUM_GPUS}"
echo "OUTPUT_ROOT: ${IMAGINAIRE_OUTPUT_ROOT}"
echo "EXPERIMENT: ${EXPERIMENT}"
echo "WANDB_MODE: ${WANDB_MODE}"
echo "=============================================="

# Verify checkpoint exists
if [ ! -d "${RESUME_CKPT}/model" ]; then
    echo "ERROR: Checkpoint not found at ${RESUME_CKPT}"
    echo "Expected subdirectories: model/ optim/ scheduler/ trainer/"
    exit 1
fi

echo ""
echo "Checkpoint verified:"
ls -la "${RESUME_CKPT}/"
echo ""

# ============================================================================
# Run Training
# ============================================================================

echo "Starting resume training..."
echo ""

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
echo "Resume training completed!"
echo "=============================================="
echo "New checkpoints saved to: ${IMAGINAIRE_OUTPUT_ROOT}/cosmos_transfer_dense/multi_control/"
echo ""
