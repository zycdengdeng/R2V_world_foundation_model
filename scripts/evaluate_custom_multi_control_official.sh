#!/bin/bash
# Evaluation script using official framework
# Based on cosmos_transfer2/_src/predict2_multiview/scripts/inference.py

set -e

# Default values
CKPT_PATH="${CKPT_PATH:-/mnt/zihanw/Output_R2V_world_foundation_model_v1/cosmos_transfer_custom/multi_control/2b_custom_multi_control_20251226_155343/checkpoints/iter_000005000}"
SAVE_ROOT="${SAVE_ROOT:-/mnt/zihanw/Output_R2V_world_foundation_model_v1/cosmos_transfer_custom/multi_control/eval_iter5000_official}"
EXPERIMENT="${EXPERIMENT:-custom_multi_control_post_train}"
MAX_SAMPLES="${MAX_SAMPLES:-100}"
GUIDANCE="${GUIDANCE:-7.0}"
NUM_STEPS="${NUM_STEPS:-35}"
CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-1}"
MASTER_PORT="${MASTER_PORT:-12345}"

# Number of GPUs (from CUDA_VISIBLE_DEVICES or default to 1)
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    NGPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
else
    NGPUS=1
fi

echo "============================================"
echo "Evaluation Configuration:"
echo "  Checkpoint: ${CKPT_PATH}"
echo "  Output: ${SAVE_ROOT}"
echo "  Experiment: ${EXPERIMENT}"
echo "  Max Samples: ${MAX_SAMPLES}"
echo "  Guidance: ${GUIDANCE}"
echo "  Num Steps: ${NUM_STEPS}"
echo "  GPUs: ${NGPUS}"
echo "  Context Parallel Size: ${CONTEXT_PARALLEL_SIZE}"
echo "============================================"

# Run evaluation
PYTHONPATH=. torchrun \
    --nproc_per_node=${NGPUS} \
    --master_port=${MASTER_PORT} \
    -m cosmos_transfer2.experiments.custom.evaluate_official \
    --experiment ${EXPERIMENT} \
    --ckpt_path ${CKPT_PATH} \
    --context_parallel_size ${CONTEXT_PARALLEL_SIZE} \
    --save_root ${SAVE_ROOT} \
    --max_samples ${MAX_SAMPLES} \
    --guidance ${GUIDANCE} \
    --num_steps ${NUM_STEPS} \
    --save_videos \
    --num_videos_to_save 10 \
    --fps 10

echo "============================================"
echo "Evaluation complete!"
echo "Results saved to: ${SAVE_ROOT}"
echo "============================================"
