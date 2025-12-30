#!/bin/bash
# Evaluation script for Custom Multi-Control Model
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0,6 bash scripts/evaluate_custom_multi_control.sh
#
# Or with custom checkpoint:
#   CUDA_VISIBLE_DEVICES=0,6 CHECKPOINT_PATH=/path/to/checkpoint bash scripts/evaluate_custom_multi_control.sh

set -e

# Default checkpoint path
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/mnt/zihanw/Output_R2V_world_foundation_model_v1/cosmos_transfer_custom/multi_control/2b_custom_multi_control_20251226_155343/checkpoints/iter_000005000}"

# Output directory
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/zihanw/Output_R2V_world_foundation_model_v1/cosmos_transfer_custom/multi_control/eval_iter5000}"

# Number of samples (empty = all)
NUM_SAMPLES="${NUM_SAMPLES:-}"

# Guidance scale
GUIDANCE="${GUIDANCE:-7.0}"

# Number of sampling steps
NUM_STEPS="${NUM_STEPS:-35}"

# Seed
SEED="${SEED:-42}"

echo "=============================================="
echo "Evaluation Configuration"
echo "=============================================="
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Output Dir: ${OUTPUT_DIR}"
echo "Num Samples: ${NUM_SAMPLES:-all}"
echo "Guidance: ${GUIDANCE}"
echo "Num Steps: ${NUM_STEPS}"
echo "Seed: ${SEED}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
echo "=============================================="

# Build command
CMD="python -m cosmos_transfer2.experiments.custom.evaluate_metrics \
    --checkpoint_path ${CHECKPOINT_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --guidance ${GUIDANCE} \
    --num_steps ${NUM_STEPS} \
    --seed ${SEED}"

if [ -n "${NUM_SAMPLES}" ]; then
    CMD="${CMD} --num_samples ${NUM_SAMPLES}"
fi

# Run evaluation
echo "Running evaluation..."
${CMD}

echo "=============================================="
echo "Evaluation complete!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "=============================================="
