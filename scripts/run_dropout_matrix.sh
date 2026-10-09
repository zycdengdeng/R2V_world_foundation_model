#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Modality-dropout inference matrix on the test scenes.
# Runs 4 settings sequentially: no_hdmap, no_blur, no_depth, no_control.
# The full-condition baseline is the existing run at
# $IMAGINAIRE_OUTPUT_ROOT/inference (same seed/guidance/steps/checkpoint).

set -e
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export WORLD_SIZE=8
export IMAGINAIRE_OUTPUT_ROOT="/mnt/zihanw/Output_R2V_world_foundation_model_v1"
export HF_HOME="/mnt/zihanw/.cache/huggingface"
export HF_HUB_CACHE="/mnt/zihanw/.cache/huggingface/hub"
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_ENABLE_HF_TRANSFER=0
export TRANSFORMERS_CACHE="/mnt/zihanw/.cache/huggingface/transformers"

CKPT=/mnt/zihanw/Output_R2V_world_foundation_model_v1/cosmos_transfer_custom/multi_control/2b_custom_multi_control_20260128_142857/checkpoints/iter_000006600
OUT=/mnt/zihanw/Output_R2V_world_foundation_model_v1/inference_dropout
SCENES="033 053 056 076 077 088 089"

run_setting () {
  local name=$1; shift
  echo ""
  echo "============================================================"
  echo "Setting: $name (dropping: $*)"
  echo "============================================================"
  torchrun --nproc_per_node=8 --master_port=12345 \
    -m cosmos_transfer2.experiments.custom.inference_control_dropout \
    --ckpt_path "$CKPT" \
    --context_parallel_size 8 --num_views 7 \
    --scene_ids $SCENES \
    --output_dir "$OUT/$name" \
    --drop_controls "$@"
}

run_setting no_hdmap   hdmap
run_setting no_blur    blur
run_setting no_depth   depth
run_setting no_control hdmap blur depth

echo ""
echo "============================================================"
echo "All dropout settings completed. Results under: $OUT"
echo "============================================================"
