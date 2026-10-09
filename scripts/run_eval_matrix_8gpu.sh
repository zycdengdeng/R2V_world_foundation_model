#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# 8-GPU evaluation of the dropout inference matrix: one shard per GPU, then merge.

set -e
cd "$(dirname "$0")/.."

GT_ROOT=/mnt/zihanw/proj_utils_pro/transfer_video_maker/output_full_data/BlurProjection/videos
OUT_ROOT=/mnt/zihanw/Output_R2V_world_foundation_model_v1
RESULTS="full=$OUT_ROOT/inference \
no_hdmap=$OUT_ROOT/inference_dropout/no_hdmap \
no_blur=$OUT_ROOT/inference_dropout/no_blur \
no_depth=$OUT_ROOT/inference_dropout/no_depth \
no_control=$OUT_ROOT/inference_dropout/no_control"
OUTPUT_CSV=$OUT_ROOT/inference_dropout/metrics.csv
N=8

echo "Launching $N evaluation shards..."
pids=()
for i in $(seq 0 $((N - 1))); do
  CUDA_VISIBLE_DEVICES=$i python3 -m cosmos_transfer2.experiments.custom.evaluate_dropout_matrix \
    --gt_videos_root "$GT_ROOT" \
    --results $RESULTS \
    --output_csv "$OUTPUT_CSV" \
    --num_shards $N --shard_idx $i \
    > /tmp/eval_shard_$i.log 2>&1 &
  pids+=($!)
done

fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=1
done
if [ $fail -ne 0 ]; then
  echo "At least one shard failed; check /tmp/eval_shard_*.log"
  exit 1
fi

echo "All shards done. Merging..."
CUDA_VISIBLE_DEVICES=0 python3 -m cosmos_transfer2.experiments.custom.evaluate_dropout_matrix \
  --gt_videos_root "$GT_ROOT" \
  --results $RESULTS \
  --output_csv "$OUTPUT_CSV" \
  --merge_shards

echo "Done. Summary: $OUTPUT_CSV"
