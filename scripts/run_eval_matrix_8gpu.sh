#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# 8-GPU evaluation of the dropout inference matrix: one shard per GPU, then merge.

set -e
cd "$(dirname "$0")/.."

N=8

echo "Launching $N evaluation shards (paths use the defaults baked into eval_dropout_matrix.py)..."
pids=()
for i in $(seq 0 $((N - 1))); do
  CUDA_VISIBLE_DEVICES=$i python3 scripts/eval_dropout_matrix.py \
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
CUDA_VISIBLE_DEVICES=0 python3 scripts/eval_dropout_matrix.py --merge_shards

echo "Done."
