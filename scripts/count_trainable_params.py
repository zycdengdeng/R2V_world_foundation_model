#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Count trainable (control-branch) vs total parameters from a DCP checkpoint's
metadata, without loading any weights (CPU, seconds).

Trainable modules under the freeze policy (freeze_base_model):
    control_blocks (incl. before_proj/after_proj), control_embedder, input_hint_block

Usage:
    python3 scripts/count_trainable_params.py \
        /path/to/checkpoints/iter_000006600/model
"""
import sys
from math import prod

from torch.distributed.checkpoint import FileSystemReader

TRAINABLE_MARKERS = ("control_blocks", "control_embedder", "input_hint_block")


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(f"Usage: {sys.argv[0]} <path/to/iter_XXXXXX/model>")

    md = FileSystemReader(sys.argv[1]).read_metadata()

    total = 0
    trainable = 0
    groups = {m: 0 for m in TRAINABLE_MARKERS}
    ema_total = 0

    for name, meta in md.state_dict_metadata.items():
        size = getattr(meta, "size", None)
        if size is None:  # non-tensor entries (e.g. bytes metadata)
            continue
        n = prod(size) if len(size) > 0 else 1
        # EMA copies duplicate every weight; count them separately, not in totals
        if "ema" in name.lower():
            ema_total += n
            continue
        total += n
        for marker in TRAINABLE_MARKERS:
            if marker in name:
                trainable += n
                groups[marker] += n
                break

    frozen = total - trainable
    print(f"Checkpoint: {sys.argv[1]}")
    print(f"Total params (excl. EMA copies): {total / 1e9:.3f} B")
    print(f"  Trainable (control branch):    {trainable / 1e6:.1f} M  ({100 * trainable / total:.2f} %)")
    for marker, n in groups.items():
        print(f"    - {marker:20s} {n / 1e6:8.1f} M")
    print(f"  Frozen (backbone & misc):      {frozen / 1e9:.3f} B  ({100 * frozen / total:.2f} %)")
    if ema_total:
        print(f"  (EMA copies in checkpoint:     {ema_total / 1e9:.3f} B, excluded above)")


if __name__ == "__main__":
    main()
