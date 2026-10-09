#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Rewrite the caption field of Dongfeng caption JSONs, keeping all other fields.

Only BlurProjection/captions/ftheta_camera_front_wide_120fov is actually read by
training (caption content + sample list), but all 7 camera folders are rewritten
for consistency.

Usage:
    # 1. Back up first (captions tree is ~14MB):
    #    cp -r .../BlurProjection/captions .../BlurProjection/captions_backup
    # 2. Edit NEW_CAPTION / SCENE_CAPTIONS below.
    # 3. python3 -m cosmos_transfer2.experiments.custom.rewrite_dongfeng_captions
"""
import json
from pathlib import Path

CAPTIONS_ROOT = Path("/mnt2/dongfeng_ego_output/transfer2/BlurProjection/captions")

# Generic scene description. Do NOT mention camera identity or heading direction:
# per-view prefixes are appended automatically by the training dataset.
NEW_CAPTION = (
    "Daytime driving scene on Chinese urban roads. "
    "Multi-lane asphalt road with white lane markings. "
    "Buildings, trees, and street infrastructure along the roadside. "
    "Mixed traffic including cars, buses, trucks, cyclists, and pedestrians."
)

# Optional per-scene overrides, keyed by scene ID (e.g. "5k_2026-01-12-10-05-16").
# Scenes not listed here fall back to NEW_CAPTION.
SCENE_CAPTIONS = {
    # "5k_2026-01-12-10-05-16": "Nighttime driving scene ...",
}


def main() -> None:
    count = 0
    for cam_dir in sorted(CAPTIONS_ROOT.iterdir()):
        if not cam_dir.is_dir():
            continue
        for jf in sorted(cam_dir.glob("*.json")):
            data = json.loads(jf.read_text(encoding="utf-8"))
            scene = data.get("scene", "")
            data["caption"] = SCENE_CAPTIONS.get(scene, NEW_CAPTION)
            jf.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            count += 1
    print(f"Rewrote {count} caption files.")


if __name__ == "__main__":
    main()
