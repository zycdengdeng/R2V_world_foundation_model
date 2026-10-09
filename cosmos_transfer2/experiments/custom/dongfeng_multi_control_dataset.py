# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Dataset for the Dongfeng ego-view data.
#
# Same directory layout as the original output_full_data
# (BlurProjection/{captions,control_input_blur,videos}, DepthSparse/control_input_depth,
#  HDMapBbox/control_input_hdmap_bbox, each with 7 ftheta camera subfolders),
# but a different sample naming scheme:
#     old: {scene}_seg{NN}            e.g. "017_seg01"        (scene = "017")
#     new: {drive}_seg{NNN}           e.g. "5w_2026-01-12-11-37-14_seg013"
#          where drive contains underscores, so scene extraction must strip the
#          trailing "_seg<digits>" instead of splitting on the first "_".

import re

from cosmos_transfer2.experiments.custom.custom_multi_control_dataset import (
    MultiControlMultiviewDataset,
)

_SEG_SUFFIX_RE = re.compile(r"_seg\d+$")


def extract_dongfeng_scene_id(sample_id: str) -> str:
    """'5w_2026-01-12-11-37-14_seg013' -> '5w_2026-01-12-11-37-14'."""
    return _SEG_SUFFIX_RE.sub("", sample_id)


class DongfengMultiControlDataset(MultiControlMultiviewDataset):
    """MultiControlMultiviewDataset with scene IDs parsed from Dongfeng-style names."""

    def _extract_scene_id(self, sample_id: str) -> str:
        return extract_dongfeng_scene_id(sample_id)
