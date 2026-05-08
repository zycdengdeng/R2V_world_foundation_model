# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Custom experiments module for multi-control post training

from cosmos_transfer2.experiments.custom.custom_multi_control_dataset import (
    MultiControlMultiviewDataset,
    collate_fn,
)
from cosmos_transfer2.experiments.custom.roadside_multi_control_dataset import (
    RoadsideMultiControlDataset,
)
from cosmos_transfer2.experiments.custom.single_view_hdmap_dataset import (
    SingleViewHDMapDataset,
)

# Import experiments to trigger registration
from cosmos_transfer2.experiments.custom import custom_multi_control_experiment
from cosmos_transfer2.experiments.custom import dense_multi_control_experiment
from cosmos_transfer2.experiments.custom import ablation_single_view_experiment
from cosmos_transfer2.experiments.custom import single_view_hdmap_experiment

__all__ = [
    "MultiControlMultiviewDataset",
    "RoadsideMultiControlDataset",
    "SingleViewHDMapDataset",
    "collate_fn",
    "custom_multi_control_experiment",
    "dense_multi_control_experiment",
    "ablation_single_view_experiment",
    "single_view_hdmap_experiment",
]
