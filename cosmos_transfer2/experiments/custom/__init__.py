# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Custom experiments module for multi-control post training

from cosmos_transfer2.experiments.custom.custom_multi_control_dataset import (
    MultiControlMultiviewDataset,
    collate_fn,
)

# Import experiment to trigger registration
from cosmos_transfer2.experiments.custom import custom_multi_control_experiment

__all__ = [
    "MultiControlMultiviewDataset",
    "collate_fn",
    "custom_multi_control_experiment",
]
