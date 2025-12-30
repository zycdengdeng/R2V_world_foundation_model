# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Config module for Custom Multi-Control experiment
# This file sets up the Hydra configuration for loading custom experiments

from typing import Any, List

import attrs

from cosmos_transfer2._src.imaginaire import config
from cosmos_transfer2._src.imaginaire.trainer import ImaginaireTrainer as Trainer
from cosmos_transfer2._src.imaginaire.utils.config_helper import import_all_modules_from_package
from cosmos_transfer2._src.predict2.configs.common.defaults.checkpoint import register_checkpoint
from cosmos_transfer2._src.predict2.configs.common.defaults.ckpt_type import register_ckpt_type
from cosmos_transfer2._src.predict2.configs.common.defaults.ema import register_ema
from cosmos_transfer2._src.predict2.configs.common.defaults.optimizer import register_optimizer
from cosmos_transfer2._src.predict2.configs.common.defaults.scheduler import register_scheduler
from cosmos_transfer2._src.predict2.configs.common.defaults.tokenizer import register_tokenizer
from cosmos_transfer2._src.transfer2.configs.vid2vid_transfer.defaults.callbacks import register_callbacks
from cosmos_transfer2._src.transfer2_multiview.configs.vid2vid_transfer.defaults.conditioner import register_conditioner
from cosmos_transfer2._src.transfer2_multiview.configs.vid2vid_transfer.defaults.dataloader import (
    register_dataloaders,
)
from cosmos_transfer2._src.transfer2_multiview.configs.vid2vid_transfer.defaults.dataloader_local import (
    register_dataloader_local,
)
from cosmos_transfer2._src.transfer2_multiview.configs.vid2vid_transfer.defaults.model import register_model
from cosmos_transfer2._src.transfer2_multiview.configs.vid2vid_transfer.defaults.net import register_net


@attrs.define(slots=False)
class Config(config.Config):
    # default config groups that will be used unless overwritten
    defaults: List[Any] = attrs.field(
        factory=lambda: [
            "_self_",
            {"data_train": "mock"},
            {"data_val": "mock"},
            {"model": "fsdp_rectified_flow_multiview_control"},
            {"net": "cosmos_v1_2B_multiview_control"},
            {"optimizer": "fusedadamw"},
            {"scheduler": "lambdalinear"},
            {"callbacks": "basic"},
            {"ckpt_type": "dcp"},
            {"tokenizer": "wan2pt1_tokenizer"},
            {"conditioner": "multi_view_video_prediction_control"},
            {"ema": "power"},
            {"checkpoint": "s3"},
            {"experiment": None},
        ]
    )


def make_config() -> Config:
    """Create and return the config object for model loading."""
    c = Config(
        model=None,
        optimizer=None,
        scheduler=None,
        dataloader_train=None,
        dataloader_val=None,
    )

    # Set up job defaults
    c.job.project = "cosmos_transfer2_custom"
    c.job.group = "eval"
    c.job.name = "eval_${now:%Y-%m-%d}_${now:%H-%M-%S}"

    # Set up trainer defaults
    c.trainer.type = Trainer
    c.trainer.straggler_detection.enabled = False
    c.trainer.max_iter = 400_000
    c.trainer.logging_iter = 100
    c.trainer.validation_iter = 100
    c.trainer.run_validation = False
    c.trainer.callbacks = None

    # Register all config groups
    register_checkpoint()
    register_ckpt_type()
    register_ema()
    register_callbacks()
    register_optimizer()
    register_scheduler()
    register_tokenizer()
    register_dataloaders()
    register_dataloader_local()
    register_conditioner()
    register_model()
    register_net()

    # Import custom experiment configurations
    # This registers the custom_multi_control_post_train experiment
    import_all_modules_from_package("cosmos_transfer2.experiments.custom", reload=True)

    return c
