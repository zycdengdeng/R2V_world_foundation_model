# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Multi-control post-training experiment configuration for zihanw.
Trains with blur, depth, and hdmap_bbox control inputs simultaneously.

Usage:
    torchrun --nproc_per_node=4 --master_port=12341 -m cosmos_transfer2.train \
        --config=cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/config.py \
        -- experiment=zihanw_multicontrol_post_train
"""

import os

from hydra.core.config_store import ConfigStore

from cosmos_transfer2._src.imaginaire.utils.checkpoint_db import get_checkpoint_path
from cosmos_transfer2.multiview_config import DEFAULT_CHECKPOINT

# Import the dataloader to ensure it's registered
import cosmos_transfer2.experiments.multiview.zihanw_multicontrol_dataloader  # noqa: F401


zihanw_multicontrol_post_train = dict(
    defaults=[
        f"/experiment/{DEFAULT_CHECKPOINT.experiment}",
        {"override /data_train": "zihanw_multicontrol_multiview"},
        {"override /data_val": "zihanw_multicontrol_multiview_val"},
        # Override conditioner to use multi-control version with blur, depth, and hdmap_bbox
        {"override /conditioner": "video_prediction_multiview_control_conditioner_multicontrol"},
    ],
    job=dict(
        project="cosmos_transfer_v2p5",
        group="zihanw_multicontrol",
        name="zihanw_multicontrol_post_train"
    ),
    checkpoint=dict(
        save_iter=200,  # Save every 200 iterations
        # NOTE: For fresh training with correct conditioner, set load_path="" and load_training_state=False
        # Previous checkpoints (iter_000000400, iter_000003000) were trained with WRONG conditioner
        # (only hdmap_bbox, missing blur and depth) - DO NOT resume from them!
        load_path="",  # Start fresh with correct conditioner
        load_training_state=False,
        strict_resume=False,
        load_from_object_store=dict(
            enabled=False,  # Loading from local filesystem
        ),
        save_to_object_store=dict(
            enabled=False,
        ),
    ),
    model=dict(
        config=dict(
            # Multiple control inputs: blur + depth + hdmap_bbox
            # hint_keys format: control names joined by "_"
            # IMPORTANT: Must match dataloader output keys exactly!
            # Dataloader outputs: control_input_blur, control_input_depth, control_input_hdmap_bbox
            # So hint_keys should be: "blur_depth_hdmap_bbox" (NOT "blur_depth_hdmap"!)
            hint_keys="blur_depth_hdmap_bbox",
            base_load_from=None,
            # Adjust for 29 frames: pixel_frames = (state_t - 1) * 4 + 1
            # 29 = (8 - 1) * 4 + 1, so state_t = 8
            state_t=8,  # latent temporal dimension for 29 frames
            # Override view sampling for 2-view training
            # Must match n_views which equals context_parallel_size
            train_sample_views_range=(2, 2),  # Always sample exactly 2 views
        ),
    ),
    trainer=dict(
        logging_iter=50,
        max_iter=10_000,
        run_validation=True,  # Enable validation
        validation_iter=200,  # Run validation every 200 iterations
        callbacks=dict(
            heart_beat=dict(
                save_s3=False,
            ),
            iter_speed=dict(
                hit_thres=100,
                save_s3=False,
            ),
            device_monitor=dict(
                save_s3=False,
            ),
            every_n_sample_reg=dict(
                every_n=200,  # Generate samples every 200 iterations
                save_s3=False,
                ctrl_hint_keys=["control_input_blur", "control_input_depth", "control_input_hdmap_bbox"],
            ),
            every_n_sample_ema=dict(
                every_n=200,  # Generate samples every 200 iterations
                save_s3=False,
                ctrl_hint_keys=["control_input_blur", "control_input_depth", "control_input_hdmap_bbox"],
            ),
            wandb=dict(
                save_s3=False,
            ),
            wandb_10x=dict(
                save_s3=False,
            ),
            dataloader_speed=dict(
                save_s3=False,
            ),
            frame_loss_log=dict(
                save_s3=False,
            ),
        ),
    ),
    model_parallel=dict(
        # For 2-view training with 2 GPUs
        # Constraints: state_t=8 AND num_heads=16
        # Valid cp_size: intersection of factors(8) and factors(16) = {1, 2, 4, 8}
        # With 29 frames, can scale to 4 or 8 GPUs later
        context_parallel_size=2,
    ),
)


# Register the experiment configuration
cs = ConfigStore.instance()

for _item in [
    zihanw_multicontrol_post_train,
]:
    experiment_name = [name.lower() for name, value in globals().items() if value is _item][0]

    cs.store(
        group="experiment",
        package="_global_",
        name=experiment_name,
        node=_item,
    )
