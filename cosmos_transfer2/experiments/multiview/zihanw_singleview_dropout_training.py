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
Single-view multi-control training with control dropout.

This experiment trains a single-view model with 20% control dropout to preserve
text generation capability. The dropout allows the model to learn both:
- Text-only generation (when controls are dropped)
- Control-guided generation (when controls are present)

Usage:
    # Single GPU training
    PYTHONPATH=. python -m cosmos_transfer2.train \
        --config=cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/config.py \
        -- experiment=zihanw_singleview_dropout_train

    # Multi-GPU training (recommended: 2 GPUs)
    torchrun --nproc_per_node=2 --master_port=12342 -m cosmos_transfer2.train \
        --config=cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/config.py \
        -- experiment=zihanw_singleview_dropout_train
"""

import torch.distributed as dist
from hydra.core.config_store import ConfigStore

from cosmos_transfer2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_transfer2._src.predict2.datasets.local_datasets.dataset_video import get_generic_dataloader, get_sampler
from cosmos_transfer2._src.predict2_multiview.datasets.multiview import (
    DEFAULT_CAMERAS,
    collate_fn,
)
from cosmos_transfer2.multiview_config import DEFAULT_CHECKPOINT

# Import the dataset class and camera definitions
from cosmos_transfer2.experiments.multiview.zihanw_multicontrol_dataloader import (
    MultiControlMultiviewDataset,
)


# Single camera for single-view training (front camera only)
CAMERAS_1VIEW: tuple[str, ...] = (
    "camera_front_wide_120fov",  # Front camera only
)


def register_singleview_dataloader() -> None:
    """Register single-view dataloader with front camera only."""

    cs = ConfigStore.instance()

    # Single-view dataset (front camera only)
    dataset = L(MultiControlMultiviewDataset)(
        base_video_dir="/mnt/zihanw/proj_utils_pro/transfer_video_maker/output/BlurProjection",
        control_dirs={
            "blur": "/mnt/zihanw/proj_utils_pro/transfer_video_maker/output/BlurProjection/control_input_blur",
            "depth": "/mnt/zihanw/proj_utils_pro/transfer_video_maker/output/DepthSparse/control_input_depth",
            "hdmap_bbox": "/mnt/zihanw/proj_utils_pro/transfer_video_maker/output/HDMapBbox/control_input_hdmap_bbox",
        },
        folder_to_camera_key={f"ftheta_{camera_name}": camera_name for camera_name in DEFAULT_CAMERAS},
        resolution_hw=(720, 1280),
        num_video_frames=29,  # 29 frames -> state_t=8
        single_caption_camera_name="camera_front_wide_120fov",
        # Single view: front camera only
        selected_cameras=CAMERAS_1VIEW,
        # Exclude clips for inference/evaluation
        exclude_clips=("075", "077"),
    )

    cs.store(
        group="data_train",
        package="dataloader_train",
        name="zihanw_singleview_multicontrol",
        node=L(get_generic_dataloader)(
            dataset=dataset,
            sampler=L(get_sampler)(dataset=dataset) if dist.is_initialized() else None,
            collate_fn=collate_fn,
            batch_size=1,
            drop_last=True,
            num_workers=4,
            pin_memory=True,
        ),
    )

    # Validation dataset
    val_dataset = L(MultiControlMultiviewDataset)(
        base_video_dir="/mnt/zihanw/proj_utils_pro/transfer_video_maker/output/BlurProjection",
        control_dirs={
            "blur": "/mnt/zihanw/proj_utils_pro/transfer_video_maker/output/BlurProjection/control_input_blur",
            "depth": "/mnt/zihanw/proj_utils_pro/transfer_video_maker/output/DepthSparse/control_input_depth",
            "hdmap_bbox": "/mnt/zihanw/proj_utils_pro/transfer_video_maker/output/HDMapBbox/control_input_hdmap_bbox",
        },
        folder_to_camera_key={f"ftheta_{camera_name}": camera_name for camera_name in DEFAULT_CAMERAS},
        resolution_hw=(720, 1280),
        num_video_frames=29,
        single_caption_camera_name="camera_front_wide_120fov",
        selected_cameras=CAMERAS_1VIEW,
        include_only_clips=("075", "077"),
    )

    cs.store(
        group="data_val",
        package="dataloader_val",
        name="zihanw_singleview_multicontrol_val",
        node=L(get_generic_dataloader)(
            dataset=val_dataset,
            sampler=L(get_sampler)(dataset=val_dataset) if dist.is_initialized() else None,
            collate_fn=collate_fn,
            batch_size=1,
            drop_last=False,
            num_workers=4,
            pin_memory=True,
        ),
    )


# Register dataloaders when module is imported
register_singleview_dataloader()


# Experiment configuration
zihanw_singleview_dropout_train = dict(
    defaults=[
        f"/experiment/{DEFAULT_CHECKPOINT.experiment}",
        {"override /data_train": "zihanw_singleview_multicontrol"},
        {"override /data_val": "zihanw_singleview_multicontrol_val"},
        # Use conditioner with 20% control dropout for text capability preservation
        {"override /conditioner": "video_prediction_multiview_control_conditioner_multicontrol_dropout"},
    ],
    job=dict(
        project="cosmos_transfer_v2p5",
        group="zihanw_singleview",
        name="zihanw_singleview_dropout_train"
    ),
    checkpoint=dict(
        save_iter=200,  # Save every 200 iterations
        # Resume from checkpoint (update iter_XXXXXX to your latest checkpoint)
        load_path="/mnt/zihanw/cosmos-transfer-output/cosmos_transfer_v2p5/zihanw_singleview/zihanw_singleview_dropout_train/checkpoints/iter_000003000",
        load_training_state=True,  # Resume optimizer and iteration counter
        strict_resume=False,
        load_from_object_store=dict(
            enabled=False,
        ),
        save_to_object_store=dict(
            enabled=False,
        ),
    ),
    model=dict(
        config=dict(
            # Multiple control inputs: blur + depth + hdmap_bbox
            hint_keys="blur_depth_hdmap_bbox",
            base_load_from=None,
            # state_t=8 for 29 frames
            state_t=8,
            # Single view training
            train_sample_views_range=(1, 1),
        ),
    ),
    trainer=dict(
        logging_iter=50,
        max_iter=10_000,  # Start with 10k iterations
        run_validation=False,  # Disable validation initially
        validation_iter=200,
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
                every_n=200,
                save_s3=False,
                ctrl_hint_keys=["control_input_blur", "control_input_depth", "control_input_hdmap_bbox"],
            ),
            every_n_sample_ema=dict(
                every_n=200,
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
        # Single view doesn't need context parallelism across views
        # But we can still use it for memory efficiency
        # For single-view with state_t=8, valid cp_size: 1, 2, 4, 8
        context_parallel_size=1,  # Single GPU for single view
    ),
)


# Register the experiment configuration
cs = ConfigStore.instance()

for _item in [
    zihanw_singleview_dropout_train,
]:
    experiment_name = [name.lower() for name, value in globals().items() if value is _item][0]

    cs.store(
        group="experiment",
        package="_global_",
        name=experiment_name,
        node=_item,
    )
