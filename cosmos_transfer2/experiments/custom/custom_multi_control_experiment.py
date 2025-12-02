# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Custom Experiment Configuration for Multi-Control Post Training
# Trains a model with 3 control types: blur, depth, hdmap_bbox

import os

import torch.distributed as dist
from hydra.core.config_store import ConfigStore

from cosmos_transfer2._src.imaginaire.flags import SMOKE
from cosmos_transfer2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_transfer2._src.imaginaire.utils.checkpoint_db import get_checkpoint_path

from cosmos_transfer2._src.predict2.datasets.local_datasets.dataset_video import get_generic_dataloader, get_sampler
from cosmos_transfer2._src.predict2_multiview.datasets.multiview import DEFAULT_CAMERAS
from cosmos_transfer2.multiview_config import DEFAULT_CHECKPOINT

from cosmos_transfer2.experiments.custom.custom_multi_control_dataset import (
    MultiControlMultiviewDataset,
    collate_fn,
)


# ============================================================================
# Configuration Constants - MODIFY THESE FOR YOUR SETUP
# ============================================================================

# Dataset paths (user needs to modify these)
BLUR_DATASET_DIR = "/mnt/zihanw/proj_utils_pro/transfer_video_maker/output/BlurProjection"
DEPTH_DATASET_DIR = "/mnt/zihanw/proj_utils_pro/transfer_video_maker/output/DepthSparse"
HDMAP_DATASET_DIR = "/mnt/zihanw/proj_utils_pro/transfer_video_maker/output/HDMapBbox"

# Output directory
OUTPUT_DIR = "/mnt/zihanw/Output_R2V_world_foundation_model_v1"

# Test scene IDs to exclude from training
TEST_SCENE_IDS = ["047", "061"]

# Training scene IDs (for reference)
TRAIN_SCENE_IDS = ["017", "019", "020", "022", "045", "049", "051", "055", "057", "059", "065", "067", "069", "073", "075", "077"]

# Number of GPUs (for context parallel)
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 8))


# ============================================================================
# Dataset Configuration
# ============================================================================

def register_custom_dataloader() -> None:
    """Register custom dataloader with multi-control dataset."""
    cs = ConfigStore.instance()

    # Create dataset
    dataset = L(MultiControlMultiviewDataset)(
        blur_dataset_dir=BLUR_DATASET_DIR,
        depth_dataset_dir=DEPTH_DATASET_DIR,
        hdmap_dataset_dir=HDMAP_DATASET_DIR,
        resolution_hw=(720, 1280),
        num_video_frames=29,
        fps_downsample_factor=1,
        camera_keys=DEFAULT_CAMERAS if not SMOKE else DEFAULT_CAMERAS[:1],
        single_caption_camera_name="camera_front_wide_120fov",
        add_view_prefix_to_caption=True,
        exclude_scene_ids=TEST_SCENE_IDS,
    )

    # Register dataloader
    cs.store(
        group="data_train",
        package="dataloader_train",
        name="custom_multi_control_train_data",
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


# ============================================================================
# Experiment Configuration
# ============================================================================

# Main experiment configuration - inherits from the default multiview experiment
custom_multi_control_post_train = dict(
    # Inherit from default multiview experiment and use custom dataloader
    defaults=[
        f"/experiment/{DEFAULT_CHECKPOINT.experiment}",
        {"override /data_train": "custom_multi_control_train_data"},
    ],
    job=dict(
        project="cosmos_transfer_custom",
        group="multi_control",
        name="2b_custom_multi_control_post_train",
    ),
    checkpoint=dict(
        save_iter=200,
        load_path=get_checkpoint_path(DEFAULT_CHECKPOINT.s3.uri),
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
            # IMPORTANT: hint_keys defines which control inputs to use
            # "vis_depth_hdmap" -> ["control_input_vis", "control_input_depth", "control_input_hdmap_bbox"]
            # We use "vis" for blur data to match the existing condition class
            hint_keys="vis_depth_hdmap",
            base_load_from=None,
        ),
    ),
    trainer=dict(
        logging_iter=10,
        max_iter=5_000,
        callbacks=dict(
            heart_beat=dict(save_s3=False),
            iter_speed=dict(hit_thres=200, save_s3=False),
            device_monitor=dict(save_s3=False),
            every_n_sample_reg=dict(every_n=200, save_s3=False),
            every_n_sample_ema=dict(every_n=200, save_s3=False),
            wandb=dict(save_s3=False),
            wandb_10x=dict(save_s3=False),
            dataloader_speed=dict(save_s3=False),
            frame_loss_log=dict(save_s3=False),
        ),
    ),
    model_parallel=dict(
        context_parallel_size=WORLD_SIZE,
    ),
)


# Smaller configuration for quick testing
custom_multi_control_post_train_small = dict(
    defaults=[
        f"/experiment/{DEFAULT_CHECKPOINT.experiment}",
        {"override /data_train": "custom_multi_control_train_data"},
    ],
    job=dict(
        project="cosmos_transfer_custom",
        group="multi_control",
        name="2b_custom_multi_control_post_train_small",
    ),
    checkpoint=dict(
        save_iter=100,
        load_path=get_checkpoint_path(DEFAULT_CHECKPOINT.s3.uri),
        load_training_state=False,
        strict_resume=False,
        load_from_object_store=dict(enabled=False),
        save_to_object_store=dict(enabled=False),
    ),
    model=dict(
        config=dict(
            hint_keys="vis_depth_hdmap",
            base_load_from=None,
        ),
    ),
    trainer=dict(
        logging_iter=10,
        max_iter=500,
        callbacks=dict(
            heart_beat=dict(save_s3=False),
            iter_speed=dict(hit_thres=50, save_s3=False),
            device_monitor=dict(save_s3=False),
            every_n_sample_reg=dict(every_n=100, save_s3=False),
            every_n_sample_ema=dict(every_n=100, save_s3=False),
            wandb=dict(save_s3=False),
            wandb_10x=dict(save_s3=False),
            dataloader_speed=dict(save_s3=False),
            frame_loss_log=dict(save_s3=False),
        ),
    ),
    model_parallel=dict(
        context_parallel_size=WORLD_SIZE,
    ),
)


# ============================================================================
# Registration
# ============================================================================

cs = ConfigStore.instance()

# Register the configurations with Hydra ConfigStore
for _item in [
    custom_multi_control_post_train,
    custom_multi_control_post_train_small,
]:
    experiment_name = [name.lower() for name, value in globals().items() if value is _item][0]
    cs.store(
        group="experiment",
        package="_global_",
        name=experiment_name,
        node=_item,
    )

# Also register the custom dataloader
register_custom_dataloader()
