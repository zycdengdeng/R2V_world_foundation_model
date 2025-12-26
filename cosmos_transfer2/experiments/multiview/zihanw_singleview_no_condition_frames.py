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
Single-view multi-control training WITHOUT conditional frames.

Key differences from dropout_training:
- No conditional frames (min=0, max=0): Model never sees GT frames during training
- No control dropout: Model always sees control inputs
- This forces the model to learn scene structure from sparse control alone

Why this helps avoid sparse artifacts:
- Without GT frames as reference, model cannot "copy" patterns
- Model must learn to interpret sparse point cloud semantically
- Training matches inference (no GT frames in either case)

Usage:
    # Single GPU training
    PYTHONPATH=. python -m cosmos_transfer2.train \
        --config=cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/config.py \
        -- experiment=zihanw_singleview_no_condition_frames

    # Multi-GPU training (recommended: 4 GPUs for context parallel)
    torchrun --nproc_per_node=4 --master_port=12342 -m cosmos_transfer2.train \
        --config=cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/config.py \
        -- experiment=zihanw_singleview_no_condition_frames
"""

import os
from datetime import datetime

import torch.distributed as dist
from hydra.core.config_store import ConfigStore

# Generate timestamp for unique run directories (prevents auto-resume from old checkpoints)
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

from cosmos_transfer2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_transfer2._src.imaginaire.utils.checkpoint_db import get_checkpoint_by_uuid
from cosmos_transfer2._src.predict2.datasets.local_datasets.dataset_video import get_generic_dataloader, get_sampler
from cosmos_transfer2._src.predict2.text_encoders.text_encoder import EmbeddingConcatStrategy
from cosmos_transfer2._src.predict2_multiview.datasets.multiview import (
    DEFAULT_CAMERAS,
    collate_fn,
)

# Import the dataset class and camera definitions
from cosmos_transfer2.experiments.multiview.zihanw_multicontrol_dataloader import (
    MultiControlMultiviewDataset,
)

# Get the Transfer2.5 multiview checkpoint (has hdmap_bbox pretrained)
# This checkpoint has:
# - Base model (text-to-video capability)
# - hdmap_bbox control head (channels 0-15)
TRANSFER2_MULTIVIEW_CHECKPOINT = get_checkpoint_by_uuid("4ecc66e9-df19-4aed-9802-0d11e057287a")

# Number of GPUs - automatically detected from WORLD_SIZE environment variable
# This is set by torchrun based on --nproc_per_node
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))


# Single camera for single-view training (front camera only)
CAMERAS_1VIEW: tuple[str, ...] = (
    "camera_front_wide_120fov",  # Front camera only
)


def register_singleview_no_cond_dataloader() -> None:
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
        name="zihanw_singleview_no_cond",
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
        name="zihanw_singleview_no_cond_val",
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
register_singleview_no_cond_dataloader()


# Experiment configuration - NO conditional frames version
# Following the same pattern as classmate's custom_multi_control_experiment.py
# Key: Don't inherit from buttercup experiment (which includes load_base_model_callbacks)
zihanw_singleview_no_condition_frames = dict(
    # Direct defaults - NOT inheriting from buttercup experiment
    # This avoids load_base_model_callbacks which uses DCP format (incompatible with .pt files)
    defaults=[
        {"override /data_train": "zihanw_singleview_no_cond"},
        {"override /data_val": "zihanw_singleview_no_cond_val"},
        {"override /model": "fsdp_rectified_flow_multiview_control"},
        {"override /net": "cosmos_v1_2B_multiview_control"},
        {"override /conditioner": "video_prediction_multiview_control_conditioner_multicontrol"},
        {"override /ckpt_type": "dcp"},
        {"override /optimizer": "fusedadamw"},
        {"override /tokenizer": "wan2pt1_tokenizer"},
        # IMPORTANT: NO load_base_model_callbacks! It uses DCP format which fails with .pt files
        {"override /callbacks": ["basic", "wandb", "cluster_speed"]},
        "_self_",
    ],
    job=dict(
        project="cosmos_transfer_v2p5",
        group="zihanw_singleview",
        name=f"zihanw_singleview_no_cond_{RUN_TIMESTAMP}"  # Unique name to prevent auto-resume
    ),
    checkpoint=dict(
        save_iter=200,  # Save every 200 iterations
        # Load pretrained weights via checkpointer
        # The checkpointer uses easy_io.load() which can handle .pt files from HuggingFace
        load_path=TRANSFER2_MULTIVIEW_CHECKPOINT.path,
        load_training_state=False,  # Don't load optimizer state
        strict_resume=False,  # Allow missing keys for new control heads
        load_from_object_store=dict(enabled=False),
        save_to_object_store=dict(enabled=False),
    ),
    # Optimizer config (same as classmate's)
    optimizer=dict(
        lr=3e-5,  # Reduced for smaller dataset
        weight_decay=1e-3,
        betas=[0.9, 0.999],
    ),
    # Scheduler config
    scheduler=dict(
        f_max=[1.0],
        f_min=[0.1],
        warm_up_steps=[250],
        cycle_lengths=[30000],  # Match max_iter
    ),
    model=dict(
        config=dict(
            # Multiple control inputs - ORDER MATTERS for pretrained weights!
            # hdmap: channels 0-15, uses pretrained weights from Transfer2.5
            # blur: channels 16-31, train from scratch
            # depth: channels 32-47, train from scratch
            hint_keys="hdmap_blur_depth",
            # No conditional frames - pure control-based generation
            min_num_conditional_frames_per_view=0,
            max_num_conditional_frames_per_view=0,
            condition_locations=["first_random_n"],
            # Single view training (1 camera)
            train_sample_views_range=[1, 1],
            conditional_frames_probs={0: 1.0},  # 100% no condition frames
            state_t=8,
            online_text_embeddings_as_dict=False,
            fsdp_shard_size=8,
            resolution="720p",
            shift=5,
            use_dynamic_shift=False,
            train_time_weight="uniform",
            train_time_distribution="logitnormal",
            base_load_from=None,  # No additional base model loading
            # Network configuration (same as classmate's)
            net=dict(
                timestep_scale=0.001,
                use_wan_fp32_strategy=True,
                concat_view_embedding=True,
                view_condition_dim=7,
                state_t=8,
                n_cameras_emb=7,
                vace_has_mask=False,
                use_input_hint_block=True,
                condition_strategy="spaced",
                vace_block_every_n=7,
                rope_enable_fps_modulation=False,
                rope_h_extrapolation_ratio=3.0,
                rope_w_extrapolation_ratio=3.0,
                rope_t_extrapolation_ratio=8.0 / 24.0,
                use_crossattn_projection=True,
                crossattn_proj_in_channels=100352,
                crossattn_emb_channels=1024,
                sac_config=dict(mode="predict2_2b_720_aggressive"),
            ),
            # Conditioner configuration
            conditioner=dict(
                use_video_condition=dict(dropout_rate=0.0),
                text=dict(dropout_rate=0.2, use_empty_string=False),
            ),
            tokenizer=dict(temporal_window=16),
            # Text encoder configuration - required for computing text embeddings from ai_caption
            text_encoder_class="reason1p1_7B",
            text_encoder_config=dict(
                embedding_concat_strategy=str(EmbeddingConcatStrategy.FULL_CONCAT),
                compute_online=True,
            ),
        ),
    ),
    trainer=dict(
        logging_iter=50,
        max_iter=30_000,  # Extended to 30k iterations
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
                # Order must match hint_keys: hdmap first, then blur, depth
                ctrl_hint_keys=["control_input_hdmap_bbox", "control_input_blur", "control_input_depth"],
            ),
            every_n_sample_ema=dict(
                every_n=200,
                save_s3=False,
                # Order must match hint_keys: hdmap first, then blur, depth
                ctrl_hint_keys=["control_input_hdmap_bbox", "control_input_blur", "control_input_depth"],
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
        # Context parallel size = number of GPUs (automatically detected)
        # For state_t=8, valid cp_size: 1, 2, 4, 8
        context_parallel_size=WORLD_SIZE,
    ),
)


# Register the experiment configuration
cs = ConfigStore.instance()

for _item in [
    zihanw_singleview_no_condition_frames,
]:
    experiment_name = [name.lower() for name, value in globals().items() if value is _item][0]

    cs.store(
        group="experiment",
        package="_global_",
        name=experiment_name,
        node=_item,
    )
