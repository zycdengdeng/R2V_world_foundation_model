# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Custom Experiment Configuration for Multi-Control Post Training
# Trains a model with 3 control types: vis (blur), depth, bbox (hdmap)
#
# Option D: Use Transfer2.5 multiview checkpoint but train all 3 heads from scratch
# By using "bbox" instead of "hdmap", the pre-trained hdmap_bbox weights are ignored

import os

import torch.distributed as dist
from hydra.core.config_store import ConfigStore

from cosmos_transfer2._src.imaginaire.flags import SMOKE
from cosmos_transfer2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_transfer2._src.imaginaire.utils.checkpoint_db import get_checkpoint_by_uuid

from cosmos_transfer2._src.predict2.datasets.local_datasets.dataset_video import get_generic_dataloader, get_sampler
from cosmos_transfer2._src.predict2.text_encoders.text_encoder import EmbeddingConcatStrategy
from cosmos_transfer2._src.predict2_multiview.datasets.multiview import DEFAULT_CAMERAS
from cosmos_transfer2._src.predict2_multiview.callbacks.every_n_draw_sample_multiviewvideo import (
    EveryNDrawSampleMultiviewVideo,
)

from cosmos_transfer2.experiments.custom.custom_multi_control_dataset import (
    MultiControlMultiviewDataset,
    collate_fn,
)

# Get the Transfer2.5 multiview checkpoint (optimized for control tasks)
# Using "bbox" instead of "hdmap" so pre-trained hdmap_bbox weights are NOT used
# All 3 control heads (vis, depth, bbox) will be randomly initialized
TRANSFER2_MULTIVIEW_CHECKPOINT = get_checkpoint_by_uuid("4ecc66e9-df19-4aed-9802-0d11e057287a")


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

# Main experiment configuration
# Option D: Use Transfer2.5 multiview (optimized for control) but train all 3 heads from scratch
# Using "bbox" instead of "hdmap" so all heads (vis, depth, bbox) are randomly initialized
custom_multi_control_post_train = dict(
    # Use Transfer2 multiview config (includes ControlNet architecture)
    # Override with custom dataloader
    defaults=[
        {"override /data_train": "custom_multi_control_train_data"},
        {"override /model": "fsdp_rectified_flow_multiview_control"},
        {"override /net": "cosmos_v1_2B_multiview_control"},
        {"override /conditioner": "video_prediction_multiview_control_conditioner"},
        {"override /ckpt_type": "dcp"},
        {"override /optimizer": "fusedadamw"},
        {"override /tokenizer": "wan2pt1_tokenizer"},
        "_self_",
    ],
    job=dict(
        project="cosmos_transfer_custom",
        group="multi_control",
        name="2b_custom_multi_control_post_train",
    ),
    checkpoint=dict(
        save_iter=500,  # Save every 500 iterations
        load_path=TRANSFER2_MULTIVIEW_CHECKPOINT.path,  # Load from Transfer2.5 multiview
        load_training_state=False,  # Don't load optimizer state
        strict_resume=False,  # Allow missing keys (ControlNet heads will be random initialized)
        load_from_object_store=dict(enabled=False),
        save_to_object_store=dict(enabled=False),
    ),
    optimizer=dict(
        lr=8.63e-5,
        weight_decay=1e-3,
        betas=[0.9, 0.999],
    ),
    scheduler=dict(
        f_max=[0.5],
        f_min=[0.2],
        warm_up_steps=[1000],
        cycle_lengths=[100_000],
    ),
    model=dict(
        config=dict(
            # CRITICAL: 3 control heads - vis (blur), depth, bbox (hdmap data)
            hint_keys="vis_depth_bbox",
            # Training configuration
            min_num_conditional_frames_per_view=0,  # t2w mode
            max_num_conditional_frames_per_view=2,  # i2w or v2v
            condition_locations=["first_random_n"],
            train_sample_views_range=[7, 7],  # All 7 views
            conditional_frames_probs={0: 0.5, 1: 0.25, 2: 0.25},
            state_t=8,
            online_text_embeddings_as_dict=False,
            fsdp_shard_size=8,
            resolution="720p",
            shift=5,
            use_dynamic_shift=False,
            train_time_weight="uniform",
            train_time_distribution="logitnormal",
            base_load_from=None,  # No additional base model loading
            # Network configuration
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
                vace_block_every_n=7,  # Apply control to every 7th layer
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
            # Text encoder - use Reason1.1
            text_encoder_class="reason1p1_7B",
            text_encoder_config=dict(
                embedding_concat_strategy=str(EmbeddingConcatStrategy.FULL_CONCAT),
                compute_online=True,
            ),
        ),
    ),
    trainer=dict(
        logging_iter=50,
        max_iter=20_000,  # More iterations since training from scratch
        callbacks=dict(
            heart_beat=dict(save_s3=False),
            iter_speed=dict(hit_thres=100, every_n=100, save_s3=False),
            device_monitor=dict(save_s3=False),
            grad_clip=dict(clip_norm=0.1),
            # Sample generation for monitoring
            every_n_sample_reg=L(EveryNDrawSampleMultiviewVideo)(
                every_n=1000,
                is_x0=False,
                is_ema=False,
                num_sampling_step=35,
                guidance=[7],
                fps=10,
                ctrl_hint_keys=["control_input_vis", "control_input_depth", "control_input_bbox"],
                control_weights=[0.0, 1.0],
                save_s3=False,
            ),
            every_n_sample_ema=L(EveryNDrawSampleMultiviewVideo)(
                every_n=1000,
                is_x0=False,
                is_ema=True,
                num_sampling_step=35,
                guidance=[7],
                fps=10,
                ctrl_hint_keys=["control_input_vis", "control_input_depth", "control_input_bbox"],
                control_weights=[0.0, 1.0],
                save_s3=False,
            ),
            wandb=dict(save_s3=False),
            wandb_10x=dict(save_s3=False),
            dataloader_speed=dict(save_s3=False),
            frame_loss_log=dict(save_s3=False),
        ),
    ),
    model_parallel=dict(
        context_parallel_size=WORLD_SIZE,
    ),
    dataloader_train=dict(
        augmentation_config=dict(
            single_caption_camera_name="camera_front_wide_120fov",
            add_view_prefix_to_caption=True,
        ),
    ),
)


# Smaller configuration for quick testing (500 iterations)
custom_multi_control_post_train_small = dict(
    defaults=[
        {"override /data_train": "custom_multi_control_train_data"},
        {"override /model": "fsdp_rectified_flow_multiview_control"},
        {"override /net": "cosmos_v1_2B_multiview_control"},
        {"override /conditioner": "video_prediction_multiview_control_conditioner"},
        {"override /ckpt_type": "dcp"},
        {"override /optimizer": "fusedadamw"},
        {"override /tokenizer": "wan2pt1_tokenizer"},
        "_self_",
    ],
    job=dict(
        project="cosmos_transfer_custom",
        group="multi_control",
        name="2b_custom_multi_control_post_train_small",
    ),
    checkpoint=dict(
        save_iter=100,  # Save every 100 iterations for testing
        load_path=TRANSFER2_MULTIVIEW_CHECKPOINT.path,
        load_training_state=False,
        strict_resume=False,
        load_from_object_store=dict(enabled=False),
        save_to_object_store=dict(enabled=False),
    ),
    optimizer=dict(
        lr=8.63e-5,
        weight_decay=1e-3,
        betas=[0.9, 0.999],
    ),
    scheduler=dict(
        f_max=[0.5],
        f_min=[0.2],
        warm_up_steps=[100],
        cycle_lengths=[1000],
    ),
    model=dict(
        config=dict(
            hint_keys="vis_depth_bbox",
            min_num_conditional_frames_per_view=0,
            max_num_conditional_frames_per_view=2,
            condition_locations=["first_random_n"],
            train_sample_views_range=[7, 7],
            conditional_frames_probs={0: 0.5, 1: 0.25, 2: 0.25},
            state_t=8,
            online_text_embeddings_as_dict=False,
            fsdp_shard_size=8,
            resolution="720p",
            shift=5,
            base_load_from=None,
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
            conditioner=dict(
                use_video_condition=dict(dropout_rate=0.0),
                text=dict(dropout_rate=0.2, use_empty_string=False),
            ),
            tokenizer=dict(temporal_window=16),
            text_encoder_class="reason1p1_7B",
            text_encoder_config=dict(
                embedding_concat_strategy=str(EmbeddingConcatStrategy.FULL_CONCAT),
                compute_online=True,
            ),
        ),
    ),
    trainer=dict(
        logging_iter=10,
        max_iter=500,
        callbacks=dict(
            heart_beat=dict(save_s3=False),
            iter_speed=dict(hit_thres=50, every_n=50, save_s3=False),
            device_monitor=dict(save_s3=False),
            grad_clip=dict(clip_norm=0.1),
            every_n_sample_reg=L(EveryNDrawSampleMultiviewVideo)(
                every_n=200,
                is_x0=False,
                is_ema=False,
                num_sampling_step=35,
                guidance=[7],
                fps=10,
                ctrl_hint_keys=["control_input_vis", "control_input_depth", "control_input_bbox"],
                control_weights=[0.0, 1.0],
                save_s3=False,
            ),
            every_n_sample_ema=L(EveryNDrawSampleMultiviewVideo)(
                every_n=200,
                is_x0=False,
                is_ema=True,
                num_sampling_step=35,
                guidance=[7],
                fps=10,
                ctrl_hint_keys=["control_input_vis", "control_input_depth", "control_input_bbox"],
                control_weights=[0.0, 1.0],
                save_s3=False,
            ),
            wandb=dict(save_s3=False),
            wandb_10x=dict(save_s3=False),
            dataloader_speed=dict(save_s3=False),
            frame_loss_log=dict(save_s3=False),
        ),
    ),
    model_parallel=dict(
        context_parallel_size=WORLD_SIZE,
    ),
    dataloader_train=dict(
        augmentation_config=dict(
            single_caption_camera_name="camera_front_wide_120fov",
            add_view_prefix_to_caption=True,
        ),
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
