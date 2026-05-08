# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Single-view HDMap fine-tuning of Cosmos-Transfer2.5 on car-side data.
#
# • Single camera: camera_front_wide_120fov
# • Single control: hdmap_bbox (continues from the pre-trained Transfer2.5 hdmap head)
# • 0 conditional frames (pure prompt + hdmap -> video)
# • 4-GPU FSDP, no context parallelism
# • Loads from the official Transfer2.5 multiview checkpoint

import os
from datetime import datetime

import torch.distributed as dist
from hydra.core.config_store import ConfigStore

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

from cosmos_transfer2._src.imaginaire.flags import SMOKE
from cosmos_transfer2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_transfer2._src.imaginaire.utils.checkpoint_db import get_checkpoint_by_uuid

from cosmos_transfer2._src.predict2.datasets.local_datasets.dataset_video import (
    get_generic_dataloader,
    get_sampler,
)
from cosmos_transfer2._src.predict2.text_encoders.text_encoder import EmbeddingConcatStrategy
from cosmos_transfer2._src.predict2_multiview.callbacks.every_n_draw_sample_multiviewvideo import (
    EveryNDrawSampleMultiviewVideo,
)

from cosmos_transfer2.experiments.custom.single_view_hdmap_dataset import (
    SingleViewHDMapDataset,
    collate_fn,
)
from cosmos_transfer2.experiments.custom.test_loss_callback import EveryNTestLoss


# Official Transfer2.5 multiview ControlNet checkpoint (has pre-trained hdmap head).
TRANSFER2_MULTIVIEW_CHECKPOINT = get_checkpoint_by_uuid("4ecc66e9-df19-4aed-9802-0d11e057287a")


# ============================================================================
# Configuration Constants - MODIFY THESE FOR YOUR SETUP
# ============================================================================

# Car-side single-view dataset directory
CARSIDE_DATASET_DIR = "/mnt/zihanw/car_side_bbox_videos"

# Output directory
OUTPUT_DIR = "/mnt/zihanw/Output_R2V_world_foundation_model_v1"

# Test scenes (same as the multi-control branch for consistency)
TEST_SCENE_IDS = ["031", "033", "053", "056", "076", "077", "088", "089"]

# Number of GPUs (single-view: only used for FSDP shard size; no context parallelism)
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 4))


# ============================================================================
# Dataset / Dataloader registration
# ============================================================================

def create_eval_dataset() -> SingleViewHDMapDataset:
    """Test set: only TEST_SCENE_IDS (excluding all train scenes)."""
    return SingleViewHDMapDataset(
        dataset_dir=CARSIDE_DATASET_DIR,
        resolution_hw=(720, 1280),
        num_video_frames=29,
        fps_downsample_factor=1,
        add_view_prefix_to_caption=True,
        # Construct the eval set as "everything except the train set" by listing
        # an empty exclude set on the *train* side. For eval we explicitly skip
        # all scenes that are NOT in TEST_SCENE_IDS by reading the directory.
        exclude_scene_ids=_compute_train_scene_ids(),
    )


def _compute_train_scene_ids() -> list:
    """Return scene IDs that should be excluded from the eval set, i.e. all
    scenes not in TEST_SCENE_IDS that are present on disk."""
    from pathlib import Path

    caption_folder = (
        Path(CARSIDE_DATASET_DIR)
        / "captions"
        / "ftheta_camera_front_wide_120fov"
    )
    scene_ids = set()
    if caption_folder.exists():
        for f in caption_folder.glob("*.json"):
            scene_ids.add(f.stem.split("_")[0])
    return sorted(scene_ids - set(TEST_SCENE_IDS))


def register_single_view_hdmap_dataloader() -> None:
    cs = ConfigStore.instance()
    dataset = L(SingleViewHDMapDataset)(
        dataset_dir=CARSIDE_DATASET_DIR,
        resolution_hw=(720, 1280),
        num_video_frames=29,
        fps_downsample_factor=1,
        add_view_prefix_to_caption=True,
        exclude_scene_ids=TEST_SCENE_IDS,
    )
    cs.store(
        group="data_train",
        package="dataloader_train",
        name="single_view_hdmap_train_data",
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
# Experiment configuration
# ============================================================================

# hint_keys="hdmap" -> control_input_hdmap_bbox (uses pre-trained head weights)
single_view_hdmap_post_train = dict(
    defaults=[
        {"override /data_train": "single_view_hdmap_train_data"},
        {"override /model": "fsdp_rectified_flow_multiview_control"},
        {"override /net": "cosmos_v1_2B_multiview_control"},
        # Default Transfer2.5 multiview control conditioner already wires
        # control_input_hdmap_bbox; no custom conditioner needed.
        {"override /conditioner": "video_prediction_multiview_control_conditioner"},
        {"override /ckpt_type": "dcp"},
        {"override /optimizer": "fusedadamw"},
        {"override /tokenizer": "wan2pt1_tokenizer"},
        {"override /callbacks": ["basic", "wandb", "cluster_speed"]},
        "_self_",
    ],
    job=dict(
        project="cosmos_transfer_custom",
        group="single_view_hdmap",
        name=f"2b_single_view_hdmap_{RUN_TIMESTAMP}",
    ),
    checkpoint=dict(
        save_iter=500,
        load_path=TRANSFER2_MULTIVIEW_CHECKPOINT.path,
        load_training_state=False,
        strict_resume=False,
        load_from_object_store=dict(enabled=False),
        save_to_object_store=dict(enabled=False),
    ),
    optimizer=dict(
        lr=1e-4,
        weight_decay=1e-3,
        betas=[0.9, 0.999],
    ),
    scheduler=dict(
        f_max=[1.0],
        f_min=[0.2],
        warm_up_steps=[200],
        cycle_lengths=[5000],
    ),
    model=dict(
        config=dict(
            # hdmap only -> control_input_hdmap_bbox; pre-trained head is loaded
            hint_keys="hdmap",
            min_num_conditional_frames_per_view=0,
            max_num_conditional_frames_per_view=0,
            condition_locations=["first_random_n"],
            # Single view
            train_sample_views_range=[1, 1],
            conditional_frames_probs={0: 1.0},
            state_t=8,
            online_text_embeddings_as_dict=False,
            # FSDP across all 4 GPUs (no context parallel for single view)
            fsdp_shard_size=WORLD_SIZE,
            resolution="720p",
            shift=5,
            use_dynamic_shift=False,
            train_time_weight="uniform",
            train_time_distribution="logitnormal",
            base_load_from=None,
            net=dict(
                timestep_scale=0.001,
                use_wan_fp32_strategy=True,
                concat_view_embedding=True,
                # Single-view embedding
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
        logging_iter=50,
        grad_accum_iter=4,
        max_iter=5000,
        callbacks=dict(
            heart_beat=dict(save_s3=False),
            iter_speed=dict(hit_thres=100, every_n=100, save_s3=False),
            device_monitor=dict(save_s3=False),
            grad_clip=dict(clip_norm=0.1),
            every_n_sample_ema=L(EveryNDrawSampleMultiviewVideo)(
                every_n=500,
                is_x0=False,
                is_ema=True,
                num_sampling_step=35,
                guidance=[7],
                fps=10,
                ctrl_hint_keys=["control_input_hdmap_bbox"],
                control_weights=[1.0],
                num_cond_frames=[0],
                save_s3=False,
            ),
            every_n_test_loss=L(EveryNTestLoss)(
                eval_dataset=create_eval_dataset(),
                every_n=200,
                num_timestep_samples=4,
                name="test_loss",
            ),
            wandb=dict(save_s3=False),
            wandb_10x=dict(save_s3=False),
            dataloader_speed=dict(save_s3=False),
            frame_loss_log=dict(save_s3=False),
        ),
    ),
    model_parallel=dict(
        # Single view -> no context parallel
        context_parallel_size=1,
    ),
    dataloader_train=dict(
        augmentation_config=dict(
            single_caption_camera_name="camera_front_wide_120fov",
            add_view_prefix_to_caption=True,
        ),
    ),
)


# Smaller / smoke configuration for quick sanity check (50 iterations).
single_view_hdmap_post_train_smoke = dict(
    defaults=[
        {"override /data_train": "single_view_hdmap_train_data"},
        {"override /model": "fsdp_rectified_flow_multiview_control"},
        {"override /net": "cosmos_v1_2B_multiview_control"},
        {"override /conditioner": "video_prediction_multiview_control_conditioner"},
        {"override /ckpt_type": "dcp"},
        {"override /optimizer": "fusedadamw"},
        {"override /tokenizer": "wan2pt1_tokenizer"},
        {"override /callbacks": ["basic", "wandb", "cluster_speed"]},
        "_self_",
    ],
    job=dict(
        project="cosmos_transfer_custom",
        group="single_view_hdmap",
        name=f"2b_single_view_hdmap_smoke_{RUN_TIMESTAMP}",
    ),
    checkpoint=dict(
        save_iter=50,
        load_path=TRANSFER2_MULTIVIEW_CHECKPOINT.path,
        load_training_state=False,
        strict_resume=False,
        load_from_object_store=dict(enabled=False),
        save_to_object_store=dict(enabled=False),
    ),
    optimizer=dict(lr=1e-4, weight_decay=1e-3, betas=[0.9, 0.999]),
    scheduler=dict(
        f_max=[1.0], f_min=[0.5], warm_up_steps=[20], cycle_lengths=[200]
    ),
    model=dict(
        config=dict(
            hint_keys="hdmap",
            min_num_conditional_frames_per_view=0,
            max_num_conditional_frames_per_view=0,
            condition_locations=["first_random_n"],
            train_sample_views_range=[1, 1],
            conditional_frames_probs={0: 1.0},
            state_t=8,
            online_text_embeddings_as_dict=False,
            fsdp_shard_size=WORLD_SIZE,
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
        grad_accum_iter=1,
        max_iter=50,
        callbacks=dict(
            heart_beat=dict(save_s3=False),
            iter_speed=dict(hit_thres=10, every_n=10, save_s3=False),
            device_monitor=dict(save_s3=False),
            grad_clip=dict(clip_norm=0.1),
            wandb=dict(save_s3=False),
            wandb_10x=dict(save_s3=False),
            dataloader_speed=dict(save_s3=False),
            frame_loss_log=dict(save_s3=False),
        ),
    ),
    model_parallel=dict(context_parallel_size=1),
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

for _item in [
    single_view_hdmap_post_train,
    single_view_hdmap_post_train_smoke,
]:
    experiment_name = [name.lower() for name, value in globals().items() if value is _item][0]
    cs.store(
        group="experiment",
        package="_global_",
        name=experiment_name,
        node=_item,
    )

register_single_view_hdmap_dataloader()
