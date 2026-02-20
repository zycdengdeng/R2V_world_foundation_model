# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Experiment Configuration for Dense (Roadside) Multi-Control Post Training
# Trains a model with 3 control types: hdmap, blur, depth
# Uses nested directory structure: {camera}/{scene}/{sample}.mp4
#
# Strategy: Use Transfer2.5 multiview checkpoint (from scratch)
# Channel order matters for pre-trained weight compatibility:
# - hdmap: channels 0-15, uses pre-trained hdmap_bbox weights from Transfer2.5
# - blur: channels 16-31, train from scratch (not in Transfer2.5)
# - depth: channels 32-47, train from scratch (not in Transfer2.5)

import copy
import os
from datetime import datetime

import torch
import torch.distributed as dist
from hydra.core.config_store import ConfigStore

# Generate timestamp for unique run directories
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

from cosmos_transfer2._src.imaginaire.flags import SMOKE
from cosmos_transfer2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_transfer2._src.imaginaire.lazy_config import LazyDict
from cosmos_transfer2._src.imaginaire.utils.checkpoint_db import get_checkpoint_by_uuid

from cosmos_transfer2._src.predict2.datasets.local_datasets.dataset_video import get_generic_dataloader, get_sampler
from cosmos_transfer2._src.predict2.text_encoders.text_encoder import EmbeddingConcatStrategy
from cosmos_transfer2._src.predict2.conditioner import ReMapkey
from cosmos_transfer2._src.predict2_multiview.callbacks.every_n_draw_sample_multiviewvideo import (
    EveryNDrawSampleMultiviewVideo,
)
from cosmos_transfer2._src.transfer2_multiview.configs.vid2vid_transfer.defaults.conditioner import (
    MultiViewControlVideo2WorldConditioner,
    _SHARED_CONFIG_AV,
)
from cosmos_transfer2._src.predict2_multiview.conditioner import MVTextAttr

from cosmos_transfer2.experiments.custom.roadside_multi_control_dataset import (
    RoadsideMultiControlDataset,
    collate_fn,
)
from cosmos_transfer2.experiments.custom.test_loss_callback import (
    EveryNTestLoss,
)

# Get the Transfer2.5 multiview checkpoint (optimized for control tasks)
# This checkpoint has pre-trained hdmap_bbox control head
TRANSFER2_MULTIVIEW_CHECKPOINT = get_checkpoint_by_uuid("4ecc66e9-df19-4aed-9802-0d11e057287a")


# ============================================================================
# Custom Conditioner Configuration (same as original)
# ============================================================================

_CUSTOM_MULTI_CONTROL_CONFIG = copy.deepcopy(_SHARED_CONFIG_AV)

# Add control_input_blur
_CUSTOM_MULTI_CONTROL_CONFIG["control_input_blur"] = L(ReMapkey)(
    input_key="control_input_blur",
    output_key="control_input_blur",
    dropout_rate=0.0,
    dtype=None,
)

# Add control_input_depth
_CUSTOM_MULTI_CONTROL_CONFIG["control_input_depth"] = L(ReMapkey)(
    input_key="control_input_depth",
    output_key="control_input_depth",
    dropout_rate=0.0,
    dtype=None,
)

# Add multiview-specific config
_CUSTOM_MULTI_CONTROL_CONFIG["view_indices_B_T"] = L(ReMapkey)(
    input_key="latent_view_indices_B_T",
    output_key="view_indices_B_T",
    dropout_rate=0.0,
    dtype=None,
)
_CUSTOM_MULTI_CONTROL_CONFIG["ref_cam_view_idx_sample_position"] = L(ReMapkey)(
    input_key="ref_cam_view_idx_sample_position",
    output_key="ref_cam_view_idx_sample_position",
    dropout_rate=0.0,
    dtype=None,
)

# Create the custom conditioner
DenseMultiControlConditioner: LazyDict = L(MultiViewControlVideo2WorldConditioner)(
    **_CUSTOM_MULTI_CONTROL_CONFIG,
)


# ============================================================================
# Configuration Constants
# ============================================================================

# Dataset paths for Dense/Roadside data (NESTED directory structure)
DENSE_BLUR_DATASET_DIR = "/mnt/zihanw/merge_dyn_stat/proj_utils_pro_Roadside_transfer/transfer_video_maker/output/BlurProjection"
DENSE_DEPTH_DATASET_DIR = "/mnt/zihanw/merge_dyn_stat/proj_utils_pro_Roadside_transfer/transfer_video_maker/output/DepthSparse"
DENSE_HDMAP_DATASET_DIR = "/mnt/zihanw/merge_dyn_stat/proj_utils_pro_Roadside_transfer/transfer_video_maker/output/HDMapBbox"

# Output directory
OUTPUT_DIR = "/mnt/zihanw/Output_R2V_world_foundation_model_v2"

# Test scene IDs (same as original - 8 scenes)
TEST_SCENE_IDS = ["031", "033", "053", "056", "076", "077", "088", "089"]

# Training scene IDs (all other scenes)
TRAIN_SCENE_IDS = [
    "001", "002", "003", "004", "006", "007", "008", "009", "010", "012", "013", "014", "015", "016", "017",
    "019", "020", "021", "022", "024", "025", "026", "027", "028", "029", "030", "032", "034", "035", "036",
    "037", "038", "039", "040", "041", "042", "043", "044", "045", "046", "047", "048", "049", "050", "051",
    "052", "054", "055", "057", "058", "059", "060", "061", "062", "063", "064", "065", "066", "067", "068",
    "069", "070", "072", "073", "074", "075", "078", "079", "080", "081", "083", "084", "085", "086", "087",
]

# Number of GPUs (for context parallel)
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 8))

# Training cameras (7 views for 8 GPUs)
TRAINING_CAMERAS = (
    "camera_front_wide_120fov",      # 前视广角 120°
    "camera_cross_right_120fov",     # 右侧 120°
    "camera_rear_right_70fov",       # 右后 70°
    "camera_rear_tele_30fov",        # 后视长焦 30°
    "camera_rear_left_70fov",        # 左后 70°
    "camera_cross_left_120fov",      # 左侧 120°
    "camera_front_tele_30fov",       # 前视长焦 30°
)


# ============================================================================
# Dataset Configuration
# ============================================================================

def create_eval_dataset():
    """Create evaluation dataset with only test scenes."""
    return RoadsideMultiControlDataset(
        blur_dataset_dir=DENSE_BLUR_DATASET_DIR,
        depth_dataset_dir=DENSE_DEPTH_DATASET_DIR,
        hdmap_dataset_dir=DENSE_HDMAP_DATASET_DIR,
        resolution_hw=(720, 1280),
        num_video_frames=29,
        fps_downsample_factor=1,
        camera_keys=TRAINING_CAMERAS if not SMOKE else TRAINING_CAMERAS[:1],
        single_caption_camera_name="camera_front_wide_120fov",
        add_view_prefix_to_caption=True,
        exclude_scene_ids=TRAIN_SCENE_IDS,  # Exclude training scenes = keep only test scenes
    )


def register_dense_dataloader() -> None:
    """Register dataloader with Roadside multi-control dataset."""
    cs = ConfigStore.instance()

    # Create training dataset (excludes test scenes)
    dataset = L(RoadsideMultiControlDataset)(
        blur_dataset_dir=DENSE_BLUR_DATASET_DIR,
        depth_dataset_dir=DENSE_DEPTH_DATASET_DIR,
        hdmap_dataset_dir=DENSE_HDMAP_DATASET_DIR,
        resolution_hw=(720, 1280),
        num_video_frames=29,
        fps_downsample_factor=1,
        camera_keys=TRAINING_CAMERAS if not SMOKE else TRAINING_CAMERAS[:1],
        single_caption_camera_name="camera_front_wide_120fov",
        add_view_prefix_to_caption=True,
        exclude_scene_ids=TEST_SCENE_IDS,
    )

    # Register dataloader
    cs.store(
        group="data_train",
        package="dataloader_train",
        name="dense_multi_control_train_data",
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

dense_multi_control_post_train = dict(
    defaults=[
        {"override /data_train": "dense_multi_control_train_data"},
        {"override /model": "fsdp_rectified_flow_multiview_control"},
        {"override /net": "cosmos_v1_2B_multiview_control"},
        {"override /conditioner": "dense_multi_control_conditioner"},
        {"override /ckpt_type": "dcp"},
        {"override /optimizer": "fusedadamw"},
        {"override /tokenizer": "wan2pt1_tokenizer"},
        {"override /callbacks": ["basic", "wandb", "cluster_speed"]},
        "_self_",
    ],
    job=dict(
        project="cosmos_transfer_dense",
        group="multi_control",
        name=f"2b_dense_multi_control_{RUN_TIMESTAMP}",
    ),
    checkpoint=dict(
        save_iter=200,
        # Start from Transfer2.5 checkpoint (from scratch)
        load_path=TRANSFER2_MULTIVIEW_CHECKPOINT.path,
        load_training_state=False,  # Don't load optimizer state - training from scratch
        strict_resume=False,
        load_from_object_store=dict(enabled=False),
        save_to_object_store=dict(enabled=False),
    ),
    optimizer=dict(
        lr=1e-4,  # Standard learning rate for training from scratch
        weight_decay=1e-3,
        betas=[0.9, 0.999],
    ),
    scheduler=dict(
        f_max=[1.0],  # Peak lr = 1e-4
        f_min=[0.1],  # Final lr = 1e-5
        warm_up_steps=[500],  # Warmup: 0-500
        cycle_lengths=[8000],  # Total 8000 iterations
    ),
    model=dict(
        config=dict(
            # 3 control heads: hdmap (pre-trained), blur (scratch), depth (scratch)
            hint_keys="hdmap_blur_depth",
            # NO condition frames - pure control-based generation
            min_num_conditional_frames_per_view=0,
            max_num_conditional_frames_per_view=0,
            condition_locations=["first_random_n"],
            train_sample_views_range=[7, 7],
            conditional_frames_probs={0: 1.0},
            state_t=8,
            online_text_embeddings_as_dict=False,
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
        max_iter=8000,
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
                ctrl_hint_keys=["control_input_hdmap_bbox", "control_input_blur", "control_input_depth"],
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

# Register the custom conditioner
cs.store(
    group="conditioner",
    package="model.config.conditioner",
    name="dense_multi_control_conditioner",
    node=DenseMultiControlConditioner,
)

# Register the experiment configuration
for _item in [dense_multi_control_post_train]:
    experiment_name = [name.lower() for name, value in globals().items() if value is _item][0]
    cs.store(
        group="experiment",
        package="_global_",
        name=experiment_name,
        node=_item,
    )

# Register the dataloader
register_dense_dataloader()
