# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Ablation Experiment Configuration for Single-View Control Head Training
# Each experiment trains a single camera view (front_wide) with different control combinations
# Designed to run on a single GPU for efficient ablation studies
#
# 7 Experiments:
#   1. hdmap only
#   2. blur only
#   3. depth only
#   4. hdmap + blur
#   5. hdmap + depth
#   6. blur + depth
#   7. hdmap + blur + depth (baseline)

import copy
import os
from datetime import datetime

import torch
import torch.distributed as dist
from hydra.core.config_store import ConfigStore

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

from cosmos_transfer2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_transfer2._src.imaginaire.lazy_config import LazyDict
from cosmos_transfer2._src.imaginaire.utils.checkpoint_db import get_checkpoint_by_uuid

from cosmos_transfer2._src.predict2.datasets.local_datasets.dataset_video import get_generic_dataloader, get_sampler
from cosmos_transfer2._src.predict2.text_encoders.text_encoder import EmbeddingConcatStrategy
from cosmos_transfer2._src.predict2.conditioner import ReMapkey
from cosmos_transfer2._src.transfer2_multiview.configs.vid2vid_transfer.defaults.conditioner import (
    MultiViewControlVideo2WorldConditioner,
    _SHARED_CONFIG_AV,
)

from cosmos_transfer2.experiments.custom.custom_multi_control_dataset import (
    MultiControlMultiviewDataset,
    collate_fn,
)

# Transfer2.5 multiview checkpoint
TRANSFER2_MULTIVIEW_CHECKPOINT = get_checkpoint_by_uuid("4ecc66e9-df19-4aed-9802-0d11e057287a")


# ============================================================================
# Dataset Configuration - Single View (Front Wide only)
# ============================================================================

BLUR_DATASET_DIR = "/mnt/zihanw/proj_utils_pro/transfer_video_maker/output_full_data/BlurProjection"
DEPTH_DATASET_DIR = "/mnt/zihanw/proj_utils_pro/transfer_video_maker/output_full_data/DepthSparse"
HDMAP_DATASET_DIR = "/mnt/zihanw/proj_utils_pro/transfer_video_maker/output_full_data/HDMapBbox"

OUTPUT_DIR = "/mnt/zihanw/Output_R2V_ablation"

TEST_SCENE_IDS = ["031", "033", "053", "056", "076", "077", "088", "089"]
TRAIN_SCENE_IDS = [
    "001", "002", "003", "004", "006", "007", "008", "009", "010", "012", "013", "014", "015", "016", "017",
    "019", "020", "021", "022", "024", "025", "026", "027", "028", "029", "030", "032", "034", "035", "036",
    "037", "038", "039", "040", "041", "042", "043", "044", "045", "046", "047", "048", "049", "050", "051",
    "052", "054", "055", "057", "058", "059", "060", "061", "062", "063", "064", "065", "066", "067", "068",
    "069", "070", "072", "073", "074", "075", "078", "079", "080", "081", "083", "084", "085", "086", "087",
]

# Single view: Front Wide only
SINGLE_VIEW_CAMERA = ("camera_front_wide_120fov",)


# ============================================================================
# Conditioner Configurations for Different Control Combinations
# ============================================================================

def create_conditioner_config(control_types: list):
    """Create conditioner config for specified control types.

    Args:
        control_types: List of control types, e.g., ["hdmap", "blur", "depth"]
    """
    config = copy.deepcopy(_SHARED_CONFIG_AV)

    # Remove default hdmap_bbox if not needed
    if "hdmap" not in control_types and "control_input_hdmap_bbox" in config:
        config.pop("control_input_hdmap_bbox")

    # Add blur if needed
    if "blur" in control_types:
        config["control_input_blur"] = L(ReMapkey)(
            input_key="control_input_blur",
            output_key="control_input_blur",
            dropout_rate=0.0,
            dtype=None,
        )

    # Add depth if needed
    if "depth" in control_types:
        config["control_input_depth"] = L(ReMapkey)(
            input_key="control_input_depth",
            output_key="control_input_depth",
            dropout_rate=0.0,
            dtype=None,
        )

    # Add hdmap if needed (already in _SHARED_CONFIG_AV, but ensure it's there)
    if "hdmap" in control_types and "control_input_hdmap_bbox" not in config:
        config["control_input_hdmap_bbox"] = L(ReMapkey)(
            input_key="control_input_hdmap_bbox",
            output_key="control_input_hdmap_bbox",
            dropout_rate=0.0,
            dtype=None,
        )

    # Add multiview-specific config (still needed even for single view)
    config["view_indices_B_T"] = L(ReMapkey)(
        input_key="latent_view_indices_B_T",
        output_key="view_indices_B_T",
        dropout_rate=0.0,
        dtype=None,
    )
    config["ref_cam_view_idx_sample_position"] = L(ReMapkey)(
        input_key="ref_cam_view_idx_sample_position",
        output_key="ref_cam_view_idx_sample_position",
        dropout_rate=0.0,
        dtype=None,
    )

    return L(MultiViewControlVideo2WorldConditioner)(**config)


# Create conditioners for all 7 combinations
CONDITIONER_HDMAP_ONLY = create_conditioner_config(["hdmap"])
CONDITIONER_BLUR_ONLY = create_conditioner_config(["blur"])
CONDITIONER_DEPTH_ONLY = create_conditioner_config(["depth"])
CONDITIONER_HDMAP_BLUR = create_conditioner_config(["hdmap", "blur"])
CONDITIONER_HDMAP_DEPTH = create_conditioner_config(["hdmap", "depth"])
CONDITIONER_BLUR_DEPTH = create_conditioner_config(["blur", "depth"])
CONDITIONER_ALL = create_conditioner_config(["hdmap", "blur", "depth"])


# ============================================================================
# Dataset Registration - Single View
# ============================================================================

def create_single_view_eval_dataset():
    """Create evaluation dataset with single view."""
    return MultiControlMultiviewDataset(
        blur_dataset_dir=BLUR_DATASET_DIR,
        depth_dataset_dir=DEPTH_DATASET_DIR,
        hdmap_dataset_dir=HDMAP_DATASET_DIR,
        resolution_hw=(720, 1280),
        num_video_frames=29,
        fps_downsample_factor=1,
        camera_keys=SINGLE_VIEW_CAMERA,
        single_caption_camera_name="camera_front_wide_120fov",
        add_view_prefix_to_caption=True,
        exclude_scene_ids=TRAIN_SCENE_IDS,
    )


def register_single_view_dataloader() -> None:
    """Register single-view dataloader."""
    cs = ConfigStore.instance()

    dataset = L(MultiControlMultiviewDataset)(
        blur_dataset_dir=BLUR_DATASET_DIR,
        depth_dataset_dir=DEPTH_DATASET_DIR,
        hdmap_dataset_dir=HDMAP_DATASET_DIR,
        resolution_hw=(720, 1280),
        num_video_frames=29,
        fps_downsample_factor=1,
        camera_keys=SINGLE_VIEW_CAMERA,
        single_caption_camera_name="camera_front_wide_120fov",
        add_view_prefix_to_caption=True,
        exclude_scene_ids=TEST_SCENE_IDS,
    )

    cs.store(
        group="data_train",
        package="dataloader_train",
        name="ablation_single_view_train_data",
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
# Base Experiment Configuration Template
# ============================================================================

def create_ablation_experiment(
    experiment_name: str,
    hint_keys: str,
    conditioner_name: str,
):
    """Create ablation experiment configuration.

    Args:
        experiment_name: Name suffix for the experiment
        hint_keys: Control head configuration (e.g., "hdmap", "blur_depth")
        conditioner_name: Registered conditioner name
    """
    return dict(
        defaults=[
            {"override /data_train": "ablation_single_view_train_data"},
            {"override /model": "fsdp_rectified_flow_multiview_control"},
            {"override /net": "cosmos_v1_2B_multiview_control"},
            {"override /conditioner": conditioner_name},
            {"override /ckpt_type": "dcp"},
            {"override /optimizer": "fusedadamw"},
            {"override /tokenizer": "wan2pt1_tokenizer"},
            {"override /callbacks": ["basic", "wandb", "cluster_speed"]},
            "_self_",
        ],
        job=dict(
            project="cosmos_ablation",
            group="single_view",
            name=f"ablation_{experiment_name}_{RUN_TIMESTAMP}",
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
            f_min=[0.1],
            warm_up_steps=[200],
            cycle_lengths=[5000],
        ),
        model=dict(
            config=dict(
                hint_keys=hint_keys,
                min_num_conditional_frames_per_view=0,
                max_num_conditional_frames_per_view=0,
                condition_locations=["first_random_n"],
                train_sample_views_range=[1, 1],  # Single view
                conditional_frames_probs={0: 1.0},
                state_t=8,
                online_text_embeddings_as_dict=False,
                fsdp_shard_size=1,  # Single GPU
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
                    view_condition_dim=1,  # Single view
                    state_t=8,
                    n_cameras_emb=7,  # Keep 7 for compatibility
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
                wandb=dict(save_s3=False),
                wandb_10x=dict(save_s3=False),
                dataloader_speed=dict(save_s3=False),
                frame_loss_log=dict(save_s3=False),
            ),
        ),
        model_parallel=dict(
            context_parallel_size=1,  # Single GPU
        ),
        dataloader_train=dict(
            augmentation_config=dict(
                single_caption_camera_name="camera_front_wide_120fov",
                add_view_prefix_to_caption=True,
            ),
        ),
    )


# ============================================================================
# Create All 7 Experiment Configurations
# ============================================================================

# 1. HDMap only
ablation_hdmap_only = create_ablation_experiment(
    experiment_name="hdmap_only",
    hint_keys="hdmap",
    conditioner_name="ablation_conditioner_hdmap_only",
)

# 2. Blur only
ablation_blur_only = create_ablation_experiment(
    experiment_name="blur_only",
    hint_keys="blur",
    conditioner_name="ablation_conditioner_blur_only",
)

# 3. Depth only
ablation_depth_only = create_ablation_experiment(
    experiment_name="depth_only",
    hint_keys="depth",
    conditioner_name="ablation_conditioner_depth_only",
)

# 4. HDMap + Blur
ablation_hdmap_blur = create_ablation_experiment(
    experiment_name="hdmap_blur",
    hint_keys="hdmap_blur",
    conditioner_name="ablation_conditioner_hdmap_blur",
)

# 5. HDMap + Depth
ablation_hdmap_depth = create_ablation_experiment(
    experiment_name="hdmap_depth",
    hint_keys="hdmap_depth",
    conditioner_name="ablation_conditioner_hdmap_depth",
)

# 6. Blur + Depth
ablation_blur_depth = create_ablation_experiment(
    experiment_name="blur_depth",
    hint_keys="blur_depth",
    conditioner_name="ablation_conditioner_blur_depth",
)

# 7. All controls (baseline)
ablation_all_controls = create_ablation_experiment(
    experiment_name="all_controls",
    hint_keys="hdmap_blur_depth",
    conditioner_name="ablation_conditioner_all",
)


# ============================================================================
# Registration
# ============================================================================

cs = ConfigStore.instance()

# Register conditioners
cs.store(group="conditioner", package="model.config.conditioner",
         name="ablation_conditioner_hdmap_only", node=CONDITIONER_HDMAP_ONLY)
cs.store(group="conditioner", package="model.config.conditioner",
         name="ablation_conditioner_blur_only", node=CONDITIONER_BLUR_ONLY)
cs.store(group="conditioner", package="model.config.conditioner",
         name="ablation_conditioner_depth_only", node=CONDITIONER_DEPTH_ONLY)
cs.store(group="conditioner", package="model.config.conditioner",
         name="ablation_conditioner_hdmap_blur", node=CONDITIONER_HDMAP_BLUR)
cs.store(group="conditioner", package="model.config.conditioner",
         name="ablation_conditioner_hdmap_depth", node=CONDITIONER_HDMAP_DEPTH)
cs.store(group="conditioner", package="model.config.conditioner",
         name="ablation_conditioner_blur_depth", node=CONDITIONER_BLUR_DEPTH)
cs.store(group="conditioner", package="model.config.conditioner",
         name="ablation_conditioner_all", node=CONDITIONER_ALL)

# Register experiments
for _item in [
    ablation_hdmap_only,
    ablation_blur_only,
    ablation_depth_only,
    ablation_hdmap_blur,
    ablation_hdmap_depth,
    ablation_blur_depth,
    ablation_all_controls,
]:
    experiment_name = [name.lower() for name, value in globals().items() if value is _item][0]
    cs.store(
        group="experiment",
        package="_global_",
        name=experiment_name,
        node=_item,
    )

# Register dataloader
register_single_view_dataloader()
