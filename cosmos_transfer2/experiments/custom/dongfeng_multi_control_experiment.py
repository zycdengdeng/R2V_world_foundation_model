# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Experiment configuration for multi-control post training on the Dongfeng ego-view data.
#
# This is a parallel setup to custom_multi_control_experiment.py and does not modify it:
# - new dataset paths under /mnt2/dongfeng_ego_output/transfer2
# - Dongfeng sample naming ({drive}_seg{NNN}) handled by DongfengMultiControlDataset
# - training starts FRESH from the official Transfer2.5 multiview checkpoint
#   (pre-trained hdmap_bbox control head), NOT from any previous custom run
#
# Channel order (must match the pre-trained checkpoint):
# - hdmap: channels 0-15, uses pre-trained hdmap_bbox weights from Transfer2.5
# - blur:  channels 16-31, train from scratch
# - depth: channels 32-47, train from scratch
#
# Train/test split: by scene (drive). By default 2 evenly spaced scenes (from the sorted
# scene list) are held out as the test set; override with the DONGFENG_TEST_SCENES env var
# (comma-separated scene IDs).

import copy
import os
from datetime import datetime
from pathlib import Path

import torch.distributed as dist
from hydra.core.config_store import ConfigStore

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

from cosmos_transfer2.experiments.custom.custom_multi_control_dataset import collate_fn
from cosmos_transfer2.experiments.custom.dongfeng_multi_control_dataset import (
    DongfengMultiControlDataset,
    extract_dongfeng_scene_id,
)
from cosmos_transfer2.experiments.custom.test_loss_callback import EveryNTestLoss
from cosmos_transfer2.experiments.custom.text_embedding_cache_callback import CacheTextEmbeddings

# Official Transfer2.5 multiview checkpoint (pre-trained hdmap_bbox control head)
TRANSFER2_MULTIVIEW_CHECKPOINT = get_checkpoint_by_uuid("4ecc66e9-df19-4aed-9802-0d11e057287a")


# ============================================================================
# Conditioner (same construction as custom_multi_control_experiment, registered
# under its own name to keep this module self-contained)
# ============================================================================

_DONGFENG_MULTI_CONTROL_CONFIG = copy.deepcopy(_SHARED_CONFIG_AV)
_DONGFENG_MULTI_CONTROL_CONFIG["control_input_blur"] = L(ReMapkey)(
    input_key="control_input_blur",
    output_key="control_input_blur",
    dropout_rate=0.0,
    dtype=None,
)
_DONGFENG_MULTI_CONTROL_CONFIG["control_input_depth"] = L(ReMapkey)(
    input_key="control_input_depth",
    output_key="control_input_depth",
    dropout_rate=0.0,
    dtype=None,
)
_DONGFENG_MULTI_CONTROL_CONFIG["view_indices_B_T"] = L(ReMapkey)(
    input_key="latent_view_indices_B_T",
    output_key="view_indices_B_T",
    dropout_rate=0.0,
    dtype=None,
)
_DONGFENG_MULTI_CONTROL_CONFIG["ref_cam_view_idx_sample_position"] = L(ReMapkey)(
    input_key="ref_cam_view_idx_sample_position",
    output_key="ref_cam_view_idx_sample_position",
    dropout_rate=0.0,
    dtype=None,
)

DongfengMultiControlConditioner: LazyDict = L(MultiViewControlVideo2WorldConditioner)(
    **_DONGFENG_MULTI_CONTROL_CONFIG,
)


# ============================================================================
# Dataset paths and train/test split
# ============================================================================

DONGFENG_ROOT = "/mnt2/dongfeng_ego_output/transfer2"
BLUR_DATASET_DIR = f"{DONGFENG_ROOT}/BlurProjection"
DEPTH_DATASET_DIR = f"{DONGFENG_ROOT}/DepthSparse"
HDMAP_DATASET_DIR = f"{DONGFENG_ROOT}/HDMapBbox"

# How many scenes to hold out when DONGFENG_TEST_SCENES is not set
NUM_AUTO_TEST_SCENES = 2

WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 8))

TRAINING_CAMERAS = (
    "camera_front_wide_120fov",
    "camera_cross_right_120fov",
    "camera_rear_right_70fov",
    "camera_rear_tele_30fov",
    "camera_rear_left_70fov",
    "camera_cross_left_120fov",
    "camera_front_tele_30fov",
)


def _scan_scene_ids() -> list:
    """List unique scene IDs by scanning front-wide captions. Returns [] if data is absent."""
    caption_dir = Path(BLUR_DATASET_DIR) / "captions" / "ftheta_camera_front_wide_120fov"
    if not caption_dir.exists():
        return []
    scenes = {extract_dongfeng_scene_id(p.stem) for p in caption_dir.glob("*.json")}
    return sorted(scenes)


_ALL_SCENE_IDS = _scan_scene_ids()
_env_test = os.environ.get("DONGFENG_TEST_SCENES", "").strip()
if _env_test:
    TEST_SCENE_IDS = [s.strip() for s in _env_test.split(",") if s.strip()]
elif _ALL_SCENE_IDS:
    # Pick evenly spaced scenes from the sorted list, e.g. for 33 scenes and k=2
    # this selects indices 11 and 22 (interior picks, never the first/last scene).
    _n = len(_ALL_SCENE_IDS)
    _k = min(NUM_AUTO_TEST_SCENES, _n)
    TEST_SCENE_IDS = [_ALL_SCENE_IDS[(i + 1) * _n // (_k + 1)] for i in range(_k)]
else:
    TEST_SCENE_IDS = []
TRAIN_SCENE_IDS = [s for s in _ALL_SCENE_IDS if s not in set(TEST_SCENE_IDS)]

if _ALL_SCENE_IDS:
    print(f"[dongfeng_multi_control] {len(_ALL_SCENE_IDS)} scenes total; "
          f"{len(TRAIN_SCENE_IDS)} train / {len(TEST_SCENE_IDS)} test. "
          f"Test scenes: {TEST_SCENE_IDS}")


def create_eval_dataset():
    """Evaluation dataset holding only the test scenes."""
    return DongfengMultiControlDataset(
        blur_dataset_dir=BLUR_DATASET_DIR,
        depth_dataset_dir=DEPTH_DATASET_DIR,
        hdmap_dataset_dir=HDMAP_DATASET_DIR,
        resolution_hw=(720, 1280),
        num_video_frames=29,
        fps_downsample_factor=1,
        camera_keys=TRAINING_CAMERAS if not SMOKE else TRAINING_CAMERAS[:1],
        single_caption_camera_name="camera_front_wide_120fov",
        add_view_prefix_to_caption=True,
        exclude_scene_ids=TRAIN_SCENE_IDS,
    )


def register_dongfeng_dataloader() -> None:
    cs = ConfigStore.instance()
    dataset = L(DongfengMultiControlDataset)(
        blur_dataset_dir=BLUR_DATASET_DIR,
        depth_dataset_dir=DEPTH_DATASET_DIR,
        hdmap_dataset_dir=HDMAP_DATASET_DIR,
        resolution_hw=(720, 1280),
        num_video_frames=29,
        fps_downsample_factor=1,
        camera_keys=TRAINING_CAMERAS if not SMOKE else TRAINING_CAMERAS[:1],
        single_caption_camera_name="camera_front_wide_120fov",
        add_view_prefix_to_caption=True,
        exclude_scene_ids=TEST_SCENE_IDS,
    )
    cs.store(
        group="data_train",
        package="dataloader_train",
        name="dongfeng_multi_control_train_data",
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
# Experiment configuration: fresh post-training from the official checkpoint
# ============================================================================

def _build_trainer_callbacks() -> dict:
    callbacks = dict(
        heart_beat=dict(save_s3=False),
        # Captions are (near-)identical across the dataset; skip redundant 7B
        # text-encoder forwards by caching embeddings per caption set.
        cache_text_embeddings=L(CacheTextEmbeddings)(store_on_gpu=False),
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
        wandb=dict(save_s3=False),
        wandb_10x=dict(save_s3=False),
        dataloader_speed=dict(save_s3=False),
        frame_loss_log=dict(save_s3=False),
    )
    # Test-loss callback needs an eagerly built dataset; only add it when the data
    # (and a non-empty test split) is actually present, so importing this module on
    # machines without /mnt2 does not fail.
    if _ALL_SCENE_IDS and TEST_SCENE_IDS:
        callbacks["every_n_test_loss"] = L(EveryNTestLoss)(
            eval_dataset=create_eval_dataset(),
            every_n=200,
            num_timestep_samples=4,
            name="test_loss",
        )
    return callbacks


dongfeng_multi_control_post_train = dict(
    defaults=[
        {"override /data_train": "dongfeng_multi_control_train_data"},
        {"override /model": "fsdp_rectified_flow_multiview_control"},
        {"override /net": "cosmos_v1_2B_multiview_control"},
        {"override /conditioner": "dongfeng_multi_control_conditioner"},
        {"override /ckpt_type": "dcp"},
        {"override /optimizer": "fusedadamw"},
        {"override /tokenizer": "wan2pt1_tokenizer"},
        {"override /callbacks": ["basic", "wandb", "cluster_speed"]},
        "_self_",
    ],
    job=dict(
        project="cosmos_transfer_custom",
        group="dongfeng",
        name=f"2b_dongfeng_multi_control_{RUN_TIMESTAMP}",
    ),
    checkpoint=dict(
        save_iter=500,
        # Fresh start from the OFFICIAL Transfer2.5 multiview checkpoint
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
        f_max=[0.5],   # peak lr ~= 4.3e-5
        f_min=[0.2],   # final lr ~= 1.7e-5
        warm_up_steps=[500],
        cycle_lengths=[8000],
    ),
    model=dict(
        config=dict(
            # hdmap must be first to match pre-trained checkpoint weights at channels 0-15
            hint_keys="hdmap_blur_depth",
            # Pure control-based generation: no conditional frames
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
        grad_accum_iter=2,
        max_iter=8000,
        callbacks=_build_trainer_callbacks(),
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

cs.store(
    group="conditioner",
    package="model.config.conditioner",
    name="dongfeng_multi_control_conditioner",
    node=DongfengMultiControlConditioner,
)

cs.store(
    group="experiment",
    package="_global_",
    name="dongfeng_multi_control_post_train",
    node=dongfeng_multi_control_post_train,
)

register_dongfeng_dataloader()
