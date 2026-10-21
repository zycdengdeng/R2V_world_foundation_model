# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Small-scale post-training on the Tianjin (roadside intersection) data.
#
# Parallel to custom_multi_control_experiment.py, with a reduced TRAIN subset:
# - TEST scenes are IDENTICAL to the full run (031 033 053 056 076 077 088 089),
#   so metrics are directly comparable with the full-data model
# - training uses NUM_SMALL_TRAIN_SCENES evenly spaced scenes out of the 75
#   full-run train scenes (override with TIANJIN_TRAIN_SCENES="001,005,...")
# - fresh start from the official Transfer2.5 multiview checkpoint
# - ~12h schedule (400 iters, grad_accum 2, ~27 epochs over 10 scenes) with the
#   OOM fixes and the text-embedding cache (per-scene captions -> larger cache)

import copy
import os
from datetime import datetime

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

from cosmos_transfer2.experiments.custom.custom_multi_control_dataset import (
    MultiControlMultiviewDataset,
    collate_fn,
)
from cosmos_transfer2.experiments.custom.custom_multi_control_experiment import (
    BLUR_DATASET_DIR,
    DEPTH_DATASET_DIR,
    HDMAP_DATASET_DIR,
    TEST_SCENE_IDS,
    TRAIN_SCENE_IDS,
    TRAINING_CAMERAS,
)
from cosmos_transfer2.experiments.custom.test_loss_callback import EveryNTestLoss
from cosmos_transfer2.experiments.custom.text_embedding_cache_callback import CacheTextEmbeddings

TRANSFER2_MULTIVIEW_CHECKPOINT = get_checkpoint_by_uuid("4ecc66e9-df19-4aed-9802-0d11e057287a")

WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 8))

# ============================================================================
# Reduced train subset (test set unchanged)
# ============================================================================

NUM_SMALL_TRAIN_SCENES = 10

_env_scenes = os.environ.get("TIANJIN_TRAIN_SCENES", "").strip()
if _env_scenes:
    SMALL_TRAIN_SCENE_IDS = [s.strip() for s in _env_scenes.split(",") if s.strip()]
else:
    _n = len(TRAIN_SCENE_IDS)
    _k = min(NUM_SMALL_TRAIN_SCENES, _n)
    # Evenly spaced picks across the sorted full train list
    SMALL_TRAIN_SCENE_IDS = [TRAIN_SCENE_IDS[i * _n // _k] for i in range(_k)]

# The dataset filters by exclusion: exclude test scenes AND unused train scenes
_EXCLUDE_FOR_SMALL_TRAIN = sorted(
    set(TEST_SCENE_IDS) | (set(TRAIN_SCENE_IDS) - set(SMALL_TRAIN_SCENE_IDS))
)

print(f"[tianjin_small] Train subset: {len(SMALL_TRAIN_SCENE_IDS)}/{len(TRAIN_SCENE_IDS)} scenes: "
      f"{SMALL_TRAIN_SCENE_IDS}")
print(f"[tianjin_small] Test scenes (same as full run): {TEST_SCENE_IDS}")

# In-run test-loss callback uses a 2-scene subset (evenly spaced from the 8)
# to keep each evaluation ~6 min; FINAL evaluation still uses all 8 test scenes.
TEST_LOSS_SCENE_IDS = [TEST_SCENE_IDS[(i + 1) * len(TEST_SCENE_IDS) // 3] for i in range(2)]
print(f"[tianjin_small] Test-loss callback subset: {TEST_LOSS_SCENE_IDS}")


# ============================================================================
# Conditioner (same construction, own registration name)
# ============================================================================

_TIANJIN_SMALL_CONFIG = copy.deepcopy(_SHARED_CONFIG_AV)
_TIANJIN_SMALL_CONFIG["control_input_blur"] = L(ReMapkey)(
    input_key="control_input_blur", output_key="control_input_blur", dropout_rate=0.0, dtype=None,
)
_TIANJIN_SMALL_CONFIG["control_input_depth"] = L(ReMapkey)(
    input_key="control_input_depth", output_key="control_input_depth", dropout_rate=0.0, dtype=None,
)
_TIANJIN_SMALL_CONFIG["view_indices_B_T"] = L(ReMapkey)(
    input_key="latent_view_indices_B_T", output_key="view_indices_B_T", dropout_rate=0.0, dtype=None,
)
_TIANJIN_SMALL_CONFIG["ref_cam_view_idx_sample_position"] = L(ReMapkey)(
    input_key="ref_cam_view_idx_sample_position", output_key="ref_cam_view_idx_sample_position",
    dropout_rate=0.0, dtype=None,
)

TianjinSmallConditioner: LazyDict = L(MultiViewControlVideo2WorldConditioner)(**_TIANJIN_SMALL_CONFIG)


# ============================================================================
# Datasets
# ============================================================================

def create_eval_dataset():
    """Test-loss subset (2 of the 8 test scenes); final eval uses all 8."""
    return MultiControlMultiviewDataset(
        blur_dataset_dir=BLUR_DATASET_DIR,
        depth_dataset_dir=DEPTH_DATASET_DIR,
        hdmap_dataset_dir=HDMAP_DATASET_DIR,
        resolution_hw=(720, 1280),
        num_video_frames=29,
        fps_downsample_factor=1,
        camera_keys=TRAINING_CAMERAS if not SMOKE else TRAINING_CAMERAS[:1],
        single_caption_camera_name="camera_front_wide_120fov",
        add_view_prefix_to_caption=True,
        exclude_scene_ids=sorted(
            set(TRAIN_SCENE_IDS) | (set(TEST_SCENE_IDS) - set(TEST_LOSS_SCENE_IDS))
        ),
    )


def register_tianjin_small_dataloader() -> None:
    cs = ConfigStore.instance()
    dataset = L(MultiControlMultiviewDataset)(
        blur_dataset_dir=BLUR_DATASET_DIR,
        depth_dataset_dir=DEPTH_DATASET_DIR,
        hdmap_dataset_dir=HDMAP_DATASET_DIR,
        resolution_hw=(720, 1280),
        num_video_frames=29,
        fps_downsample_factor=1,
        camera_keys=TRAINING_CAMERAS if not SMOKE else TRAINING_CAMERAS[:1],
        single_caption_camera_name="camera_front_wide_120fov",
        add_view_prefix_to_caption=True,
        exclude_scene_ids=_EXCLUDE_FOR_SMALL_TRAIN,
    )
    cs.store(
        group="data_train",
        package="dataloader_train",
        name="tianjin_small_train_data",
        node=L(get_generic_dataloader)(
            dataset=dataset,
            sampler=L(get_sampler)(dataset=dataset) if dist.is_initialized() else None,
            collate_fn=collate_fn,
            batch_size=1,
            drop_last=True,
            num_workers=2,
            persistent_workers=True,
            pin_memory=True,
        ),
    )


# ============================================================================
# Experiment: short fresh post-training on the reduced subset
# ============================================================================

tianjin_multi_control_small = dict(
    defaults=[
        {"override /data_train": "tianjin_small_train_data"},
        {"override /model": "fsdp_rectified_flow_multiview_control"},
        {"override /net": "cosmos_v1_2B_multiview_control"},
        {"override /conditioner": "tianjin_small_conditioner"},
        {"override /ckpt_type": "dcp"},
        {"override /optimizer": "fusedadamw"},
        {"override /tokenizer": "wan2pt1_tokenizer"},
        {"override /callbacks": ["basic", "wandb", "cluster_speed"]},
        "_self_",
    ],
    job=dict(
        project="cosmos_transfer_custom",
        group="tianjin_small",
        name=f"2b_tianjin_small_{RUN_TIMESTAMP}",
    ),
    checkpoint=dict(
        save_iter=100,
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
        warm_up_steps=[50],
        cycle_lengths=[400],
    ),
    model=dict(
        config=dict(
            hint_keys="hdmap_blur_depth",
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
        max_iter=400,
        callbacks=dict(
            heart_beat=dict(save_s3=False),
            # Tianjin captions differ per scene -> cache up to 40 caption sets
            cache_text_embeddings=L(CacheTextEmbeddings)(store_on_gpu=False, max_entries=40),
            iter_speed=dict(hit_thres=100, every_n=100, save_s3=False),
            device_monitor=dict(save_s3=False),
            grad_clip=dict(clip_norm=0.1),
            every_n_sample_ema=L(EveryNDrawSampleMultiviewVideo)(
                every_n=200,
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
                every_n=100,
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

cs.store(
    group="conditioner",
    package="model.config.conditioner",
    name="tianjin_small_conditioner",
    node=TianjinSmallConditioner,
)

cs.store(
    group="experiment",
    package="_global_",
    name="tianjin_multi_control_small",
    node=tianjin_multi_control_small,
)

register_tianjin_small_dataloader()
