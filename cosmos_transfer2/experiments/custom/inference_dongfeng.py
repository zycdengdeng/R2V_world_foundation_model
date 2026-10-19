#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Inference for the Dongfeng-trained multi-control model.

Same flow as inference.py, with the Dongfeng data layer:
- DongfengMultiControlDataset (handles {drive}_seg{NNN} sample naming)
- dataset paths / train-test split from dongfeng_multi_control_experiment
  (set DONGFENG_TEST_SCENES to the same value used during training if you
  overrode the automatic split)

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 WORLD_SIZE=8 \
    torchrun --nproc_per_node=8 --master_port=12350 \
        -m cosmos_transfer2.experiments.custom.inference_dongfeng \
        --ckpt_path /mnt2/Output_R2V_dongfeng/cosmos_transfer_custom/dongfeng/2b_dongfeng_multi_control_*/checkpoints/iter_XXXXXX \
        --output_dir /mnt2/Output_R2V_dongfeng/inference \
        --context_parallel_size 8 --num_views 7
    # Default: all test-split samples. Filter with --scene_ids <scene...>
    # (full scene names, e.g. 5w_2026-01-12-11-29-05) or --sample_names.
"""

import argparse
import gc

import torch
import torch.distributed as dist
from loguru import logger

# Reuse the proven inference machinery (also sets NVTE_FUSED_ATTN=0)
from cosmos_transfer2.experiments.custom.inference import (
    cleanup_distributed,
    init_distributed,
    load_model_and_config,
    load_sample,
    run_inference_single,
    save_per_view_results,
)
from cosmos_transfer2.experiments.custom.dongfeng_multi_control_dataset import (
    DongfengMultiControlDataset,
    extract_dongfeng_scene_id,
)
from cosmos_transfer2.experiments.custom.dongfeng_multi_control_experiment import (
    BLUR_DATASET_DIR,
    DEPTH_DATASET_DIR,
    HDMAP_DATASET_DIR,
    TEST_SCENE_IDS,
    TRAIN_SCENE_IDS,
    TRAINING_CAMERAS,
)


def parse_args():
    p = argparse.ArgumentParser(description="Dongfeng multi-control inference")
    p.add_argument("--ckpt_path", type=str, required=True)
    p.add_argument("--experiment", type=str, default="dongfeng_multi_control_post_train")
    p.add_argument("--output_dir", type=str, default="/mnt2/Output_R2V_dongfeng/inference")
    p.add_argument("--scene_ids", type=str, nargs="+", default=None,
                   help="Full scene names to run (e.g. 5w_2026-01-12-11-29-05); default: all test samples")
    p.add_argument("--sample_names", type=str, nargs="+", default=None,
                   help="Exact sample names (e.g. 5w_2026-01-12-11-29-05_seg003)")
    p.add_argument("--use_train_set", action="store_true", default=False)
    p.add_argument("--context_parallel_size", type=int, default=1)
    p.add_argument("--num_views", type=int, default=None)
    p.add_argument("--guidance", type=float, default=7.0)
    p.add_argument("--num_steps", type=int, default=35)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--load_ema", action="store_true", default=True)
    return p.parse_args()


def create_dongfeng_dataset(num_views: int, use_train_set: bool):
    exclude_ids = TEST_SCENE_IDS if use_train_set else TRAIN_SCENE_IDS
    which = "training" if use_train_set else "test"
    logger.info(f"Creating Dongfeng {which} dataset ({num_views} views); excluding {len(exclude_ids)} scenes")

    dataset = DongfengMultiControlDataset(
        blur_dataset_dir=BLUR_DATASET_DIR,
        depth_dataset_dir=DEPTH_DATASET_DIR,
        hdmap_dataset_dir=HDMAP_DATASET_DIR,
        resolution_hw=(720, 1280),
        num_video_frames=29,
        fps_downsample_factor=1,
        camera_keys=TRAINING_CAMERAS[:num_views],
        single_caption_camera_name="camera_front_wide_120fov",
        add_view_prefix_to_caption=True,
        exclude_scene_ids=exclude_ids,
    )
    logger.info(f"Dongfeng {which} dataset has {len(dataset)} samples")
    if len(dataset) == 0:
        raise ValueError(f"No {which} samples found!")
    return dataset


def select_samples(dataset, scene_ids, sample_names):
    if sample_names:
        wanted = set(sample_names)
        indices = [i for i, s in enumerate(dataset.samples) if s in wanted]
    elif scene_ids:
        wanted = set(scene_ids)
        indices = [i for i, s in enumerate(dataset.samples)
                   if extract_dongfeng_scene_id(s) in wanted]
    else:
        indices = list(range(len(dataset)))
    logger.info(f"Selected {len(indices)} sample(s): {[dataset.samples[i] for i in indices]}")
    return indices


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    num_views = args.num_views if args.num_views is not None else args.context_parallel_size
    camera_names = list(TRAINING_CAMERAS[:num_views])

    logger.info("=" * 60)
    logger.info("Dongfeng Multi-Control Inference")
    logger.info("=" * 60)
    logger.info(f"Checkpoint: {args.ckpt_path}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Dataset: {'TRAIN' if args.use_train_set else 'TEST'} split "
                f"(test scenes: {TEST_SCENE_IDS})")
    logger.info(f"Views: {num_views}, Guidance: {args.guidance}, Steps: {args.num_steps}")
    logger.info("=" * 60)

    if num_views > args.context_parallel_size:
        raise ValueError(f"num_views ({num_views}) must be <= context_parallel_size ({args.context_parallel_size})")

    process_group, is_rank0 = init_distributed(args.context_parallel_size)

    try:
        model, config = load_model_and_config(
            experiment_name=args.experiment,
            ckpt_path=args.ckpt_path,
            context_parallel_size=args.context_parallel_size,
            load_ema=args.load_ema,
            process_group=process_group,
        )

        dataset = create_dongfeng_dataset(num_views=num_views, use_train_set=args.use_train_set)
        sample_indices = select_samples(dataset, args.scene_ids, args.sample_names)
        if not sample_indices:
            raise ValueError("No samples matched the given filters")

        for i, sample_idx in enumerate(sample_indices):
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Sample {i + 1}/{len(sample_indices)}: {dataset.samples[sample_idx]}")
            logger.info(f"{'=' * 60}")

            torch.cuda.empty_cache()
            gc.collect()

            batch = load_sample(dataset, sample_idx, model)

            generated, raw_data = run_inference_single(
                model=model,
                batch=batch,
                guidance=args.guidance,
                num_steps=args.num_steps,
                num_conditional_frames=0,
                control_weight=1.0,
            )

            if is_rank0:
                save_per_view_results(
                    generated=generated,
                    batch=batch,
                    output_dir=args.output_dir,
                    sample_idx=sample_idx,
                    camera_names=camera_names,
                    fps=args.fps,
                )

            del batch, generated, raw_data
            torch.cuda.empty_cache()
            gc.collect()

            if dist.is_initialized():
                dist.barrier()

        logger.info("=" * 60)
        logger.info(f"SUCCESS! {len(sample_indices)} sample(s) completed. Results: {args.output_dir}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Inference failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
