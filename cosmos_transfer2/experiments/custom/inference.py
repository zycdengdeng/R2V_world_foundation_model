#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Standalone inference script for custom multi-control model.

Supports both single-GPU and multi-GPU inference with DCP checkpoints.
Flow aligned with evaluation_callback.py (proven working during training).
Saves per-view generated and ground truth videos.

Usage:
    # Inference on specific scene(s) (8 GPUs, 7 views)
    torchrun --nproc_per_node=8 --master_port=12345 \
        -m cosmos_transfer2.experiments.custom.inference \
        --ckpt_path /path/to/checkpoints/iter_000006600 \
        --output_dir /path/to/inference \
        --context_parallel_size 8 \
        --num_views 7 \
        --scene_ids 031

    # Full test set inference
    torchrun --nproc_per_node=8 --master_port=12345 \
        -m cosmos_transfer2.experiments.custom.inference \
        --ckpt_path /path/to/checkpoints/iter_000006600 \
        --output_dir /path/to/inference \
        --context_parallel_size 8 \
        --num_views 7 \
        --all_samples
"""

import argparse
import gc
import importlib
import os
from pathlib import Path
from typing import Dict, Any, List

import torch
import torch.distributed as dist
from einops import rearrange
from loguru import logger

# Disable fused attention for compatibility
os.environ["NVTE_FUSED_ATTN"] = "0"

# Use the same constants as the official callbacks
from cosmos_transfer2._src.predict2.models.video2world_model import NUM_CONDITIONAL_FRAMES_KEY

CONTROL_WEIGHT_KEY = "control_weight"


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-control model inference")
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="Path to checkpoint directory (e.g., .../iter_000006600)")
    parser.add_argument("--experiment", type=str, default="custom_multi_control_post_train",
                        help="Experiment name")
    parser.add_argument("--output_dir", type=str, default="/mnt/zihanw/Output_R2V_world_foundation_model_v1/inference",
                        help="Output directory for generated videos")
    # Sample selection: --scene_ids, --all_samples, or --sample_idx
    parser.add_argument("--scene_ids", type=str, nargs="+", default=None,
                        help="Scene IDs to run inference on (e.g., 031 033)")
    parser.add_argument("--all_samples", action="store_true", default=False,
                        help="Run inference on ALL test samples")
    parser.add_argument("--sample_idx", type=int, default=None,
                        help="Single test sample index (fallback if no --scene_ids or --all_samples)")
    # Dataset selection
    parser.add_argument("--use_train_set", action="store_true", default=False,
                        help="Use training set instead of test set (for debugging/visualization)")
    # Model & inference params
    parser.add_argument("--context_parallel_size", type=int, default=1,
                        help="Context parallel size (number of GPUs)")
    parser.add_argument("--num_views", type=int, default=None,
                        help="Number of camera views (default: same as context_parallel_size)")
    parser.add_argument("--guidance", type=float, default=7.0,
                        help="CFG guidance scale")
    parser.add_argument("--num_steps", type=int, default=35,
                        help="Number of sampling steps")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--fps", type=int, default=10,
                        help="FPS for saved videos")
    parser.add_argument("--load_ema", action="store_true", default=True,
                        help="Load EMA weights to regular model")
    return parser.parse_args()


def init_distributed(context_parallel_size: int):
    """Initialize distributed processing."""
    from megatron.core import parallel_state

    if "RANK" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
        logger.info("Initialized single-GPU distributed (gloo backend)")
    else:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(dist.get_rank())

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    parallel_state.initialize_model_parallel(
        context_parallel_size=context_parallel_size,
    )
    process_group = parallel_state.get_context_parallel_group()

    logger.info(f"Initialized distributed: rank={rank}, world_size={world_size}, cp_size={context_parallel_size}")
    return process_group, rank == 0


def load_model_and_config(experiment_name: str, ckpt_path: str, context_parallel_size: int,
                          load_ema: bool = True, process_group=None):
    """Load model from DCP checkpoint."""
    from cosmos_transfer2._src.imaginaire.lazy_config import instantiate
    from cosmos_transfer2._src.imaginaire.utils.config_helper import get_config_module, override
    from cosmos_transfer2._src.predict2.checkpointer.dcp import (
        DefaultLoadPlanner,
        ModelWrapper,
        dcp_load_state_dict,
    )
    from torch.distributed.checkpoint import FileSystemReader

    logger.info("Loading config...")
    config_file = "cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/config.py"
    config_module = get_config_module(config_file)
    config = importlib.import_module(config_module).make_config()
    config = override(config, ["--", f"experiment={experiment_name}"])

    config.model.config.ema.enabled = False
    config.model.config.fsdp_shard_size = context_parallel_size

    config.validate()
    config.freeze()
    logger.info("Config loaded")

    logger.info("Instantiating model...")
    model = instantiate(config.model).cuda()
    model.on_train_start()
    logger.info(f"Model type: {type(model).__name__}")

    if context_parallel_size > 1 and process_group is not None:
        model.net.enable_context_parallel(process_group)
        logger.info("Context parallel enabled")

    logger.info(f"Loading checkpoint from: {ckpt_path}")
    ckpt_path = Path(ckpt_path)
    model_ckpt_path = ckpt_path / "model"

    if not model_ckpt_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_ckpt_path}")

    model_wrapper = ModelWrapper(model, load_ema_to_reg=load_ema)
    state_dict = model_wrapper.state_dict()

    storage_reader = FileSystemReader(str(model_ckpt_path))
    load_planner = DefaultLoadPlanner(allow_partial_load=True)
    dcp_load_state_dict(state_dict, storage_reader, load_planner)
    model_wrapper.load_state_dict(state_dict)

    logger.info("Checkpoint loaded successfully!")

    model.eval()
    return model, config


def create_test_dataset(num_views: int = 7, use_train_set: bool = False):
    """Create dataset for inference.

    Args:
        num_views: Number of camera views
        use_train_set: If True, use training set; if False, use test set
    """
    from cosmos_transfer2.experiments.custom.custom_multi_control_experiment import (
        BLUR_DATASET_DIR,
        DEPTH_DATASET_DIR,
        HDMAP_DATASET_DIR,
        TRAINING_CAMERAS,
        TRAIN_SCENE_IDS,
        TEST_SCENE_IDS,
    )
    from cosmos_transfer2.experiments.custom.custom_multi_control_dataset import (
        MultiControlMultiviewDataset,
    )

    camera_keys = TRAINING_CAMERAS[:num_views]

    if use_train_set:
        # For training set: exclude test scenes
        exclude_ids = TEST_SCENE_IDS
        dataset_type = "training"
    else:
        # For test set: exclude training scenes
        exclude_ids = TRAIN_SCENE_IDS
        dataset_type = "test"

    logger.info(f"Creating {dataset_type} dataset with {num_views} view(s): {camera_keys}")

    dataset = MultiControlMultiviewDataset(
        blur_dataset_dir=BLUR_DATASET_DIR,
        depth_dataset_dir=DEPTH_DATASET_DIR,
        hdmap_dataset_dir=HDMAP_DATASET_DIR,
        resolution_hw=(720, 1280),
        num_video_frames=29,
        fps_downsample_factor=1,
        camera_keys=camera_keys,
        single_caption_camera_name="camera_front_wide_120fov",
        add_view_prefix_to_caption=True,
        exclude_scene_ids=exclude_ids,
    )
    logger.info(f"{dataset_type.capitalize()} dataset has {len(dataset)} samples")

    if len(dataset) == 0:
        raise ValueError(f"No {dataset_type} samples found!")

    return dataset


def get_sample_indices_for_scenes(test_dataset, scene_ids: List[str]) -> List[int]:
    """Get dataset indices for specific scene IDs.

    Sample IDs are like "031_seg01", scene ID is the part before "_".
    """
    scene_set = set(scene_ids)
    indices = []
    for i, sample_id in enumerate(test_dataset.samples):
        scene_id = sample_id.split("_")[0]
        if scene_id in scene_set:
            indices.append(i)
    logger.info(f"Found {len(indices)} samples for scenes {scene_ids}: "
                f"{[test_dataset.samples[i] for i in indices]}")
    return indices


def load_sample(test_dataset, sample_idx: int, model) -> Dict[str, Any]:
    """Load a single sample and prepare for inference.

    Follows evaluation_callback.py:_load_eval_samples() pattern.
    Each rank loads the same data independently (dataset is deterministic).
    """
    from cosmos_transfer2.experiments.custom.custom_multi_control_dataset import collate_fn

    uint8_keys = {'video', 'control_input_blur', 'control_input_depth', 'control_input_hdmap_bbox'}

    sample = test_dataset[sample_idx]
    batch = collate_fn([sample])

    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            if key in uint8_keys:
                batch[key] = value.to(device=torch.device("cuda"))
            else:
                batch[key] = value.to(**model.tensor_kwargs)

    logger.info(f"Loaded sample {sample_idx}: {sample['__key__']}")
    return batch


@torch.no_grad()
def run_inference_single(
    model,
    batch: Dict[str, Any],
    guidance: float = 7.0,
    num_steps: int = 35,
    num_conditional_frames: int = 0,
    control_weight: float = 1.0,
):
    """Run inference on a single batch.

    Flow aligned with evaluation_callback.py:_run_evaluation():
    1. Compute text embeddings online
    2. Get data and condition
    3. Set control parameters (AFTER get_data_and_condition)
    4. Generate samples
    5. Decode
    """
    # Step 1: Compute text embeddings online
    if hasattr(model, 'inplace_compute_text_embeddings_online'):
        model.inplace_compute_text_embeddings_online(batch)
        logger.info("Text embeddings computed")

    # Step 2: Get data and condition
    raw_data, x0, condition = model.get_data_and_condition(batch)
    logger.info(f"raw_data shape: {raw_data.shape}, x0 (latent) shape: {x0.shape}")

    # Step 3: Set control parameters AFTER get_data_and_condition
    batch[NUM_CONDITIONAL_FRAMES_KEY] = num_conditional_frames
    batch[CONTROL_WEIGHT_KEY] = control_weight

    # Step 4: Generate samples
    logger.info(f"Generating with guidance={guidance}, steps={num_steps}, "
                f"cond_frames={num_conditional_frames}, ctrl_weight={control_weight}...")
    sample = model.generate_samples_from_batch(
        batch,
        guidance=guidance,
        state_shape=x0.shape[1:],
        n_sample=x0.shape[0],
        num_steps=num_steps,
        is_negative_prompt=False,
    )
    logger.info(f"Generated latent shape: {sample.shape}")

    # Step 5: Decode
    if hasattr(model, "decode"):
        generated = model.decode(sample)
        logger.info(f"Decoded video shape: {generated.shape}")
    else:
        generated = sample

    return generated, raw_data


def save_per_view_results(
    generated: torch.Tensor,
    raw_data: torch.Tensor,
    batch: Dict[str, Any],
    output_dir: str,
    sample_idx: int,
    camera_names: List[str],
    fps: int = 10,
):
    """Save per-view generated and ground truth videos.

    Output structure:
        {output_dir}/{sample_id}/
            {camera_name}_generated.mp4
            {camera_name}_gt.mp4
            {camera_name}_control_blur.mp4
            {camera_name}_control_depth.mp4
            {camera_name}_control_hdmap.mp4

    Video data is (B, C, V*T, H, W) where V views are concatenated along time.
    """
    from cosmos_transfer2._src.imaginaire.visualize.video import save_img_or_video

    n_views = len(batch.get("view_indices_selection", [[0]])[0])
    sample_id = batch.get("__key__", [f"sample_{sample_idx}"])[0]

    # Create per-sample directory
    sample_dir = Path(output_dir) / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Convert from [-1, 1] to [0, 1]
    generated_01 = ((generated.float() + 1.0) / 2.0).clamp(0, 1)
    gt_01 = ((raw_data.float() + 1.0) / 2.0).clamp(0, 1)

    logger.info(f"Saving per-view results for {sample_id}: {n_views} views to {sample_dir}")

    # Split by view: (B, C, V*T, H, W) -> (V, B, C, T, H, W)
    gen_views = rearrange(generated_01, "B C (V T) H W -> V B C T H W", V=n_views)
    gt_views = rearrange(gt_01, "B C (V T) H W -> V B C T H W", V=n_views)

    for v in range(n_views):
        cam_name = camera_names[v] if v < len(camera_names) else f"view_{v}"
        # Remove "camera_" prefix for shorter filenames
        short_name = cam_name.replace("camera_", "")

        # Save generated video for this view: (B, C, T, H, W) -> use [0] for batch dim
        gen_path = sample_dir / f"{short_name}_generated"
        save_img_or_video(gen_views[v, 0], str(gen_path), fps=fps)

        # Save ground truth video for this view
        gt_path = sample_dir / f"{short_name}_gt"
        save_img_or_video(gt_views[v, 0], str(gt_path), fps=fps)

        logger.info(f"  View {v} ({short_name}): generated + gt saved")

    logger.info(f"All per-view results saved to: {sample_dir}")


def cleanup_distributed():
    """Clean up distributed resources."""
    if dist.is_initialized():
        from megatron.core import parallel_state
        if parallel_state.is_initialized():
            parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    num_views = args.num_views if args.num_views is not None else args.context_parallel_size

    # Get camera names for saving
    from cosmos_transfer2.experiments.custom.custom_multi_control_experiment import TRAINING_CAMERAS
    camera_names = list(TRAINING_CAMERAS[:num_views])

    logger.info("=" * 60)
    logger.info("Multi-Control Model Inference")
    logger.info("=" * 60)
    logger.info(f"Checkpoint: {args.ckpt_path}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Dataset: {'TRAINING SET' if args.use_train_set else 'TEST SET'}")
    logger.info(f"Scene IDs: {args.scene_ids or 'all' if args.all_samples else args.sample_idx}")
    logger.info(f"Context parallel size: {args.context_parallel_size}")
    logger.info(f"Views: {num_views} -> {camera_names}")
    logger.info(f"Guidance: {args.guidance}, Steps: {args.num_steps}")
    logger.info("=" * 60)

    if num_views > args.context_parallel_size:
        raise ValueError(
            f"num_views ({num_views}) must be <= context_parallel_size ({args.context_parallel_size}). "
            f"Use --context_parallel_size {num_views} or --num_views {args.context_parallel_size}"
        )

    process_group, is_rank0 = init_distributed(args.context_parallel_size)

    try:
        # Load model
        model, config = load_model_and_config(
            experiment_name=args.experiment,
            ckpt_path=args.ckpt_path,
            context_parallel_size=args.context_parallel_size,
            load_ema=args.load_ema,
            process_group=process_group,
        )

        # Create dataset (test set by default, or train set if --use_train_set)
        test_dataset = create_test_dataset(num_views=num_views, use_train_set=args.use_train_set)

        # Determine which samples to process
        if args.scene_ids:
            sample_indices = get_sample_indices_for_scenes(test_dataset, args.scene_ids)
            if not sample_indices:
                raise ValueError(f"No samples found for scene IDs: {args.scene_ids}")
        elif args.all_samples:
            sample_indices = list(range(len(test_dataset)))
        elif args.sample_idx is not None:
            if args.sample_idx >= len(test_dataset):
                raise ValueError(f"Sample index {args.sample_idx} out of range (max {len(test_dataset)-1})")
            sample_indices = [args.sample_idx]
        else:
            # Default: first sample
            sample_indices = [0]

        logger.info(f"Will process {len(sample_indices)} sample(s)")
        for idx in sample_indices:
            logger.info(f"  [{idx}] {test_dataset.samples[idx]}")

        # Process each sample
        for i, sample_idx in enumerate(sample_indices):
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing sample {i+1}/{len(sample_indices)}: "
                        f"{test_dataset.samples[sample_idx]} (idx={sample_idx})")
            logger.info(f"{'='*60}")

            # Aggressive GPU cleanup before each sample
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()

            # Load single sample
            batch = load_sample(test_dataset, sample_idx, model)

            # Run inference
            generated, raw_data = run_inference_single(
                model=model,
                batch=batch,
                guidance=args.guidance,
                num_steps=args.num_steps,
                num_conditional_frames=0,
                control_weight=1.0,
            )

            # Save per-view results (only on rank 0)
            if is_rank0:
                save_per_view_results(
                    generated=generated,
                    raw_data=raw_data,
                    batch=batch,
                    output_dir=args.output_dir,
                    sample_idx=sample_idx,
                    camera_names=camera_names,
                    fps=args.fps,
                )

            # Aggressive cleanup after each sample
            del batch, generated, raw_data
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()

            # Synchronize all ranks
            if dist.is_initialized():
                dist.barrier()

            logger.info(f"Sample {test_dataset.samples[sample_idx]} completed")

        logger.info("=" * 60)
        logger.info(f"SUCCESS! All {len(sample_indices)} sample(s) completed.")
        logger.info(f"Results saved to: {args.output_dir}")
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
