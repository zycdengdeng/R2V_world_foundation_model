#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Standalone inference script for custom multi-control model.

Supports both single-GPU and multi-GPU inference with DCP checkpoints.
Flow aligned with evaluation_callback.py (proven working during training).

Usage:
    # Single sample inference (8 GPUs, 7 views)
    torchrun --nproc_per_node=8 --master_port=12345 \
        -m cosmos_transfer2.experiments.custom.inference \
        --ckpt_path /path/to/checkpoints/iter_000005000 \
        --output_dir ./inference_output \
        --context_parallel_size 8 \
        --num_views 7 \
        --sample_idx 0

    # Full test set inference (all samples)
    torchrun --nproc_per_node=8 --master_port=12345 \
        -m cosmos_transfer2.experiments.custom.inference \
        --ckpt_path /path/to/checkpoints/iter_000005000 \
        --output_dir ./inference_output \
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
                        help="Path to checkpoint directory (e.g., .../iter_000005000)")
    parser.add_argument("--experiment", type=str, default="custom_multi_control_post_train",
                        help="Experiment name")
    parser.add_argument("--output_dir", type=str, default="/mnt/zihanw/Output_R2V_world_foundation_model_v1/inference_output",
                        help="Output directory for generated videos")
    parser.add_argument("--sample_idx", type=int, default=0,
                        help="Test sample index to use (ignored if --all_samples)")
    parser.add_argument("--all_samples", action="store_true", default=False,
                        help="Run inference on ALL test samples")
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
    """Initialize distributed processing.

    NOTE: Even for single-GPU inference, we need to initialize megatron parallel state
    because the model's encode() method uses context parallel groups.
    """
    from megatron.core import parallel_state

    if "RANK" not in os.environ:
        # Single GPU mode - still need to init distributed for megatron
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
        logger.info("Initialized single-GPU distributed (gloo backend)")
    else:
        # Multi-GPU mode
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(dist.get_rank())

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Initialize megatron parallel state (required for model's encode_cp)
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

    # Step 1: Load config
    logger.info("Loading config...")
    config_file = "cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/config.py"
    config_module = get_config_module(config_file)
    config = importlib.import_module(config_module).make_config()
    config = override(config, ["--", f"experiment={experiment_name}"])

    # Disable EMA in model config (we'll load EMA weights to regular model)
    config.model.config.ema.enabled = False
    # Set FSDP shard size to match context parallel size
    config.model.config.fsdp_shard_size = context_parallel_size

    config.validate()
    config.freeze()
    logger.info("Config loaded")

    # Step 2: Instantiate model
    logger.info("Instantiating model...")
    model = instantiate(config.model).cuda()
    model.on_train_start()
    logger.info(f"Model type: {type(model).__name__}")

    # Enable context parallel if needed
    if context_parallel_size > 1 and process_group is not None:
        model.net.enable_context_parallel(process_group)
        logger.info("Context parallel enabled")

    # Step 3: Load DCP checkpoint
    logger.info(f"Loading checkpoint from: {ckpt_path}")
    ckpt_path = Path(ckpt_path)
    model_ckpt_path = ckpt_path / "model"

    if not model_ckpt_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_ckpt_path}")

    # Create model wrapper for loading EMA weights to regular model
    model_wrapper = ModelWrapper(model, load_ema_to_reg=load_ema)
    state_dict = model_wrapper.state_dict()

    # Load using FileSystemReader
    storage_reader = FileSystemReader(str(model_ckpt_path))
    load_planner = DefaultLoadPlanner(allow_partial_load=True)
    dcp_load_state_dict(state_dict, storage_reader, load_planner)
    model_wrapper.load_state_dict(state_dict)

    logger.info("Checkpoint loaded successfully!")

    model.eval()
    return model, config


def create_test_dataset(num_views: int = 7):
    """Create test dataset (all test samples).

    Args:
        num_views: Number of camera views to use
    Returns:
        test_dataset: The full test dataset
    """
    from cosmos_transfer2.experiments.custom.custom_multi_control_experiment import (
        BLUR_DATASET_DIR,
        DEPTH_DATASET_DIR,
        HDMAP_DATASET_DIR,
        TRAINING_CAMERAS,
        TRAIN_SCENE_IDS,
    )
    from cosmos_transfer2.experiments.custom.custom_multi_control_dataset import (
        MultiControlMultiviewDataset,
    )

    # Use only the first num_views cameras
    camera_keys = TRAINING_CAMERAS[:num_views]
    logger.info(f"Creating test dataset with {num_views} view(s): {camera_keys}")

    test_dataset = MultiControlMultiviewDataset(
        blur_dataset_dir=BLUR_DATASET_DIR,
        depth_dataset_dir=DEPTH_DATASET_DIR,
        hdmap_dataset_dir=HDMAP_DATASET_DIR,
        resolution_hw=(720, 1280),
        num_video_frames=29,
        fps_downsample_factor=1,
        camera_keys=camera_keys,
        single_caption_camera_name="camera_front_wide_120fov",
        add_view_prefix_to_caption=True,
        exclude_scene_ids=TRAIN_SCENE_IDS,  # Exclude training scenes = keep only test scenes
    )
    logger.info(f"Test dataset has {len(test_dataset)} samples")

    if len(test_dataset) == 0:
        raise ValueError("No test samples found!")

    return test_dataset


def load_sample(test_dataset, sample_idx: int, model) -> Dict[str, Any]:
    """Load a single sample from the test dataset and prepare for inference.

    Follows the same pattern as evaluation_callback.py:_load_eval_samples().
    Each rank loads the same data independently (dataset is deterministic).
    """
    from cosmos_transfer2.experiments.custom.custom_multi_control_dataset import collate_fn

    uint8_keys = {'video', 'control_input_blur', 'control_input_depth', 'control_input_hdmap_bbox'}

    sample = test_dataset[sample_idx]
    batch = collate_fn([sample])

    # Move to device (same as eval callback)
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
    # Step 1: Compute text embeddings online (same as training & eval callback)
    if hasattr(model, 'inplace_compute_text_embeddings_online'):
        model.inplace_compute_text_embeddings_online(batch)
        logger.info("Text embeddings computed")

    # Step 2: Get data and condition
    raw_data, x0, condition = model.get_data_and_condition(batch)
    logger.info(f"raw_data shape: {raw_data.shape}, x0 (latent) shape: {x0.shape}")

    # Step 3: Set control parameters AFTER get_data_and_condition
    # (matches evaluation_callback.py and EveryNDrawSampleMultiviewVideo)
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


def save_results(
    generated: torch.Tensor,
    raw_data: torch.Tensor,
    batch: Dict[str, Any],
    output_dir: str,
    sample_idx: int,
    fps: int = 10,
):
    """Save generated videos and comparisons."""
    from cosmos_transfer2._src.imaginaire.visualize.video import save_img_or_video

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Convert from [-1, 1] to [0, 1]
    generated_01 = ((generated + 1.0) / 2.0).clamp(0, 1)
    gt_01 = ((raw_data + 1.0) / 2.0).clamp(0, 1)

    # Get number of views
    n_views = len(batch.get("view_indices_selection", [[0]])[0])

    sample_id = batch.get("__key__", [f"sample_{sample_idx}"])[0]
    logger.info(f"Saving results for sample: {sample_id}, n_views={n_views}")

    # Save generated video
    gen_path = output_path / f"{sample_id}_generated"
    save_img_or_video(generated_01[0], str(gen_path), fps=fps)
    logger.info(f"Saved: {gen_path}.mp4")

    # Save ground truth video
    gt_path = output_path / f"{sample_id}_ground_truth"
    save_img_or_video(gt_01[0], str(gt_path), fps=fps)
    logger.info(f"Saved: {gt_path}.mp4")

    # Save side-by-side comparison (width)
    comparison_h = torch.cat([gt_01, generated_01], dim=-1)
    comp_path = output_path / f"{sample_id}_comparison_side_by_side"
    save_img_or_video(comparison_h[0], str(comp_path), fps=fps)
    logger.info(f"Saved: {comp_path}.mp4")

    # Rearrange to show views vertically for better visualization
    if n_views > 1:
        # Data is (B, C, V*T, H, W), rearrange to (B, C, T, V*H, W)
        B, C, VT, H, W = generated_01.shape
        T = VT // n_views

        def views_vertical(x):
            return rearrange(x, "B C (V T) H W -> B C T (V H) W", V=n_views)

        gen_v = views_vertical(generated_01)
        gt_v = views_vertical(gt_01)

        # Save views stacked vertically
        gen_views_path = output_path / f"{sample_id}_generated_views"
        save_img_or_video(gen_v[0], str(gen_views_path), fps=fps)
        logger.info(f"Saved: {gen_views_path}.mp4")

        gt_views_path = output_path / f"{sample_id}_gt_views"
        save_img_or_video(gt_v[0], str(gt_views_path), fps=fps)
        logger.info(f"Saved: {gt_views_path}.mp4")

        # Comparison: GT | Generated side by side, views vertical
        comp_views = torch.cat([gt_v, gen_v], dim=-1)
        comp_views_path = output_path / f"{sample_id}_comparison_views"
        save_img_or_video(comp_views[0], str(comp_views_path), fps=fps)
        logger.info(f"Saved: {comp_views_path}.mp4")

    # Save control inputs
    control_keys = ['control_input_blur', 'control_input_depth', 'control_input_hdmap_bbox']
    for key in control_keys:
        if key in batch and batch[key] is not None:
            ctrl = batch[key].float()
            if ctrl.max() > 1:
                ctrl = ctrl / 255.0
            ctrl_path = output_path / f"{sample_id}_{key}"
            save_img_or_video(ctrl[0].cpu(), str(ctrl_path), fps=fps)
            logger.info(f"Saved: {ctrl_path}.mp4")

    logger.info(f"All results saved to: {output_path}")


def cleanup_distributed():
    """Clean up distributed resources."""
    if dist.is_initialized():
        from megatron.core import parallel_state
        if parallel_state.is_initialized():
            parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


def main():
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Determine number of views (default: same as context_parallel_size)
    num_views = args.num_views if args.num_views is not None else args.context_parallel_size

    logger.info("=" * 60)
    logger.info("Multi-Control Model Inference")
    logger.info("=" * 60)
    logger.info(f"Checkpoint: {args.ckpt_path}")
    logger.info(f"Experiment: {args.experiment}")
    logger.info(f"Mode: {'all samples' if args.all_samples else f'sample_idx={args.sample_idx}'}")
    logger.info(f"Context parallel size: {args.context_parallel_size}")
    logger.info(f"Number of views: {num_views}")
    logger.info(f"Guidance: {args.guidance}, Steps: {args.num_steps}")
    logger.info("=" * 60)

    # Validate: num_views must be <= context_parallel_size
    if num_views > args.context_parallel_size:
        raise ValueError(
            f"num_views ({num_views}) must be <= context_parallel_size ({args.context_parallel_size}). "
            f"Use --context_parallel_size {num_views} or --num_views {args.context_parallel_size}"
        )

    # Initialize distributed
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

        # Create test dataset
        test_dataset = create_test_dataset(num_views=num_views)

        # Determine which samples to process
        if args.all_samples:
            sample_indices = list(range(len(test_dataset)))
        else:
            if args.sample_idx >= len(test_dataset):
                logger.warning(f"Sample index {args.sample_idx} out of range, using 0")
                sample_indices = [0]
            else:
                sample_indices = [args.sample_idx]

        logger.info(f"Will process {len(sample_indices)} sample(s): {sample_indices}")

        # Process each sample (following eval callback pattern: one at a time with cleanup)
        for i, sample_idx in enumerate(sample_indices):
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing sample {i+1}/{len(sample_indices)} (idx={sample_idx})")
            logger.info(f"{'='*60}")

            # Aggressive GPU cleanup before each sample (same as eval callback)
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()

            # Load single sample (each rank loads same data independently)
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

            # Save results (only on rank 0)
            if is_rank0:
                save_results(
                    generated=generated,
                    raw_data=raw_data,
                    batch=batch,
                    output_dir=args.output_dir,
                    sample_idx=sample_idx,
                    fps=args.fps,
                )

            # Aggressive cleanup after each sample (same as eval callback)
            del batch, generated, raw_data
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()

            # Synchronize all ranks after each sample
            if dist.is_initialized():
                dist.barrier()

            logger.info(f"Sample {sample_idx} completed")

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
