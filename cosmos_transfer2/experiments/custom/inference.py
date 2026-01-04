#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Standalone inference script for custom multi-control model.

Supports both single-GPU and multi-GPU inference with DCP checkpoints.

Usage:
    # Single GPU inference on one test sample
    CUDA_VISIBLE_DEVICES=0 python -m cosmos_transfer2.experiments.custom.inference \
        --ckpt_path /mnt/zihanw/Output_R2V_world_foundation_model_v1/cosmos_transfer_custom/multi_control/2b_custom_multi_control_20251226_155343/checkpoints/iter_000005000 \
        --output_dir ./inference_output \
        --sample_idx 0

    # Multi-GPU inference (4 GPUs with context parallelism)
    torchrun --nproc_per_node=4 --master_port=12345 \
        -m cosmos_transfer2.experiments.custom.inference \
        --ckpt_path /path/to/checkpoints/iter_000005000 \
        --output_dir ./inference_output \
        --context_parallel_size 4
"""

import argparse
import importlib
import os
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.distributed as dist
from einops import rearrange
from loguru import logger

# Disable fused attention for compatibility
os.environ["NVTE_FUSED_ATTN"] = "0"


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-control model inference")
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="Path to checkpoint directory (e.g., .../iter_000005000)")
    parser.add_argument("--experiment", type=str, default="custom_multi_control_post_train",
                        help="Experiment name")
    parser.add_argument("--output_dir", type=str, default="./inference_output",
                        help="Output directory for generated videos")
    parser.add_argument("--sample_idx", type=int, default=0,
                        help="Test sample index to use")
    parser.add_argument("--context_parallel_size", type=int, default=1,
                        help="Context parallel size (number of GPUs)")
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
    """Initialize distributed processing for multi-GPU inference."""
    if "RANK" not in os.environ:
        return None, True  # Single GPU mode

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    torch.cuda.set_device(rank)

    # Initialize megatron parallel state
    from megatron.core import parallel_state
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


def create_test_dataset(sample_idx: int = 0):
    """Create test dataset and get one sample."""
    from cosmos_transfer2.experiments.custom.custom_multi_control_experiment import (
        BLUR_DATASET_DIR,
        DEPTH_DATASET_DIR,
        HDMAP_DATASET_DIR,
        TRAINING_CAMERAS,
        TRAIN_SCENE_IDS,
    )
    from cosmos_transfer2.experiments.custom.custom_multi_control_dataset import (
        MultiControlMultiviewDataset,
        collate_fn,
    )

    logger.info("Creating test dataset...")
    test_dataset = MultiControlMultiviewDataset(
        blur_dataset_dir=BLUR_DATASET_DIR,
        depth_dataset_dir=DEPTH_DATASET_DIR,
        hdmap_dataset_dir=HDMAP_DATASET_DIR,
        resolution_hw=(720, 1280),
        num_video_frames=29,
        fps_downsample_factor=1,
        camera_keys=TRAINING_CAMERAS,
        single_caption_camera_name="camera_front_wide_120fov",
        add_view_prefix_to_caption=True,
        exclude_scene_ids=TRAIN_SCENE_IDS,  # Exclude training scenes = keep only test scenes
    )
    logger.info(f"Test dataset has {len(test_dataset)} samples")

    if len(test_dataset) == 0:
        raise ValueError("No test samples found!")

    if sample_idx >= len(test_dataset):
        logger.warning(f"Sample index {sample_idx} out of range, using index 0")
        sample_idx = 0

    # Get sample and collate
    sample = test_dataset[sample_idx]
    batch = collate_fn([sample])

    logger.info(f"Loaded sample {sample_idx}: {sample['__key__']}")
    logger.info(f"Batch keys: {list(batch.keys())}")

    return batch, test_dataset


def prepare_batch(batch: Dict[str, Any], model) -> Dict[str, Any]:
    """Move batch to GPU and prepare for inference."""
    device = torch.device("cuda")
    uint8_keys = {'video', 'control_input_blur', 'control_input_depth', 'control_input_hdmap_bbox'}

    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            if key in uint8_keys:
                batch[key] = value.to(device=device)
            else:
                batch[key] = value.to(**model.tensor_kwargs)

    return batch


@torch.no_grad()
def run_inference(
    model,
    batch: Dict[str, Any],
    guidance: float = 7.0,
    num_steps: int = 35,
    num_conditional_frames: int = 0,
    control_weight: float = 1.0,
):
    """Run inference on a batch.

    Based on the evaluation callback flow in evaluation_callback.py
    """
    # Set control parameters
    batch["num_conditional_frames"] = num_conditional_frames
    batch["control_weight"] = control_weight

    # Compute text embeddings online (same as training)
    if hasattr(model, 'inplace_compute_text_embeddings_online'):
        model.inplace_compute_text_embeddings_online(batch)
        logger.info("Text embeddings computed")

    # Get data and condition
    raw_data, x0, condition = model.get_data_and_condition(batch)
    logger.info(f"raw_data shape: {raw_data.shape}")
    logger.info(f"x0 (latent) shape: {x0.shape}")

    # Generate samples
    logger.info(f"Generating with guidance={guidance}, steps={num_steps}...")
    sample = model.generate_samples_from_batch(
        batch,
        guidance=guidance,
        state_shape=x0.shape[1:],
        n_sample=x0.shape[0],
        num_steps=num_steps,
        is_negative_prompt=False,
    )
    logger.info(f"Generated latent shape: {sample.shape}")

    # Decode
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

    logger.info("=" * 60)
    logger.info("Multi-Control Model Inference")
    logger.info("=" * 60)
    logger.info(f"Checkpoint: {args.ckpt_path}")
    logger.info(f"Experiment: {args.experiment}")
    logger.info(f"Sample index: {args.sample_idx}")
    logger.info(f"Context parallel size: {args.context_parallel_size}")
    logger.info("=" * 60)

    # Initialize distributed if needed
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

        # Create test dataset and get sample
        batch, test_dataset = create_test_dataset(args.sample_idx)

        # Prepare batch for inference
        batch = prepare_batch(batch, model)

        # Run inference
        logger.info("Running inference...")
        generated, raw_data = run_inference(
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
                sample_idx=args.sample_idx,
                fps=args.fps,
            )

        logger.info("=" * 60)
        logger.info("SUCCESS! Inference completed.")
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
