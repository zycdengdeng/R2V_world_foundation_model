#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Inference script for single-view ablation experiments.

Usage:
    # Inference for a specific ablation experiment
    CUDA_VISIBLE_DEVICES=0 python -m cosmos_transfer2.experiments.custom.inference_ablation_single_view \
        --experiment ablation_hdmap_only \
        --ckpt_path /mnt/zihanw/Output_R2V_world_foundation_model_v1/ablation_all/cosmos_ablation/single_view/ablation_hdmap_only_20260506_191239/checkpoints/iter_000001000 \
        --output_dir /mnt/zihanw/Output_R2V_world_foundation_model_v1/ablation_all/inference/hdmap_only
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

os.environ["NVTE_FUSED_ATTN"] = "0"

from cosmos_transfer2._src.predict2.models.video2world_model import NUM_CONDITIONAL_FRAMES_KEY

CONTROL_WEIGHT_KEY = "control_weight"


def parse_args():
    parser = argparse.ArgumentParser(description="Single-view ablation inference")
    parser.add_argument("--experiment", type=str, required=True,
                        choices=[
                            "ablation_hdmap_only",
                            "ablation_blur_only",
                            "ablation_depth_only",
                            "ablation_hdmap_blur",
                            "ablation_hdmap_depth",
                            "ablation_blur_depth",
                            "ablation_all_controls",
                        ],
                        help="Experiment name")
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="Path to checkpoint directory (e.g., .../iter_000001000)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for generated videos")
    parser.add_argument("--scene_ids", type=str, nargs="+",
                        default=["031", "033", "053", "056", "076", "077", "088", "089"],
                        help="Scene IDs to run inference on (default: all test scenes)")
    parser.add_argument("--guidance", type=float, default=7.0,
                        help="CFG guidance scale")
    parser.add_argument("--num_steps", type=int, default=35,
                        help="Number of sampling steps")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--fps", type=int, default=10,
                        help="FPS for saved videos")
    parser.add_argument("--load_ema", action="store_true", default=True,
                        help="Load EMA weights")
    return parser.parse_args()


def init_distributed():
    """Initialize for single GPU (no distributed)."""
    if not dist.is_initialized():
        os.environ["MASTER_ADDR"] = "localhost"
        # Use a random port to avoid conflicts when running multiple experiments
        import random
        port = random.randint(29500, 29999)
        os.environ["MASTER_PORT"] = str(port)
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        dist.init_process_group(backend="gloo", rank=0, world_size=1)

    from megatron.core import parallel_state
    if not parallel_state.is_initialized():
        parallel_state.initialize_model_parallel(context_parallel_size=1)

    logger.info("Initialized single-GPU mode")


def load_model_and_config(experiment_name: str, ckpt_path: str, load_ema: bool = True):
    """Load model from DCP checkpoint."""
    from cosmos_transfer2._src.imaginaire.lazy_config import instantiate
    from cosmos_transfer2._src.imaginaire.utils.config_helper import get_config_module, override
    from cosmos_transfer2._src.predict2.checkpointer.dcp import (
        DefaultLoadPlanner,
        ModelWrapper,
        dcp_load_state_dict,
    )
    from torch.distributed.checkpoint import FileSystemReader

    logger.info(f"Loading config for experiment: {experiment_name}")
    config_file = "cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/config.py"
    config_module = get_config_module(config_file)
    config = importlib.import_module(config_module).make_config()
    config = override(config, ["--", f"experiment={experiment_name}"])

    config.model.config.ema.enabled = False
    config.model.config.fsdp_shard_size = 1

    config.validate()
    config.freeze()
    logger.info("Config loaded")

    logger.info("Instantiating model...")
    model = instantiate(config.model).cuda()
    model.on_train_start()
    logger.info(f"Model type: {type(model).__name__}")
    logger.info(f"Model hint_keys: {model.hint_keys}")

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


def create_test_dataset():
    """Create single-view test dataset."""
    from cosmos_transfer2.experiments.custom.ablation_single_view_experiment import (
        BLUR_DATASET_DIR,
        DEPTH_DATASET_DIR,
        HDMAP_DATASET_DIR,
        SINGLE_VIEW_CAMERA,
        TRAIN_SCENE_IDS,
    )
    from cosmos_transfer2.experiments.custom.custom_multi_control_dataset import (
        MultiControlMultiviewDataset,
    )

    logger.info(f"Creating single-view test dataset: {SINGLE_VIEW_CAMERA}")

    dataset = MultiControlMultiviewDataset(
        blur_dataset_dir=BLUR_DATASET_DIR,
        depth_dataset_dir=DEPTH_DATASET_DIR,
        hdmap_dataset_dir=HDMAP_DATASET_DIR,
        resolution_hw=(720, 1280),
        num_video_frames=29,
        fps_downsample_factor=1,
        camera_keys=SINGLE_VIEW_CAMERA,
        single_caption_camera_name="camera_front_wide_120fov",
        add_view_prefix_to_caption=True,
        exclude_scene_ids=TRAIN_SCENE_IDS,  # Exclude training = keep test
    )
    logger.info(f"Test dataset has {len(dataset)} samples")

    return dataset


def get_sample_indices_for_scenes(test_dataset, scene_ids: List[str]) -> List[int]:
    """Get dataset indices for specific scene IDs."""
    scene_set = set(scene_ids)
    indices = []
    for i, sample_id in enumerate(test_dataset.samples):
        scene_id = sample_id.split("_")[0]
        if scene_id in scene_set:
            indices.append(i)
    logger.info(f"Found {len(indices)} samples for scenes {scene_ids}")
    return indices


def load_sample(test_dataset, sample_idx: int, model) -> Dict[str, Any]:
    """Load a single sample."""
    from cosmos_transfer2.experiments.custom.custom_multi_control_dataset import collate_fn

    uint8_keys = {'video', 'control_input_blur', 'control_input_depth', 'control_input_hdmap_bbox'}
    # Index tensors must stay as int64/long (not converted to bfloat16)
    index_keys = {'front_cam_view_idx_sample_position', 'ref_cam_view_idx_sample_position',
                  'latent_view_indices_B_T', 'view_indices_B_T'}

    sample = test_dataset[sample_idx]
    batch = collate_fn([sample])

    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            if key in uint8_keys:
                batch[key] = value.to(device=torch.device("cuda"))
            elif key in index_keys:
                # Index tensors: move to device but keep dtype as int64
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
):
    """Run inference on a single batch."""
    # Compute text embeddings
    if hasattr(model, 'inplace_compute_text_embeddings_online'):
        model.inplace_compute_text_embeddings_online(batch)
        logger.info("Text embeddings computed")

    # Get data and condition
    raw_data, x0, condition = model.get_data_and_condition(batch)
    logger.info(f"raw_data shape: {raw_data.shape}, x0 shape: {x0.shape}")

    # Set control parameters
    batch[NUM_CONDITIONAL_FRAMES_KEY] = 0
    batch[CONTROL_WEIGHT_KEY] = 1.0

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
    """Save generated video."""
    from cosmos_transfer2._src.imaginaire.visualize.video import save_img_or_video

    sample_id = batch.get("__key__", [f"sample_{sample_idx}"])[0]

    sample_dir = Path(output_dir) / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Convert from [-1, 1] to [0, 1]
    generated_01 = ((generated.float() + 1.0) / 2.0).clamp(0, 1)
    raw_01 = ((raw_data.float() + 1.0) / 2.0).clamp(0, 1)

    # Save generated video (B, C, T, H, W) -> take first batch
    gen_path = sample_dir / "generated"
    save_img_or_video(generated_01[0], str(gen_path), fps=fps)

    # Save ground truth
    gt_path = sample_dir / "ground_truth"
    save_img_or_video(raw_01[0], str(gt_path), fps=fps)

    logger.info(f"Saved results to: {sample_dir}")


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

    logger.info("=" * 70)
    logger.info("Single-View Ablation Inference")
    logger.info("=" * 70)
    logger.info(f"Experiment: {args.experiment}")
    logger.info(f"Checkpoint: {args.ckpt_path}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Scene IDs: {args.scene_ids}")
    logger.info(f"Guidance: {args.guidance}, Steps: {args.num_steps}")
    logger.info("=" * 70)

    init_distributed()

    try:
        # Load model
        model, config = load_model_and_config(
            experiment_name=args.experiment,
            ckpt_path=args.ckpt_path,
            load_ema=args.load_ema,
        )

        # Create dataset
        test_dataset = create_test_dataset()

        # Get sample indices
        sample_indices = get_sample_indices_for_scenes(test_dataset, args.scene_ids)
        if not sample_indices:
            raise ValueError(f"No samples found for scene IDs: {args.scene_ids}")

        logger.info(f"Will process {len(sample_indices)} sample(s)")

        # Process each sample
        for i, sample_idx in enumerate(sample_indices):
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing sample {i+1}/{len(sample_indices)}: "
                        f"{test_dataset.samples[sample_idx]}")
            logger.info(f"{'='*60}")

            torch.cuda.empty_cache()
            gc.collect()

            # Load sample
            batch = load_sample(test_dataset, sample_idx, model)

            # Run inference
            generated, raw_data = run_inference_single(
                model=model,
                batch=batch,
                guidance=args.guidance,
                num_steps=args.num_steps,
            )

            # Save results
            save_results(
                generated=generated,
                raw_data=raw_data,
                batch=batch,
                output_dir=args.output_dir,
                sample_idx=sample_idx,
                fps=args.fps,
            )

            del batch, generated, raw_data
            torch.cuda.empty_cache()
            gc.collect()

            logger.info(f"Sample completed")

        logger.info("=" * 70)
        logger.info(f"SUCCESS! All {len(sample_indices)} sample(s) completed.")
        logger.info(f"Results saved to: {args.output_dir}")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"Inference failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
