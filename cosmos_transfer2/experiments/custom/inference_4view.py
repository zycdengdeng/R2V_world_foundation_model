#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Inference script for 4-view multi-control model (4 GPUs).

Uses the 4-view training configuration:
- 4 cameras: front_wide, cross_left, cross_right, rear_right
- 20 test scenes, 59 training scenes
- 4 GPUs with context parallelism

Usage:
    # Inference on specific scene(s)
    CUDA_VISIBLE_DEVICES=0,1,2,3 WORLD_SIZE=4 \
    IMAGINAIRE_OUTPUT_ROOT="/mnt/zihanw/Output_R2V_world_foundation_model_v1" \
    HF_HOME="/mnt/zihanw/.cache/huggingface" \
    HF_HUB_CACHE="/mnt/zihanw/.cache/huggingface/hub" \
    HF_ENDPOINT="https://hf-mirror.com" \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    TRANSFORMERS_CACHE="/mnt/zihanw/.cache/huggingface/transformers" \
    torchrun --nproc_per_node=4 --master_port=12346 \
        -m cosmos_transfer2.experiments.custom.inference_4view \
        --ckpt_path /path/to/checkpoints/iter_000003400 \
        --output_dir /path/to/inference_4view \
        --scene_ids 031

    # Full test set inference
    ... --all_samples
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

# ============================================================================
# 4-view specific constants (from git history: before commit bdcdb7b)
# ============================================================================

CAMERAS_4VIEW = (
    "camera_front_wide_120fov",
    "camera_cross_left_120fov",
    "camera_cross_right_120fov",
    "camera_rear_right_70fov",
)

# 4-view train/test split (different from 7-view!)
TRAIN_SCENE_IDS_4VIEW = [
    "001", "002", "003", "004", "006", "007", "009", "010", "013", "015", "017", "019", "020",
    "024", "025", "026", "027", "028", "029", "032", "033", "034", "035", "036", "038", "039",
    "040", "041", "042", "045", "046", "047", "048", "049", "050", "052", "055", "056", "057",
    "058", "059", "060", "061", "064", "066", "068", "069", "070", "073", "074", "077", "078",
    "079", "080", "081", "082", "085", "088", "089",
]

TEST_SCENE_IDS_4VIEW = [
    "008", "012", "022", "030", "031", "037", "043", "044",
    "051", "054", "062", "065", "067", "072", "075", "076",
    "083", "084", "086", "087",
]

BLUR_DATASET_DIR = "/mnt/zihanw/proj_utils_pro/transfer_video_maker/output_full_data/BlurProjection"
DEPTH_DATASET_DIR = "/mnt/zihanw/proj_utils_pro/transfer_video_maker/output_full_data/DepthSparse"
HDMAP_DATASET_DIR = "/mnt/zihanw/proj_utils_pro/transfer_video_maker/output_full_data/HDMapBbox"


def parse_args():
    parser = argparse.ArgumentParser(description="4-view multi-control model inference")
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="Path to checkpoint directory")
    parser.add_argument("--experiment", type=str, default="custom_multi_control_post_train",
                        help="Experiment name")
    parser.add_argument("--output_dir", type=str,
                        default="/mnt/zihanw/Output_R2V_world_foundation_model_v1/inference_4view",
                        help="Output directory")
    parser.add_argument("--scene_ids", type=str, nargs="+", default=None,
                        help="Scene IDs to run inference on (e.g., 031 037)")
    parser.add_argument("--all_samples", action="store_true", default=False,
                        help="Run inference on ALL test samples")
    parser.add_argument("--sample_idx", type=int, default=None,
                        help="Single test sample index")
    parser.add_argument("--guidance", type=float, default=7.0)
    parser.add_argument("--num_steps", type=int, default=35)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--load_ema", action="store_true", default=True)
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


def create_test_dataset():
    """Create 4-view test dataset using 4-view train/test split."""
    from cosmos_transfer2.experiments.custom.custom_multi_control_dataset import (
        MultiControlMultiviewDataset,
    )

    logger.info(f"Creating 4-view test dataset with cameras: {CAMERAS_4VIEW}")
    logger.info(f"Test scenes ({len(TEST_SCENE_IDS_4VIEW)}): {TEST_SCENE_IDS_4VIEW}")

    test_dataset = MultiControlMultiviewDataset(
        blur_dataset_dir=BLUR_DATASET_DIR,
        depth_dataset_dir=DEPTH_DATASET_DIR,
        hdmap_dataset_dir=HDMAP_DATASET_DIR,
        resolution_hw=(720, 1280),
        num_video_frames=29,
        fps_downsample_factor=1,
        camera_keys=CAMERAS_4VIEW,
        single_caption_camera_name="camera_front_wide_120fov",
        add_view_prefix_to_caption=True,
        exclude_scene_ids=TRAIN_SCENE_IDS_4VIEW,
    )
    logger.info(f"Test dataset has {len(test_dataset)} samples")

    if len(test_dataset) == 0:
        raise ValueError("No test samples found!")

    return test_dataset


def get_sample_indices_for_scenes(test_dataset, scene_ids: List[str]) -> List[int]:
    """Get dataset indices for specific scene IDs."""
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
    """Load a single sample and prepare for inference."""
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
def run_inference_single(model, batch, guidance=7.0, num_steps=35,
                         num_conditional_frames=0, control_weight=1.0):
    """Run inference (same flow as evaluation_callback.py)."""
    if hasattr(model, 'inplace_compute_text_embeddings_online'):
        model.inplace_compute_text_embeddings_online(batch)
        logger.info("Text embeddings computed")

    raw_data, x0, condition = model.get_data_and_condition(batch)
    logger.info(f"raw_data shape: {raw_data.shape}, x0 (latent) shape: {x0.shape}")

    batch[NUM_CONDITIONAL_FRAMES_KEY] = num_conditional_frames
    batch[CONTROL_WEIGHT_KEY] = control_weight

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

    if hasattr(model, "decode"):
        generated = model.decode(sample)
        logger.info(f"Decoded video shape: {generated.shape}")
    else:
        generated = sample

    return generated, raw_data


def save_per_view_results(generated, batch, output_dir, sample_idx, fps=10):
    """Save per-view generated videos."""
    from cosmos_transfer2._src.imaginaire.visualize.video import save_img_or_video

    n_views = len(batch.get("view_indices_selection", [[0]])[0])
    sample_id = batch.get("__key__", [f"sample_{sample_idx}"])[0]

    sample_dir = Path(output_dir) / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    generated_01 = ((generated.float() + 1.0) / 2.0).clamp(0, 1)
    logger.info(f"Saving per-view results for {sample_id}: {n_views} views to {sample_dir}")

    gen_views = rearrange(generated_01, "B C (V T) H W -> V B C T H W", V=n_views)

    for v in range(n_views):
        cam_name = CAMERAS_4VIEW[v] if v < len(CAMERAS_4VIEW) else f"view_{v}"
        short_name = cam_name.replace("camera_", "")

        gen_path = sample_dir / f"{short_name}_generated"
        save_img_or_video(gen_views[v, 0], str(gen_path), fps=fps)

        logger.info(f"  View {v} ({short_name}): generated saved")

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
    num_views = 4  # Fixed for 4-view model
    context_parallel_size = 4  # Fixed for 4-view model

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    logger.info("=" * 60)
    logger.info("4-View Multi-Control Model Inference")
    logger.info("=" * 60)
    logger.info(f"Checkpoint: {args.ckpt_path}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Cameras: {CAMERAS_4VIEW}")
    logger.info(f"Test scenes: {TEST_SCENE_IDS_4VIEW}")
    logger.info(f"Context parallel size: {context_parallel_size}")
    logger.info(f"Guidance: {args.guidance}, Steps: {args.num_steps}")
    logger.info("=" * 60)

    process_group, is_rank0 = init_distributed(context_parallel_size)

    try:
        model, config = load_model_and_config(
            experiment_name=args.experiment,
            ckpt_path=args.ckpt_path,
            context_parallel_size=context_parallel_size,
            load_ema=args.load_ema,
            process_group=process_group,
        )

        test_dataset = create_test_dataset()

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
            sample_indices = [0]

        logger.info(f"Will process {len(sample_indices)} sample(s)")
        for idx in sample_indices:
            logger.info(f"  [{idx}] {test_dataset.samples[idx]}")

        for i, sample_idx in enumerate(sample_indices):
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing sample {i+1}/{len(sample_indices)}: "
                        f"{test_dataset.samples[sample_idx]} (idx={sample_idx})")
            logger.info(f"{'='*60}")

            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()

            batch = load_sample(test_dataset, sample_idx, model)

            generated, raw_data = run_inference_single(
                model=model,
                batch=batch,
                guidance=args.guidance,
                num_steps=args.num_steps,
            )

            if is_rank0:
                save_per_view_results(
                    generated=generated,
                    batch=batch,
                    output_dir=args.output_dir,
                    sample_idx=sample_idx,
                    fps=args.fps,
                )

            del batch, generated, raw_data
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()

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
