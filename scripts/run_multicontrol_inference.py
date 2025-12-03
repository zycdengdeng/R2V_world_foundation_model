# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Inference script for multi-control (blur, depth, hdmap) multiview model.

Usage:
    cd /home/user/R2V_world_foundation_model

    # Single GPU inference
    PYTHONPATH=. python scripts/run_multicontrol_inference.py \
        --ckpt_path /mnt/zihanw/cosmos-transfer-output/cosmos_transfer_v2p5/zihanw_multicontrol/zihanw_multicontrol_post_train/checkpoints/iter_000003000 \
        --save_root /mnt/zihanw/cosmos-transfer-output/inference_results/iter_3000

    # 2 GPU inference (context parallel)
    PYTHONPATH=. torchrun --nproc_per_node=2 --master_port=12345 scripts/run_multicontrol_inference.py \
        --ckpt_path /mnt/zihanw/cosmos-transfer-output/cosmos_transfer_v2p5/zihanw_multicontrol/zihanw_multicontrol_post_train/checkpoints/iter_000003000 \
        --context_parallel_size 2 \
        --save_root /mnt/zihanw/cosmos-transfer-output/inference_results/iter_3000
"""

import argparse
import os
import sys

# Add project root to path FIRST
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set environment variables BEFORE any other imports
os.environ["NVTE_FUSED_ATTN"] = "0"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

# Import HF mirror config BEFORE any huggingface imports
import cosmos_transfer2._src.imaginaire.utils.hf_mirror  # noqa: F401

import torch
from einops import rearrange
from loguru import logger

from cosmos_transfer2._src.imaginaire.lazy_config import instantiate
from cosmos_transfer2._src.imaginaire.utils import distributed
from cosmos_transfer2._src.imaginaire.visualize.video import save_img_or_video
from cosmos_transfer2._src.predict2.utils.model_loader import load_model_from_checkpoint

# Import dataloader to register it
import cosmos_transfer2.experiments.multiview.zihanw_multicontrol_dataloader  # noqa: F401


def to_model_input(data_batch, model):
    """Move data to GPU and convert to model dtype."""
    for k, v in data_batch.items():
        if isinstance(v, torch.Tensor):
            v = v.cuda()
            if torch.is_floating_point(v):
                v = v.to(**model.tensor_kwargs)
            data_batch[k] = v
    return data_batch


def time_to_width_dimension(mv_video, n_views):
    """Reshape video from (B, C, V*T, H, W) to (B, C, T, H, V*W)."""
    return rearrange(mv_video, "B C (V T) H W -> B C T H (V W)", V=n_views)


class MultiControlInference:
    """Inference handler for multi-control multiview model."""

    def __init__(
        self,
        experiment_name: str,
        ckpt_path: str,
        context_parallel_size: int = 1,
    ):
        self.experiment_name = experiment_name
        self.ckpt_path = ckpt_path
        self.context_parallel_size = context_parallel_size
        self.process_group = None
        self.rank0 = True

        if "RANK" in os.environ:
            self._init_distributed()

        # Load the model and config
        logger.info(f"Loading model from checkpoint: {ckpt_path}")
        model, config = load_model_from_checkpoint(
            experiment_name=self.experiment_name,
            s3_checkpoint_dir=self.ckpt_path,
            config_file="cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/config.py",
            load_ema_to_reg=True,
            experiment_opts=[],
        )

        # Enable context parallel on the model if using context parallelism
        if self.context_parallel_size > 1:
            from megatron.core import parallel_state
            model.net.enable_context_parallel(self.process_group, cp_comm_type="p2p")
            self.rank0 = distributed.get_rank() == 0

        self.model = model
        self.config = config
        logger.info("Model loaded successfully")

    def _init_distributed(self):
        """Initialize distributed processing for context parallelism."""
        from megatron.core import parallel_state

        distributed.init()
        parallel_state.initialize_model_parallel(
            context_parallel_size=self.context_parallel_size,
        )
        self.process_group = parallel_state.get_context_parallel_group()
        logger.info(f"Initialized context parallel with size {self.context_parallel_size}")
        logger.info(f"Current rank: {distributed.get_rank()}, World size: {distributed.get_world_size()}")

    def generate_from_batch(
        self,
        data_batch,
        guidance: float = 7.0,
        seed: int = 42,
        num_steps: int = 35,
        use_negative_prompt: bool = True,
    ):
        """Generate video from a batch of data.

        Returns:
            Tensor of shape (1, 3, V*T, H, W) with values in [0, 1]
        """
        data_batch = to_model_input(data_batch, self.model)

        # Compute text embeddings if needed
        if self.model.config.text_encoder_config is not None and self.model.config.text_encoder_config.compute_online:
            self.model.inplace_compute_text_embeddings_online(data_batch)

        raw_data, x0, condition = self.model.get_data_and_condition(data_batch)

        self.model.eval()
        sample = self.model.generate_samples_from_batch(
            data_batch,
            guidance=guidance,
            state_shape=x0.shape[1:],
            n_sample=x0.shape[0],
            seed=seed,
            num_steps=num_steps,
            is_negative_prompt=use_negative_prompt,
        )

        # Decode and normalize to [0, 1]
        video = ((self.model.decode(sample) + 1.0) / 2.0).clamp(0, 1)
        return video

    def cleanup(self):
        """Clean up distributed resources."""
        if "RANK" in os.environ:
            import torch.distributed as dist
            from megatron.core import parallel_state

            if parallel_state.is_initialized():
                parallel_state.destroy_model_parallel()
            dist.destroy_process_group()


def create_eval_dataloader():
    """Create a dataloader for evaluation clips (075, 077)."""
    from torch.utils.data import DataLoader
    from cosmos_transfer2.experiments.multiview.zihanw_multicontrol_dataloader import (
        MultiControlMultiviewDataset,
        CAMERAS_2VIEW,
        DEFAULT_CAMERAS,
        collate_fn,
    )

    dataset = MultiControlMultiviewDataset(
        base_video_dir="/mnt/zihanw/proj_utils_pro/transfer_video_maker/output/BlurProjection",
        control_dirs={
            "blur": "/mnt/zihanw/proj_utils_pro/transfer_video_maker/output/BlurProjection/control_input_blur",
            "depth": "/mnt/zihanw/proj_utils_pro/transfer_video_maker/output/DepthSparse/control_input_depth",
            "hdmap_bbox": "/mnt/zihanw/proj_utils_pro/transfer_video_maker/output/HDMapBbox/control_input_hdmap_bbox",
        },
        folder_to_camera_key={f"ftheta_{camera_name}": camera_name for camera_name in DEFAULT_CAMERAS},
        resolution_hw=(720, 1280),
        num_video_frames=29,
        single_caption_camera_name="camera_front_wide_120fov",
        selected_cameras=CAMERAS_2VIEW,
        # Only include eval clips
        include_only_clips=("075", "077"),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    return dataloader


def parse_arguments():
    parser = argparse.ArgumentParser(description="Multi-control multiview inference")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to checkpoint directory",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="zihanw_multicontrol_post_train",
        help="Experiment name",
    )
    parser.add_argument(
        "--context_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs for context parallelism",
    )
    parser.add_argument(
        "--save_root",
        type=str,
        default="/mnt/zihanw/cosmos-transfer-output/inference_results",
        help="Directory to save results",
    )
    parser.add_argument("--guidance", type=float, default=7.0, help="Guidance scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_steps", type=int, default=35, help="Number of diffusion steps")
    parser.add_argument("--fps", type=int, default=10, help="FPS for saved videos")
    parser.add_argument("--max_samples", type=int, default=100, help="Maximum samples to generate")
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Disable gradient computation
    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Initialize inference handler
    inference = MultiControlInference(
        experiment_name=args.experiment,
        ckpt_path=args.ckpt_path,
        context_parallel_size=args.context_parallel_size,
    )

    # Create eval dataloader
    logger.info("Creating evaluation dataloader for clips 075 and 077...")
    dataloader = create_eval_dataloader()
    logger.info(f"Found {len(dataloader.dataset)} samples for evaluation")

    # Create save directory
    os.makedirs(args.save_root, exist_ok=True)

    # Generate samples
    for i, batch in enumerate(dataloader):
        if i >= args.max_samples:
            break

        sample_name = batch.get("__url__", [f"sample_{i}"])[0]
        logger.info(f"Processing sample {i+1}/{min(len(dataloader.dataset), args.max_samples)}: {sample_name}")

        # Set number of conditional frames (0 = unconditional)
        batch["num_conditional_frames"] = 0

        # Generate video
        video = inference.generate_from_batch(
            batch,
            guidance=args.guidance,
            seed=args.seed,
            num_steps=args.num_steps,
        )

        # Save results (only on rank 0)
        if inference.rank0:
            n_views = batch["sample_n_views"].item()

            # Save generated video (all views stacked horizontally)
            video_wide = time_to_width_dimension(video, n_views)
            save_path = os.path.join(args.save_root, f"{sample_name}_generated")
            save_img_or_video(video_wide[0], save_path, fps=args.fps)
            logger.info(f"Saved generated video to {save_path}.mp4")

            # Save ground truth video for comparison
            gt_video = batch["video"].float() / 255.0  # Normalize to [0, 1]
            # batch["video"] already has batch dimension from dataloader
            if gt_video.dim() == 4:  # [C, T, H, W]
                gt_video = gt_video.unsqueeze(0)  # Add batch dimension
            gt_wide = time_to_width_dimension(gt_video, n_views)
            gt_path = os.path.join(args.save_root, f"{sample_name}_ground_truth")
            save_img_or_video(gt_wide[0], gt_path, fps=args.fps)
            logger.info(f"Saved ground truth video to {gt_path}.mp4")

            # Save control inputs for reference
            for ctrl_name in ["blur", "depth", "hdmap_bbox"]:
                ctrl_key = f"control_input_{ctrl_name}"
                if ctrl_key in batch:
                    ctrl_video = batch[ctrl_key].float() / 255.0
                    if ctrl_video.dim() == 4:  # [C, T, H, W]
                        ctrl_video = ctrl_video.unsqueeze(0)
                    ctrl_wide = time_to_width_dimension(ctrl_video, n_views)
                    ctrl_path = os.path.join(args.save_root, f"{sample_name}_control_{ctrl_name}")
                    save_img_or_video(ctrl_wide[0], ctrl_path, fps=args.fps)

    # Cleanup
    inference.cleanup()
    logger.info("Inference completed!")


if __name__ == "__main__":
    main()
