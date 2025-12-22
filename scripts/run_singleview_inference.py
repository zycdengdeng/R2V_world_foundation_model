# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Inference script for single-view multi-control (blur, depth, hdmap) model with dropout.

Usage:
    cd /home/user/R2V_world_foundation_model

    # Single GPU inference
    PYTHONPATH=. python scripts/run_singleview_inference.py \
        --ckpt_path /mnt/zihanw/cosmos-transfer-output/cosmos_transfer_v2p5/zihanw_singleview/zihanw_singleview_dropout_train/checkpoints/iter_000030000/model_ema_bf16.pt \
        --save_root /mnt/zihanw/cosmos-transfer-output/inference_results/singleview_iter_30000

    # Multi-GPU inference (context parallel)
    PYTHONPATH=. torchrun --nproc_per_node=4 --master_port=12346 scripts/run_singleview_inference.py \
        --ckpt_path /mnt/zihanw/cosmos-transfer-output/cosmos_transfer_v2p5/zihanw_singleview/zihanw_singleview_dropout_train/checkpoints/iter_000030000/model_ema_bf16.pt \
        --context_parallel_size 4 \
        --save_root /mnt/zihanw/cosmos-transfer-output/inference_results/singleview_iter_30000
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
from loguru import logger

from cosmos_transfer2._src.imaginaire.lazy_config import instantiate
from cosmos_transfer2._src.imaginaire.utils import distributed
from cosmos_transfer2._src.imaginaire.visualize.video import save_img_or_video
from cosmos_transfer2._src.predict2.utils.model_loader import load_model_from_checkpoint

# Import dataloader to register it
import cosmos_transfer2.experiments.multiview.zihanw_singleview_dropout_training  # noqa: F401


def to_model_input(data_batch, model):
    """Move data to GPU and convert to model dtype."""
    for k, v in data_batch.items():
        if isinstance(v, torch.Tensor):
            v = v.cuda()
            if torch.is_floating_point(v):
                v = v.to(**model.tensor_kwargs)
            data_batch[k] = v
    return data_batch


# Single camera for single-view inference
CAMERAS_1VIEW = ("camera_front_wide_120fov",)


class SingleViewInference:
    """Inference handler for single-view multi-control model."""

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

    def _init_distributed(self):
        """Initialize distributed training."""
        distributed.init()

        if self.context_parallel_size > 1:
            from megatron.core import parallel_state

            parallel_state.initialize_model_parallel(
                context_parallel_size=self.context_parallel_size
            )
            self.process_group = parallel_state.get_context_parallel_group()

    @torch.no_grad()
    def generate_from_batch(
        self,
        batch: dict,
        guidance: float = 7.0,
        seed: int = 42,
        num_steps: int = 35,
        use_negative_prompt: bool = True,
    ) -> torch.Tensor:
        """Generate video from a data batch.

        Returns:
            Tensor of shape (1, 3, T, H, W) with values in [0, 1]
        """
        batch = to_model_input(batch, self.model)

        # Compute text embeddings if needed (tokenizes the text)
        if self.model.config.text_encoder_config is not None and self.model.config.text_encoder_config.compute_online:
            self.model.inplace_compute_text_embeddings_online(batch)

        # Get data and condition to determine state shape
        raw_data, x0, condition = self.model.get_data_and_condition(batch)

        self.model.eval()
        # Generate samples
        sample = self.model.generate_samples_from_batch(
            data_batch=batch,
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
        """Cleanup distributed state."""
        if "RANK" in os.environ:
            import torch.distributed as dist
            from megatron.core import parallel_state

            if parallel_state.is_initialized():
                parallel_state.destroy_model_parallel()
            dist.destroy_process_group()


def create_singleview_eval_dataloader():
    """Create a dataloader for single-view evaluation clips (075, 077)."""
    from torch.utils.data import DataLoader
    from cosmos_transfer2.experiments.multiview.zihanw_multicontrol_dataloader import (
        MultiControlMultiviewDataset,
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
        # Single view: front camera only
        selected_cameras=CAMERAS_1VIEW,
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
    parser = argparse.ArgumentParser(description="Single-view multi-control inference")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to checkpoint file (.pt)",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="zihanw_singleview_dropout_train",
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
        default="/mnt/zihanw/cosmos-transfer-output/inference_results/singleview",
        help="Directory to save results",
    )
    parser.add_argument("--guidance", type=float, default=7.0, help="Guidance scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_steps", type=int, default=35, help="Number of diffusion steps")
    parser.add_argument("--fps", type=int, default=10, help="FPS for saved videos")
    parser.add_argument("--max_samples", type=int, default=100, help="Maximum samples to generate")
    parser.add_argument(
        "--control_weight",
        type=float,
        default=1.0,
        help="Control weight for all control inputs (0.0-1.0). Default: 1.0",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Disable gradient computation
    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Initialize inference handler
    inference = SingleViewInference(
        experiment_name=args.experiment,
        ckpt_path=args.ckpt_path,
        context_parallel_size=args.context_parallel_size,
    )

    # Create eval dataloader
    dataloader = create_singleview_eval_dataloader()
    logger.info(f"Loaded {len(dataloader.dataset)} samples for evaluation")

    # Create output directory
    os.makedirs(args.save_root, exist_ok=True)

    # Generate samples
    for i, batch in enumerate(dataloader):
        if i >= args.max_samples:
            break

        sample_name = batch.get("__url__", [f"sample_{i}"])[0]
        if inference.rank0:
            logger.info(f"Processing sample {i+1}/{min(len(dataloader), args.max_samples)}: {sample_name}")

        # Save ground truth and control videos BEFORE model processing
        # (because generate_from_batch modifies batch tensors in-place)
        if inference.rank0:
            # Save ground truth video
            gt_video = batch["video"].clone().float() / 255.0  # Clone and normalize to [0, 1]
            if gt_video.dim() == 4:  # [C, T, H, W]
                gt_video = gt_video.unsqueeze(0)
            gt_path = os.path.join(args.save_root, f"{sample_name}_ground_truth")
            save_img_or_video(gt_video[0], gt_path, fps=args.fps)
            logger.info(f"Saved ground truth video to {gt_path}.mp4")

            # Save control videos
            for ctrl_name in ["blur", "depth", "hdmap_bbox"]:
                ctrl_key = f"control_input_{ctrl_name}"
                if ctrl_key in batch:
                    ctrl_video = batch[ctrl_key].clone().float() / 255.0
                    if ctrl_video.dim() == 4:
                        ctrl_video = ctrl_video.unsqueeze(0)
                    ctrl_path = os.path.join(args.save_root, f"{sample_name}_control_{ctrl_name}")
                    save_img_or_video(ctrl_video[0], ctrl_path, fps=args.fps)
                    logger.info(f"Saved control video to {ctrl_path}.mp4")

        # Set number of conditional frames (0 = unconditional)
        batch["num_conditional_frames"] = 0

        # Set control weight
        batch["control_weight"] = args.control_weight
        if inference.rank0:
            logger.info(f"Control weight: {args.control_weight}")

        # Generate video
        video = inference.generate_from_batch(
            batch,
            guidance=args.guidance,
            seed=args.seed,
            num_steps=args.num_steps,
        )

        # Save generated video (only on rank 0)
        if inference.rank0:
            # For single view, video shape is (B, C, T, H, W) - no view concatenation needed
            save_path = os.path.join(args.save_root, f"{sample_name}_generated")
            save_img_or_video(video[0], save_path, fps=args.fps)
            logger.info(f"Saved generated video to {save_path}.mp4")

    # Cleanup
    inference.cleanup()
    logger.info("Inference completed!")


if __name__ == "__main__":
    main()
