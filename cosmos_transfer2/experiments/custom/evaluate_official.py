# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Evaluation script for custom multi-control model using official framework.

Based on cosmos_transfer2/_src/predict2_multiview/scripts/inference.py

Usage:
```bash
# Single GPU evaluation (context_parallel_size=1)
PYTHONPATH=. torchrun --nproc_per_node=1 --master_port=12345 \
    -m cosmos_transfer2.experiments.custom.evaluate_official \
    --experiment custom_multi_control_post_train \
    --ckpt_path /path/to/checkpoints/iter_000005000 \
    --save_root /path/to/eval_output \
    --max_samples 100

# Multi-GPU evaluation with context parallelism
PYTHONPATH=. torchrun --nproc_per_node=8 --master_port=12345 \
    -m cosmos_transfer2.experiments.custom.evaluate_official \
    --experiment custom_multi_control_post_train \
    --ckpt_path /path/to/checkpoints/iter_000005000 \
    --context_parallel_size 8 \
    --save_root /path/to/eval_output
```
"""

import argparse
import json
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from loguru import logger
from megatron.core import parallel_state
from tqdm import tqdm

from cosmos_transfer2._src.imaginaire.lazy_config import instantiate
from cosmos_transfer2._src.imaginaire.utils import distributed, log
from cosmos_transfer2._src.imaginaire.visualize.video import save_img_or_video
from cosmos_transfer2._src.predict2.utils.model_loader import load_model_from_checkpoint

# Metrics
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    logger.warning("lpips not available, LPIPS metric will be skipped")

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.inception import InceptionScore
    FID_AVAILABLE = True
except ImportError:
    FID_AVAILABLE = False
    logger.warning("torchmetrics not available, FID metric will be skipped")


def to_model_input(data_batch, model):
    """Convert batch to model input format."""
    for k, v in data_batch.items():
        _v = v
        if isinstance(v, torch.Tensor):
            _v = _v.cuda()
            if torch.is_floating_point(v):
                _v = _v.to(**model.tensor_kwargs)
        data_batch[k] = _v
    return data_batch


class MetricsCalculator:
    """Calculate evaluation metrics: LPIPS, PSNR, SSIM."""

    def __init__(self, device):
        self.device = device
        self.lpips_fn = None
        if LPIPS_AVAILABLE:
            self.lpips_fn = lpips.LPIPS(net='alex').to(device)
            self.lpips_fn.eval()

    def compute_psnr(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Compute PSNR between prediction and target."""
        mse = F.mse_loss(pred, target)
        if mse == 0:
            return float('inf')
        return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()

    def compute_ssim(self, pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> float:
        """Compute SSIM between prediction and target."""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        # Create gaussian window
        sigma = 1.5
        gauss = torch.exp(-torch.arange(window_size).float().sub(window_size // 2).pow(2) / (2 * sigma ** 2))
        gauss = gauss / gauss.sum()
        window = gauss.unsqueeze(1) * gauss.unsqueeze(0)
        window = window.expand(pred.shape[1], 1, window_size, window_size).to(pred.device)

        mu1 = F.conv2d(pred, window, padding=window_size//2, groups=pred.shape[1])
        mu2 = F.conv2d(target, window, padding=window_size//2, groups=target.shape[1])

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(pred * pred, window, padding=window_size//2, groups=pred.shape[1]) - mu1_sq
        sigma2_sq = F.conv2d(target * target, window, padding=window_size//2, groups=target.shape[1]) - mu2_sq
        sigma12 = F.conv2d(pred * target, window, padding=window_size//2, groups=pred.shape[1]) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean().item()

    def compute_lpips(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Compute LPIPS between prediction and target."""
        if self.lpips_fn is None:
            return 0.0
        # LPIPS expects input in range [-1, 1]
        pred_scaled = pred * 2 - 1
        target_scaled = target * 2 - 1
        with torch.no_grad():
            return self.lpips_fn(pred_scaled, target_scaled).mean().item()


class MultiControlInference:
    """Handles inference for multi-control model using official framework."""

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

        # Load model using official framework
        logger.info(f"Loading model from checkpoint: {ckpt_path}")
        model, config = load_model_from_checkpoint(
            experiment_name=self.experiment_name,
            s3_checkpoint_dir=self.ckpt_path,
            config_file="cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/config.py",
            load_ema_to_reg=True,
        )

        # Enable context parallel if needed
        if self.context_parallel_size > 1:
            model.net.enable_context_parallel(self.process_group)
            self.rank0 = distributed.get_rank() == 0

        self.model = model
        self.config = config
        logger.info("Model loaded successfully")

    def _init_distributed(self):
        """Initialize distributed processing."""
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
        use_negative_prompt: bool = False,
        num_conditional_frames: int = 0,
        control_weight: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate video from batch using the same flow as evaluation callback.

        Based on cosmos_transfer2/experiments/custom/evaluation_callback.py

        Returns:
            Tuple of (generated_video, raw_gt_video)
            Both tensors are in range [-1, 1] with shape (B, C, V*T, H, W)
        """
        # Move data to model device (same as callback)
        data_batch = to_model_input(data_batch, self.model)

        # Set control parameters
        data_batch["num_conditional_frames"] = num_conditional_frames
        data_batch["control_weight"] = control_weight

        # Compute text embeddings online (same as training/callback)
        if hasattr(self.model, 'inplace_compute_text_embeddings_online'):
            self.model.inplace_compute_text_embeddings_online(data_batch)

        # Get raw data and condition (same as callback line 205)
        raw_data, x0, condition = self.model.get_data_and_condition(data_batch)

        # Generate samples (same as callback line 238-247)
        self.model.eval()
        sample = self.model.generate_samples_from_batch(
            data_batch,
            guidance=guidance,
            state_shape=x0.shape[1:],
            n_sample=x0.shape[0],
            num_steps=num_steps,
            is_negative_prompt=use_negative_prompt,
        )

        # Decode (same as callback line 246-247)
        if hasattr(self.model, "decode"):
            generated = self.model.decode(sample)
        else:
            generated = sample

        return generated, raw_data

    def cleanup(self):
        """Clean up distributed resources."""
        if "RANK" in os.environ:
            import torch.distributed as dist
            if parallel_state.is_initialized():
                parallel_state.destroy_model_parallel()
            dist.destroy_process_group()


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Multi-control model evaluation script")
    parser.add_argument("--experiment", type=str, required=True,
                        help="Experiment name (e.g., custom_multi_control_post_train)")
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="Path to checkpoint directory")
    parser.add_argument("--context_parallel_size", type=int, default=1,
                        help="Context parallel size (number of GPUs)")
    parser.add_argument("--guidance", type=float, default=7.0, help="Guidance scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_steps", type=int, default=35, help="Number of sampling steps")
    parser.add_argument("--max_samples", type=int, default=100, help="Maximum samples to evaluate")
    parser.add_argument("--save_root", type=str, default="results/eval", help="Output directory")
    parser.add_argument("--save_videos", action="store_true", help="Save comparison videos")
    parser.add_argument("--num_videos_to_save", type=int, default=10, help="Number of videos to save")
    parser.add_argument("--fps", type=int, default=10, help="FPS for saved videos")
    return parser.parse_args()


def main():
    os.environ["NVTE_FUSED_ATTN"] = "0"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_grad_enabled(False)

    args = parse_arguments()

    # Initialize inference
    inference = MultiControlInference(
        experiment_name=args.experiment,
        ckpt_path=args.ckpt_path,
        context_parallel_size=args.context_parallel_size,
    )

    # Create output directory
    save_root = Path(args.save_root)
    if inference.rank0:
        save_root.mkdir(parents=True, exist_ok=True)
        (save_root / "videos").mkdir(exist_ok=True)

    # Initialize metrics calculator
    device = torch.device("cuda")
    metrics_calc = MetricsCalculator(device)

    # Get dataloader - use validation/test data
    # Try to get test dataloader, fall back to train
    try:
        dataloader = instantiate(inference.config.dataloader_val)
        logger.info("Using validation dataloader")
    except Exception:
        dataloader = instantiate(inference.config.dataloader_train)
        logger.info("Using training dataloader (validation not available)")

    # Metrics storage
    all_psnr = []
    all_ssim = []
    all_lpips = []

    # Determine which samples to save videos for
    video_save_indices = set()
    if args.save_videos and args.num_videos_to_save > 0:
        num_to_save = min(args.num_videos_to_save, args.max_samples)
        if args.max_samples > 1:
            video_save_indices = set(
                int(i * (args.max_samples - 1) / (num_to_save - 1))
                for i in range(num_to_save)
            )

    # Evaluation loop
    logger.info(f"Starting evaluation on {args.max_samples} samples...")
    num_frames_per_sample = 0

    for i, batch in enumerate(tqdm(dataloader, total=args.max_samples, disable=not inference.rank0)):
        if i >= args.max_samples:
            break

        # Generate video using same flow as evaluation callback
        try:
            generated, raw_gt = inference.generate_from_batch(
                batch,
                guidance=args.guidance,
                num_steps=args.num_steps,
                use_negative_prompt=False,
                num_conditional_frames=0,  # No conditioning for evaluation
                control_weight=1.0,
            )
        except Exception as e:
            logger.error(f"Error generating sample {i}: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Convert from [-1, 1] to [0, 1] for metrics
        generated_01 = ((generated + 1.0) / 2.0).clamp(0, 1)
        gt_01 = ((raw_gt + 1.0) / 2.0).clamp(0, 1)

        # Ensure same shape
        if generated_01.shape != gt_01.shape:
            logger.warning(f"Shape mismatch: generated {generated_01.shape}, gt {gt_01.shape}")
            continue

        # Compute metrics (frame-wise)
        B, C, T, H, W = generated_01.shape
        num_frames_per_sample = T

        for t in range(T):
            pred_frame = generated_01[:, :, t:t+1, :, :]
            gt_frame = gt_01[:, :, t:t+1, :, :]

            # Reshape for metrics (B, C, H, W)
            pred_2d = pred_frame.squeeze(2)
            gt_2d = gt_frame.squeeze(2)

            psnr = metrics_calc.compute_psnr(pred_2d, gt_2d)
            ssim = metrics_calc.compute_ssim(pred_2d, gt_2d)
            lpips_val = metrics_calc.compute_lpips(pred_2d, gt_2d)

            all_psnr.append(psnr)
            all_ssim.append(ssim)
            all_lpips.append(lpips_val)

        # Save comparison video
        if inference.rank0 and i in video_save_indices:
            # Stack GT and generated side by side (in [0,1] range for save_img_or_video)
            comparison = torch.cat([gt_01, generated_01], dim=-1)  # Concat on width
            save_path = save_root / "videos" / f"sample_{i:04d}"
            save_img_or_video(comparison[0], str(save_path), fps=args.fps)
            logger.info(f"Saved comparison video: {save_path}")

    # Compute final metrics
    if inference.rank0:
        num_samples = len(all_psnr) // num_frames_per_sample if (all_psnr and num_frames_per_sample > 0) else 0
        metrics = {
            "psnr_mean": float(np.mean(all_psnr)) if all_psnr else 0,
            "psnr_std": float(np.std(all_psnr)) if all_psnr else 0,
            "ssim_mean": float(np.mean(all_ssim)) if all_ssim else 0,
            "ssim_std": float(np.std(all_ssim)) if all_ssim else 0,
            "lpips_mean": float(np.mean(all_lpips)) if all_lpips else 0,
            "lpips_std": float(np.std(all_lpips)) if all_lpips else 0,
            "num_samples": num_samples,
            "num_frames": len(all_psnr),
        }

        # Save metrics
        metrics_path = save_root / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info("=" * 50)
        logger.info("Evaluation Results:")
        logger.info(f"  PSNR:  {metrics['psnr_mean']:.4f} ± {metrics['psnr_std']:.4f}")
        logger.info(f"  SSIM:  {metrics['ssim_mean']:.4f} ± {metrics['ssim_std']:.4f}")
        logger.info(f"  LPIPS: {metrics['lpips_mean']:.4f} ± {metrics['lpips_std']:.4f}")
        logger.info(f"  Samples: {metrics['num_samples']}, Frames: {metrics['num_frames']}")
        logger.info(f"Results saved to: {metrics_path}")
        logger.info("=" * 50)

    # Cleanup
    inference.cleanup()


if __name__ == "__main__":
    main()
