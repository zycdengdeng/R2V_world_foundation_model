# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Evaluation script for Multi-Control Post Training
# Calculates FID, FVD, LPIPS metrics on test set
#
# Usage:
#   # First, install dependencies in your conda environment:
#   pip install lpips scipy
#
#   # Then run evaluation:
#   CUDA_VISIBLE_DEVICES=0,6 python -m cosmos_transfer2.experiments.custom.evaluate_metrics \
#       --checkpoint_path /path/to/checkpoint \
#       --output_dir /path/to/output

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from loguru import logger
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Check optional dependencies
LPIPS_AVAILABLE = False
SCIPY_AVAILABLE = False

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    logger.warning("lpips not installed. LPIPS metric will use MSE fallback. Install with: pip install lpips")

try:
    from scipy import linalg
    SCIPY_AVAILABLE = True
except ImportError:
    logger.warning("scipy not installed. FID/FVD will use simplified calculation. Install with: pip install scipy")


# ============================================================================
# Metric Implementations
# ============================================================================

def calculate_psnr(real: torch.Tensor, fake: torch.Tensor) -> float:
    """Calculate Peak Signal-to-Noise Ratio.

    Args:
        real: (N, C, H, W) tensor with values in [0, 1]
        fake: (N, C, H, W) tensor with values in [0, 1]

    Returns:
        Average PSNR in dB
    """
    mse = F.mse_loss(real, fake, reduction='none').mean(dim=[1, 2, 3])
    psnr = 10 * torch.log10(1.0 / (mse + 1e-10))
    return psnr.mean().item()


def calculate_ssim(real: torch.Tensor, fake: torch.Tensor, window_size: int = 11) -> float:
    """Calculate Structural Similarity Index.

    Args:
        real: (N, C, H, W) tensor with values in [0, 1]
        fake: (N, C, H, W) tensor with values in [0, 1]
        window_size: Size of the Gaussian window

    Returns:
        Average SSIM
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Create Gaussian window
    sigma = 1.5
    gauss = torch.exp(-torch.arange(window_size).float().sub(window_size // 2).pow(2) / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()
    window = gauss.unsqueeze(1) @ gauss.unsqueeze(0)
    window = window.unsqueeze(0).unsqueeze(0).to(real.device)

    # Expand window for all channels
    C = real.shape[1]
    window = window.expand(C, 1, window_size, window_size)

    # Calculate means
    mu1 = F.conv2d(real, window, padding=window_size // 2, groups=C)
    mu2 = F.conv2d(fake, window, padding=window_size // 2, groups=C)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # Calculate variances and covariance
    sigma1_sq = F.conv2d(real * real, window, padding=window_size // 2, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(fake * fake, window, padding=window_size // 2, groups=C) - mu2_sq
    sigma12 = F.conv2d(real * fake, window, padding=window_size // 2, groups=C) - mu1_mu2

    # Calculate SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean().item()


class FIDCalculator:
    """Calculate Frechet Inception Distance using InceptionV3."""

    def __init__(self, device: torch.device):
        self.device = device
        self.inception = None
        self._load_inception()

    def _load_inception(self):
        """Load InceptionV3 model for feature extraction."""
        try:
            from torchvision.models import inception_v3, Inception_V3_Weights
            self.inception = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
            self.inception.fc = nn.Identity()  # Remove final classification layer
            self.inception.eval()
            self.inception.to(self.device)
            logger.info("Loaded InceptionV3 for FID calculation")
        except Exception as e:
            logger.error(f"Failed to load InceptionV3: {e}")
            raise

    @torch.no_grad()
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract features from images.

        Args:
            images: (N, C, H, W) tensor with values in [0, 1]

        Returns:
            (N, 2048) feature tensor
        """
        # Resize to 299x299 for InceptionV3
        images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
        # Normalize to ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        images = (images - mean) / std

        features = self.inception(images)
        return features

    def calculate_statistics(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate mean and covariance of features."""
        mu = features.mean(dim=0)
        features_centered = features - mu
        cov = features_centered.T @ features_centered / (features.shape[0] - 1)
        return mu, cov

    def calculate_fid(self, real_features: torch.Tensor, fake_features: torch.Tensor) -> float:
        """Calculate FID between real and fake feature distributions."""
        mu1, sigma1 = self.calculate_statistics(real_features)
        mu2, sigma2 = self.calculate_statistics(fake_features)

        # Calculate FID
        diff = mu1 - mu2

        # Matrix square root using eigendecomposition
        sigma1_np = sigma1.cpu().numpy()
        sigma2_np = sigma2.cpu().numpy()

        if SCIPY_AVAILABLE:
            try:
                covmean, _ = linalg.sqrtm(sigma1_np @ sigma2_np, disp=False)
                if np.iscomplexobj(covmean):
                    covmean = covmean.real
                fid = float(diff @ diff + np.trace(sigma1_np) + np.trace(sigma2_np) - 2 * np.trace(covmean))
            except Exception as e:
                logger.warning(f"scipy sqrtm failed: {e}. Using simplified FID.")
                fid = float(diff @ diff + np.trace(sigma1_np) + np.trace(sigma2_np))
        else:
            # Simplified FID without matrix square root
            fid = float(diff @ diff + np.trace(sigma1_np) + np.trace(sigma2_np))

        return fid


class LPIPSCalculator:
    """Calculate LPIPS (Learned Perceptual Image Patch Similarity)."""

    def __init__(self, device: torch.device, net: str = 'alex'):
        self.device = device
        self.net = net
        self.model = None
        self.use_fallback = False
        self._load_model()

    def _load_model(self):
        """Load LPIPS model."""
        if LPIPS_AVAILABLE:
            try:
                self.model = lpips.LPIPS(net=self.net).to(self.device)
                self.model.eval()
                logger.info(f"Loaded LPIPS model with {self.net} backbone")
            except Exception as e:
                logger.warning(f"Failed to load LPIPS model: {e}. Using MSE fallback.")
                self.use_fallback = True
        else:
            logger.warning("lpips not available. Using MSE-based perceptual similarity as fallback.")
            self.use_fallback = True

    @torch.no_grad()
    def calculate(self, real: torch.Tensor, fake: torch.Tensor) -> float:
        """Calculate LPIPS between real and fake images.

        Args:
            real: (N, C, H, W) tensor with values in [0, 1]
            fake: (N, C, H, W) tensor with values in [0, 1]

        Returns:
            Average LPIPS score (or MSE-based fallback)
        """
        if self.use_fallback:
            # Use MSE as fallback (normalized to similar range as LPIPS)
            mse = F.mse_loss(real, fake, reduction='none').mean(dim=[1, 2, 3])
            return mse.mean().item()

        # LPIPS expects values in [-1, 1]
        real = real * 2 - 1
        fake = fake * 2 - 1

        # Calculate per-sample LPIPS
        lpips_values = []
        for i in range(real.shape[0]):
            lpips_val = self.model(real[i:i+1], fake[i:i+1])
            lpips_values.append(lpips_val.item())

        return np.mean(lpips_values)


class FVDCalculator:
    """Calculate Frechet Video Distance using I3D features."""

    def __init__(self, device: torch.device):
        self.device = device
        self.i3d = None
        self._load_i3d()

    def _load_i3d(self):
        """Load I3D model for video feature extraction."""
        try:
            # Try to use torchvision's video models or a simple 3D CNN
            from torchvision.models.video import r3d_18, R3D_18_Weights
            self.i3d = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
            self.i3d.fc = nn.Identity()  # Remove classification layer
            self.i3d.eval()
            self.i3d.to(self.device)
            logger.info("Loaded R3D-18 for FVD calculation")
        except Exception as e:
            logger.warning(f"Could not load R3D-18: {e}. Using simplified FVD.")
            self.i3d = None

    @torch.no_grad()
    def extract_features(self, video: torch.Tensor) -> torch.Tensor:
        """Extract features from video.

        Args:
            video: (N, C, T, H, W) tensor with values in [0, 1]

        Returns:
            (N, D) feature tensor
        """
        if self.i3d is None:
            # Fallback: use frame-wise features averaged over time
            N, C, T, H, W = video.shape
            video_flat = rearrange(video, 'n c t h w -> (n t) c h w')
            # Simple feature extraction using average pooling
            features = F.adaptive_avg_pool2d(video_flat, (7, 7))
            features = rearrange(features, '(n t) c h w -> n (t c h w)', n=N, t=T)
            return features

        # Resize for I3D (expects 112x112 or similar)
        video = F.interpolate(
            rearrange(video, 'n c t h w -> (n t) c h w'),
            size=(112, 112),
            mode='bilinear',
            align_corners=False
        )
        video = rearrange(video, '(n t) c h w -> n c t h w', n=video.shape[0] // video.shape[2] if hasattr(video, 'shape') else 1)

        # Normalize
        mean = torch.tensor([0.43216, 0.394666, 0.37645], device=self.device).view(1, 3, 1, 1, 1)
        std = torch.tensor([0.22803, 0.22145, 0.216989], device=self.device).view(1, 3, 1, 1, 1)
        video = (video - mean) / std

        features = self.i3d(video)
        return features

    def calculate_fvd(self, real_features: torch.Tensor, fake_features: torch.Tensor) -> float:
        """Calculate FVD between real and fake video features."""
        mu1 = real_features.mean(dim=0)
        mu2 = fake_features.mean(dim=0)

        sigma1 = torch.cov(real_features.T)
        sigma2 = torch.cov(fake_features.T)

        diff = mu1 - mu2

        # Matrix square root
        sigma1_np = sigma1.cpu().numpy()
        sigma2_np = sigma2.cpu().numpy()

        # Add small epsilon for numerical stability
        eps = 1e-6
        sigma1_np = sigma1_np + eps * np.eye(sigma1_np.shape[0])
        sigma2_np = sigma2_np + eps * np.eye(sigma2_np.shape[0])

        if SCIPY_AVAILABLE:
            try:
                covmean, _ = linalg.sqrtm(sigma1_np @ sigma2_np, disp=False)
                if np.iscomplexobj(covmean):
                    covmean = covmean.real
                fvd = float(diff @ diff + np.trace(sigma1_np) + np.trace(sigma2_np) - 2 * np.trace(covmean))
            except Exception as e:
                logger.warning(f"FVD calculation failed: {e}. Using simplified metric.")
                fvd = float(torch.norm(mu1 - mu2).item())
        else:
            # Simplified FVD without matrix square root
            fvd = float(diff @ diff + np.trace(sigma1_np) + np.trace(sigma2_np))

        return fvd


# ============================================================================
# Data Loading
# ============================================================================

def load_test_dataset():
    """Load the test dataset."""
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

    # Create test dataset (exclude training scenes = keep only test scenes)
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

    logger.info(f"Loaded test dataset with {len(test_dataset)} samples")
    return test_dataset


# ============================================================================
# Model Loading and Inference
# ============================================================================

def load_model(checkpoint_path: str, device: torch.device):
    """Load the trained model from checkpoint.

    Handles both .pt files and DCP (Distributed Checkpoint) directories.
    """
    import importlib
    import os
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint import FileSystemReader

    from cosmos_transfer2._src.imaginaire.lazy_config import instantiate
    from cosmos_transfer2._src.imaginaire.utils import misc
    from cosmos_transfer2._src.imaginaire.utils.config_helper import get_config_module, override
    from cosmos_transfer2._src.predict2.checkpointer.dcp import DefaultLoadPlanner, ModelWrapper

    # Import custom experiment to register it
    from cosmos_transfer2.experiments.custom import custom_multi_control_experiment

    # Load config
    config_file = "cosmos_transfer2/experiments/custom/config.py"
    config_module = get_config_module(config_file)
    config = importlib.import_module(config_module).make_config()
    config = override(config, ["--", "experiment=custom_multi_control_post_train"])

    # Override checkpoint path
    config.checkpoint.load_path = str(checkpoint_path)

    # Disable EMA since we're loading EMA weights to regular model
    config.model.config.ema.enabled = False

    # Disable FSDP for single-GPU evaluation
    config.model.config.fsdp_shard_size = 1

    # Validate and freeze config
    config.validate()
    config.freeze()
    misc.set_random_seed(seed=42, by_rank=True)

    # Set up CUDA
    torch.backends.cudnn.deterministic = config.trainer.cudnn.deterministic
    torch.backends.cudnn.benchmark = config.trainer.cudnn.benchmark
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = True

    # Instantiate model
    logger.info("Instantiating model...")
    with misc.timer("instantiate model"):
        model = instantiate(config.model).cuda()
        model.on_train_start()

    # Determine checkpoint format
    is_dcp = os.path.isdir(checkpoint_path)

    if is_dcp:
        # Load DCP checkpoint
        logger.info(f"Loading DCP checkpoint from {checkpoint_path}")

        # The model/ subdirectory contains the model state
        model_ckpt_path = os.path.join(checkpoint_path, "model")
        if not os.path.isdir(model_ckpt_path):
            model_ckpt_path = checkpoint_path

        # Load checkpoint state dict
        storage_reader = FileSystemReader(model_ckpt_path)
        load_planner = DefaultLoadPlanner()

        # Get the model's net state dict structure for loading
        # Skip _extra_state keys (TransformerEngine FP8 metadata, not needed for inference)
        net_state_dict = {}
        for k, v in model.net.state_dict().items():
            if "_extra_state" in k:
                continue  # Skip TransformerEngine extra state
            # Map model keys to checkpoint keys (net_ema. prefix for EMA weights)
            ckpt_key = f"net_ema.{k}"
            net_state_dict[ckpt_key] = torch.zeros_like(v)

        # Load from checkpoint
        dcp.load(
            net_state_dict,
            storage_reader=storage_reader,
            planner=load_planner,
        )

        # Remap keys: remove net_ema. prefix, skip _extra_state
        remapped_state_dict = {}
        for k, v in net_state_dict.items():
            if "_extra_state" in k:
                continue  # Skip TransformerEngine extra state
            if k.startswith("net_ema."):
                new_key = k[len("net_ema."):]
                remapped_state_dict[new_key] = v
            else:
                remapped_state_dict[k] = v

        # Load into model.net with strict=False (will have missing _extra_state keys, that's OK)
        missing, unexpected = model.net.load_state_dict(remapped_state_dict, strict=False)
        # Filter out _extra_state from missing keys for cleaner logging
        missing = [k for k in missing if "_extra_state" not in k]
        if missing:
            logger.warning(f"Missing keys (first 5): {missing[:5]}")
        if unexpected:
            logger.warning(f"Unexpected keys (first 5): {unexpected[:5]}")

        logger.info(f"Successfully loaded DCP checkpoint from {model_ckpt_path}")
    else:
        # Load .pt checkpoint
        logger.info(f"Loading .pt checkpoint from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model.load_state_dict(state_dict, strict=False)

    model.eval()
    model.to(device)

    logger.info(f"Model loaded and ready for evaluation")
    return model, config


def prepare_batch_for_inference(batch: Dict, model, device: torch.device) -> Dict:
    """Prepare a batch for inference."""
    from cosmos_transfer2.experiments.custom.custom_multi_control_dataset import collate_fn

    # Move to device
    uint8_keys = {'video', 'control_input_blur', 'control_input_depth', 'control_input_hdmap_bbox'}

    prepared = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            if key in uint8_keys:
                prepared[key] = value.to(device=device)
            else:
                prepared[key] = value.to(device=device, dtype=torch.bfloat16)
        else:
            prepared[key] = value

    return prepared


@torch.no_grad()
def run_inference(model, batch: Dict, guidance: float = 7.0, num_steps: int = 35, seed: int = 42) -> torch.Tensor:
    """Run inference on a batch and return generated video.

    Returns:
        Generated video tensor (B, C, T, H, W) with values in [0, 1]
    """
    from cosmos_transfer2._src.transfer2_multiview.inference.inference import (
        to_model_input,
    )

    # Prepare input
    batch = to_model_input(batch, model)

    # Compute text embeddings if needed
    if model.config.text_encoder_config is not None and model.config.text_encoder_config.compute_online:
        model.inplace_compute_text_embeddings_online(batch)

    # Get data and condition
    raw_data, x0, condition = model.get_data_and_condition(batch)

    # Generate samples
    model.eval()
    sample = model.generate_samples_from_batch(
        batch,
        guidance=guidance,
        state_shape=x0.shape[1:],
        n_sample=x0.shape[0],
        seed=seed,
        num_steps=num_steps,
        is_negative_prompt=True,
    )

    # Decode and normalize to [0, 1]
    generated = ((model.decode(sample) + 1.0) / 2.0).clamp(0, 1)

    return generated


# ============================================================================
# Main Evaluation Loop
# ============================================================================

def evaluate(
    checkpoint_path: str,
    output_dir: str,
    num_samples: Optional[int] = None,
    guidance: float = 7.0,
    num_steps: int = 35,
    seed: int = 42,
    save_videos: bool = True,
):
    """Run evaluation on test set and compute metrics.

    Args:
        checkpoint_path: Path to model checkpoint
        output_dir: Directory to save results
        num_samples: Number of samples to evaluate (None = all)
        guidance: CFG guidance scale
        num_steps: Number of sampling steps
        seed: Random seed
        save_videos: Whether to save generated videos
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Starting evaluation on device: {device}")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Output directory: {output_dir}")

    # Load test dataset
    test_dataset = load_test_dataset()

    if num_samples is not None:
        num_samples = min(num_samples, len(test_dataset))
    else:
        num_samples = len(test_dataset)

    logger.info(f"Evaluating {num_samples} samples")

    # Determine which samples to save videos for (10 evenly spaced)
    num_videos_to_save = min(10, num_samples)
    if num_samples > 1:
        video_save_indices = set(
            int(i * (num_samples - 1) / (num_videos_to_save - 1))
            for i in range(num_videos_to_save)
        )
    else:
        video_save_indices = {0}
    logger.info(f"Will save comparison videos for {len(video_save_indices)} samples: {sorted(video_save_indices)}")

    # Load model
    model, config = load_model(checkpoint_path, device)

    # Initialize distributed and Megatron parallel state for single-GPU inference
    import torch.distributed as dist
    from megatron.core import parallel_state

    if not dist.is_initialized():
        # Initialize torch.distributed for single process
        os.environ.setdefault('MASTER_ADDR', 'localhost')
        os.environ.setdefault('MASTER_PORT', '12355')
        os.environ.setdefault('RANK', '0')
        os.environ.setdefault('WORLD_SIZE', '1')
        dist.init_process_group(backend='nccl', rank=0, world_size=1)
        logger.info("Initialized torch.distributed for single-GPU inference")

    if not parallel_state.is_initialized():
        parallel_state.initialize_model_parallel(
            context_parallel_size=1,  # Single GPU, no context parallelism
        )
        logger.info("Initialized Megatron parallel state with context_parallel_size=1")

    # Set cp_mesh to None for single-GPU text encoder inference
    # Structure: model.text_encoder.model (QwenVLBaseModel) -> model.text_encoder.model.model (Qwen2VLModel)
    if hasattr(model, 'text_encoder') and model.text_encoder is not None:
        te = model.text_encoder
        # TextEncoder.model is QwenVLBaseModel, QwenVLBaseModel.model is Qwen2VLModel
        if hasattr(te, 'model') and te.model is not None:
            qwen_vl_base = te.model  # QwenVLBaseModel
            if hasattr(qwen_vl_base, 'model') and qwen_vl_base.model is not None:
                inner_model = qwen_vl_base.model  # Qwen2VLModel or Qwen2_5_VLModel
                if hasattr(inner_model, 'set_cp_mesh'):
                    inner_model.set_cp_mesh(None)
                    logger.info("Set text_encoder.model.model.cp_mesh = None via set_cp_mesh() for single-GPU inference")
                elif hasattr(inner_model, 'cp_mesh'):
                    try:
                        inner_model.cp_mesh = None
                        logger.info("Set text_encoder.model.model.cp_mesh = None for single-GPU inference")
                    except Exception as e:
                        logger.warning(f"Could not set cp_mesh: {e}")

    # Initialize metric calculators
    fid_calc = FIDCalculator(device)
    lpips_calc = LPIPSCalculator(device)
    fvd_calc = FVDCalculator(device)

    # Collect features and metrics
    real_frame_features = []
    fake_frame_features = []
    real_video_features = []
    fake_video_features = []
    lpips_scores = []
    psnr_scores = []
    ssim_scores = []

    from cosmos_transfer2.experiments.custom.custom_multi_control_dataset import collate_fn

    for idx in tqdm(range(num_samples), desc="Evaluating"):
        try:
            # Load sample
            sample = test_dataset[idx]
            batch = collate_fn([sample])
            batch = prepare_batch_for_inference(batch, model, device)

            # Get ground truth video
            gt_video = batch['video'].float() / 255.0  # (B, C, V*T, H, W)

            # Run inference
            gen_video = run_inference(model, batch, guidance=guidance, num_steps=num_steps, seed=seed)

            # Extract frames for FID
            B, C, VT, H, W = gt_video.shape
            gt_frames = rearrange(gt_video, 'b c t h w -> (b t) c h w')
            gen_frames = rearrange(gen_video, 'b c t h w -> (b t) c h w')

            # Sample frames for efficiency (every 4th frame)
            sample_indices = list(range(0, gt_frames.shape[0], 4))
            gt_frames_sampled = gt_frames[sample_indices]
            gen_frames_sampled = gen_frames[sample_indices]

            # Extract FID features
            gt_feats = fid_calc.extract_features(gt_frames_sampled)
            gen_feats = fid_calc.extract_features(gen_frames_sampled)
            real_frame_features.append(gt_feats.cpu())
            fake_frame_features.append(gen_feats.cpu())

            # Calculate LPIPS for this sample
            lpips_score = lpips_calc.calculate(gt_frames_sampled, gen_frames_sampled)
            lpips_scores.append(lpips_score)

            # Calculate PSNR and SSIM
            psnr_score = calculate_psnr(gt_frames_sampled, gen_frames_sampled)
            ssim_score = calculate_ssim(gt_frames_sampled, gen_frames_sampled)
            psnr_scores.append(psnr_score)
            ssim_scores.append(ssim_score)

            # Extract video features for FVD
            # Reshape to (B, C, T, H, W) format
            n_views = 4  # Number of camera views
            T = VT // n_views
            gt_video_reshaped = rearrange(gt_video, 'b c (v t) h w -> (b v) c t h w', v=n_views)
            gen_video_reshaped = rearrange(gen_video, 'b c (v t) h w -> (b v) c t h w', v=n_views)

            gt_vid_feats = fvd_calc.extract_features(gt_video_reshaped)
            gen_vid_feats = fvd_calc.extract_features(gen_video_reshaped)
            real_video_features.append(gt_vid_feats.cpu())
            fake_video_features.append(gen_vid_feats.cpu())

            # Save videos for selected samples only (10 evenly spaced)
            if save_videos and idx in video_save_indices:
                save_comparison_video(
                    gt_video[0], gen_video[0], batch,
                    os.path.join(output_dir, f"sample_{idx:04d}.mp4"),
                    fps=10
                )
                logger.info(f"  Saved comparison video for sample {idx}")

            logger.info(f"Sample {idx}: LPIPS={lpips_score:.4f}, PSNR={psnr_score:.2f}dB, SSIM={ssim_score:.4f}")

        except Exception as e:
            logger.error(f"Error processing sample {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Calculate final metrics
    logger.info("Calculating final metrics...")

    # Concatenate all features
    real_frame_features = torch.cat(real_frame_features, dim=0)
    fake_frame_features = torch.cat(fake_frame_features, dim=0)
    real_video_features = torch.cat(real_video_features, dim=0)
    fake_video_features = torch.cat(fake_video_features, dim=0)

    # Calculate FID
    fid_score = fid_calc.calculate_fid(
        real_frame_features.to(device),
        fake_frame_features.to(device)
    )

    # Calculate FVD
    fvd_score = fvd_calc.calculate_fvd(
        real_video_features.to(device),
        fake_video_features.to(device)
    )

    # Calculate average metrics
    avg_lpips = np.mean(lpips_scores)
    avg_psnr = np.mean(psnr_scores)
    avg_ssim = np.mean(ssim_scores)

    # Print and save results
    results = {
        'FID': fid_score,
        'FVD': fvd_score,
        'LPIPS': avg_lpips,
        'PSNR': avg_psnr,
        'SSIM': avg_ssim,
        'num_samples': num_samples,
        'checkpoint': checkpoint_path,
    }

    logger.info("=" * 50)
    logger.info("Evaluation Results:")
    logger.info(f"  FID:   {fid_score:.4f}")
    logger.info(f"  FVD:   {fvd_score:.4f}")
    logger.info(f"  LPIPS: {avg_lpips:.4f}")
    logger.info(f"  PSNR:  {avg_psnr:.2f} dB")
    logger.info(f"  SSIM:  {avg_ssim:.4f}")
    logger.info("=" * 50)

    # Save results to file
    results_path = os.path.join(output_dir, "metrics.txt")
    with open(results_path, 'w') as f:
        f.write("Evaluation Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Num samples: {num_samples}\n")
        f.write(f"Guidance: {guidance}\n")
        f.write(f"Num steps: {num_steps}\n")
        f.write("=" * 50 + "\n")
        f.write(f"FID:   {fid_score:.4f}\n")
        f.write(f"FVD:   {fvd_score:.4f}\n")
        f.write(f"LPIPS: {avg_lpips:.4f}\n")
        f.write(f"PSNR:  {avg_psnr:.2f} dB\n")
        f.write(f"SSIM:  {avg_ssim:.4f}\n")

    logger.info(f"Results saved to {results_path}")

    return results


def save_comparison_video(
    gt_video: torch.Tensor,
    gen_video: torch.Tensor,
    batch: Dict,
    output_path: str,
    fps: int = 10,
):
    """Save a comparison video with ground truth and generated side by side.

    Args:
        gt_video: (C, V*T, H, W) ground truth video
        gen_video: (C, V*T, H, W) generated video
        batch: Original batch with control inputs
        output_path: Path to save video
        fps: Frames per second
    """
    import cv2

    C, VT, H, W = gt_video.shape
    n_views = 4
    T = VT // n_views

    # Reshape to (V, T, H, W, C)
    gt = rearrange(gt_video, 'c (v t) h w -> v t h w c', v=n_views)
    gen = rearrange(gen_video, 'c (v t) h w -> v t h w c', v=n_views)

    # Convert to numpy uint8
    gt = (gt.cpu().numpy() * 255).astype(np.uint8)
    gen = (gen.cpu().numpy() * 255).astype(np.uint8)

    # Create video writer
    # Stack views horizontally, GT and Gen vertically
    frame_h = H * 2
    frame_w = W * n_views

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))

    for t in range(T):
        # Stack views horizontally
        gt_frame = np.concatenate([gt[v, t] for v in range(n_views)], axis=1)
        gen_frame = np.concatenate([gen[v, t] for v in range(n_views)], axis=1)

        # Stack GT and Gen vertically
        frame = np.concatenate([gt_frame, gen_frame], axis=0)

        # Convert RGB to BGR for OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)

    out.release()


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate model on test set")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to model checkpoint directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./eval_results",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (default: all)"
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=7.0,
        help="CFG guidance scale"
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=35,
        help="Number of sampling steps"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--no_save_videos",
        action="store_true",
        help="Don't save comparison videos"
    )

    args = parser.parse_args()

    evaluate(
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        guidance=args.guidance,
        num_steps=args.num_steps,
        seed=args.seed,
        save_videos=not args.no_save_videos,
    )


if __name__ == "__main__":
    main()
