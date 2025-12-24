# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Test script to analyze VAE encoder response to different inputs.

This script helps understand:
1. What happens when encoder receives zero input (pixel space)
2. How sparse point cloud inputs are encoded
3. The difference between "true zero latent" vs "encoded zero"

Usage:
    cd /mnt/zihanw/R2V_world_foundation_model_zz
    PYTHONPATH=. python scripts/test_encoder_zero_response.py
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set environment variables
os.environ["NVTE_FUSED_ATTN"] = "0"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import cv2

# Import HF mirror config
import cosmos_transfer2._src.imaginaire.utils.hf_mirror  # noqa: F401


def load_real_image(image_path, target_height=128, target_width=256, num_frames=5):
    """Load a real image and convert to video format.

    Returns tensor of shape (1, 3, T, H, W) in range [-1, 1]
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize((target_width, target_height), Image.LANCZOS)
    img_np = np.array(img).astype(np.float32) / 255.0  # [0, 1]
    img_np = img_np * 2 - 1  # [-1, 1]

    # Convert to (C, H, W)
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)  # (3, H, W)

    # Expand to video (repeat the same frame)
    video = img_tensor.unsqueeze(0).unsqueeze(2).expand(1, 3, num_frames, target_height, target_width)
    return video.clone()


def load_sparse_from_dataset(data_root, target_height=128, target_width=256, num_frames=5):
    """Load sparse point cloud data from the dataset.

    Looks for blur/depth control inputs which are sparse LiDAR point clouds.
    """
    # Try to find a sample from the dataset
    possible_paths = [
        os.path.join(data_root, "blur"),
        os.path.join(data_root, "control_input_blur"),
        data_root,
    ]

    for path in possible_paths:
        if os.path.exists(path):
            # Find first image/video file
            for ext in ['*.png', '*.jpg', '*.mp4']:
                import glob
                files = glob.glob(os.path.join(path, "**", ext), recursive=True)
                if files:
                    file_path = files[0]
                    print(f"  Found sparse data: {file_path}")

                    if file_path.endswith('.mp4'):
                        # Load video
                        cap = cv2.VideoCapture(file_path)
                        frames = []
                        for _ in range(num_frames):
                            ret, frame = cap.read()
                            if not ret:
                                break
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frame = cv2.resize(frame, (target_width, target_height))
                            frames.append(frame)
                        cap.release()

                        if len(frames) < num_frames:
                            frames = frames + [frames[-1]] * (num_frames - len(frames))

                        video_np = np.stack(frames, axis=0).astype(np.float32) / 255.0
                        video_np = video_np * 2 - 1
                        video_tensor = torch.from_numpy(video_np).permute(3, 0, 1, 2).unsqueeze(0)
                        return video_tensor
                    else:
                        # Load image
                        return load_real_image(file_path, target_height, target_width, num_frames)

    return None


def create_test_inputs(normal_image_path, sparse_data_path=None,
                       target_height=128, target_width=256, num_frames=5):
    """Create test inputs including real data."""

    shape = (1, 3, num_frames, target_height, target_width)
    inputs = {}

    # 1. All zeros
    inputs["zeros"] = torch.zeros(shape)

    # 2. Load real normal image
    if os.path.exists(normal_image_path):
        print(f"  Loading normal image: {normal_image_path}")
        inputs["real_image"] = load_real_image(normal_image_path, target_height, target_width, num_frames)
    else:
        print(f"  WARNING: Normal image not found: {normal_image_path}")
        inputs["real_image"] = torch.rand(shape) * 2 - 1

    # 3. Try to load sparse data from dataset
    if sparse_data_path and os.path.exists(sparse_data_path):
        sparse = load_sparse_from_dataset(sparse_data_path, target_height, target_width, num_frames)
        if sparse is not None:
            inputs["real_sparse"] = sparse

    # 4. Create synthetic sparse (1% points)
    sparse = torch.zeros(shape)
    mask = torch.rand(shape) < 0.01
    sparse[mask] = torch.rand(mask.sum()) * 2 - 1
    inputs["synthetic_sparse_1pct"] = sparse

    # 5. Create very sparse (0.1% points)
    very_sparse = torch.zeros(shape)
    mask = torch.rand(shape) < 0.001
    very_sparse[mask] = torch.rand(mask.sum()) * 2 - 1
    inputs["synthetic_sparse_0.1pct"] = very_sparse

    return inputs


def analyze_latent(name, latent):
    """Analyze latent tensor statistics."""
    return {
        "name": name,
        "shape": list(latent.shape),
        "mean": latent.mean().item(),
        "std": latent.std().item(),
        "min": latent.min().item(),
        "max": latent.max().item(),
        "abs_mean": latent.abs().mean().item(),
        "zero_ratio": (latent == 0).sum().item() / latent.numel(),
    }


def print_analysis(all_stats, latents):
    """Print analysis in a formatted table."""
    print("\n" + "=" * 100)
    print("ENCODER ZERO RESPONSE ANALYSIS")
    print("=" * 100)
    print(f"{'Input Type':<25} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'AbsMean':>10}")
    print("-" * 100)

    for stats in all_stats:
        print(f"{stats['name']:<25} {stats['mean']:>10.4f} {stats['std']:>10.4f} "
              f"{stats['min']:>10.4f} {stats['max']:>10.4f} {stats['abs_mean']:>10.4f}")

    print("=" * 100)

    # Key comparison
    zero_latent = latents['zeros']
    true_zero = torch.zeros_like(zero_latent)

    print("\n" + "=" * 100)
    print("KEY FINDING: Distance from encoder output to TRUE ZERO latent")
    print("=" * 100)

    for name, latent in latents.items():
        l1 = (latent - true_zero).abs().mean().item()
        l2 = ((latent - true_zero) ** 2).mean().sqrt().item()
        print(f"  {name:<25} L1={l1:.6f}  L2={l2:.6f}")

    print("\n" + "-" * 100)
    zero_l1 = (zero_latent - true_zero).abs().mean().item()
    print(f"\n  CONCLUSION: encoder(pixel_zeros) produces latent with L1={zero_l1:.4f} from true zeros")
    print(f"  This means pixel-space dropout does NOT produce zero latent!")


def visualize_comparison(inputs, latents, save_dir):
    """Create side-by-side visualization of inputs and their latents."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    frame_idx = 0

    # Main comparison figure
    n_inputs = len(inputs)
    fig, axes = plt.subplots(n_inputs, 6, figsize=(24, 4 * n_inputs))

    if n_inputs == 1:
        axes = axes.reshape(1, -1)

    for row, (name, input_tensor) in enumerate(inputs.items()):
        # Column 0: Input RGB
        input_np = input_tensor[0, :, frame_idx].cpu().numpy()
        rgb = (input_np.transpose(1, 2, 0) + 1) / 2
        axes[row, 0].imshow(np.clip(rgb, 0, 1))
        axes[row, 0].set_title(f"Input: {name}")
        axes[row, 0].axis('off')

        # Column 1: Input sparsity visualization (how many non-zero pixels)
        non_zero_mask = (input_tensor[0, :, frame_idx].abs() > 0.01).any(dim=0).cpu().numpy()
        axes[row, 1].imshow(non_zero_mask, cmap='gray')
        sparsity = non_zero_mask.mean() * 100
        axes[row, 1].set_title(f"Non-zero pixels: {sparsity:.2f}%")
        axes[row, 1].axis('off')

        # Columns 2-5: First 4 latent channels
        latent_np = latents[name][0, :, frame_idx].cpu().numpy()
        for i in range(4):
            im = axes[row, 2+i].imshow(latent_np[i], cmap='RdBu', vmin=-2, vmax=2)
            ch_mean = latent_np[i].mean()
            ch_std = latent_np[i].std()
            axes[row, 2+i].set_title(f"Latent Ch{i}\nmean={ch_mean:.2f}, std={ch_std:.2f}")
            axes[row, 2+i].axis('off')

    plt.tight_layout()
    plt.savefig(save_dir / "comparison_all.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_dir / 'comparison_all.png'}")

    # Create a focused comparison: zeros vs real_image
    if 'real_image' in inputs:
        fig, axes = plt.subplots(2, 6, figsize=(24, 8))
        fig.suptitle("Comparison: Zero Input vs Real Image", fontsize=16)

        for row, name in enumerate(['zeros', 'real_image']):
            input_np = inputs[name][0, :, frame_idx].cpu().numpy()
            rgb = (input_np.transpose(1, 2, 0) + 1) / 2
            axes[row, 0].imshow(np.clip(rgb, 0, 1))
            axes[row, 0].set_title(f"Input: {name}")
            axes[row, 0].axis('off')

            non_zero_mask = (inputs[name][0, :, frame_idx].abs() > 0.01).any(dim=0).cpu().numpy()
            axes[row, 1].imshow(non_zero_mask, cmap='gray')
            axes[row, 1].set_title(f"Non-zero: {non_zero_mask.mean()*100:.1f}%")
            axes[row, 1].axis('off')

            latent_np = latents[name][0, :, frame_idx].cpu().numpy()
            for i in range(4):
                im = axes[row, 2+i].imshow(latent_np[i], cmap='RdBu', vmin=-2, vmax=2)
                axes[row, 2+i].set_title(f"Latent Ch{i}: {latent_np[i].mean():.2f}")
                axes[row, 2+i].axis('off')

        plt.tight_layout()
        plt.savefig(save_dir / "comparison_zeros_vs_real.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_dir / 'comparison_zeros_vs_real.png'}")


def main():
    print("=" * 100)
    print("VAE ENCODER ZERO RESPONSE TEST")
    print("=" * 100)

    # Check CUDA
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return

    device = torch.device("cuda")
    print(f"Using device: {device}")

    # ========== USER CONFIG ==========
    # Normal image path (provided by user)
    NORMAL_IMAGE_PATH = "/mnt/zihanw/R2V_world_foundation_model_zz/rgborg.png"

    # Optional: path to sparse data from dataset
    SPARSE_DATA_PATH = None  # Set this if you want to load real sparse data

    # Output directory
    SAVE_DIR = "/mnt/zihanw/encoder_analysis"
    # =================================

    # Load tokenizer
    print("\nLoading VAE tokenizer (Wan2pt1)...")
    from cosmos_transfer2._src.predict2.tokenizers.wan2pt1 import Wan2pt1VAEInterface

    tokenizer = Wan2pt1VAEInterface(
        chunk_duration=81,
        load_mean_std=False,
        temporal_window=4,
        is_parallel=False,
    )

    print(f"Tokenizer loaded!")
    print(f"  Spatial compression: {tokenizer.spatial_compression_factor}x")
    print(f"  Temporal compression: {tokenizer.temporal_compression_factor}x")
    print(f"  Latent channels: {tokenizer.latent_ch}")

    # Create test inputs
    print("\nCreating test inputs...")
    test_height = 128
    test_width = 256
    num_frames = 5

    inputs = create_test_inputs(
        normal_image_path=NORMAL_IMAGE_PATH,
        sparse_data_path=SPARSE_DATA_PATH,
        target_height=test_height,
        target_width=test_width,
        num_frames=num_frames,
    )

    print(f"\nInput shapes:")
    for name, tensor in inputs.items():
        print(f"  {name}: {tensor.shape}")

    # Encode all inputs
    print("\nEncoding inputs...")
    latents = {}
    all_stats = []

    with torch.no_grad():
        for name, input_tensor in inputs.items():
            input_tensor = input_tensor.to(device).to(torch.bfloat16)
            latent = tokenizer.encode(input_tensor)
            latents[name] = latent.float().cpu()
            stats = analyze_latent(name, latent)
            all_stats.append(stats)
            print(f"  Encoded {name}: latent shape = {latent.shape}")

    # Print analysis
    print_analysis(all_stats, latents)

    # Visualize
    print("\nGenerating visualizations...")
    visualize_comparison(inputs, latents, SAVE_DIR)

    # Save text results
    result_path = os.path.join(SAVE_DIR, "encoder_test_results.txt")
    with open(result_path, 'w') as f:
        f.write("=== VAE Encoder Zero Response Test ===\n\n")
        for stats in all_stats:
            f.write(f"{stats['name']}:\n")
            f.write(f"  mean={stats['mean']:.6f}, std={stats['std']:.6f}\n")
            f.write(f"  min={stats['min']:.6f}, max={stats['max']:.6f}\n\n")

        f.write("\n=== Distance to True Zero Latent ===\n")
        true_zero = torch.zeros_like(latents['zeros'])
        for name, latent in latents.items():
            l1 = (latent - true_zero).abs().mean().item()
            f.write(f"{name}: L1={l1:.6f}\n")

        zero_l1 = (latents['zeros'] - true_zero).abs().mean().item()
        f.write(f"\nCONCLUSION: encoder(zeros) has L1={zero_l1:.4f} from true zeros\n")
        f.write("Pixel-space dropout does NOT produce zero latent!\n")

    print(f"\nResults saved to: {result_path}")

    print("\n" + "=" * 100)
    print("TEST COMPLETE")
    print("=" * 100)
    print(f"\nVisualization saved to: {SAVE_DIR}/")


if __name__ == "__main__":
    main()
