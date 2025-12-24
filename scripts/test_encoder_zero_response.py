# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Test script to analyze VAE encoder response to different inputs.

This script helps understand:
1. What happens when encoder receives zero input (pixel space)
2. How sparse point cloud inputs are encoded
3. The difference between "true zero latent" vs "encoded zero"

Usage:
    cd /home/user/R2V_world_foundation_model
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

# Import HF mirror config
import cosmos_transfer2._src.imaginaire.utils.hf_mirror  # noqa: F401


def create_test_inputs(batch_size=1, num_frames=5, height=720, width=1280):
    """Create different test inputs for encoder analysis."""

    # Input shape: (B, C, T, H, W), range [-1, 1]
    shape = (batch_size, 3, num_frames, height, width)

    inputs = {}

    # 1. All zeros (pixel space)
    inputs["zeros"] = torch.zeros(shape)

    # 2. All ones (maximum value)
    inputs["ones"] = torch.ones(shape)

    # 3. All negative ones (minimum value)
    inputs["neg_ones"] = -torch.ones(shape)

    # 4. Random noise (uniform)
    inputs["random"] = torch.rand(shape) * 2 - 1  # [-1, 1]

    # 5. Sparse point cloud simulation (1% of pixels have values)
    sparse = torch.zeros(shape)
    mask = torch.rand(shape) < 0.01  # 1% sparse
    sparse[mask] = torch.rand(mask.sum()) * 2 - 1
    inputs["sparse_1pct"] = sparse

    # 6. Very sparse point cloud (0.1% of pixels)
    very_sparse = torch.zeros(shape)
    mask = torch.rand(shape) < 0.001  # 0.1% sparse
    very_sparse[mask] = torch.rand(mask.sum()) * 2 - 1
    inputs["sparse_0.1pct"] = very_sparse

    # 7. Single point (only center pixel has value)
    single_point = torch.zeros(shape)
    single_point[:, :, :, height//2, width//2] = 1.0
    inputs["single_point"] = single_point

    # 8. Horizontal gradient
    gradient = torch.linspace(-1, 1, width).view(1, 1, 1, 1, width).expand(shape)
    inputs["gradient"] = gradient

    return inputs


def analyze_latent(name, latent):
    """Analyze latent tensor statistics."""
    stats = {
        "name": name,
        "shape": list(latent.shape),
        "mean": latent.mean().item(),
        "std": latent.std().item(),
        "min": latent.min().item(),
        "max": latent.max().item(),
        "abs_mean": latent.abs().mean().item(),
        "num_zeros": (latent == 0).sum().item(),
        "total_elements": latent.numel(),
        "zero_ratio": (latent == 0).sum().item() / latent.numel(),
        "near_zero_ratio": (latent.abs() < 0.01).sum().item() / latent.numel(),
    }
    return stats


def print_analysis(all_stats):
    """Print analysis in a formatted table."""
    print("\n" + "=" * 100)
    print("ENCODER ZERO RESPONSE ANALYSIS")
    print("=" * 100)
    print(f"{'Input Type':<20} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'AbsMean':>10} {'ZeroRatio':>10}")
    print("-" * 100)

    for stats in all_stats:
        print(f"{stats['name']:<20} {stats['mean']:>10.4f} {stats['std']:>10.4f} "
              f"{stats['min']:>10.4f} {stats['max']:>10.4f} {stats['abs_mean']:>10.4f} "
              f"{stats['zero_ratio']:>10.4f}")

    print("=" * 100)

    # Key findings
    print("\nKEY FINDINGS:")
    zero_stats = next(s for s in all_stats if s['name'] == 'zeros')
    print(f"  1. Zero input -> Latent mean: {zero_stats['mean']:.6f} (should be 0 if encoder is linear)")
    print(f"  2. Zero input -> Latent std:  {zero_stats['std']:.6f} (should be 0 if encoder is linear)")
    print(f"  3. Zero input -> Zero ratio:  {zero_stats['zero_ratio']:.6f} (ratio of exactly-zero values)")

    if abs(zero_stats['mean']) > 0.001 or zero_stats['std'] > 0.001:
        print("\n  CONCLUSION: encoder(zeros) != zeros_latent")
        print("  This means dropout in pixel space does NOT produce zero latent!")
    else:
        print("\n  CONCLUSION: encoder(zeros) ≈ zeros_latent")
        print("  Dropout in pixel space produces near-zero latent.")


def visualize_latents(inputs, latents, save_dir):
    """Visualize input and latent feature maps."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Select middle frame for visualization
    frame_idx = 0

    for name in inputs.keys():
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle(f"Input: {name}", fontsize=16)

        # Row 1: Input (3 channels + combined)
        input_tensor = inputs[name][0, :, frame_idx].cpu().numpy()  # (3, H, W)
        for i in range(3):
            axes[0, i].imshow(input_tensor[i], cmap='RdBu', vmin=-1, vmax=1)
            axes[0, i].set_title(f"Input Channel {i}")
            axes[0, i].axis('off')

        # Combined RGB view
        rgb = (input_tensor.transpose(1, 2, 0) + 1) / 2  # [-1,1] -> [0,1]
        axes[0, 3].imshow(np.clip(rgb, 0, 1))
        axes[0, 3].set_title("Input RGB")
        axes[0, 3].axis('off')

        # Row 2: Latent (first 4 channels)
        latent_tensor = latents[name][0, :, frame_idx].cpu().numpy()  # (C, H, W)
        num_latent_ch = min(4, latent_tensor.shape[0])

        for i in range(num_latent_ch):
            im = axes[1, i].imshow(latent_tensor[i], cmap='RdBu')
            axes[1, i].set_title(f"Latent Ch{i} (mean={latent_tensor[i].mean():.3f})")
            axes[1, i].axis('off')
            plt.colorbar(im, ax=axes[1, i], fraction=0.046)

        plt.tight_layout()
        plt.savefig(save_dir / f"analysis_{name}.png", dpi=100, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_dir / f'analysis_{name}.png'}")

    # Create comparison figure for zeros vs sparse
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("Comparison: Zeros vs Sparse (Latent Space)", fontsize=16)

    for row, name in enumerate(['zeros', 'sparse_1pct']):
        latent_tensor = latents[name][0, :, frame_idx].cpu().numpy()
        for i in range(4):
            im = axes[row, i].imshow(latent_tensor[i], cmap='RdBu')
            axes[row, i].set_title(f"{name} - Ch{i}")
            axes[row, i].axis('off')
            plt.colorbar(im, ax=axes[row, i], fraction=0.046)

    plt.tight_layout()
    plt.savefig(save_dir / "comparison_zeros_vs_sparse.png", dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_dir / 'comparison_zeros_vs_sparse.png'}")


def compute_latent_difference(latents):
    """Compute difference between different latent representations."""
    print("\n" + "=" * 100)
    print("LATENT DIFFERENCE ANALYSIS")
    print("=" * 100)

    zero_latent = latents['zeros']

    print(f"{'Comparison':<30} {'L1 Diff':>12} {'L2 Diff':>12} {'Max Diff':>12}")
    print("-" * 100)

    for name, latent in latents.items():
        if name == 'zeros':
            continue

        l1_diff = (latent - zero_latent).abs().mean().item()
        l2_diff = ((latent - zero_latent) ** 2).mean().sqrt().item()
        max_diff = (latent - zero_latent).abs().max().item()

        print(f"zeros vs {name:<20} {l1_diff:>12.6f} {l2_diff:>12.6f} {max_diff:>12.6f}")


def generate_copyable_results(all_stats, latents):
    """Generate results in a format easy to copy and share."""
    print("\n" + "=" * 100)
    print("COPYABLE RESULTS (for sharing)")
    print("=" * 100)

    result_text = []
    result_text.append("=== VAE Encoder Zero Response Test Results ===\n")

    for stats in all_stats:
        result_text.append(f"{stats['name']}:")
        result_text.append(f"  mean={stats['mean']:.6f}, std={stats['std']:.6f}")
        result_text.append(f"  min={stats['min']:.6f}, max={stats['max']:.6f}")
        result_text.append(f"  abs_mean={stats['abs_mean']:.6f}")
        result_text.append("")

    # Add key comparison
    zero_latent = latents['zeros']
    true_zero = torch.zeros_like(zero_latent)

    l1_from_true_zero = (zero_latent - true_zero).abs().mean().item()
    l2_from_true_zero = ((zero_latent - true_zero) ** 2).mean().sqrt().item()

    result_text.append("=== Key Finding ===")
    result_text.append(f"Distance from encoder(zeros) to true_zeros_latent:")
    result_text.append(f"  L1: {l1_from_true_zero:.6f}")
    result_text.append(f"  L2: {l2_from_true_zero:.6f}")

    if l1_from_true_zero > 0.01:
        result_text.append("\nCONCLUSION: encoder(zeros) != zeros")
        result_text.append("Pixel-space dropout does NOT produce zero latent!")
    else:
        result_text.append("\nCONCLUSION: encoder(zeros) ≈ zeros")

    full_result = "\n".join(result_text)
    print(full_result)

    # Save to file
    save_path = "/mnt/zihanw/encoder_test_results.txt"
    with open(save_path, 'w') as f:
        f.write(full_result)
    print(f"\nResults saved to: {save_path}")

    return full_result


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

    # Load model to get the tokenizer/encoder
    print("\nLoading model...")

    from cosmos_transfer2._src.imaginaire.lazy_config import instantiate
    from cosmos_transfer2._src.predict2.utils.model_loader import load_model_from_checkpoint

    # Import experiment config
    import cosmos_transfer2.experiments.multiview.zihanw_singleview_dropout_training  # noqa: F401

    # Use a checkpoint path (adjust as needed)
    ckpt_path = "/mnt/zihanw/cosmos-transfer-output/cosmos_transfer_v2p5/zihanw_singleview/zihanw_singleview_dropout_train/checkpoints/iter_000010000"

    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        print("Trying alternative path...")
        ckpt_path = "/mnt/zihanw/cosmos-transfer-output/cosmos_transfer_v2p5/zihanw_singleview/zihanw_singleview_dropout_train/checkpoints/iter_000005000"

    if not os.path.exists(ckpt_path):
        print("No checkpoint found. Please provide a valid checkpoint path.")
        return

    print(f"Loading from: {ckpt_path}")

    model, config = load_model_from_checkpoint(
        experiment_name="zihanw_singleview_dropout_train",
        s3_checkpoint_dir=ckpt_path,
        config_file="cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/config.py",
        load_ema_to_reg=True,
        experiment_opts=[],
    )

    model.eval()
    tokenizer = model.tokenizer
    print(f"Tokenizer type: {type(tokenizer)}")
    print(f"Tokenizer spatial compression: {tokenizer.spatial_compression_factor}")
    print(f"Tokenizer temporal compression: {tokenizer.temporal_compression_factor}")

    # Create test inputs (smaller size for faster testing)
    print("\nCreating test inputs...")
    # Use smaller resolution for faster testing
    test_height = 180  # 720 / 4
    test_width = 320   # 1280 / 4
    num_frames = 5

    inputs = create_test_inputs(
        batch_size=1,
        num_frames=num_frames,
        height=test_height,
        width=test_width
    )

    print(f"Input shape: {inputs['zeros'].shape}")
    print(f"Expected latent shape: ({1}, {tokenizer.latent_ch}, "
          f"{num_frames // tokenizer.temporal_compression_factor}, "
          f"{test_height // tokenizer.spatial_compression_factor}, "
          f"{test_width // tokenizer.spatial_compression_factor})")

    # Encode all inputs
    print("\nEncoding inputs...")
    latents = {}
    all_stats = []

    with torch.no_grad():
        for name, input_tensor in inputs.items():
            input_tensor = input_tensor.to(device).to(torch.bfloat16)

            # Encode
            latent = tokenizer.encode(input_tensor)
            latents[name] = latent.float().cpu()

            # Analyze
            stats = analyze_latent(name, latent)
            all_stats.append(stats)
            print(f"  Encoded {name}: latent shape = {latent.shape}")

    # Print analysis
    print_analysis(all_stats)

    # Compute differences
    compute_latent_difference(latents)

    # Visualize
    print("\nGenerating visualizations...")
    save_dir = "/mnt/zihanw/encoder_analysis"
    visualize_latents(inputs, latents, save_dir)

    # Generate copyable results
    generate_copyable_results(all_stats, latents)

    print("\n" + "=" * 100)
    print("TEST COMPLETE")
    print("=" * 100)
    print(f"\nVisualization saved to: {save_dir}/")
    print(f"Text results saved to: /mnt/zihanw/encoder_test_results.txt")


if __name__ == "__main__":
    main()
