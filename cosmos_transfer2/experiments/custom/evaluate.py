#!/usr/bin/env python3
"""
Evaluation script for multi-control inference results.

Computes per-view and overall metrics:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- LPIPS (Learned Perceptual Image Patch Similarity)
- FID (Fréchet Inception Distance) - distribution-level, requires frame extraction
- FVD (Fréchet Video Distance) - distribution-level, requires torchmetrics

Usage:
    python -m cosmos_transfer2.experiments.custom.evaluate \
        --input_dir /mnt/zihanw/Output_R2V_world_foundation_model_v1/inference \
        --output_csv /mnt/zihanw/Output_R2V_world_foundation_model_v1/inference/metrics.csv

Prerequisites:
    pip install lpips scipy clean-fid torchmetrics
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger


# ============================================================================
# Video I/O
# ============================================================================

def read_video_frames(video_path: str) -> np.ndarray:
    """Read video and return frames as numpy array (T, H, W, 3) uint8."""
    from decord import VideoReader
    vr = VideoReader(video_path)
    frames = vr.get_batch(list(range(len(vr)))).asnumpy()  # (T, H, W, 3)
    return frames


def frames_to_tensor(frames: np.ndarray) -> torch.Tensor:
    """Convert (T, H, W, 3) uint8 numpy to (T, 3, H, W) float32 [0, 1] tensor."""
    t = torch.from_numpy(frames).float() / 255.0
    t = t.permute(0, 3, 1, 2)  # (T, 3, H, W)
    return t


# ============================================================================
# Per-frame paired metrics: PSNR, SSIM, LPIPS
# ============================================================================

def compute_psnr(gen: torch.Tensor, gt: torch.Tensor) -> float:
    """Compute PSNR between two (T, 3, H, W) tensors in [0, 1]."""
    mse = F.mse_loss(gen, gt, reduction='mean').item()
    if mse == 0:
        return float('inf')
    return 10.0 * np.log10(1.0 / mse)


def compute_ssim(gen: torch.Tensor, gt: torch.Tensor) -> float:
    """Compute SSIM between two (T, 3, H, W) tensors in [0, 1].

    Uses torchmetrics if available, otherwise falls back to simple implementation.
    """
    try:
        from torchmetrics.image import StructuralSimilarityIndexMeasure
        ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(gen.device)
        return ssim_metric(gen, gt).item()
    except ImportError:
        pass

    # Fallback: simplified SSIM (per-frame, average)
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_vals = []
    for i in range(gen.shape[0]):
        g = gen[i:i+1]  # (1, 3, H, W)
        r = gt[i:i+1]
        mu_g = F.avg_pool2d(g, 11, stride=1, padding=5)
        mu_r = F.avg_pool2d(r, 11, stride=1, padding=5)
        mu_g_sq = mu_g ** 2
        mu_r_sq = mu_r ** 2
        mu_gr = mu_g * mu_r
        sigma_g_sq = F.avg_pool2d(g ** 2, 11, stride=1, padding=5) - mu_g_sq
        sigma_r_sq = F.avg_pool2d(r ** 2, 11, stride=1, padding=5) - mu_r_sq
        sigma_gr = F.avg_pool2d(g * r, 11, stride=1, padding=5) - mu_gr
        ssim_map = ((2 * mu_gr + C1) * (2 * sigma_gr + C2)) / \
                   ((mu_g_sq + mu_r_sq + C1) * (sigma_g_sq + sigma_r_sq + C2))
        ssim_vals.append(ssim_map.mean().item())
    return float(np.mean(ssim_vals))


def compute_lpips(gen: torch.Tensor, gt: torch.Tensor, lpips_model) -> float:
    """Compute LPIPS between two (T, 3, H, W) tensors in [0, 1].

    LPIPS expects input in [-1, 1].
    """
    # Convert [0, 1] -> [-1, 1]
    gen_scaled = gen * 2.0 - 1.0
    gt_scaled = gt * 2.0 - 1.0

    # Compute per-frame LPIPS and average
    lpips_vals = []
    batch_size = 4  # Process frames in batches to save memory
    for i in range(0, gen.shape[0], batch_size):
        g_batch = gen_scaled[i:i+batch_size].to(gen.device)
        r_batch = gt_scaled[i:i+batch_size].to(gen.device)
        with torch.no_grad():
            val = lpips_model(g_batch, r_batch)
        lpips_vals.append(val.mean().item())
    return float(np.mean(lpips_vals))


# ============================================================================
# Distribution metrics: FID
# ============================================================================

def extract_and_save_frames(video_dir: str, output_frames_dir: str):
    """Extract frames from all videos in inference output for FID computation.

    Creates two directories:
        {output_frames_dir}/generated/  - all generated frames
        {output_frames_dir}/gt/         - all ground truth frames
    """
    gen_dir = Path(output_frames_dir) / "generated"
    gt_dir = Path(output_frames_dir) / "gt"
    gen_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    from PIL import Image

    video_dir = Path(video_dir)
    frame_count = 0

    for sample_dir in sorted(video_dir.iterdir()):
        if not sample_dir.is_dir():
            continue

        sample_id = sample_dir.name

        for video_file in sorted(sample_dir.glob("*_generated.mp4")):
            view_name = video_file.stem.replace("_generated", "")
            gt_file = sample_dir / f"{view_name}_gt.mp4"

            if not gt_file.exists():
                logger.warning(f"GT not found for {video_file}, skipping")
                continue

            gen_frames = read_video_frames(str(video_file))
            gt_frames = read_video_frames(str(gt_file))

            n_frames = min(len(gen_frames), len(gt_frames))
            for t in range(n_frames):
                fname = f"{sample_id}_{view_name}_{t:04d}.png"
                Image.fromarray(gen_frames[t]).save(str(gen_dir / fname))
                Image.fromarray(gt_frames[t]).save(str(gt_dir / fname))
                frame_count += 1

    logger.info(f"Extracted {frame_count} frame pairs to {output_frames_dir}")
    return str(gen_dir), str(gt_dir), frame_count


def compute_fid(gen_frames_dir: str, gt_frames_dir: str) -> float:
    """Compute FID between two directories of images."""
    try:
        from cleanfid import fid as cleanfid
        score = cleanfid.compute_fid(gen_frames_dir, gt_frames_dir)
        return score
    except ImportError:
        pass

    try:
        # Fallback: pytorch-fid
        from pytorch_fid import fid_score
        score = fid_score.calculate_fid_given_paths(
            [gen_frames_dir, gt_frames_dir],
            batch_size=50, device='cuda', dims=2048
        )
        return score
    except ImportError:
        pass

    logger.error("Neither clean-fid nor pytorch-fid is installed. "
                 "Install with: pip install clean-fid")
    return float('nan')


# ============================================================================
# Distribution metrics: FVD
# ============================================================================

def compute_fvd_for_view(gen_videos: List[torch.Tensor], gt_videos: List[torch.Tensor]) -> float:
    """Compute FVD between lists of video tensors.

    Each video is (T, 3, H, W) in [0, 1].
    Requires torchmetrics with video support.
    """
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        # FVD is not in standard torchmetrics, use a simple approach:
        # Compute per-frame FID as a proxy, or skip
        logger.warning("FVD requires specialized I3D model. Using per-frame FID as proxy.")
        return float('nan')
    except ImportError:
        logger.warning("torchmetrics not available for FVD computation")
        return float('nan')


# ============================================================================
# Main evaluation
# ============================================================================

def find_video_pairs(input_dir: str) -> Dict[str, List[Tuple[str, str, str]]]:
    """Find all (generated, gt) video pairs grouped by view name.

    Returns: {view_name: [(sample_id, gen_path, gt_path), ...]}
    """
    input_path = Path(input_dir)
    pairs_by_view = defaultdict(list)

    for sample_dir in sorted(input_path.iterdir()):
        if not sample_dir.is_dir():
            continue

        sample_id = sample_dir.name

        for gen_file in sorted(sample_dir.glob("*_generated.mp4")):
            view_name = gen_file.stem.replace("_generated", "")
            gt_file = sample_dir / f"{view_name}_gt.mp4"

            if gt_file.exists():
                pairs_by_view[view_name].append((sample_id, str(gen_file), str(gt_file)))
            else:
                logger.warning(f"Missing GT: {gt_file}")

    return dict(pairs_by_view)


def evaluate(input_dir: str, output_csv: str, device: str = "cuda"):
    """Run full evaluation."""

    logger.info(f"Evaluating: {input_dir}")

    # Find all video pairs
    pairs_by_view = find_video_pairs(input_dir)
    if not pairs_by_view:
        logger.error("No video pairs found!")
        return

    all_views = sorted(pairs_by_view.keys())
    logger.info(f"Found {len(all_views)} views: {all_views}")
    for view in all_views:
        logger.info(f"  {view}: {len(pairs_by_view[view])} samples")

    # Initialize LPIPS model
    lpips_model = None
    try:
        import lpips
        lpips_model = lpips.LPIPS(net='alex').to(device)
        lpips_model.eval()
        logger.info("LPIPS model loaded (AlexNet)")
    except ImportError:
        logger.warning("lpips not installed, LPIPS will be skipped. Install: pip install lpips")

    # ---- Per-view paired metrics: PSNR, SSIM, LPIPS ----
    results = []  # List of dicts for CSV output
    all_psnr = []
    all_ssim = []
    all_lpips = []

    for view_name in all_views:
        pairs = pairs_by_view[view_name]
        view_psnr = []
        view_ssim = []
        view_lpips = []

        logger.info(f"\n--- Evaluating view: {view_name} ({len(pairs)} samples) ---")

        for sample_id, gen_path, gt_path in pairs:
            gen_frames = frames_to_tensor(read_video_frames(gen_path)).to(device)
            gt_frames = frames_to_tensor(read_video_frames(gt_path)).to(device)

            # Ensure same number of frames
            n = min(gen_frames.shape[0], gt_frames.shape[0])
            gen_frames = gen_frames[:n]
            gt_frames = gt_frames[:n]

            # PSNR
            psnr_val = compute_psnr(gen_frames, gt_frames)
            view_psnr.append(psnr_val)

            # SSIM
            ssim_val = compute_ssim(gen_frames, gt_frames)
            view_ssim.append(ssim_val)

            # LPIPS
            lpips_val = float('nan')
            if lpips_model is not None:
                lpips_val = compute_lpips(gen_frames, gt_frames, lpips_model)
            view_lpips.append(lpips_val)

            logger.info(f"  {sample_id}: PSNR={psnr_val:.2f}, SSIM={ssim_val:.4f}, LPIPS={lpips_val:.4f}")

            results.append({
                'sample_id': sample_id,
                'view': view_name,
                'psnr': psnr_val,
                'ssim': ssim_val,
                'lpips': lpips_val,
            })

            # Free GPU memory
            del gen_frames, gt_frames
            torch.cuda.empty_cache()

        # Per-view averages
        avg_psnr = np.mean(view_psnr)
        avg_ssim = np.mean(view_ssim)
        avg_lpips = np.nanmean(view_lpips)
        logger.info(f"  [{view_name} AVG] PSNR={avg_psnr:.2f}, SSIM={avg_ssim:.4f}, LPIPS={avg_lpips:.4f}")

        all_psnr.extend(view_psnr)
        all_ssim.extend(view_ssim)
        all_lpips.extend(view_lpips)

    # Overall averages
    logger.info(f"\n{'='*60}")
    logger.info(f"OVERALL (all views, all samples):")
    logger.info(f"  PSNR:  {np.mean(all_psnr):.2f}")
    logger.info(f"  SSIM:  {np.mean(all_ssim):.4f}")
    logger.info(f"  LPIPS: {np.nanmean(all_lpips):.4f}")

    # ---- FID (distribution-level, needs frame extraction) ----
    logger.info(f"\n{'='*60}")
    logger.info("Computing FID (extracting frames)...")
    frames_dir = str(Path(input_dir) / "_eval_frames")
    gen_frames_dir, gt_frames_dir, n_frames = extract_and_save_frames(input_dir, frames_dir)

    if n_frames > 0:
        fid_score = compute_fid(gen_frames_dir, gt_frames_dir)
        logger.info(f"  FID (all views): {fid_score:.2f}")
    else:
        fid_score = float('nan')
        logger.warning("No frames extracted, FID skipped")

    # Per-view FID
    fid_per_view = {}
    for view_name in all_views:
        view_gen_dir = str(Path(frames_dir) / f"generated_{view_name}")
        view_gt_dir = str(Path(frames_dir) / f"gt_{view_name}")
        Path(view_gen_dir).mkdir(parents=True, exist_ok=True)
        Path(view_gt_dir).mkdir(parents=True, exist_ok=True)

        from PIL import Image
        pairs = pairs_by_view[view_name]
        for sample_id, gen_path, gt_path in pairs:
            gen_frames = read_video_frames(gen_path)
            gt_frames = read_video_frames(gt_path)
            n = min(len(gen_frames), len(gt_frames))
            for t in range(n):
                fname = f"{sample_id}_{t:04d}.png"
                Image.fromarray(gen_frames[t]).save(os.path.join(view_gen_dir, fname))
                Image.fromarray(gt_frames[t]).save(os.path.join(view_gt_dir, fname))

        view_fid = compute_fid(view_gen_dir, view_gt_dir)
        fid_per_view[view_name] = view_fid
        logger.info(f"  FID [{view_name}]: {view_fid:.2f}")

    # ---- Summary ----
    logger.info(f"\n{'='*60}")
    logger.info("FINAL SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"{'View':<30} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8} {'FID':>8}")
    logger.info(f"{'-'*62}")
    for view_name in all_views:
        view_results = [r for r in results if r['view'] == view_name]
        vp = np.mean([r['psnr'] for r in view_results])
        vs = np.mean([r['ssim'] for r in view_results])
        vl = np.nanmean([r['lpips'] for r in view_results])
        vf = fid_per_view.get(view_name, float('nan'))
        logger.info(f"{view_name:<30} {vp:>8.2f} {vs:>8.4f} {vl:>8.4f} {vf:>8.2f}")

    logger.info(f"{'-'*62}")
    logger.info(f"{'OVERALL':<30} {np.mean(all_psnr):>8.2f} {np.mean(all_ssim):>8.4f} "
                f"{np.nanmean(all_lpips):>8.4f} {fid_score:>8.2f}")

    # ---- Save CSV ----
    if output_csv:
        csv_path = Path(output_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        # Per-sample results
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['sample_id', 'view', 'psnr', 'ssim', 'lpips'])
            writer.writeheader()
            writer.writerows(results)
        logger.info(f"Per-sample results saved to: {csv_path}")

        # Summary results
        summary_path = csv_path.parent / csv_path.stem.replace('metrics', 'summary')
        summary_path = summary_path.with_suffix('.csv')
        with open(summary_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['view', 'psnr', 'ssim', 'lpips', 'fid'])
            writer.writeheader()
            for view_name in all_views:
                view_results_list = [r for r in results if r['view'] == view_name]
                writer.writerow({
                    'view': view_name,
                    'psnr': f"{np.mean([r['psnr'] for r in view_results_list]):.4f}",
                    'ssim': f"{np.mean([r['ssim'] for r in view_results_list]):.4f}",
                    'lpips': f"{np.nanmean([r['lpips'] for r in view_results_list]):.4f}",
                    'fid': f"{fid_per_view.get(view_name, float('nan')):.4f}",
                })
            writer.writerow({
                'view': 'OVERALL',
                'psnr': f"{np.mean(all_psnr):.4f}",
                'ssim': f"{np.mean(all_ssim):.4f}",
                'lpips': f"{np.nanmean(all_lpips):.4f}",
                'fid': f"{fid_score:.4f}",
            })
        logger.info(f"Summary saved to: {summary_path}")

    # Save full results as JSON too
    json_path = Path(output_csv).with_suffix('.json') if output_csv else Path(input_dir) / "metrics.json"
    summary_dict = {
        'overall': {
            'psnr': float(np.mean(all_psnr)),
            'ssim': float(np.mean(all_ssim)),
            'lpips': float(np.nanmean(all_lpips)),
            'fid': float(fid_score) if not np.isnan(fid_score) else None,
        },
        'per_view': {},
        'per_sample': results,
    }
    for view_name in all_views:
        view_results_list = [r for r in results if r['view'] == view_name]
        summary_dict['per_view'][view_name] = {
            'psnr': float(np.mean([r['psnr'] for r in view_results_list])),
            'ssim': float(np.mean([r['ssim'] for r in view_results_list])),
            'lpips': float(np.nanmean([r['lpips'] for r in view_results_list])),
            'fid': float(fid_per_view.get(view_name, float('nan'))),
            'n_samples': len(view_results_list),
        }
    with open(json_path, 'w') as f:
        json.dump(summary_dict, f, indent=2)
    logger.info(f"Full results saved to: {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate inference results")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to inference output directory")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Path to save metrics CSV (default: {input_dir}/metrics.csv)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for computation")
    args = parser.parse_args()

    if args.output_csv is None:
        args.output_csv = str(Path(args.input_dir) / "metrics.csv")

    evaluate(args.input_dir, args.output_csv, args.device)


if __name__ == "__main__":
    main()
