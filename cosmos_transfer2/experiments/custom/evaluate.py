#!/usr/bin/env python3
"""
Evaluation script for multi-control inference results.

Computes per-view and overall metrics for paper reporting:
- PSNR (Peak Signal-to-Noise Ratio) - mean ± std
- SSIM (Structural Similarity Index) - mean ± std
- LPIPS (Learned Perceptual Image Patch Similarity) - mean ± std
- FID (Fréchet Inception Distance) - single value (distribution-level, image)
- FVD (Fréchet Video Distance) - single value (distribution-level, video)

Usage:
    python cosmos_transfer2/experiments/custom/evaluate.py \
        --input_dir /mnt/zihanw/Output_R2V_world_foundation_model_v1/inference \
        --device cuda

Prerequisites:
    pip install lpips scipy clean-fid torchmetrics decord pillow

    # For FVD (optional, will skip if not available):
    pip install pytorch-fvd
    # Or manually download I3D weights

Output format (for paper):
    | View | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FID ↓ | FVD ↓ |
    |------|--------|--------|---------|-------|-------|
    | front_wide | 22.35 ± 1.2 | 0.782 ± 0.03 | 0.215 ± 0.02 | 45.2 | 123.4 |
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

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

    Requires torchmetrics for accurate computation.
    """
    try:
        from torchmetrics.image import StructuralSimilarityIndexMeasure
        ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(gen.device)
        return ssim_metric(gen, gt).item()
    except ImportError:
        raise ImportError(
            "torchmetrics is required for accurate SSIM computation. "
            "Install with: pip install torchmetrics"
        )


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

def extract_and_save_frames(video_dir: str, output_frames_dir: str,
                            pairs_by_view: Dict[str, List[Tuple[str, str, str]]]) -> Tuple[str, str, int]:
    """Extract frames from all videos for FID computation.

    Creates:
        {output_frames_dir}/generated/  - all generated frames
        {output_frames_dir}/gt/         - all ground truth frames
    """
    gen_dir = Path(output_frames_dir) / "generated"
    gt_dir = Path(output_frames_dir) / "gt"
    gen_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    from PIL import Image

    frame_count = 0
    for view_name, pairs in pairs_by_view.items():
        for sample_id, gen_path, gt_path in pairs:
            gen_frames = read_video_frames(gen_path)
            gt_frames = read_video_frames(gt_path)
            n = min(len(gen_frames), len(gt_frames))
            for t in range(n):
                fname = f"{sample_id}_{view_name}_{t:04d}.png"
                Image.fromarray(gen_frames[t]).save(str(gen_dir / fname))
                Image.fromarray(gt_frames[t]).save(str(gt_dir / fname))
                frame_count += 1

    logger.info(f"Extracted {frame_count} frame pairs to {output_frames_dir}")
    return str(gen_dir), str(gt_dir), frame_count


def extract_view_frames(pairs: List[Tuple[str, str, str]], output_dir: str) -> Tuple[str, str]:
    """Extract frames for a single view."""
    gen_dir = Path(output_dir) / "gen"
    gt_dir = Path(output_dir) / "gt"
    gen_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    from PIL import Image

    for sample_id, gen_path, gt_path in pairs:
        gen_frames = read_video_frames(gen_path)
        gt_frames = read_video_frames(gt_path)
        n = min(len(gen_frames), len(gt_frames))
        for t in range(n):
            fname = f"{sample_id}_{t:04d}.png"
            Image.fromarray(gen_frames[t]).save(str(gen_dir / fname))
            Image.fromarray(gt_frames[t]).save(str(gt_dir / fname))

    return str(gen_dir), str(gt_dir)


def compute_fid(gen_frames_dir: str, gt_frames_dir: str) -> float:
    """Compute FID between two directories of images."""
    try:
        from cleanfid import fid as cleanfid
        score = cleanfid.compute_fid(gen_frames_dir, gt_frames_dir)
        return score
    except ImportError:
        pass

    try:
        from pytorch_fid import fid_score
        score = fid_score.calculate_fid_given_paths(
            [gen_frames_dir, gt_frames_dir],
            batch_size=50, device='cuda', dims=2048
        )
        return score
    except ImportError:
        pass

    logger.error("Neither clean-fid nor pytorch-fid installed. Install: pip install clean-fid")
    return float('nan')


# ============================================================================
# Distribution metrics: FVD (Fréchet Video Distance)
# ============================================================================

# I3D weights URL (from StyleGAN-V / TATS)
I3D_WEIGHTS_URL = "https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1"
I3D_WEIGHTS_PATH = Path("/mnt/zihanw/R2V_world_foundation_model_v1/.cache/fvd/i3d_torchscript.pt")


def download_i3d_weights() -> str:
    """Download I3D weights if not cached."""
    if I3D_WEIGHTS_PATH.exists():
        return str(I3D_WEIGHTS_PATH)

    logger.info(f"Downloading I3D weights to {I3D_WEIGHTS_PATH}...")
    I3D_WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    import urllib.request
    try:
        urllib.request.urlretrieve(I3D_WEIGHTS_URL, str(I3D_WEIGHTS_PATH))
        logger.info("I3D weights downloaded successfully")
        return str(I3D_WEIGHTS_PATH)
    except Exception as e:
        logger.warning(f"Failed to download I3D weights: {e}")
        return None


def compute_fvd(gen_videos: List[torch.Tensor], gt_videos: List[torch.Tensor],
                device: str = "cuda") -> float:
    """Compute FVD between two lists of video tensors.

    Each video should be (T, C, H, W) in [0, 1].
    FVD uses I3D features (standard) to measure video quality distribution.

    Args:
        gen_videos: List of generated video tensors
        gt_videos: List of ground truth video tensors
        device: Computation device

    Returns:
        FVD score (lower is better)
    """
    from scipy import linalg

    def frechet_distance(mu1, sigma1, mu2, sigma2):
        """Compute Fréchet distance between two Gaussians."""
        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))

    # Try I3D (standard for FVD)
    i3d_path = download_i3d_weights()
    if i3d_path is not None:
        try:
            logger.info("Computing FVD using I3D features (standard)...")
            i3d = torch.jit.load(i3d_path).eval().to(device)

            def extract_i3d_features(videos: List[torch.Tensor]) -> np.ndarray:
                features = []
                with torch.no_grad():
                    for video in videos:
                        # video: (T, C, H, W) in [0, 1]
                        T, C, H, W = video.shape
                        v = video.to(device)

                        # Resize each frame to 224x224 (bilinear for 4D)
                        v = F.interpolate(v, size=(224, 224), mode='bilinear', align_corners=False)
                        # v: (T, C, 224, 224)

                        # Convert to I3D input format: (N, T, H, W, C)
                        v = v.permute(0, 2, 3, 1).unsqueeze(0)  # (1, T, 224, 224, C)

                        # Scale to [-1, 1] as expected by I3D
                        v = v * 2 - 1

                        # Extract features
                        feat = i3d(v)  # (1, 400) logits
                        features.append(feat.cpu().numpy())

                return np.concatenate(features, axis=0)

            gen_feats = extract_i3d_features(gen_videos)
            gt_feats = extract_i3d_features(gt_videos)

            # Compute statistics
            mu_gen, mu_gt = np.mean(gen_feats, axis=0), np.mean(gt_feats, axis=0)
            sigma_gen = np.cov(gen_feats, rowvar=False) if gen_feats.shape[0] > 1 else np.zeros((gen_feats.shape[1], gen_feats.shape[1]))
            sigma_gt = np.cov(gt_feats, rowvar=False) if gt_feats.shape[0] > 1 else np.zeros((gt_feats.shape[1], gt_feats.shape[1]))

            return frechet_distance(mu_gen, sigma_gen, mu_gt, sigma_gt)

        except Exception as e:
            logger.warning(f"I3D failed: {e}, falling back to R3D-18")

    # Fallback: R3D-18
    try:
        import torchvision.models.video as video_models

        logger.info("Computing FVD using R3D-18 features (fallback)...")
        model = video_models.r3d_18(pretrained=True).to(device)
        model.eval()
        model.fc = torch.nn.Identity()

        def extract_r3d_features(videos: List[torch.Tensor]) -> np.ndarray:
            features = []
            with torch.no_grad():
                for video in videos:
                    v = video.unsqueeze(0).permute(0, 2, 1, 3, 4).to(device)
                    v = F.interpolate(v, size=(v.shape[2], 112, 112), mode='trilinear', align_corners=False)
                    feat = model(v)
                    features.append(feat.cpu().numpy())
            return np.concatenate(features, axis=0)

        gen_feats = extract_r3d_features(gen_videos)
        gt_feats = extract_r3d_features(gt_videos)

        mu_gen, mu_gt = np.mean(gen_feats, axis=0), np.mean(gt_feats, axis=0)
        sigma_gen = np.cov(gen_feats, rowvar=False) if gen_feats.shape[0] > 1 else np.zeros((gen_feats.shape[1], gen_feats.shape[1]))
        sigma_gt = np.cov(gt_feats, rowvar=False) if gt_feats.shape[0] > 1 else np.zeros((gt_feats.shape[1], gt_feats.shape[1]))

        return frechet_distance(mu_gen, sigma_gen, mu_gt, sigma_gt)

    except Exception as e:
        logger.warning(f"FVD computation failed: {e}")
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
        if not sample_dir.is_dir() or sample_dir.name.startswith('_'):
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


def format_mean_std(values: List[float], precision: int = 4) -> str:
    """Format as 'mean ± std' for paper."""
    mean = np.mean(values)
    std = np.std(values)
    if precision == 2:
        return f"{mean:.2f} ± {std:.2f}"
    return f"{mean:.{precision}f} ± {std:.{precision}f}"


def evaluate(input_dir: str, output_csv: str, device: str = "cuda", fvd_frames: int = 16):
    """Run full evaluation.

    Args:
        input_dir: Path to inference output directory
        output_csv: Path to save metrics CSV
        device: Computation device
        fvd_frames: Number of consecutive frames for FVD (default: 16, standard for paper comparison)
    """

    logger.info(f"Evaluating: {input_dir}")
    logger.info(f"Device: {device}")
    logger.info(f"FVD frames: {fvd_frames} (consecutive, center-cropped)")

    # Find all video pairs
    pairs_by_view = find_video_pairs(input_dir)
    if not pairs_by_view:
        logger.error("No video pairs found!")
        return

    all_views = sorted(pairs_by_view.keys())
    total_samples = sum(len(pairs) for pairs in pairs_by_view.values())
    logger.info(f"Found {len(all_views)} views, {total_samples} total video pairs")
    for view in all_views:
        logger.info(f"  {view}: {len(pairs_by_view[view])} samples")

    # Check dependencies
    try:
        from torchmetrics.image import StructuralSimilarityIndexMeasure
        logger.info("torchmetrics available for SSIM")
    except ImportError:
        logger.error("torchmetrics required! Install: pip install torchmetrics")
        return

    # Initialize LPIPS model
    lpips_model = None
    try:
        import lpips
        lpips_model = lpips.LPIPS(net='alex').to(device)
        lpips_model.eval()
        logger.info("LPIPS model loaded (AlexNet)")
    except ImportError:
        logger.error("lpips required! Install: pip install lpips")
        return

    # ========================================================================
    # Per-sample paired metrics: PSNR, SSIM, LPIPS
    # ========================================================================
    results = []  # Per-sample results
    all_psnr = []
    all_ssim = []
    all_lpips = []

    for view_name in all_views:
        pairs = pairs_by_view[view_name]
        view_psnr = []
        view_ssim = []
        view_lpips = []

        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating view: {view_name} ({len(pairs)} samples)")
        logger.info(f"{'='*60}")

        for sample_id, gen_path, gt_path in pairs:
            gen_frames_np = read_video_frames(gen_path)
            gt_frames_np = read_video_frames(gt_path)

            # Check frame count consistency
            n_gen, n_gt = len(gen_frames_np), len(gt_frames_np)
            if n_gen != n_gt:
                logger.warning(f"Frame count mismatch for {sample_id}/{view_name}: "
                               f"gen={n_gen}, gt={n_gt}. Using min={min(n_gen, n_gt)}")

            n = min(n_gen, n_gt)
            gen_frames = frames_to_tensor(gen_frames_np[:n]).to(device)
            gt_frames = frames_to_tensor(gt_frames_np[:n]).to(device)

            # PSNR
            psnr_val = compute_psnr(gen_frames, gt_frames)
            view_psnr.append(psnr_val)

            # SSIM
            ssim_val = compute_ssim(gen_frames, gt_frames)
            view_ssim.append(ssim_val)

            # LPIPS
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

        # Per-view summary
        logger.info(f"\n  [{view_name}] PSNR: {format_mean_std(view_psnr, 2)}")
        logger.info(f"  [{view_name}] SSIM: {format_mean_std(view_ssim, 4)}")
        logger.info(f"  [{view_name}] LPIPS: {format_mean_std(view_lpips, 4)}")

        all_psnr.extend(view_psnr)
        all_ssim.extend(view_ssim)
        all_lpips.extend(view_lpips)

    # ========================================================================
    # FID (distribution-level)
    # ========================================================================
    logger.info(f"\n{'='*60}")
    logger.info("Computing FID (extracting frames)...")
    logger.info(f"{'='*60}")

    frames_dir = str(Path(input_dir) / "_eval_frames")

    # Overall FID
    gen_frames_dir, gt_frames_dir, n_frames = extract_and_save_frames(
        input_dir, frames_dir, pairs_by_view
    )
    fid_overall = compute_fid(gen_frames_dir, gt_frames_dir) if n_frames > 0 else float('nan')
    logger.info(f"Overall FID: {fid_overall:.2f}")

    # Per-view FID
    fid_per_view = {}
    for view_name in all_views:
        view_frames_dir = str(Path(frames_dir) / view_name)
        gen_dir, gt_dir = extract_view_frames(pairs_by_view[view_name], view_frames_dir)
        view_fid = compute_fid(gen_dir, gt_dir)
        fid_per_view[view_name] = view_fid
        logger.info(f"  FID [{view_name}]: {view_fid:.2f}")

    # ========================================================================
    # FVD (distribution-level, video)
    # ========================================================================
    logger.info(f"\n{'='*60}")
    logger.info("Computing FVD (Fréchet Video Distance)...")
    logger.info(f"Using {fvd_frames} consecutive frames (center-cropped)")
    logger.info(f"{'='*60}")

    # Collect all videos for FVD
    all_gen_videos = []
    all_gt_videos = []
    fvd_per_view = {}

    for view_name in all_views:
        pairs = pairs_by_view[view_name]
        view_gen_videos = []
        view_gt_videos = []

        for sample_id, gen_path, gt_path in pairs:
            gen_frames_np = read_video_frames(gen_path)
            gt_frames_np = read_video_frames(gt_path)
            n = min(len(gen_frames_np), len(gt_frames_np))

            # Extract center consecutive frames for FVD (standard: 16 frames)
            if n >= fvd_frames:
                # Center crop: take middle fvd_frames consecutive frames
                start_idx = (n - fvd_frames) // 2
                end_idx = start_idx + fvd_frames
                gen_video = frames_to_tensor(gen_frames_np[start_idx:end_idx])  # (fvd_frames, C, H, W)
                gt_video = frames_to_tensor(gt_frames_np[start_idx:end_idx])
            else:
                # Video too short, use all frames (with warning)
                logger.warning(f"Video {sample_id}/{view_name} has only {n} frames, "
                               f"using all (need {fvd_frames} for standard FVD)")
                gen_video = frames_to_tensor(gen_frames_np[:n])
                gt_video = frames_to_tensor(gt_frames_np[:n])

            view_gen_videos.append(gen_video)
            view_gt_videos.append(gt_video)
            all_gen_videos.append(gen_video)
            all_gt_videos.append(gt_video)

        # Per-view FVD
        if len(view_gen_videos) >= 2:  # Need at least 2 samples for covariance
            view_fvd = compute_fvd(view_gen_videos, view_gt_videos, device)
            fvd_per_view[view_name] = view_fvd
            logger.info(f"  FVD [{view_name}]: {view_fvd:.2f}")
        else:
            fvd_per_view[view_name] = float('nan')
            logger.warning(f"  FVD [{view_name}]: skipped (need >= 2 samples)")

    # Overall FVD
    if len(all_gen_videos) >= 2:
        fvd_overall = compute_fvd(all_gen_videos, all_gt_videos, device)
        logger.info(f"Overall FVD: {fvd_overall:.2f}")
    else:
        fvd_overall = float('nan')

    # ========================================================================
    # Final Summary (Paper Format)
    # ========================================================================
    logger.info(f"\n{'='*90}")
    logger.info("FINAL RESULTS (Paper Format)")
    logger.info(f"{'='*90}")
    logger.info(f"{'View':<25} {'PSNR ↑':>15} {'SSIM ↑':>15} {'LPIPS ↓':>15} {'FID ↓':>10} {'FVD ↓':>10}")
    logger.info(f"{'-'*90}")

    per_view_summary = {}
    for view_name in all_views:
        view_results = [r for r in results if r['view'] == view_name]
        vp = [r['psnr'] for r in view_results]
        vs = [r['ssim'] for r in view_results]
        vl = [r['lpips'] for r in view_results]
        vf = fid_per_view.get(view_name, float('nan'))
        vfvd = fvd_per_view.get(view_name, float('nan'))

        per_view_summary[view_name] = {
            'psnr_mean': float(np.mean(vp)),
            'psnr_std': float(np.std(vp)),
            'ssim_mean': float(np.mean(vs)),
            'ssim_std': float(np.std(vs)),
            'lpips_mean': float(np.mean(vl)),
            'lpips_std': float(np.std(vl)),
            'fid': float(vf),
            'fvd': float(vfvd),
            'n_samples': len(view_results),
        }

        logger.info(f"{view_name:<25} {format_mean_std(vp, 2):>15} {format_mean_std(vs, 4):>15} "
                    f"{format_mean_std(vl, 4):>15} {vf:>10.2f} {vfvd:>10.2f}")

    logger.info(f"{'-'*90}")
    logger.info(f"{'OVERALL':<25} {format_mean_std(all_psnr, 2):>15} {format_mean_std(all_ssim, 4):>15} "
                f"{format_mean_std(all_lpips, 4):>15} {fid_overall:>10.2f} {fvd_overall:>10.2f}")
    logger.info(f"{'='*90}")

    # ========================================================================
    # Save results
    # ========================================================================
    output_dir = Path(output_csv).parent if output_csv else Path(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Per-sample CSV
    csv_path = Path(output_csv) if output_csv else output_dir / "metrics.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['sample_id', 'view', 'psnr', 'ssim', 'lpips'])
        writer.writeheader()
        writer.writerows(results)
    logger.info(f"Per-sample results: {csv_path}")

    # 2. Summary CSV (for paper table)
    summary_csv = output_dir / "summary.csv"
    with open(summary_csv, 'w', newline='') as f:
        fieldnames = ['view', 'psnr_mean', 'psnr_std', 'ssim_mean', 'ssim_std',
                      'lpips_mean', 'lpips_std', 'fid', 'fvd', 'n_samples']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for view_name in all_views:
            row = {'view': view_name, **per_view_summary[view_name]}
            writer.writerow(row)
        # Overall row
        writer.writerow({
            'view': 'OVERALL',
            'psnr_mean': float(np.mean(all_psnr)),
            'psnr_std': float(np.std(all_psnr)),
            'ssim_mean': float(np.mean(all_ssim)),
            'ssim_std': float(np.std(all_ssim)),
            'lpips_mean': float(np.mean(all_lpips)),
            'lpips_std': float(np.std(all_lpips)),
            'fid': float(fid_overall),
            'fvd': float(fvd_overall),
            'n_samples': len(results),
        })
    logger.info(f"Summary (for paper): {summary_csv}")

    # 3. Full JSON
    json_path = output_dir / "metrics.json"
    full_results = {
        'overall': {
            'psnr': {'mean': float(np.mean(all_psnr)), 'std': float(np.std(all_psnr))},
            'ssim': {'mean': float(np.mean(all_ssim)), 'std': float(np.std(all_ssim))},
            'lpips': {'mean': float(np.mean(all_lpips)), 'std': float(np.std(all_lpips))},
            'fid': float(fid_overall),
            'fvd': float(fvd_overall),
            'n_samples': len(results),
        },
        'per_view': per_view_summary,
        'per_sample': results,
    }
    with open(json_path, 'w') as f:
        json.dump(full_results, f, indent=2)
    logger.info(f"Full results (JSON): {json_path}")

    # 4. LaTeX table snippet
    latex_path = output_dir / "table.tex"
    newline = " \\\\\n"  # LaTeX row ending
    with open(latex_path, 'w') as f:
        f.write("% Auto-generated LaTeX table\n")
        f.write("\\begin{tabular}{lccccc}\n")
        f.write("\\toprule\n")
        f.write("View & PSNR $\\uparrow$ & SSIM $\\uparrow$ & LPIPS $\\downarrow$ & FID $\\downarrow$ & FVD $\\downarrow$" + newline)
        f.write("\\midrule\n")
        for view_name in all_views:
            s = per_view_summary[view_name]
            view_escaped = view_name.replace('_', '\\_')
            f.write(f"{view_escaped} & "
                    f"{s['psnr_mean']:.2f} $\\pm$ {s['psnr_std']:.2f} & "
                    f"{s['ssim_mean']:.4f} $\\pm$ {s['ssim_std']:.4f} & "
                    f"{s['lpips_mean']:.4f} $\\pm$ {s['lpips_std']:.4f} & "
                    f"{s['fid']:.2f} & "
                    f"{s['fvd']:.2f}" + newline)
        f.write("\\midrule\n")
        f.write(f"Overall & "
                f"{np.mean(all_psnr):.2f} $\\pm$ {np.std(all_psnr):.2f} & "
                f"{np.mean(all_ssim):.4f} $\\pm$ {np.std(all_ssim):.4f} & "
                f"{np.mean(all_lpips):.4f} $\\pm$ {np.std(all_lpips):.4f} & "
                f"{fid_overall:.2f} & "
                f"{fvd_overall:.2f}" + newline)
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
    logger.info(f"LaTeX table: {latex_path}")

    logger.info("\nEvaluation complete!")


def main():
    parser = argparse.ArgumentParser(description="Evaluate inference results")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to inference output directory")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Path to save metrics CSV (default: {input_dir}/metrics.csv)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for computation")
    parser.add_argument("--fvd_frames", type=int, default=16,
                        help="Number of consecutive frames for FVD computation "
                             "(default: 16, standard for paper comparison). "
                             "Uses center-cropped consecutive frames.")
    args = parser.parse_args()

    if args.output_csv is None:
        args.output_csv = str(Path(args.input_dir) / "metrics.csv")

    evaluate(args.input_dir, args.output_csv, args.device, args.fvd_frames)


if __name__ == "__main__":
    main()
