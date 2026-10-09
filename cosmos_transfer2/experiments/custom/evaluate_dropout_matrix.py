#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Folder-vs-folder evaluation of the modality-dropout inference matrix.

Compares each setting's generated videos against ground-truth videos and
reports paired PSNR / SSIM / LPIPS plus distribution-level FID / FVD.

Expected layout per setting directory (output of inference*.py):
    {setting_dir}/{sample_id}/{camera_short}_generated.mp4

Usage (single GPU is enough):
    CUDA_VISIBLE_DEVICES=0 python3 -m cosmos_transfer2.experiments.custom.evaluate_dropout_matrix \
        --gt_videos_root /mnt/zihanw/proj_utils_pro/transfer_video_maker/output_full_data/BlurProjection/videos \
        --results full=/mnt/zihanw/Output_R2V_world_foundation_model_v1/inference \
                  no_hdmap=/mnt/zihanw/Output_R2V_world_foundation_model_v1/inference_dropout/no_hdmap \
                  no_blur=/mnt/zihanw/Output_R2V_world_foundation_model_v1/inference_dropout/no_blur \
                  no_depth=/mnt/zihanw/Output_R2V_world_foundation_model_v1/inference_dropout/no_depth \
                  no_control=/mnt/zihanw/Output_R2V_world_foundation_model_v1/inference_dropout/no_control \
        --output_csv /mnt/zihanw/Output_R2V_world_foundation_model_v1/inference_dropout/metrics.csv
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from loguru import logger

from cosmos_transfer2.experiments.custom.evaluate_metrics import (
    FIDCalculator,
    FVDCalculator,
    calculate_psnr,
    calculate_ssim,
)

try:
    import lpips as lpips_lib
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False

CAMERAS = (
    "camera_front_wide_120fov",
    "camera_cross_right_120fov",
    "camera_rear_right_70fov",
    "camera_rear_tele_30fov",
    "camera_rear_left_70fov",
    "camera_cross_left_120fov",
    "camera_front_tele_30fov",
)

NUM_FRAMES = 29


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate dropout inference matrix")
    p.add_argument("--gt_videos_root", type=str, required=True,
                   help="BlurProjection/videos root containing ftheta_camera_* folders")
    p.add_argument("--results", type=str, nargs="+", required=True,
                   help="name=path pairs; include 'full' to enable paired deltas")
    p.add_argument("--output_csv", type=str, required=True)
    p.add_argument("--num_views", type=int, default=7)
    p.add_argument("--lpips_size", type=int, default=256,
                   help="Frames are resized to this square size for LPIPS only")
    p.add_argument("--frame_batch", type=int, default=8)
    return p.parse_args()


def load_video_frames(path: Path, num_frames: int) -> torch.Tensor:
    """Load first num_frames as float tensor (T, C, H, W) in [0, 1]."""
    from decord import VideoReader
    vr = VideoReader(str(path))
    n = min(num_frames, len(vr))
    frames = vr.get_batch(list(range(n))).asnumpy()  # (T, H, W, C) uint8
    t = torch.from_numpy(frames).float() / 255.0
    return rearrange(t, "t h w c -> t c h w")


@torch.no_grad()
def extract_fvd_features(fvd: FVDCalculator, video_T_C_H_W: torch.Tensor, device) -> torch.Tensor:
    """R3D-18 clip feature for one video. Avoids the buggy multi-clip reshape
    in FVDCalculator.extract_features by handling N=1 explicitly."""
    if fvd.i3d is None:
        feat = F.adaptive_avg_pool2d(video_T_C_H_W.to(device), (7, 7))
        return feat.flatten().unsqueeze(0)
    v = F.interpolate(video_T_C_H_W.to(device), size=(112, 112), mode="bilinear", align_corners=False)
    v = rearrange(v, "t c h w -> 1 c t h w")
    mean = torch.tensor([0.43216, 0.394666, 0.37645], device=device).view(1, 3, 1, 1, 1)
    std = torch.tensor([0.22803, 0.22145, 0.216989], device=device).view(1, 3, 1, 1, 1)
    return fvd.i3d((v - mean) / std)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    settings = {}
    for spec in args.results:
        name, _, path = spec.partition("=")
        if not path:
            raise ValueError(f"--results entries must be name=path, got: {spec}")
        settings[name] = Path(path)
    gt_root = Path(args.gt_videos_root)
    cameras = CAMERAS[:args.num_views]

    # Samples = intersection of per-setting sample dirs
    sample_sets = []
    for name, d in settings.items():
        if not d.exists():
            raise FileNotFoundError(f"Results dir for '{name}' not found: {d}")
        sample_sets.append({p.name for p in d.iterdir() if p.is_dir()})
    samples = sorted(set.intersection(*sample_sets))
    if not samples:
        raise ValueError("No common sample directories across settings")
    logger.info(f"Settings: {list(settings)} | {len(samples)} common samples | {len(cameras)} views")

    lpips_model = None
    if LPIPS_AVAILABLE:
        lpips_model = lpips_lib.LPIPS(net="alex").to(device).eval()
    else:
        logger.warning("lpips not installed -> LPIPS column will be NaN")

    fid = FIDCalculator(device)
    fvd = FVDCalculator(device)

    per_pair = defaultdict(list)       # (setting, metric) -> per (sample,view) values
    per_sample_rows = []
    fid_feats = defaultdict(list)      # setting -> inception features; "GT" for real
    fvd_feats = defaultdict(list)

    for si, sample in enumerate(samples):
        for cam in cameras:
            cam_short = cam.replace("camera_", "")
            gt_path = gt_root / f"ftheta_{cam}" / f"{sample}.mp4"
            if not gt_path.exists():
                logger.warning(f"GT missing, skipping: {gt_path}")
                continue
            gt = load_video_frames(gt_path, NUM_FRAMES)

            gt_dev_cache = None
            for name, d in settings.items():
                gen_path = d / sample / f"{cam_short}_generated.mp4"
                if not gen_path.exists():
                    logger.warning(f"[{name}] missing: {gen_path}")
                    continue
                gen = load_video_frames(gen_path, NUM_FRAMES)
                T = min(gen.shape[0], gt.shape[0])
                gen_t = gen[:T]
                gt_t = gt[:T]
                if gt_t.shape[-2:] != gen_t.shape[-2:]:
                    gt_t = F.interpolate(gt_t, size=gen_t.shape[-2:], mode="bilinear", align_corners=False)

                psnr_vals, ssim_vals, lpips_vals = [], [], []
                for s in range(0, T, args.frame_batch):
                    g = gen_t[s:s + args.frame_batch].to(device)
                    r = gt_t[s:s + args.frame_batch].to(device)
                    psnr_vals.append(calculate_psnr(r, g))
                    ssim_vals.append(calculate_ssim(r, g))
                    if lpips_model is not None:
                        gs = F.interpolate(g, size=(args.lpips_size, args.lpips_size),
                                           mode="bilinear", align_corners=False)
                        rs = F.interpolate(r, size=(args.lpips_size, args.lpips_size),
                                           mode="bilinear", align_corners=False)
                        with torch.no_grad():
                            lpips_vals.append(lpips_model(rs * 2 - 1, gs * 2 - 1).mean().item())
                    # FID features (generated; GT once per pair)
                    with torch.no_grad():
                        fid_feats[name].append(fid.extract_features(g).cpu())
                        if gt_dev_cache is None:
                            fid_feats["GT"].append(fid.extract_features(r).cpu())
                gt_dev_cache = True

                row = {
                    "setting": name, "sample": sample, "camera": cam_short,
                    "psnr": float(np.mean(psnr_vals)),
                    "ssim": float(np.mean(ssim_vals)),
                    "lpips": float(np.mean(lpips_vals)) if lpips_vals else float("nan"),
                }
                per_sample_rows.append(row)
                for m in ("psnr", "ssim", "lpips"):
                    per_pair[(name, m)].append(row[m])

                fvd_feats[name].append(extract_fvd_features(fvd, gen_t, device).cpu())
            fvd_feats["GT"].append(extract_fvd_features(fvd, gt_t, device).cpu())

        logger.info(f"[{si + 1}/{len(samples)}] {sample} done")

    # Distribution metrics
    gt_fid_feat = torch.cat(fid_feats.pop("GT"), dim=0)
    gt_fvd_feat = torch.cat(fvd_feats.pop("GT"), dim=0)
    dist_metrics = {}
    for name in settings:
        f_fid = torch.cat(fid_feats[name], dim=0)
        f_fvd = torch.cat(fvd_feats[name], dim=0)
        dist_metrics[name] = {
            "fid": fid.calculate_fid(gt_fid_feat, f_fid),
            "fvd": fvd.calculate_fvd(gt_fvd_feat, f_fvd),
        }

    # Report
    header = f"{'setting':<12} {'PSNR↑':>8} {'SSIM↑':>8} {'LPIPS↓':>8} {'FID↓':>8} {'FVD↓':>10}"
    logger.info("=" * len(header))
    logger.info(header)
    logger.info("-" * len(header))
    summary_rows = []
    for name in settings:
        p = np.mean(per_pair[(name, "psnr")])
        s = np.mean(per_pair[(name, "ssim")])
        l = np.nanmean(per_pair[(name, "lpips")])
        row = dict(setting=name, psnr=p, ssim=s, lpips=l, **dist_metrics[name])
        summary_rows.append(row)
        logger.info(f"{name:<12} {p:>8.3f} {s:>8.4f} {l:>8.4f} "
                    f"{dist_metrics[name]['fid']:>8.2f} {dist_metrics[name]['fvd']:>10.2f}")
    logger.info("=" * len(header))

    # Paired deltas and win counts vs full
    if "full" in settings:
        logger.info("Paired deltas vs 'full' (positive = worse than full):")
        for name in settings:
            if name == "full":
                continue
            msgs = []
            for m, better_low in (("psnr", False), ("ssim", False), ("lpips", True)):
                a = np.array(per_pair[("full", m)])
                b = np.array(per_pair[(name, m)])
                n = min(len(a), len(b))
                d = (a[:n] - b[:n]) if not better_low else (b[:n] - a[:n])
                wins = int((d > 0).sum())
                msgs.append(f"{m}: Δ={d.mean():+.4f} (full better on {wins}/{n})")
            logger.info(f"  {name:<12} " + " | ".join(msgs))

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["setting", "psnr", "ssim", "lpips", "fid", "fvd"])
        w.writeheader()
        w.writerows(summary_rows)
    detail = out.with_name(out.stem + "_per_sample.csv")
    with open(detail, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["setting", "sample", "camera", "psnr", "ssim", "lpips"])
        w.writeheader()
        w.writerows(per_sample_rows)
    logger.info(f"Saved summary to {out} and per-sample details to {detail}")


if __name__ == "__main__":
    main()
