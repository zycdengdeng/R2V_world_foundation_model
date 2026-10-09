#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Folder-vs-folder evaluation of the modality-dropout inference matrix.

Compares each setting's generated videos against ground-truth videos and
reports paired PSNR / SSIM / LPIPS plus distribution-level FID / FVD,
per-seg aggregates, and a visualization-pick ranking.

Expected layout per setting directory (output of inference*.py):
    {setting_dir}/{sample_id}/{camera_short}_generated.mp4

Single-GPU usage:
    CUDA_VISIBLE_DEVICES=0 python3 -m cosmos_transfer2.experiments.custom.evaluate_dropout_matrix \
        --gt_videos_root .../BlurProjection/videos \
        --results full=.../inference no_hdmap=.../inference_dropout/no_hdmap ... \
        --output_csv .../inference_dropout/metrics.csv

Multi-GPU usage (shard by samples, then merge):
    bash scripts/run_eval_matrix_8gpu.sh     # launches 8 shards + merge
Or manually:
    CUDA_VISIBLE_DEVICES=$i python3 -m ...evaluate_dropout_matrix ... \
        --num_shards 8 --shard_idx $i        # for i in 0..7, in parallel
    python3 -m ...evaluate_dropout_matrix ... --merge_shards
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
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
    # Sharded execution
    p.add_argument("--num_shards", type=int, default=1,
                   help=">1: process only this process's share of samples and save a shard file")
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--merge_shards", action="store_true", default=False,
                   help="Load all shard files next to --output_csv and produce the final report")
    return p.parse_args()


def shard_path(output_csv: str, shard_idx: int) -> Path:
    out = Path(output_csv)
    return out.with_name(out.stem + f"_shard{shard_idx}.pt")


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


def compute(args, settings, samples, device):
    """Compute paired metrics and FID/FVD features for the given samples."""
    cameras = CAMERAS[:args.num_views]
    gt_root = Path(args.gt_videos_root)

    lpips_model = None
    if LPIPS_AVAILABLE:
        lpips_model = lpips_lib.LPIPS(net="alex").to(device).eval()
    else:
        logger.warning("lpips not installed -> LPIPS column will be NaN")

    fid = FIDCalculator(device)
    fvd = FVDCalculator(device)

    per_sample_rows = []
    fid_feats = defaultdict(list)
    fvd_feats = defaultdict(list)

    for si, sample in enumerate(samples):
        for cam in cameras:
            cam_short = cam.replace("camera_", "")
            gt_path = gt_root / f"ftheta_{cam}" / f"{sample}.mp4"
            if not gt_path.exists():
                logger.warning(f"GT missing, skipping: {gt_path}")
                continue
            gt = load_video_frames(gt_path, NUM_FRAMES)

            gt_feat_done = False
            gt_t = None
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
                    with torch.no_grad():
                        fid_feats[name].append(fid.extract_features(g).cpu())
                        if not gt_feat_done:
                            fid_feats["GT"].append(fid.extract_features(r).cpu())
                gt_feat_done = True

                per_sample_rows.append({
                    "setting": name, "sample": sample, "camera": cam_short,
                    "psnr": float(np.mean(psnr_vals)),
                    "ssim": float(np.mean(ssim_vals)),
                    "lpips": float(np.mean(lpips_vals)) if lpips_vals else float("nan"),
                })
                fvd_feats[name].append(extract_fvd_features(fvd, gen_t, device).cpu())
            if gt_t is not None:
                fvd_feats["GT"].append(extract_fvd_features(fvd, gt_t, device).cpu())

        logger.info(f"[shard {args.shard_idx}] [{si + 1}/{len(samples)}] {sample} done")

    fid_cat = {k: torch.cat(v, dim=0) for k, v in fid_feats.items()}
    fvd_cat = {k: torch.cat(v, dim=0) for k, v in fvd_feats.items()}
    return per_sample_rows, fid_cat, fvd_cat


def report(args, setting_names, per_sample_rows, fid_cat, fvd_cat, device):
    """Final tables, rankings, and CSV outputs."""
    per_pair = defaultdict(list)
    for row in per_sample_rows:
        for m in ("psnr", "ssim", "lpips"):
            per_pair[(row["setting"], m)].append(row[m])
    samples = sorted({r["sample"] for r in per_sample_rows})

    fid = FIDCalculator(device)
    fvd = FVDCalculator(device)
    gt_fid_feat = fid_cat.pop("GT")
    gt_fvd_feat = fvd_cat.pop("GT")
    dist_metrics = {}
    for name in setting_names:
        dist_metrics[name] = {
            "fid": fid.calculate_fid(gt_fid_feat, fid_cat[name]),
            "fvd": fvd.calculate_fvd(gt_fvd_feat, fvd_cat[name]),
        }

    header = f"{'setting':<12} {'PSNR↑':>8} {'SSIM↑':>8} {'LPIPS↓':>8} {'FID↓':>8} {'FVD↓':>10}"
    logger.info("=" * len(header))
    logger.info(header)
    logger.info("-" * len(header))
    summary_rows = []
    for name in setting_names:
        p = np.mean(per_pair[(name, "psnr")])
        s = np.mean(per_pair[(name, "ssim")])
        l = np.nanmean(per_pair[(name, "lpips")])
        summary_rows.append(dict(setting=name, psnr=p, ssim=s, lpips=l, **dist_metrics[name]))
        logger.info(f"{name:<12} {p:>8.3f} {s:>8.4f} {l:>8.4f} "
                    f"{dist_metrics[name]['fid']:>8.2f} {dist_metrics[name]['fvd']:>10.2f}")
    logger.info("=" * len(header))

    if "full" in setting_names:
        logger.info("Paired deltas vs 'full' (positive = worse than full):")
        for name in setting_names:
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

    # Per-seg aggregation + visualization ranking
    per_seg = defaultdict(lambda: defaultdict(list))
    for row in per_sample_rows:
        for m in ("psnr", "ssim", "lpips"):
            per_seg[row["sample"]][(row["setting"], m)].append(row[m])

    seg_rows = []
    for sample in samples:
        for name in setting_names:
            if (name, "psnr") not in per_seg[sample]:
                continue
            seg_rows.append({
                "sample": sample, "setting": name,
                "psnr": float(np.mean(per_seg[sample][(name, "psnr")])),
                "ssim": float(np.mean(per_seg[sample][(name, "ssim")])),
                "lpips": float(np.nanmean(per_seg[sample][(name, "lpips")])),
            })

    seg_metric = {(r["sample"], r["setting"]): r for r in seg_rows}
    if "full" in setting_names:
        dropout_names = [n for n in setting_names if n != "full"]
        ranking = []
        for sample in samples:
            if (sample, "full") not in seg_metric:
                continue
            full_r = seg_metric[(sample, "full")]
            gaps = {}
            for n in dropout_names:
                r = seg_metric.get((sample, n))
                if r is not None:
                    gaps[n] = r["lpips"] - full_r["lpips"]
            if gaps:
                ranking.append({"sample": sample,
                                "viz_score": float(np.nansum(list(gaps.values()))),
                                "gaps": gaps})
        ranking.sort(key=lambda r: r["viz_score"], reverse=True)

        logger.info("")
        logger.info("Per-seg visualization ranking (viz_score = sum of LPIPS gaps vs full; "
                    "higher = dropout damage more visible = better figure candidate):")
        hdr = f"{'rank':<5} {'sample':<14} {'viz_score':>9} " + " ".join(
            f"{('ΔL_' + n):>14}" for n in dropout_names)
        logger.info(hdr)
        for rank, r in enumerate(ranking, 1):
            gap_str = " ".join(f"{r['gaps'].get(n, float('nan')):>14.4f}" for n in dropout_names)
            logger.info(f"{rank:<5} {r['sample']:<14} {r['viz_score']:>9.4f} {gap_str}")
        if ranking:
            logger.info(f"Suggested figure candidates: {[r['sample'] for r in ranking[:3]]}")

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
    seg_csv = out.with_name(out.stem + "_per_seg.csv")
    with open(seg_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sample", "setting", "psnr", "ssim", "lpips"])
        w.writeheader()
        w.writerows(seg_rows)
    logger.info(f"Saved: {out} (summary), {detail} (per sample+view), {seg_csv} (per seg)")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    settings = {}
    for spec in args.results:
        name, _, path = spec.partition("=")
        if not path:
            raise ValueError(f"--results entries must be name=path, got: {spec}")
        settings[name] = Path(path)
    setting_names = list(settings)

    if args.merge_shards:
        per_sample_rows = []
        fid_parts, fvd_parts = defaultdict(list), defaultdict(list)
        n_loaded = 0
        for i in range(256):
            sp = shard_path(args.output_csv, i)
            if not sp.exists():
                continue
            blob = torch.load(sp, map_location="cpu")
            per_sample_rows.extend(blob["per_sample_rows"])
            for k, v in blob["fid_cat"].items():
                fid_parts[k].append(v)
            for k, v in blob["fvd_cat"].items():
                fvd_parts[k].append(v)
            n_loaded += 1
        if n_loaded == 0:
            raise FileNotFoundError(f"No shard files found next to {args.output_csv}")
        logger.info(f"Merged {n_loaded} shard file(s), {len(per_sample_rows)} rows")
        fid_cat = {k: torch.cat(v, dim=0) for k, v in fid_parts.items()}
        fvd_cat = {k: torch.cat(v, dim=0) for k, v in fvd_parts.items()}
        report(args, setting_names, per_sample_rows, fid_cat, fvd_cat, device)
        return

    # Discover common samples
    sample_sets = []
    for name, d in settings.items():
        if not d.exists():
            raise FileNotFoundError(f"Results dir for '{name}' not found: {d}")
        sample_sets.append({p.name for p in d.iterdir() if p.is_dir()})
    samples = sorted(set.intersection(*sample_sets))
    if not samples:
        raise ValueError("No common sample directories across settings")

    if args.num_shards > 1:
        samples = samples[args.shard_idx::args.num_shards]
        logger.info(f"Shard {args.shard_idx}/{args.num_shards}: {len(samples)} samples")

    logger.info(f"Settings: {setting_names} | {len(samples)} samples | {args.num_views} views")
    per_sample_rows, fid_cat, fvd_cat = compute(args, settings, samples, device)

    if args.num_shards > 1:
        sp = shard_path(args.output_csv, args.shard_idx)
        sp.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"per_sample_rows": per_sample_rows, "fid_cat": fid_cat, "fvd_cat": fvd_cat}, sp)
        logger.info(f"Shard saved: {sp} (run with --merge_shards after all shards finish)")
    else:
        report(args, setting_names, per_sample_rows, fid_cat, fvd_cat, device)


if __name__ == "__main__":
    main()
