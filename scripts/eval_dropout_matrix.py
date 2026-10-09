#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Standalone evaluation of the modality-dropout inference matrix.

No cosmos_transfer2 imports -- only torch / torchvision / decord / numpy
(+ optional lpips, scipy). Defaults are set for the R2V setup, so the
simplest invocation is just:

    CUDA_VISIBLE_DEVICES=0 python3 scripts/eval_dropout_matrix.py

Multi-GPU (8 shards + merge): bash scripts/run_eval_matrix_8gpu.sh

Outputs: summary table, paired deltas vs 'full', per-seg table with a
visualization-pick ranking, and three CSVs next to --output_csv.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import lpips as lpips_lib
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False

try:
    from scipy import linalg as scipy_linalg
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# ----------------------------------------------------------------------------
# Defaults for the R2V setup (override via CLI if paths change)
# ----------------------------------------------------------------------------
OUT_ROOT = "/mnt/zihanw/Output_R2V_world_foundation_model_v1"
DEFAULT_GT_ROOT = "/mnt/zihanw/proj_utils_pro/transfer_video_maker/output_full_data/BlurProjection/videos"
DEFAULT_RESULTS = [
    f"full={OUT_ROOT}/inference",
    f"no_hdmap={OUT_ROOT}/inference_dropout/no_hdmap",
    f"no_blur={OUT_ROOT}/inference_dropout/no_blur",
    f"no_depth={OUT_ROOT}/inference_dropout/no_depth",
    f"no_control={OUT_ROOT}/inference_dropout/no_control",
]
DEFAULT_OUTPUT_CSV = f"{OUT_ROOT}/inference_dropout/metrics.csv"

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
    p = argparse.ArgumentParser(description="Standalone dropout-matrix evaluation")
    p.add_argument("--gt_videos_root", type=str, default=DEFAULT_GT_ROOT)
    p.add_argument("--results", type=str, nargs="+", default=DEFAULT_RESULTS,
                   help="name=path pairs; include 'full' for paired deltas")
    p.add_argument("--output_csv", type=str, default=DEFAULT_OUTPUT_CSV)
    p.add_argument("--num_views", type=int, default=7)
    p.add_argument("--lpips_size", type=int, default=256)
    p.add_argument("--frame_batch", type=int, default=8)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--merge_shards", action="store_true", default=False)
    return p.parse_args()


# ----------------------------------------------------------------------------
# Metrics (self-contained)
# ----------------------------------------------------------------------------

def calc_psnr(real: torch.Tensor, fake: torch.Tensor) -> float:
    mse = F.mse_loss(fake, real)
    return (10 * torch.log10(1.0 / (mse + 1e-10))).item()


def calc_ssim(real: torch.Tensor, fake: torch.Tensor, window: int = 11) -> float:
    """Uniform-window SSIM on (N, C, H, W) tensors in [0, 1]."""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    pad = window // 2
    mu1 = F.avg_pool2d(real, window, stride=1, padding=pad)
    mu2 = F.avg_pool2d(fake, window, stride=1, padding=pad)
    mu1_sq, mu2_sq, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    s1 = F.avg_pool2d(real * real, window, stride=1, padding=pad) - mu1_sq
    s2 = F.avg_pool2d(fake * fake, window, stride=1, padding=pad) - mu2_sq
    s12 = F.avg_pool2d(real * fake, window, stride=1, padding=pad) - mu12
    ssim = ((2 * mu12 + C1) * (2 * s12 + C2)) / ((mu1_sq + mu2_sq + C1) * (s1 + s2 + C2))
    return ssim.mean().item()


def frechet_distance(feat_a: torch.Tensor, feat_b: torch.Tensor) -> float:
    mu1, mu2 = feat_a.mean(0), feat_b.mean(0)
    c1 = (feat_a - mu1).T @ (feat_a - mu1) / (feat_a.shape[0] - 1)
    c2 = (feat_b - mu2).T @ (feat_b - mu2) / (feat_b.shape[0] - 1)
    diff = (mu1 - mu2).double().numpy()
    c1n, c2n = c1.double().numpy(), c2.double().numpy()
    if SCIPY_AVAILABLE:
        covmean, _ = scipy_linalg.sqrtm(c1n @ c2n, disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        return float(diff @ diff + np.trace(c1n) + np.trace(c2n) - 2 * np.trace(covmean))
    return float(diff @ diff + np.trace(c1n) + np.trace(c2n))


class FrameFeatures:
    """InceptionV3 pool features for FID."""

    def __init__(self, device):
        self.device = device
        self.model = None
        try:
            from torchvision.models import inception_v3, Inception_V3_Weights
            m = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
            m.fc = nn.Identity()
            self.model = m.eval().to(device)
        except Exception as e:
            print(f"[warn] InceptionV3 unavailable ({e}); FID uses avg-pool fallback features")

    @torch.no_grad()
    def __call__(self, frames: torch.Tensor) -> torch.Tensor:  # (N,C,H,W) in [0,1]
        if self.model is None:
            return F.adaptive_avg_pool2d(frames, (8, 8)).flatten(1).cpu()
        x = F.interpolate(frames, size=(299, 299), mode="bilinear", align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        return self.model((x - mean) / std).cpu()


class ClipFeatures:
    """R3D-18 clip features for FVD."""

    def __init__(self, device):
        self.device = device
        self.model = None
        try:
            from torchvision.models.video import r3d_18, R3D_18_Weights
            m = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
            m.fc = nn.Identity()
            self.model = m.eval().to(device)
        except Exception as e:
            print(f"[warn] R3D-18 unavailable ({e}); FVD uses avg-pool fallback features")

    @torch.no_grad()
    def __call__(self, video: torch.Tensor) -> torch.Tensor:  # (T,C,H,W) in [0,1]
        if self.model is None:
            return F.adaptive_avg_pool2d(video, (7, 7)).flatten().unsqueeze(0).cpu()
        v = F.interpolate(video.to(self.device), size=(112, 112), mode="bilinear", align_corners=False)
        v = v.permute(1, 0, 2, 3).unsqueeze(0)  # (1, C, T, H, W)
        mean = torch.tensor([0.43216, 0.394666, 0.37645], device=self.device).view(1, 3, 1, 1, 1)
        std = torch.tensor([0.22803, 0.22145, 0.216989], device=self.device).view(1, 3, 1, 1, 1)
        return self.model((v - mean) / std).cpu()


def load_video(path: Path, num_frames: int) -> torch.Tensor:
    from decord import VideoReader
    vr = VideoReader(str(path))
    n = min(num_frames, len(vr))
    frames = vr.get_batch(list(range(n))).asnumpy()
    return torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 255.0  # (T,C,H,W)


def shard_file(output_csv: str, idx: int) -> Path:
    out = Path(output_csv)
    return out.with_name(out.stem + f"_shard{idx}.pt")


# ----------------------------------------------------------------------------
# Compute and report
# ----------------------------------------------------------------------------

def compute(args, settings, samples, device):
    cameras = CAMERAS[:args.num_views]
    gt_root = Path(args.gt_videos_root)
    lpips_model = lpips_lib.LPIPS(net="alex").to(device).eval() if LPIPS_AVAILABLE else None
    if lpips_model is None:
        print("[warn] lpips not installed -> LPIPS column will be NaN")
    fid_extract = FrameFeatures(device)
    fvd_extract = ClipFeatures(device)

    rows, fid_feats, fvd_feats = [], defaultdict(list), defaultdict(list)

    for si, sample in enumerate(samples):
        for cam in cameras:
            cam_short = cam.replace("camera_", "")
            gt_path = gt_root / f"ftheta_{cam}" / f"{sample}.mp4"
            if not gt_path.exists():
                print(f"[warn] GT missing, skipping: {gt_path}")
                continue
            gt = load_video(gt_path, NUM_FRAMES)

            gt_feat_done = False
            gt_t = None
            for name, d in settings.items():
                gen_path = d / sample / f"{cam_short}_generated.mp4"
                if not gen_path.exists():
                    print(f"[warn] [{name}] missing: {gen_path}")
                    continue
                gen = load_video(gen_path, NUM_FRAMES)
                T = min(gen.shape[0], gt.shape[0])
                gen_t, gt_t = gen[:T], gt[:T]
                if gt_t.shape[-2:] != gen_t.shape[-2:]:
                    gt_t = F.interpolate(gt_t, size=gen_t.shape[-2:], mode="bilinear", align_corners=False)

                ps, ss, lp = [], [], []
                for s in range(0, T, args.frame_batch):
                    g = gen_t[s:s + args.frame_batch].to(device)
                    r = gt_t[s:s + args.frame_batch].to(device)
                    ps.append(calc_psnr(r, g))
                    ss.append(calc_ssim(r, g))
                    if lpips_model is not None:
                        gs = F.interpolate(g, size=(args.lpips_size,) * 2, mode="bilinear", align_corners=False)
                        rs = F.interpolate(r, size=(args.lpips_size,) * 2, mode="bilinear", align_corners=False)
                        with torch.no_grad():
                            lp.append(lpips_model(rs * 2 - 1, gs * 2 - 1).mean().item())
                    fid_feats[name].append(fid_extract(g))
                    if not gt_feat_done:
                        fid_feats["GT"].append(fid_extract(r))
                gt_feat_done = True

                rows.append({"setting": name, "sample": sample, "camera": cam_short,
                             "psnr": float(np.mean(ps)), "ssim": float(np.mean(ss)),
                             "lpips": float(np.mean(lp)) if lp else float("nan")})
                fvd_feats[name].append(fvd_extract(gen_t))
            if gt_t is not None:
                fvd_feats["GT"].append(fvd_extract(gt_t))
        print(f"[shard {args.shard_idx}] [{si + 1}/{len(samples)}] {sample} done", flush=True)

    fid_cat = {k: torch.cat(v, dim=0) for k, v in fid_feats.items()}
    fvd_cat = {k: torch.cat(v, dim=0) for k, v in fvd_feats.items()}
    return rows, fid_cat, fvd_cat


def report(args, setting_names, rows, fid_cat, fvd_cat):
    per_pair = defaultdict(list)
    for r in rows:
        for m in ("psnr", "ssim", "lpips"):
            per_pair[(r["setting"], m)].append(r[m])
    samples = sorted({r["sample"] for r in rows})

    gt_fid, gt_fvd = fid_cat.pop("GT"), fvd_cat.pop("GT")
    dist = {n: {"fid": frechet_distance(gt_fid, fid_cat[n]),
                "fvd": frechet_distance(gt_fvd, fvd_cat[n])} for n in setting_names}

    header = f"{'setting':<12} {'PSNR↑':>8} {'SSIM↑':>8} {'LPIPS↓':>8} {'FID↓':>8} {'FVD↓':>10}"
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    summary = []
    for n in setting_names:
        p, s = np.mean(per_pair[(n, "psnr")]), np.mean(per_pair[(n, "ssim")])
        l = np.nanmean(per_pair[(n, "lpips")])
        summary.append(dict(setting=n, psnr=p, ssim=s, lpips=l, **dist[n]))
        print(f"{n:<12} {p:>8.3f} {s:>8.4f} {l:>8.4f} {dist[n]['fid']:>8.2f} {dist[n]['fvd']:>10.2f}")
    print("=" * len(header))

    if "full" in setting_names:
        print("\nPaired deltas vs 'full' (positive = worse than full):")
        for n in setting_names:
            if n == "full":
                continue
            msgs = []
            for m, low in (("psnr", False), ("ssim", False), ("lpips", True)):
                a, b = np.array(per_pair[("full", m)]), np.array(per_pair[(n, m)])
                k = min(len(a), len(b))
                d = (b[:k] - a[:k]) if low else (a[:k] - b[:k])
                msgs.append(f"{m}: Δ={np.nanmean(d):+.4f} (full better on {int((d > 0).sum())}/{k})")
            print(f"  {n:<12} " + " | ".join(msgs))

    # Per-seg aggregation + viz ranking
    per_seg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for m in ("psnr", "ssim", "lpips"):
            per_seg[r["sample"]][(r["setting"], m)].append(r[m])
    seg_rows = []
    for sample in samples:
        for n in setting_names:
            if (n, "psnr") in per_seg[sample]:
                seg_rows.append({"sample": sample, "setting": n,
                                 "psnr": float(np.mean(per_seg[sample][(n, "psnr")])),
                                 "ssim": float(np.mean(per_seg[sample][(n, "ssim")])),
                                 "lpips": float(np.nanmean(per_seg[sample][(n, "lpips")]))})
    seg_metric = {(r["sample"], r["setting"]): r for r in seg_rows}

    if "full" in setting_names:
        drops = [n for n in setting_names if n != "full"]
        ranking = []
        for sample in samples:
            fr = seg_metric.get((sample, "full"))
            if fr is None:
                continue
            gaps = {n: seg_metric[(sample, n)]["lpips"] - fr["lpips"]
                    for n in drops if (sample, n) in seg_metric}
            if gaps:
                ranking.append({"sample": sample, "viz_score": float(np.nansum(list(gaps.values()))),
                                "gaps": gaps})
        ranking.sort(key=lambda r: r["viz_score"], reverse=True)
        print("\nPer-seg visualization ranking (viz_score = sum of LPIPS gaps vs full; "
              "higher = dropout damage more visible):")
        print(f"{'rank':<5} {'sample':<14} {'viz_score':>9} " +
              " ".join(f"{('ΔL_' + n):>14}" for n in drops))
        for rank, r in enumerate(ranking, 1):
            print(f"{rank:<5} {r['sample']:<14} {r['viz_score']:>9.4f} " +
                  " ".join(f"{r['gaps'].get(n, float('nan')):>14.4f}" for n in drops))
        if ranking:
            print(f"Suggested figure candidates: {[r['sample'] for r in ranking[:3]]}")

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["setting", "psnr", "ssim", "lpips", "fid", "fvd"])
        w.writeheader()
        w.writerows(summary)
    with open(out.with_name(out.stem + "_per_sample.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["setting", "sample", "camera", "psnr", "ssim", "lpips"])
        w.writeheader()
        w.writerows(rows)
    with open(out.with_name(out.stem + "_per_seg.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sample", "setting", "psnr", "ssim", "lpips"])
        w.writeheader()
        w.writerows(seg_rows)
    print(f"\nSaved: {out}, *_per_sample.csv, *_per_seg.csv")


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
        rows, fid_p, fvd_p = [], defaultdict(list), defaultdict(list)
        n_loaded = 0
        for i in range(256):
            sp = shard_file(args.output_csv, i)
            if sp.exists():
                blob = torch.load(sp, map_location="cpu")
                rows.extend(blob["rows"])
                for k, v in blob["fid_cat"].items():
                    fid_p[k].append(v)
                for k, v in blob["fvd_cat"].items():
                    fvd_p[k].append(v)
                n_loaded += 1
        if n_loaded == 0:
            raise FileNotFoundError(f"No shard files found next to {args.output_csv}")
        print(f"Merged {n_loaded} shard file(s), {len(rows)} rows")
        report(args, setting_names,
               rows,
               {k: torch.cat(v) for k, v in fid_p.items()},
               {k: torch.cat(v) for k, v in fvd_p.items()})
        return

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
        print(f"Shard {args.shard_idx}/{args.num_shards}: {len(samples)} samples")
    print(f"Settings: {setting_names} | {len(samples)} samples | {args.num_views} views")

    rows, fid_cat, fvd_cat = compute(args, settings, samples, device)

    if args.num_shards > 1:
        sp = shard_file(args.output_csv, args.shard_idx)
        sp.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"rows": rows, "fid_cat": fid_cat, "fvd_cat": fvd_cat}, sp)
        print(f"Shard saved: {sp} (run with --merge_shards after all shards finish)")
    else:
        report(args, setting_names, rows, fid_cat, fvd_cat)


if __name__ == "__main__":
    main()
