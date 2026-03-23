#!/usr/bin/env python3
"""
Merge 3 segments (29 frames each) into 85 frames for long video conditioning.

29 frames per segment × 3 = 87 frames total
87 does not satisfy 4k+1 format (87 = 4*21 + 3)
So we need 85 frames (85 = 4*21 + 1) ✓

Strategy options:
- Option 1: Keep all of seg01 (29) + all of seg02 (29) + first 27 of seg03 = 85 frames
- Option 2: Keep first 28 of seg01 + all of seg02 (29) + first 28 of seg03 = 85 frames
- Option 3: Custom distribution

Usage:
    python merge_segments_for_85frames.py \
        --blur_dir /path/to/blur_dataset \
        --depth_dir /path/to/depth_dataset \
        --hdmap_dir /path/to/hdmap_dataset \
        --scene_id 031 \
        --output_dir /path/to/output \
        --strategy keep_seg01_seg02_trim_seg03
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from decord import VideoReader
from torchvision.io import write_video

# Default cameras
DEFAULT_CAMERAS = (
    "camera_front_wide_120fov",
    "camera_cross_right_120fov",
    "camera_rear_right_70fov",
    "camera_rear_tele_30fov",
    "camera_rear_left_70fov",
    "camera_cross_left_120fov",
    "camera_front_tele_30fov",
)


def validate_frame_count(num_frames: int) -> Tuple[bool, int, int]:
    """
    Validate if frame count satisfies 4k+1 format.

    Returns:
        (is_valid, latent_frames, remainder)
    """
    remainder = (num_frames - 1) % 4
    latent_frames = (num_frames - 1) // 4 + 1
    is_valid = remainder == 0
    return is_valid, latent_frames, remainder


def load_video_frames(video_path: Path) -> np.ndarray:
    """Load all frames from a video file."""
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    vr = VideoReader(str(video_path))
    frames = vr.get_batch(range(len(vr))).asnumpy()  # (T, H, W, C)
    return frames


def merge_segments_frames(
    seg01_frames: np.ndarray,
    seg02_frames: np.ndarray,
    seg03_frames: np.ndarray,
    target_frames: int = 85,
    strategy: str = "keep_seg01_seg02_trim_seg03",
) -> np.ndarray:
    """
    Merge 3 segments into target number of frames.

    Args:
        seg01_frames: First segment frames (T1, H, W, C)
        seg02_frames: Second segment frames (T2, H, W, C)
        seg03_frames: Third segment frames (T3, H, W, C)
        target_frames: Target number of frames (must satisfy 4k+1)
        strategy: Merging strategy
            - "keep_seg01_seg02_trim_seg03": Keep seg01+seg02, trim end of seg03
            - "trim_seg01_keep_seg02_seg03": Trim start of seg01, keep seg02+seg03
            - "symmetric_trim": Trim equally from seg01 and seg03

    Returns:
        Merged frames (target_frames, H, W, C)
    """
    total_available = len(seg01_frames) + len(seg02_frames) + len(seg03_frames)
    trim_count = total_available - target_frames

    print(f"  Total available: {total_available} frames")
    print(f"  Target: {target_frames} frames")
    print(f"  Need to trim: {trim_count} frames")

    if strategy == "keep_seg01_seg02_trim_seg03":
        # Keep all of seg01 and seg02, trim end of seg03
        seg03_keep = len(seg03_frames) - trim_count
        print(f"  Strategy: keep seg01({len(seg01_frames)}) + seg02({len(seg02_frames)}) + seg03[:{seg03_keep}]")
        merged = np.concatenate([
            seg01_frames,
            seg02_frames,
            seg03_frames[:seg03_keep],
        ], axis=0)

    elif strategy == "trim_seg01_keep_seg02_seg03":
        # Trim start of seg01, keep seg02 and seg03
        seg01_start = trim_count
        print(f"  Strategy: seg01[{seg01_start}:] + seg02({len(seg02_frames)}) + seg03({len(seg03_frames)})")
        merged = np.concatenate([
            seg01_frames[seg01_start:],
            seg02_frames,
            seg03_frames,
        ], axis=0)

    elif strategy == "symmetric_trim":
        # Trim equally from seg01 (start) and seg03 (end)
        trim_each = trim_count // 2
        trim_extra = trim_count % 2
        seg01_start = trim_each
        seg03_keep = len(seg03_frames) - trim_each - trim_extra
        print(f"  Strategy: seg01[{seg01_start}:] + seg02({len(seg02_frames)}) + seg03[:{seg03_keep}]")
        merged = np.concatenate([
            seg01_frames[seg01_start:],
            seg02_frames,
            seg03_frames[:seg03_keep],
        ], axis=0)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    assert len(merged) == target_frames, f"Expected {target_frames}, got {len(merged)}"
    return merged


def process_scene(
    scene_id: str,
    blur_dir: Path,
    depth_dir: Path,
    hdmap_dir: Path,
    output_dir: Path,
    cameras: List[str],
    target_frames: int = 85,
    strategy: str = "keep_seg01_seg02_trim_seg03",
    fps: int = 10,
):
    """
    Process a single scene, merging 3 segments for all control types and cameras.
    """
    # Validate target frames
    is_valid, latent_frames, remainder = validate_frame_count(target_frames)
    if not is_valid:
        raise ValueError(
            f"Target frames {target_frames} does not satisfy 4k+1 format. "
            f"Remainder is {remainder}, not 0. "
            f"Suggested alternatives: {target_frames - remainder} or {target_frames + (4 - remainder)}"
        )

    print(f"\n{'='*60}")
    print(f"Processing scene: {scene_id}")
    print(f"Target frames: {target_frames} (latent: {latent_frames})")
    print(f"Strategy: {strategy}")
    print(f"{'='*60}")

    segment_ids = ["seg01", "seg02", "seg03"]

    # Control types and their directories
    control_configs = [
        ("blur", blur_dir, "control_input_blur"),
        ("depth", depth_dir, "control_input_depth"),
        ("hdmap", hdmap_dir, "control_input_hdmap_bbox"),
    ]

    # Also process videos (ground truth)
    video_configs = [
        ("video", blur_dir, "videos"),  # Videos should be same across all control dirs
    ]

    all_configs = control_configs + video_configs

    for camera in cameras:
        print(f"\nCamera: {camera}")
        camera_folder = f"ftheta_{camera}"

        for output_type, base_dir, subdir in all_configs:
            print(f"  Processing {output_type}...")

            # Load all 3 segments
            segment_frames = []
            all_found = True

            for seg_id in segment_ids:
                sample_id = f"{scene_id}_{seg_id}"
                video_path = base_dir / subdir / camera_folder / f"{sample_id}.mp4"

                if not video_path.exists():
                    print(f"    Warning: {video_path} not found, skipping {output_type}")
                    all_found = False
                    break

                frames = load_video_frames(video_path)
                segment_frames.append(frames)
                print(f"    {seg_id}: {len(frames)} frames")

            if not all_found:
                continue

            # Merge segments
            merged_frames = merge_segments_frames(
                segment_frames[0],
                segment_frames[1],
                segment_frames[2],
                target_frames=target_frames,
                strategy=strategy,
            )

            # Save merged video
            output_sample_id = f"{scene_id}_merged_{target_frames}f"
            output_path = output_dir / subdir / camera_folder
            output_path.mkdir(parents=True, exist_ok=True)

            output_file = output_path / f"{output_sample_id}.mp4"

            # Convert to tensor format for write_video: (T, H, W, C) uint8
            video_tensor = torch.from_numpy(merged_frames)
            write_video(str(output_file), video_tensor, fps=fps)

            print(f"    Saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Merge 3 segments into 85-frame condition")
    parser.add_argument("--blur_dir", type=str, required=True,
                        help="Path to blur dataset directory")
    parser.add_argument("--depth_dir", type=str, required=True,
                        help="Path to depth dataset directory")
    parser.add_argument("--hdmap_dir", type=str, required=True,
                        help="Path to hdmap dataset directory")
    parser.add_argument("--scene_id", type=str, required=True,
                        help="Scene ID to process (e.g., 031)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for merged videos")
    parser.add_argument("--target_frames", type=int, default=85,
                        help="Target number of frames (must satisfy 4k+1)")
    parser.add_argument("--strategy", type=str,
                        default="keep_seg01_seg02_trim_seg03",
                        choices=["keep_seg01_seg02_trim_seg03",
                                "trim_seg01_keep_seg02_seg03",
                                "symmetric_trim"],
                        help="Merging strategy")
    parser.add_argument("--cameras", type=str, nargs="+",
                        default=list(DEFAULT_CAMERAS),
                        help="Camera names to process")
    parser.add_argument("--fps", type=int, default=10,
                        help="Output video FPS")
    args = parser.parse_args()

    # Validate target frames
    is_valid, latent_frames, remainder = validate_frame_count(args.target_frames)
    if not is_valid:
        suggestions = [
            args.target_frames - remainder,
            args.target_frames + (4 - remainder)
        ]
        print(f"ERROR: Target frames {args.target_frames} does not satisfy 4k+1 format!")
        print(f"  {args.target_frames} = 4 * {(args.target_frames-1)//4} + {remainder + 1}")
        print(f"  Valid alternatives: {suggestions}")
        return

    print(f"Target: {args.target_frames} frames = {latent_frames} latent frames")

    process_scene(
        scene_id=args.scene_id,
        blur_dir=Path(args.blur_dir),
        depth_dir=Path(args.depth_dir),
        hdmap_dir=Path(args.hdmap_dir),
        output_dir=Path(args.output_dir),
        cameras=args.cameras,
        target_frames=args.target_frames,
        strategy=args.strategy,
        fps=args.fps,
    )

    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
