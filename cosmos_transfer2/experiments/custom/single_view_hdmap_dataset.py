# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Single-view HDMap Dataset for Cosmos Transfer2.5 fine-tuning on car-side data.
# Single camera (camera_front_wide_120fov), single control (hdmap_bbox).

import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from einops import rearrange
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode, Resize


SINGLE_CAMERA: str = "camera_front_wide_120fov"
SINGLE_CAMERA_FOLDER: str = f"ftheta_{SINGLE_CAMERA}"
SINGLE_CAMERA_CAPTION_PREFIX: str = (
    "The video is captured from a camera mounted on a car. The camera is facing forward."
)


class SingleViewHDMapDataset(Dataset):
    """
    Single-view dataset with a single hdmap_bbox control input.

    Expected directory structure (single root):
        dataset_dir/
        ├── captions/ftheta_camera_front_wide_120fov/{sample_id}.json
        ├── videos/ftheta_camera_front_wide_120fov/{sample_id}.mp4
        └── control_input_hdmap_bbox/ftheta_camera_front_wide_120fov/{sample_id}.mp4

    The caption JSON is expected to contain at least a ``caption`` field.
    Sample IDs follow ``<scene_id>_seg<NN>``. Train/test scene split is done by
    ``exclude_scene_ids``.
    """

    def __init__(
        self,
        dataset_dir: str,
        resolution_hw: Tuple[int, int] = (720, 1280),
        num_video_frames: int = 29,
        fps_downsample_factor: int = 1,
        add_view_prefix_to_caption: bool = True,
        exclude_scene_ids: Optional[List[str]] = None,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.resolution_hw = resolution_hw
        self.num_video_frames = num_video_frames
        self.fps_downsample_factor = fps_downsample_factor
        self.add_view_prefix_to_caption = add_view_prefix_to_caption
        self.exclude_scene_ids = set(exclude_scene_ids or [])

        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory {self.dataset_dir} does not exist!")

        self.samples = self._build_sample_list()
        print(f"[SingleViewHDMapDataset] Found {len(self.samples)} samples")
        print(f"[SingleViewHDMapDataset] Excluded scene IDs: {sorted(self.exclude_scene_ids)}")

    def _check_video_frames(self, video_path: Path, required_frames: int) -> Tuple[bool, int]:
        if not video_path.exists():
            return False, 0
        try:
            from decord import VideoReader

            vr = VideoReader(str(video_path))
            num_frames = len(vr)
            return num_frames >= required_frames, num_frames
        except Exception:
            return False, 0

    def _build_sample_list(self) -> List[str]:
        caption_folder = self.dataset_dir / "captions" / SINGLE_CAMERA_FOLDER
        if not caption_folder.exists():
            raise FileNotFoundError(f"Caption folder {caption_folder} does not exist!")

        samples: List[str] = []
        skipped_short: List[Tuple[str, int]] = []
        skipped_missing: List[str] = []
        required_frames = self.num_video_frames * self.fps_downsample_factor

        print(f"[SingleViewHDMapDataset] Scanning videos (need >= {required_frames} frames)...")

        for caption_file in caption_folder.glob("*.json"):
            sample_id = caption_file.stem
            scene_id = sample_id.split("_")[0]

            if scene_id in self.exclude_scene_ids:
                continue

            video_path = self.dataset_dir / "videos" / SINGLE_CAMERA_FOLDER / f"{sample_id}.mp4"
            hdmap_path = (
                self.dataset_dir
                / "control_input_hdmap_bbox"
                / SINGLE_CAMERA_FOLDER
                / f"{sample_id}.mp4"
            )

            v_valid, v_frames = self._check_video_frames(video_path, required_frames)
            h_valid, h_frames = self._check_video_frames(hdmap_path, required_frames)

            if v_valid and h_valid:
                samples.append(sample_id)
            elif v_frames == 0 or h_frames == 0:
                skipped_missing.append(sample_id)
            else:
                skipped_short.append((sample_id, int(min(v_frames, h_frames))))

        samples.sort()

        if skipped_short:
            print(
                f"[SingleViewHDMapDataset] Skipped {len(skipped_short)} samples with < {required_frames} frames:"
            )
            for sid, nframes in skipped_short[:10]:
                print(f"  - {sid}: {nframes} frames")
            if len(skipped_short) > 10:
                print(f"  ... and {len(skipped_short) - 10} more")
        if skipped_missing:
            print(f"[SingleViewHDMapDataset] Skipped {len(skipped_missing)} samples with missing files")

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _extract_frames(
        self, video_bytes: bytes, frame_indices: List[int], resolution_hw: Tuple[int, int]
    ) -> Tuple[torch.Tensor, float, Tuple[int, int]]:
        from decord import VideoReader

        video_reader = VideoReader(io.BytesIO(video_bytes))
        fps = video_reader.get_avg_fps()
        frames = video_reader.get_batch(frame_indices).asnumpy()
        frames = rearrange(torch.from_numpy(frames), "t h w c -> t c h w")
        original_h, original_w = frames.shape[-2:]
        resized_frames = Resize(
            resolution_hw, interpolation=InterpolationMode.BILINEAR, antialias=True
        )(frames)
        return resized_frames, fps, (original_h, original_w)

    def _load_video(self, video_path: Path) -> bytes:
        with open(video_path, "rb") as f:
            return f.read()

    def _load_caption(self, caption_path: Path) -> str:
        with open(caption_path, "r") as f:
            caption_json = json.load(f)
        return caption_json["caption"]

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample_id = self.samples[index]

        frame_start = 0
        frame_end = frame_start + self.num_video_frames * self.fps_downsample_factor
        frame_indices = list(range(frame_start, frame_end, self.fps_downsample_factor))

        caption_path = (
            self.dataset_dir / "captions" / SINGLE_CAMERA_FOLDER / f"{sample_id}.json"
        )
        caption = self._load_caption(caption_path)
        if self.add_view_prefix_to_caption:
            caption = f"{SINGLE_CAMERA_CAPTION_PREFIX} {caption}"

        video_path = self.dataset_dir / "videos" / SINGLE_CAMERA_FOLDER / f"{sample_id}.mp4"
        frames, fps, original_hw = self._extract_frames(
            self._load_video(video_path), frame_indices, self.resolution_hw
        )

        hdmap_path = (
            self.dataset_dir
            / "control_input_hdmap_bbox"
            / SINGLE_CAMERA_FOLDER
            / f"{sample_id}.mp4"
        )
        hdmap_frames, _, _ = self._extract_frames(
            self._load_video(hdmap_path), frame_indices, self.resolution_hw
        )

        fps_out = fps / self.fps_downsample_factor

        sample: Dict[str, Any] = {
            "__key__": sample_id,
            "__url__": str(self.dataset_dir),
            # Video: CTHW
            "video": rearrange(frames, "t c h w -> c t h w"),
            # Caption per view (only one view)
            "ai_caption": [caption],
            # View bookkeeping (single view)
            "view_indices": torch.zeros(self.num_video_frames, dtype=torch.int64),
            "fps": torch.tensor(fps_out, dtype=torch.float64),
            "chunk_index": torch.tensor(0, dtype=torch.int64),
            "frame_indices": torch.tensor(frame_indices, dtype=torch.int64),
            "num_video_frames_per_view": torch.tensor(len(frame_indices), dtype=torch.int64),
            "view_indices_selection": torch.tensor([0], dtype=torch.int64),
            "camera_keys_selection": [SINGLE_CAMERA],
            "sample_n_views": torch.tensor(1, dtype=torch.int64),
            "padding_mask": torch.zeros((1, *self.resolution_hw), dtype=torch.float32),
            "ref_cam_view_idx_sample_position": torch.tensor(-1, dtype=torch.int64),
            "front_cam_view_idx_sample_position": torch.tensor(0, dtype=torch.int64),
            "original_hw": torch.tensor([list(original_hw)], dtype=torch.int64),
            # HDMap control input keyed as hdmap_bbox so that the pre-trained
            # Transfer2.5 ControlNet weights at channels 0-15 are reused.
            "control_input_hdmap_bbox": rearrange(hdmap_frames, "t c h w -> c t h w"),
        }
        return sample


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for DataLoader (mirrors custom_multi_control_dataset.collate_fn)."""
    merged: Dict[str, Any] = dict()
    is_tensor: Dict[str, bool] = dict()
    for row in batch:
        for key, value in row.items():
            if key not in merged:
                merged[key] = []
            if isinstance(value, torch.Tensor):
                is_tensor[key] = True
            merged[key].append(value)
    for key, value in merged.items():
        if is_tensor.get(key, False):
            merged[key] = torch.stack(value, dim=0)
    return merged
