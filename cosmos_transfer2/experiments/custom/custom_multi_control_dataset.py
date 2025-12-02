# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Custom Dataset for Multi-Control Post Training
# Supports 3 control types: blur, depth, hdmap_bbox

import io
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from einops import rearrange
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode, Resize


# Default camera configuration (same as original)
DEFAULT_CAMERAS: Tuple[str, ...] = (
    "camera_front_wide_120fov",
    "camera_cross_right_120fov",
    "camera_rear_right_70fov",
    "camera_rear_tele_30fov",
    "camera_rear_left_70fov",
    "camera_cross_left_120fov",
    "camera_front_tele_30fov",
)

DEFAULT_CAMERA_VIEW_MAPPING: Dict[str, int] = dict(zip(DEFAULT_CAMERAS, range(len(DEFAULT_CAMERAS))))

DEFAULT_CAPTION_PREFIXES: Dict[str, str] = {
    "camera_front_wide_120fov": "The video is captured from a camera mounted on a car. The camera is facing forward.",
    "camera_cross_right_120fov": "The video is captured from a camera mounted on a car. The camera is facing to the right.",
    "camera_rear_right_70fov": "The video is captured from a camera mounted on a car. The camera is facing the rear right side.",
    "camera_rear_tele_30fov": "The video is captured from a camera mounted on a car. The camera is facing backwards.",
    "camera_rear_left_70fov": "The video is captured from a camera mounted on a car. The camera is facing the rear left side.",
    "camera_cross_left_120fov": "The video is captured from a camera mounted on a car. The camera is facing to the left.",
    "camera_front_tele_30fov": "The video is captured from a telephoto camera mounted on a car. The camera is facing forward.",
}


class MultiControlMultiviewDataset(Dataset):
    """
    Dataset for multiview video generation with multiple control inputs.

    Supports loading data from three separate dataset directories, each containing
    a different control type (blur, depth, hdmap).

    Directory structure expected for each control type:
        dataset_dir/
        ├── captions/
        │   └── ftheta_camera_xxx/
        │       └── {sample_id}.json
        ├── control_input_{type}/
        │   └── ftheta_camera_xxx/
        │       └── {sample_id}.mp4
        └── videos/
            └── ftheta_camera_xxx/
                └── {sample_id}.mp4
    """

    def __init__(
        self,
        blur_dataset_dir: str,
        depth_dataset_dir: str,
        hdmap_dataset_dir: str,
        resolution_hw: Tuple[int, int] = (720, 1280),
        num_video_frames: int = 29,
        fps_downsample_factor: int = 1,
        camera_keys: Tuple[str, ...] = DEFAULT_CAMERAS,
        single_caption_camera_name: str = "camera_front_wide_120fov",
        add_view_prefix_to_caption: bool = True,
        exclude_scene_ids: Optional[List[str]] = None,
    ) -> None:
        """
        Args:
            blur_dataset_dir: Path to BlurProjection dataset directory
            depth_dataset_dir: Path to DepthSparse dataset directory
            hdmap_dataset_dir: Path to HDMapBbox dataset directory
            resolution_hw: Output resolution (height, width)
            num_video_frames: Number of frames to extract
            fps_downsample_factor: FPS downsample factor
            camera_keys: Tuple of camera names to use
            single_caption_camera_name: Camera to use for caption
            add_view_prefix_to_caption: Whether to add camera prefix to caption
            exclude_scene_ids: List of scene IDs to exclude (for test set)
        """
        self.blur_dataset_dir = Path(blur_dataset_dir)
        self.depth_dataset_dir = Path(depth_dataset_dir)
        self.hdmap_dataset_dir = Path(hdmap_dataset_dir)
        self.resolution_hw = resolution_hw
        self.num_video_frames = num_video_frames
        self.fps_downsample_factor = fps_downsample_factor
        self.camera_keys = camera_keys
        self.single_caption_camera_name = single_caption_camera_name
        self.add_view_prefix_to_caption = add_view_prefix_to_caption
        self.exclude_scene_ids = set(exclude_scene_ids or [])

        self.camera_view_mapping = DEFAULT_CAMERA_VIEW_MAPPING
        self.camera_prefix_mapping = DEFAULT_CAPTION_PREFIXES

        # Validate directories
        for dataset_dir, name in [
            (self.blur_dataset_dir, "blur"),
            (self.depth_dataset_dir, "depth"),
            (self.hdmap_dataset_dir, "hdmap"),
        ]:
            if not dataset_dir.exists():
                raise FileNotFoundError(f"{name} dataset directory {dataset_dir} does not exist!")

        # Build sample list from blur dataset (assumes all three have same samples)
        self.samples = self._build_sample_list()
        print(f"[MultiControlMultiviewDataset] Found {len(self.samples)} samples")
        print(f"[MultiControlMultiviewDataset] Excluded scene IDs: {self.exclude_scene_ids}")

    def _build_sample_list(self) -> List[str]:
        """Build list of sample IDs from the dataset."""
        caption_path = self.blur_dataset_dir / "captions"
        first_camera_folder = f"ftheta_{self.camera_keys[0]}"
        caption_folder = caption_path / first_camera_folder

        if not caption_folder.exists():
            raise FileNotFoundError(f"Caption folder {caption_folder} does not exist!")

        samples = []
        for caption_file in caption_folder.glob("*.json"):
            sample_id = caption_file.stem  # e.g., "017_seg01"
            scene_id = sample_id.split("_")[0]  # e.g., "017"

            # Skip excluded scenes
            if scene_id in self.exclude_scene_ids:
                continue

            samples.append(sample_id)

        samples.sort()
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _extract_frames(
        self,
        video_bytes: bytes,
        frame_indices: List[int],
        resolution_hw: Tuple[int, int]
    ) -> Tuple[torch.Tensor, float, Tuple[int, int]]:
        """Extract frames from video bytes."""
        from decord import VideoReader

        video_reader = VideoReader(io.BytesIO(video_bytes))
        fps = video_reader.get_avg_fps()
        frames = video_reader.get_batch(frame_indices).asnumpy()
        frames = rearrange(torch.from_numpy(frames), "t h w c -> t c h w")
        original_h, original_w = frames.shape[-2:]

        resized_frames = Resize(resolution_hw, interpolation=InterpolationMode.BILINEAR, antialias=True)(frames)
        return resized_frames, fps, (original_h, original_w)

    def _load_video(self, video_path: Path) -> bytes:
        """Load video file as bytes."""
        with open(video_path, "rb") as f:
            return f.read()

    def _load_caption(self, caption_path: Path) -> str:
        """Load caption from JSON file."""
        with open(caption_path, "r") as f:
            caption_json = json.load(f)
        return caption_json["caption"]

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample_id = self.samples[index]

        # Determine frame indices
        frame_start = 0
        frame_end = frame_start + self.num_video_frames * self.fps_downsample_factor
        frame_indices = list(range(frame_start, frame_end, self.fps_downsample_factor))

        # Initialize containers
        captions = []
        multiview_frames = []
        multiview_control_blur = []
        multiview_control_depth = []
        multiview_control_hdmap = []
        view_indices = []
        view_indices_selection = []
        camera_keys_selection = []
        original_sizes = []
        video_fps = None

        for camera_name in self.camera_keys:
            folder_name = f"ftheta_{camera_name}"

            # Load caption (from blur dataset, they should be the same)
            if camera_name == self.single_caption_camera_name:
                caption_path = self.blur_dataset_dir / "captions" / folder_name / f"{sample_id}.json"
                caption = self._load_caption(caption_path)
            else:
                caption = ""

            if self.add_view_prefix_to_caption:
                caption = f"{self.camera_prefix_mapping[camera_name]} {caption}"
            captions.append(caption)

            # Load video (from blur dataset, they should be the same for all control types)
            video_path = self.blur_dataset_dir / "videos" / folder_name / f"{sample_id}.mp4"
            video_bytes = self._load_video(video_path)
            frames, fps, original_hw = self._extract_frames(video_bytes, frame_indices, self.resolution_hw)
            multiview_frames.append(frames)

            if video_fps is None:
                video_fps = fps
            original_sizes.append(list(original_hw))

            # Load control inputs from each dataset
            # Blur control
            blur_path = self.blur_dataset_dir / "control_input_blur" / folder_name / f"{sample_id}.mp4"
            blur_bytes = self._load_video(blur_path)
            blur_frames, _, _ = self._extract_frames(blur_bytes, frame_indices, self.resolution_hw)
            multiview_control_blur.append(blur_frames)

            # Depth control
            depth_path = self.depth_dataset_dir / "control_input_depth" / folder_name / f"{sample_id}.mp4"
            depth_bytes = self._load_video(depth_path)
            depth_frames, _, _ = self._extract_frames(depth_bytes, frame_indices, self.resolution_hw)
            multiview_control_depth.append(depth_frames)

            # HDMap control
            hdmap_path = self.hdmap_dataset_dir / "control_input_hdmap_bbox" / folder_name / f"{sample_id}.mp4"
            hdmap_bytes = self._load_video(hdmap_path)
            hdmap_frames, _, _ = self._extract_frames(hdmap_bytes, frame_indices, self.resolution_hw)
            multiview_control_hdmap.append(hdmap_frames)

            # View indices
            view_indices.extend([self.camera_view_mapping[camera_name]] * self.num_video_frames)
            view_indices_selection.append(self.camera_view_mapping[camera_name])
            camera_keys_selection.append(camera_name)

        # Only keep caption from single_caption_camera
        if not self.add_view_prefix_to_caption:
            caption_idx = self.camera_keys.index(self.single_caption_camera_name)
            captions = [captions[caption_idx]]

        fps = video_fps / self.fps_downsample_factor

        sample = {
            "__key__": sample_id,
            "__url__": str(self.blur_dataset_dir),
            # Video: concatenate all views along time dimension, then rearrange to CTHW
            "video": rearrange(torch.cat(multiview_frames, dim=0), "t c h w -> c t h w"),
            "ai_caption": captions,
            "view_indices": torch.tensor(view_indices, dtype=torch.int64),
            "fps": torch.tensor(fps, dtype=torch.float64),
            "chunk_index": torch.tensor(0, dtype=torch.int64),
            "frame_indices": torch.tensor(frame_indices, dtype=torch.int64),
            "num_video_frames_per_view": torch.tensor(len(frame_indices), dtype=torch.int64),
            "view_indices_selection": torch.tensor(view_indices_selection, dtype=torch.int64),
            "camera_keys_selection": camera_keys_selection,
            "sample_n_views": torch.tensor(len(camera_keys_selection), dtype=torch.int64),
            "padding_mask": torch.zeros((1, *self.resolution_hw), dtype=torch.float32),
            "ref_cam_view_idx_sample_position": torch.tensor(-1, dtype=torch.int64),
            "front_cam_view_idx_sample_position": torch.tensor(
                self.camera_keys.index(self.single_caption_camera_name), dtype=torch.int64
            ),
            "original_hw": torch.tensor(original_sizes, dtype=torch.int64),
            # Three control inputs
            # NOTE: We use "control_input_vis" for blur data, "control_input_bbox" for hdmap data
            # The model's hint_keys should be set to "vis_depth_bbox"
            "control_input_vis": rearrange(torch.cat(multiview_control_blur, dim=0), "t c h w -> c t h w"),
            "control_input_depth": rearrange(torch.cat(multiview_control_depth, dim=0), "t c h w -> c t h w"),
            "control_input_bbox": rearrange(torch.cat(multiview_control_hdmap, dim=0), "t c h w -> c t h w"),
        }

        return sample


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for DataLoader."""
    merged = dict()
    is_tensor = dict()
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
