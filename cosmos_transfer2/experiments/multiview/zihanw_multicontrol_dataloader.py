# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Multi-control input dataloader for zihanw's multiview post-training.
Supports blur, depth, and hdmap control inputs simultaneously.
Supports loading from separate directories (no need to merge data).
"""

import io
import json
from pathlib import Path

import torch
import torch.distributed as dist
from einops import rearrange
from hydra.core.config_store import ConfigStore
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode, Resize

from cosmos_transfer2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_transfer2._src.predict2.datasets.local_datasets.dataset_video import get_generic_dataloader, get_sampler
from cosmos_transfer2._src.predict2_multiview.datasets.multiview import (
    DEFAULT_CAMERA_VIEW_MAPPING,
    DEFAULT_CAMERAS,
    DEFAULT_CAPTION_PREFIXES,
    collate_fn,
)

# 4-camera subset for training with 4 GPUs (360° coverage)
# This allows context_parallel_size=4 with n_views=4
CAMERAS_4VIEW: tuple[str, ...] = (
    "camera_front_wide_120fov",   # Front
    "camera_cross_right_120fov",  # Right
    "camera_rear_tele_30fov",     # Rear
    "camera_cross_left_120fov",   # Left
)

# 2-camera subset for training with 2 GPUs (front-rear coverage)
# This allows context_parallel_size=2 with n_views=2
# Required because state_t=6 (21 frames) and 6 % cp_size must equal 0
# Valid cp_size values for state_t=6: 1, 2, 3, 6
CAMERAS_2VIEW: tuple[str, ...] = (
    "camera_front_wide_120fov",   # Front
    "camera_rear_tele_30fov",     # Rear
)

# 3-camera subset for training with 3 GPUs (front-left-right coverage)
# This allows context_parallel_size=3 with n_views=3
# Provides good front hemisphere coverage
CAMERAS_3VIEW: tuple[str, ...] = (
    "camera_front_wide_120fov",   # Front Wide
    "camera_cross_left_120fov",   # Front Left
    "camera_cross_right_120fov",  # Front Right
)

# 6-camera subset for training with 6 GPUs
# This allows context_parallel_size=6 with n_views=6
CAMERAS_6VIEW: tuple[str, ...] = (
    "camera_front_wide_120fov",   # Front
    "camera_cross_right_120fov",  # Right
    "camera_rear_right_120fov",   # Rear Right
    "camera_rear_tele_30fov",     # Rear
    "camera_rear_left_120fov",    # Rear Left
    "camera_cross_left_120fov",   # Left
)

# View mapping for 4-camera setup (indices 0-3)
CAMERA_VIEW_MAPPING_4VIEW: dict[str, int] = {
    camera: idx for idx, camera in enumerate(CAMERAS_4VIEW)
}


class MultiControlMultiviewDataset(Dataset):
    """
    Dataset for multiview training with multiple control inputs (blur, depth, hdmap).

    Supports loading from separate directories - no need to merge your data!

    Example usage with separate directories:
        dataset = MultiControlMultiviewDataset(
            base_video_dir="/path/to/BlurProjection",  # Contains videos/ and captions/
            control_dirs={
                "blur": "/path/to/BlurProjection/control_input_blur",
                "depth": "/path/to/DepthSparse/control_input_depth",
                "hdmap_bbox": "/path/to/HDMapBbox/control_input_hdmap_bbox",
            },
            ...
        )
    """

    def __init__(
        self,
        base_video_dir: str,  # Directory containing videos/ and captions/
        control_dirs: dict[str, str],  # {"blur": "/path/to/control_input_blur", ...}
        folder_to_camera_key: dict[str, str],
        resolution_hw: tuple[int, int] = (720, 1280),
        num_video_frames: int = 21,
        single_caption_camera_name: str = "camera_front_wide_120fov",
        selected_cameras: tuple[str, ...] | None = None,  # Subset of cameras to use
        exclude_clips: tuple[str, ...] = (),  # Clip prefixes to exclude (e.g., ("075", "077"))
        include_only_clips: tuple[str, ...] = (),  # If set, ONLY include clips matching these prefixes
    ) -> None:
        self.base_video_dir = base_video_dir
        self.control_dirs = control_dirs
        self.control_input_names = list(control_dirs.keys())
        self.folder_to_camera_key = folder_to_camera_key
        self.camera_key_to_folder = {v: k for k, v in folder_to_camera_key.items()}
        self.resolution_hw = resolution_hw
        self.num_video_frames = num_video_frames
        self.single_caption_camera_name = single_caption_camera_name
        self.exclude_clips = exclude_clips
        self.include_only_clips = include_only_clips

        # Use selected cameras or default to all 7
        self.selected_cameras = selected_cameras if selected_cameras else DEFAULT_CAMERAS
        self.n_views = len(self.selected_cameras)

        # Build view mapping for selected cameras (0-indexed)
        self.camera_view_mapping = {
            camera: idx for idx, camera in enumerate(self.selected_cameras)
        }

        base_path = Path(base_video_dir)
        if not base_path.exists():
            raise FileNotFoundError(f"Base directory {base_video_dir} does not exist!")

        # Verify directories exist
        video_path = base_path / "videos"
        if not video_path.exists():
            raise FileNotFoundError(f"Video directory {video_path} does not exist!")

        caption_path = base_path / "captions"
        if not caption_path.exists():
            raise FileNotFoundError(f"Caption directory {caption_path} does not exist!")

        # Verify control input directories
        self.control_paths = {}
        for control_name, control_dir in control_dirs.items():
            ctrl_path = Path(control_dir)
            if not ctrl_path.exists():
                raise FileNotFoundError(f"Control input directory {ctrl_path} does not exist!")
            self.control_paths[control_name] = ctrl_path

        # Build file lists
        captions_files = list(caption_path.glob("**/*.json"))
        unique_names = sorted(set(f.stem for f in captions_files))

        # Filter clips based on include/exclude rules
        if self.include_only_clips:
            # Only include clips matching these prefixes (for validation)
            original_count = len(unique_names)
            unique_names = [
                name for name in unique_names
                if any(name.startswith(prefix) for prefix in self.include_only_clips)
            ]
            included_count = len(unique_names)
            print(f"Included {included_count} clips matching prefixes: {self.include_only_clips} (from {original_count} total)")
        elif self.exclude_clips:
            # Exclude clips matching these prefixes (for training)
            original_count = len(unique_names)
            unique_names = [
                name for name in unique_names
                if not any(name.startswith(prefix) for prefix in self.exclude_clips)
            ]
            excluded_count = original_count - len(unique_names)
            print(f"Excluded {excluded_count} clips matching prefixes: {self.exclude_clips}")

        # Filter folder_to_camera_key to only include selected cameras
        self.filtered_folder_to_camera = {
            folder: camera for folder, camera in self.folder_to_camera_key.items()
            if camera in self.selected_cameras
        }

        skipped_short_videos = 0
        self.samples = []
        for name in unique_names:
            sample = {
                "name": name,
                "videos": {},
                "captions": {},
                "controls": {ctrl_name: {} for ctrl_name in self.control_input_names},
            }

            valid_sample = True
            for folder, camera_key in self.filtered_folder_to_camera.items():
                # Caption
                caption_file = caption_path / folder / f"{name}.json"
                if caption_file.exists():
                    with open(caption_file, "r") as f:
                        caption_json = json.load(f)
                    sample["captions"][camera_key] = caption_json["caption"]

                # Video
                video_file = video_path / folder / f"{name}.mp4"
                if not video_file.exists():
                    valid_sample = False
                    break
                # Check if video has enough frames
                try:
                    from decord import VideoReader
                    with open(video_file, "rb") as f:
                        video_reader = VideoReader(io.BytesIO(f.read()))
                    if len(video_reader) < self.num_video_frames:
                        valid_sample = False
                        skipped_short_videos += 1
                        break
                except Exception:
                    valid_sample = False
                    break
                sample["videos"][camera_key] = video_file

                # Control inputs from separate directories
                for control_name in self.control_input_names:
                    control_file = self.control_paths[control_name] / folder / f"{name}.mp4"
                    if not control_file.exists():
                        valid_sample = False
                        break
                    # Check if control video has enough frames
                    try:
                        from decord import VideoReader
                        with open(control_file, "rb") as f:
                            ctrl_reader = VideoReader(io.BytesIO(f.read()))
                        if len(ctrl_reader) < self.num_video_frames:
                            valid_sample = False
                            skipped_short_videos += 1
                            break
                    except Exception:
                        valid_sample = False
                        break
                    sample["controls"][control_name][camera_key] = control_file

                if not valid_sample:
                    break

            if valid_sample and len(sample["videos"]) == len(self.filtered_folder_to_camera):
                self.samples.append(sample)

        print(f"=" * 60)
        print(f"MultiControlMultiviewDataset initialized:")
        print(f"  Base directory: {base_video_dir}")
        print(f"  Loaded {len(self.samples)} samples")
        print(f"  Skipped {skipped_short_videos} samples with insufficient frames (need {self.num_video_frames})")
        print(f"  Number of views: {self.n_views}")
        print(f"  Selected cameras: {list(self.selected_cameras)}")
        print(f"  Control inputs:")
        for ctrl_name, ctrl_path in self.control_paths.items():
            print(f"    - {ctrl_name}: {ctrl_path}")
        print(f"=" * 60)

    def __len__(self) -> int:
        return len(self.samples)

    def _read_video_frames(self, filepath: Path, frame_indices: list[int]) -> torch.Tensor:
        """Read video frames from file."""
        from decord import VideoReader

        with open(filepath, "rb") as f:
            video_bytes = f.read()

        video_reader = VideoReader(io.BytesIO(video_bytes))
        frames = video_reader.get_batch(frame_indices).asnumpy()
        frames = rearrange(torch.from_numpy(frames), "t h w c -> t c h w")
        frames = Resize(self.resolution_hw, interpolation=InterpolationMode.BILINEAR, antialias=True)(frames)
        return frames

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]

        # Determine frame indices (start from 0, sample num_video_frames)
        frame_indices = list(range(self.num_video_frames))

        # Load videos for all selected views
        multiview_frames = []
        view_indices = []

        # Use selected cameras (e.g., 4 cameras for 4 GPUs)
        camera_keys = list(self.selected_cameras)
        for camera_key in camera_keys:
            video_file = sample["videos"].get(camera_key)
            if video_file is None:
                continue

            frames = self._read_video_frames(video_file, frame_indices)
            multiview_frames.append(frames)
            # Use local view mapping (0-indexed for selected cameras)
            view_indices.extend([self.camera_view_mapping[camera_key]] * self.num_video_frames)

        # Stack all views
        video_tensor = rearrange(torch.cat(multiview_frames, dim=0), "t c h w -> c t h w")

        # Load control inputs for all selected views
        control_tensors = {}
        for control_name in self.control_input_names:
            multiview_control = []
            for camera_key in camera_keys:
                control_file = sample["controls"][control_name].get(camera_key)
                if control_file is None:
                    continue

                frames = self._read_video_frames(control_file, frame_indices)
                multiview_control.append(frames)

            control_tensors[f"control_input_{control_name}"] = rearrange(
                torch.cat(multiview_control, dim=0), "t c h w -> c t h w"
            )

        # Get caption
        caption = sample["captions"].get(self.single_caption_camera_name, "")

        # Build captions with view prefixes
        captions = []
        for camera_key in camera_keys:
            if camera_key in sample["videos"]:
                prefix = DEFAULT_CAPTION_PREFIXES.get(camera_key, "")
                captions.append(f"{prefix} {caption}")

        # Find front camera position in selected cameras
        front_cam_position = 0  # Default to first camera
        if self.single_caption_camera_name in camera_keys:
            front_cam_position = camera_keys.index(self.single_caption_camera_name)

        # Build output dict
        output = {
            "__key__": str(index),
            "__url__": str(sample["name"]),
            "video": video_tensor,
            "ai_caption": captions,
            "view_indices": torch.tensor(view_indices, dtype=torch.int64),
            "fps": torch.tensor(10.0, dtype=torch.float64),  # Assuming 10 fps
            "chunk_index": torch.tensor(0, dtype=torch.int64),
            "frame_indices": torch.tensor(frame_indices, dtype=torch.int64),
            "num_video_frames_per_view": torch.tensor(self.num_video_frames, dtype=torch.int64),
            "view_indices_selection": torch.tensor(
                [self.camera_view_mapping[k] for k in camera_keys if k in sample["videos"]],
                dtype=torch.int64
            ),
            "camera_keys_selection": [k for k in camera_keys if k in sample["videos"]],
            "sample_n_views": torch.tensor(len([k for k in camera_keys if k in sample["videos"]]), dtype=torch.int64),
            "padding_mask": torch.zeros((1, *self.resolution_hw), dtype=torch.float32),
            "ref_cam_view_idx_sample_position": torch.tensor(-1, dtype=torch.int64),
            "front_cam_view_idx_sample_position": torch.tensor(front_cam_position, dtype=torch.int64),
            "original_hw": torch.tensor(
                [[720, 1280] for _ in camera_keys if _ in sample["videos"]], dtype=torch.int64
            ),
        }

        # Add control inputs
        output.update(control_tensors)

        return output


def register_zihanw_multicontrol_dataloader() -> None:
    """Register the multi-control dataloader for zihanw's training."""

    cs = ConfigStore.instance()

    # Dataset configuration - loading from separate directories
    # Using 4 cameras for 4 GPU training (n_views <= context_parallel_size)
    dataset = L(MultiControlMultiviewDataset)(
        base_video_dir="/mnt/zihanw/proj_utils_pro/transfer_video_maker/output/BlurProjection",
        control_dirs={
            "blur": "/mnt/zihanw/proj_utils_pro/transfer_video_maker/output/BlurProjection/control_input_blur",
            "depth": "/mnt/zihanw/proj_utils_pro/transfer_video_maker/output/DepthSparse/control_input_depth",
            "hdmap_bbox": "/mnt/zihanw/proj_utils_pro/transfer_video_maker/output/HDMapBbox/control_input_hdmap_bbox",
        },
        folder_to_camera_key={f"ftheta_{camera_name}": camera_name for camera_name in DEFAULT_CAMERAS},
        resolution_hw=(720, 1280),
        num_video_frames=29,  # 29 frames -> state_t=8
        single_caption_camera_name="camera_front_wide_120fov",
        # Use 2 cameras for 2 GPU training (front-rear coverage)
        # Constraints: state_t=8 (factors: 1,2,4,8) AND num_heads=16 (factors: 1,2,4,8,16)
        # Valid cp_size values: intersection = {1, 2, 4, 8}
        selected_cameras=CAMERAS_2VIEW,
        # Exclude clips for inference/evaluation
        exclude_clips=("075", "077"),
    )

    cs.store(
        group="data_train",
        package="dataloader_train",
        name="zihanw_multicontrol_multiview",
        node=L(get_generic_dataloader)(
            dataset=dataset,
            sampler=L(get_sampler)(dataset=dataset) if dist.is_initialized() else None,
            collate_fn=collate_fn,
            batch_size=1,
            drop_last=True,
            num_workers=4,
            pin_memory=True,
        ),
    )


def register_zihanw_multicontrol_val_dataloader() -> None:
    """Register the validation dataloader using eval clips (075, 077)."""

    cs = ConfigStore.instance()

    # Validation dataset - ONLY use clips 075 and 077
    val_dataset = L(MultiControlMultiviewDataset)(
        base_video_dir="/mnt/zihanw/proj_utils_pro/transfer_video_maker/output/BlurProjection",
        control_dirs={
            "blur": "/mnt/zihanw/proj_utils_pro/transfer_video_maker/output/BlurProjection/control_input_blur",
            "depth": "/mnt/zihanw/proj_utils_pro/transfer_video_maker/output/DepthSparse/control_input_depth",
            "hdmap_bbox": "/mnt/zihanw/proj_utils_pro/transfer_video_maker/output/HDMapBbox/control_input_hdmap_bbox",
        },
        folder_to_camera_key={f"ftheta_{camera_name}": camera_name for camera_name in DEFAULT_CAMERAS},
        resolution_hw=(720, 1280),
        num_video_frames=29,
        single_caption_camera_name="camera_front_wide_120fov",
        selected_cameras=CAMERAS_2VIEW,
        # Only include eval clips (inverse of training exclusion)
        include_only_clips=("075", "077"),
    )

    cs.store(
        group="data_val",
        package="dataloader_val",
        name="zihanw_multicontrol_multiview_val",
        node=L(get_generic_dataloader)(
            dataset=val_dataset,
            sampler=L(get_sampler)(dataset=val_dataset) if dist.is_initialized() else None,
            collate_fn=collate_fn,
            batch_size=1,
            drop_last=False,  # Don't drop last for validation
            num_workers=4,
            pin_memory=True,
        ),
    )


# Auto-register when module is imported
register_zihanw_multicontrol_dataloader()
register_zihanw_multicontrol_val_dataloader()
