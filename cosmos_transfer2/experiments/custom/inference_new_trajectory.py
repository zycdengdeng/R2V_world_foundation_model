#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Inference script for new trajectory data (Roadside dataset).

Data structure (flat - no scene subdirectories):
    /mnt/zihanw/proj_utils_pro_Roadside_Generation/transfer_video_maker/output/
    ├── BlurProjection/
    │   ├── captions/{camera}/{sample}.json
    │   ├── control_input_blur/{camera}/{sample}.mp4
    │   └── videos/{camera}/{sample}.mp4
    ├── DepthSparse/
    │   └── control_input_depth/{camera}/{sample}.mp4
    └── HDMapBbox/
        └── control_input_hdmap_bbox/{camera}/{sample}.mp4

Sample naming format: {scene_id}_id{track_id}_seg{seg_id}
    Example: 001_id13_seg01, 002_id4_seg01, 010_id107_seg02

Cameras (8 views in data, use 7 for model):
    ftheta_camera_front_wide_120fov, ftheta_camera_front_tele_30fov,
    ftheta_camera_cross_left_120fov, ftheta_camera_cross_right_120fov,
    ftheta_camera_rear_left_70fov, ftheta_camera_rear_right_70fov,
    ftheta_camera_rear_tele_30fov

Usage:
    # Inference on specific scene(s) by scene ID prefix
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 WORLD_SIZE=8 \
    IMAGINAIRE_OUTPUT_ROOT="/mnt/zihanw/Output_R2V_world_foundation_model_v1" \
    HF_HOME="/mnt/zihanw/.cache/huggingface" \
    HF_HUB_CACHE="/mnt/zihanw/.cache/huggingface/hub" \
    HF_ENDPOINT="https://hf-mirror.com" \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    TRANSFORMERS_CACHE="/mnt/zihanw/.cache/huggingface/transformers" \
    torchrun --nproc_per_node=8 --master_port=12347 \
        -m cosmos_transfer2.experiments.custom.inference_new_trajectory \
        --ckpt_path /mnt/zihanw/Output_R2V_world_foundation_model_v1/cosmos_transfer_custom/multi_control/2b_custom_multi_control_20260120_122826/checkpoints/iter_000005000 \
        --scene_ids 001 003 004

    # Inference on specific sample(s) by name
    ... --sample_names 001_id13_seg01 001_id14_seg01

    # Inference on all samples
    ... --all_samples
"""

import argparse
import gc
import importlib
import io
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from einops import rearrange
from loguru import logger
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode, Resize

os.environ["NVTE_FUSED_ATTN"] = "0"

from cosmos_transfer2._src.predict2.models.video2world_model import NUM_CONDITIONAL_FRAMES_KEY

CONTROL_WEIGHT_KEY = "control_weight"


# ============================================================================
# New Trajectory Dataset Configuration
# ============================================================================

# Base path for new trajectory data
NEW_TRAJECTORY_BASE = "/mnt/zihanw/proj_utils_pro_Roadside_Generation/transfer_video_maker/output"

# 8 cameras in new data (ftheta_ prefix)
NEW_TRAJECTORY_CAMERAS_ALL = (
    "ftheta_camera_front_wide_120fov",
    "ftheta_camera_front_tele_30fov",
    "ftheta_camera_cross_left_120fov",
    "ftheta_camera_cross_right_120fov",
    "ftheta_camera_rear_left_70fov",
    "ftheta_camera_rear_right_70fov",
    "ftheta_camera_rear_tele_30fov",
)

# Map new camera names to model camera names (remove ftheta_ prefix)
# Model expects: camera_front_wide_120fov, etc.
CAMERA_NAME_MAPPING = {
    "ftheta_camera_front_wide_120fov": "camera_front_wide_120fov",
    "ftheta_camera_front_tele_30fov": "camera_front_tele_30fov",
    "ftheta_camera_cross_left_120fov": "camera_cross_left_120fov",
    "ftheta_camera_cross_right_120fov": "camera_cross_right_120fov",
    "ftheta_camera_rear_left_70fov": "camera_rear_left_70fov",
    "ftheta_camera_rear_right_70fov": "camera_rear_right_70fov",
    "ftheta_camera_rear_tele_30fov": "camera_rear_tele_30fov",
}

# Model's expected camera order (same as training)
MODEL_CAMERAS = (
    "camera_front_wide_120fov",
    "camera_cross_right_120fov",
    "camera_rear_right_70fov",
    "camera_rear_tele_30fov",
    "camera_rear_left_70fov",
    "camera_cross_left_120fov",
    "camera_front_tele_30fov",
)

MODEL_CAMERA_VIEW_MAPPING = dict(zip(MODEL_CAMERAS, range(len(MODEL_CAMERAS))))

MODEL_CAPTION_PREFIXES = {
    "camera_front_wide_120fov": "The video is captured from a camera mounted on a car. The camera is facing forward.",
    "camera_cross_right_120fov": "The video is captured from a camera mounted on a car. The camera is facing to the right.",
    "camera_rear_right_70fov": "The video is captured from a camera mounted on a car. The camera is facing the rear right side.",
    "camera_rear_tele_30fov": "The video is captured from a camera mounted on a car. The camera is facing backwards.",
    "camera_rear_left_70fov": "The video is captured from a camera mounted on a car. The camera is facing the rear left side.",
    "camera_cross_left_120fov": "The video is captured from a camera mounted on a car. The camera is facing to the left.",
    "camera_front_tele_30fov": "The video is captured from a telephoto camera mounted on a car. The camera is facing forward.",
}


# ============================================================================
# Dataset for New Trajectory Data
# ============================================================================

class NewTrajectoryDataset(Dataset):
    """
    Dataset for new trajectory data with flat directory structure (no scene subdirectories).

    Directory structure:
        base_path/
        ├── BlurProjection/
        │   ├── captions/{camera}/{sample}.json
        │   ├── control_input_blur/{camera}/{sample}.mp4
        │   └── videos/{camera}/{sample}.mp4
        ├── DepthSparse/
        │   └── control_input_depth/{camera}/{sample}.mp4
        └── HDMapBbox/
            └── control_input_hdmap_bbox/{camera}/{sample}.mp4

    Sample naming format: {scene_id}_id{track_id}_seg{seg_id}
        Examples: 001_id13_seg01, 002_id4_seg01, 010_id107_seg02
        - scene_id: 3-digit scene identifier (e.g., "001", "010")
        - track_id: variable-length track number (e.g., "13", "107")
        - seg_id: 2-digit segment number (e.g., "01", "02")
    """

    def __init__(
        self,
        base_path: str = NEW_TRAJECTORY_BASE,
        resolution_hw: Tuple[int, int] = (720, 1280),
        num_video_frames: int = 29,
        fps_downsample_factor: int = 1,
        camera_keys: Tuple[str, ...] = MODEL_CAMERAS[:7],  # Use first 7 cameras
        scene_ids: Optional[List[str]] = None,  # Filter by scene IDs (e.g., ["001", "003"])
        sample_names: Optional[List[str]] = None,  # Filter by specific sample names
    ) -> None:
        self.base_path = Path(base_path)
        self.blur_path = self.base_path / "BlurProjection"
        self.depth_path = self.base_path / "DepthSparse"
        self.hdmap_path = self.base_path / "HDMapBbox"

        self.resolution_hw = resolution_hw
        self.num_video_frames = num_video_frames
        self.fps_downsample_factor = fps_downsample_factor
        self.camera_keys = camera_keys  # Model camera names (without ftheta_)
        self.scene_ids = set(scene_ids) if scene_ids else None
        self.sample_names = set(sample_names) if sample_names else None

        # Validate directories
        for path, name in [(self.blur_path, "BlurProjection"),
                           (self.depth_path, "DepthSparse"),
                           (self.hdmap_path, "HDMapBbox")]:
            if not path.exists():
                raise FileNotFoundError(f"{name} not found at {path}")

        # Build sample list
        self.samples = self._build_sample_list()
        logger.info(f"[NewTrajectoryDataset] Found {len(self.samples)} samples")
        if self.scene_ids:
            logger.info(f"[NewTrajectoryDataset] Filtered to scenes: {self.scene_ids}")
        if self.sample_names:
            logger.info(f"[NewTrajectoryDataset] Filtered to samples: {self.sample_names}")

    def _get_ftheta_camera_name(self, model_camera: str) -> str:
        """Convert model camera name to ftheta camera name."""
        return f"ftheta_{model_camera}"

    def _extract_scene_id(self, sample_name: str) -> str:
        """Extract scene ID from sample name.

        Sample naming format: {scene_id}_id{track_id}_seg{seg_id}
        Examples: 001_id13_seg01 -> "001", 010_id107_seg02 -> "010"
        """
        # Split by "_id" to get the scene_id prefix
        parts = sample_name.split("_id")
        if len(parts) >= 2:
            return parts[0]
        return sample_name

    def _build_sample_list(self) -> List[Dict[str, str]]:
        """Build list of samples from the dataset (flat structure).

        Returns list of dicts with:
            - sample_id: unique sample identifier (e.g., "001_id17_seg01")
            - scene_id: scene identifier extracted from name (e.g., "001")
            - name: sample name (e.g., "001_id17_seg01")
        """
        # Use first camera to find all samples
        first_camera = self._get_ftheta_camera_name(self.camera_keys[0])

        # Try videos path first (for BlurProjection)
        videos_path = self.blur_path / "videos" / first_camera
        if not videos_path.exists():
            # Fallback to control_input_blur
            videos_path = self.blur_path / "control_input_blur" / first_camera

        if not videos_path.exists():
            raise FileNotFoundError(f"Data path not found: {videos_path}")

        samples = []

        # Flat structure: samples are directly in camera folder (no scene subdirectories)
        for video_file in sorted(videos_path.glob("*.mp4")):
            sample_name = video_file.stem  # e.g., "001_id17_seg01"
            scene_id = self._extract_scene_id(sample_name)  # e.g., "001"

            # Filter by scene_ids if specified
            if self.scene_ids and scene_id not in self.scene_ids:
                continue

            # Filter by sample_names if specified
            if self.sample_names and sample_name not in self.sample_names:
                continue

            # Verify all cameras and control types exist
            is_valid = True
            for camera in self.camera_keys:
                ftheta_camera = self._get_ftheta_camera_name(camera)

                # Check control inputs (flat structure)
                blur_path = self.blur_path / "control_input_blur" / ftheta_camera / f"{sample_name}.mp4"
                depth_path = self.depth_path / "control_input_depth" / ftheta_camera / f"{sample_name}.mp4"
                hdmap_path = self.hdmap_path / "control_input_hdmap_bbox" / ftheta_camera / f"{sample_name}.mp4"

                if not (blur_path.exists() and depth_path.exists() and hdmap_path.exists()):
                    logger.debug(f"Missing files for {sample_name}, camera {camera}")
                    is_valid = False
                    break

            if is_valid:
                samples.append({
                    "sample_id": sample_name,
                    "scene_id": scene_id,
                    "name": sample_name,
                })

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
        try:
            with open(caption_path, "r") as f:
                caption_json = json.load(f)
            return caption_json.get("caption", "")
        except Exception:
            return ""

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample_info = self.samples[index]
        sample_name = sample_info["name"]
        sample_id = sample_info["sample_id"]

        # Frame indices
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
            ftheta_camera = self._get_ftheta_camera_name(camera_name)

            # Load caption (only from front_wide camera) - flat structure
            if camera_name == "camera_front_wide_120fov":
                caption_path = self.blur_path / "captions" / ftheta_camera / f"{sample_name}.json"
                caption = self._load_caption(caption_path)
            else:
                caption = ""

            # Add view prefix to caption
            caption = f"{MODEL_CAPTION_PREFIXES[camera_name]} {caption}"
            captions.append(caption)

            # Load video frames (flat structure: directly in camera folder)
            video_path = self.blur_path / "videos" / ftheta_camera / f"{sample_name}.mp4"
            if video_path.exists():
                video_bytes = self._load_video(video_path)
                video_frames, fps, original_hw = self._extract_frames(video_bytes, frame_indices, self.resolution_hw)
                multiview_frames.append(video_frames)
            else:
                # Fallback: use blur frames as video frames
                blur_path_video = self.blur_path / "control_input_blur" / ftheta_camera / f"{sample_name}.mp4"
                video_bytes = self._load_video(blur_path_video)
                video_frames, fps, original_hw = self._extract_frames(video_bytes, frame_indices, self.resolution_hw)
                multiview_frames.append(video_frames)

            # Load control inputs
            # Blur - flat structure
            blur_path = self.blur_path / "control_input_blur" / ftheta_camera / f"{sample_name}.mp4"
            blur_bytes = self._load_video(blur_path)
            blur_frames, _, _ = self._extract_frames(blur_bytes, frame_indices, self.resolution_hw)
            multiview_control_blur.append(blur_frames)

            if video_fps is None:
                video_fps = fps
            original_sizes.append(list(original_hw))

            # Depth - flat structure
            depth_path = self.depth_path / "control_input_depth" / ftheta_camera / f"{sample_name}.mp4"
            depth_bytes = self._load_video(depth_path)
            depth_frames, _, _ = self._extract_frames(depth_bytes, frame_indices, self.resolution_hw)
            multiview_control_depth.append(depth_frames)

            # HDMap - flat structure
            hdmap_path = self.hdmap_path / "control_input_hdmap_bbox" / ftheta_camera / f"{sample_name}.mp4"
            hdmap_bytes = self._load_video(hdmap_path)
            hdmap_frames, _, _ = self._extract_frames(hdmap_bytes, frame_indices, self.resolution_hw)
            multiview_control_hdmap.append(hdmap_frames)

            # View indices
            view_idx = MODEL_CAMERA_VIEW_MAPPING[camera_name]
            view_indices.extend([view_idx] * self.num_video_frames)
            view_indices_selection.append(view_idx)
            camera_keys_selection.append(camera_name)

        fps = video_fps / self.fps_downsample_factor

        sample = {
            "__key__": sample_id,
            "__url__": str(self.base_path),
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
                self.camera_keys.index("camera_front_wide_120fov") if "camera_front_wide_120fov" in self.camera_keys else 0,
                dtype=torch.int64
            ),
            "original_hw": torch.tensor(original_sizes, dtype=torch.int64),
            "control_input_blur": rearrange(torch.cat(multiview_control_blur, dim=0), "t c h w -> c t h w"),
            "control_input_depth": rearrange(torch.cat(multiview_control_depth, dim=0), "t c h w -> c t h w"),
            "control_input_hdmap_bbox": rearrange(torch.cat(multiview_control_hdmap, dim=0), "t c h w -> c t h w"),
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


# ============================================================================
# Inference Functions (same as inference.py)
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="New trajectory inference")
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="Path to checkpoint directory")
    parser.add_argument("--experiment", type=str, default="custom_multi_control_post_train",
                        help="Experiment name")
    parser.add_argument("--output_dir", type=str,
                        default="/mnt/zihanw/Output_R2V_world_foundation_model_v1/inference_new_trajectory_generation",
                        help="Output directory")
    parser.add_argument("--data_path", type=str, default=NEW_TRAJECTORY_BASE,
                        help="Path to new trajectory data")
    parser.add_argument("--scene_ids", type=str, nargs="+", default=None,
                        help="Scene IDs to process (e.g., 001 003 004). "
                             "Matches samples starting with these prefixes.")
    parser.add_argument("--sample_names", type=str, nargs="+", default=None,
                        help="Specific sample names to process (e.g., 001_id13_seg01 002_id4_seg01)")
    parser.add_argument("--all_samples", action="store_true", default=False,
                        help="Process all samples")
    parser.add_argument("--sample_idx", type=int, default=None,
                        help="Single sample index")
    parser.add_argument("--guidance", type=float, default=7.0)
    parser.add_argument("--num_steps", type=int, default=35)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--load_ema", action="store_true", default=True)
    parser.add_argument("--num_views", type=int, default=7,
                        help="Number of views (max 7 due to model)")
    return parser.parse_args()


def init_distributed(context_parallel_size: int):
    """Initialize distributed processing."""
    from megatron.core import parallel_state

    if "RANK" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
        logger.info("Initialized single-GPU distributed (gloo backend)")
    else:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(dist.get_rank())

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    parallel_state.initialize_model_parallel(
        context_parallel_size=context_parallel_size,
    )
    process_group = parallel_state.get_context_parallel_group()

    logger.info(f"Initialized distributed: rank={rank}, world_size={world_size}, cp_size={context_parallel_size}")
    return process_group, rank == 0


def load_model_and_config(experiment_name: str, ckpt_path: str, context_parallel_size: int,
                          load_ema: bool = True, process_group=None):
    """Load model from DCP checkpoint."""
    from cosmos_transfer2._src.imaginaire.lazy_config import instantiate
    from cosmos_transfer2._src.imaginaire.utils.config_helper import get_config_module, override
    from cosmos_transfer2._src.predict2.checkpointer.dcp import (
        DefaultLoadPlanner,
        ModelWrapper,
        dcp_load_state_dict,
    )
    from torch.distributed.checkpoint import FileSystemReader

    logger.info("Loading config...")
    config_file = "cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/config.py"
    config_module = get_config_module(config_file)
    config = importlib.import_module(config_module).make_config()
    config = override(config, ["--", f"experiment={experiment_name}"])

    config.model.config.ema.enabled = False
    config.model.config.fsdp_shard_size = context_parallel_size

    config.validate()
    config.freeze()
    logger.info("Config loaded")

    logger.info("Instantiating model...")
    model = instantiate(config.model).cuda()
    model.on_train_start()
    logger.info(f"Model type: {type(model).__name__}")

    if context_parallel_size > 1 and process_group is not None:
        model.net.enable_context_parallel(process_group)
        logger.info("Context parallel enabled")

    logger.info(f"Loading checkpoint from: {ckpt_path}")
    ckpt_path = Path(ckpt_path)
    model_ckpt_path = ckpt_path / "model"

    if not model_ckpt_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_ckpt_path}")

    model_wrapper = ModelWrapper(model, load_ema_to_reg=load_ema)
    state_dict = model_wrapper.state_dict()

    storage_reader = FileSystemReader(str(model_ckpt_path))
    load_planner = DefaultLoadPlanner(allow_partial_load=True)
    dcp_load_state_dict(state_dict, storage_reader, load_planner)
    model_wrapper.load_state_dict(state_dict)

    logger.info("Checkpoint loaded successfully!")

    model.eval()
    return model, config


def load_sample(dataset, sample_idx: int, model) -> Dict[str, Any]:
    """Load a single sample and prepare for inference."""
    uint8_keys = {'video', 'control_input_blur', 'control_input_depth', 'control_input_hdmap_bbox'}

    sample = dataset[sample_idx]
    batch = collate_fn([sample])

    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            if key in uint8_keys:
                batch[key] = value.to(device=torch.device("cuda"))
            else:
                batch[key] = value.to(**model.tensor_kwargs)

    logger.info(f"Loaded sample {sample_idx}: {sample['__key__']}")
    return batch


@torch.no_grad()
def run_inference_single(model, batch, guidance=7.0, num_steps=35,
                         num_conditional_frames=0, control_weight=1.0):
    """Run inference on a single batch."""
    if hasattr(model, 'inplace_compute_text_embeddings_online'):
        model.inplace_compute_text_embeddings_online(batch)
        logger.info("Text embeddings computed")

    raw_data, x0, condition = model.get_data_and_condition(batch)
    logger.info(f"raw_data shape: {raw_data.shape}, x0 (latent) shape: {x0.shape}")

    batch[NUM_CONDITIONAL_FRAMES_KEY] = num_conditional_frames
    batch[CONTROL_WEIGHT_KEY] = control_weight

    logger.info(f"Generating with guidance={guidance}, steps={num_steps}...")
    sample = model.generate_samples_from_batch(
        batch,
        guidance=guidance,
        state_shape=x0.shape[1:],
        n_sample=x0.shape[0],
        num_steps=num_steps,
        is_negative_prompt=False,
    )
    logger.info(f"Generated latent shape: {sample.shape}")

    if hasattr(model, "decode"):
        generated = model.decode(sample)
        logger.info(f"Decoded video shape: {generated.shape}")
    else:
        generated = sample

    return generated, raw_data


def save_per_view_results(generated, raw_data, batch, output_dir, sample_idx,
                          camera_names, fps=10):
    """Save per-view generated and ground truth videos."""
    from cosmos_transfer2._src.imaginaire.visualize.video import save_img_or_video

    n_views = len(batch.get("view_indices_selection", [[0]])[0])
    sample_id = batch.get("__key__", [f"sample_{sample_idx}"])[0]

    # Replace "/" with "_" for directory name
    sample_dir_name = sample_id.replace("/", "_")
    sample_dir = Path(output_dir) / sample_dir_name
    sample_dir.mkdir(parents=True, exist_ok=True)

    generated_01 = ((generated.float() + 1.0) / 2.0).clamp(0, 1)
    gt_01 = ((raw_data.float() + 1.0) / 2.0).clamp(0, 1)

    logger.info(f"Saving per-view results for {sample_id}: {n_views} views to {sample_dir}")

    gen_views = rearrange(generated_01, "B C (V T) H W -> V B C T H W", V=n_views)
    gt_views = rearrange(gt_01, "B C (V T) H W -> V B C T H W", V=n_views)

    for v in range(n_views):
        cam_name = camera_names[v] if v < len(camera_names) else f"view_{v}"
        short_name = cam_name.replace("camera_", "")

        gen_path = sample_dir / f"{short_name}_generated"
        save_img_or_video(gen_views[v, 0], str(gen_path), fps=fps)

        gt_path = sample_dir / f"{short_name}_gt"
        save_img_or_video(gt_views[v, 0], str(gt_path), fps=fps)

        logger.info(f"  View {v} ({short_name}): generated + gt saved")

    logger.info(f"All per-view results saved to: {sample_dir}")


def cleanup_distributed():
    """Clean up distributed resources."""
    if dist.is_initialized():
        from megatron.core import parallel_state
        if parallel_state.is_initialized():
            parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


def main():
    args = parse_args()
    num_views = args.num_views
    context_parallel_size = 8  # Fixed for 8 GPUs

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Get camera names
    camera_names = list(MODEL_CAMERAS[:num_views])

    logger.info("=" * 60)
    logger.info("New Trajectory Inference (Generation)")
    logger.info("=" * 60)
    logger.info(f"Checkpoint: {args.ckpt_path}")
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Scene IDs: {args.scene_ids or 'all'}")
    logger.info(f"Sample names: {args.sample_names or 'all matching'}")
    logger.info(f"Context parallel size: {context_parallel_size}")
    logger.info(f"Views: {num_views} -> {camera_names}")
    logger.info(f"Guidance: {args.guidance}, Steps: {args.num_steps}")
    logger.info("=" * 60)

    process_group, is_rank0 = init_distributed(context_parallel_size)

    try:
        model, config = load_model_and_config(
            experiment_name=args.experiment,
            ckpt_path=args.ckpt_path,
            context_parallel_size=context_parallel_size,
            load_ema=args.load_ema,
            process_group=process_group,
        )

        # Create dataset
        dataset = NewTrajectoryDataset(
            base_path=args.data_path,
            resolution_hw=(720, 1280),
            num_video_frames=29,
            fps_downsample_factor=1,
            camera_keys=camera_names,
            scene_ids=args.scene_ids,
            sample_names=args.sample_names,
        )

        logger.info(f"Dataset has {len(dataset)} samples")

        # Determine which samples to process
        if args.all_samples:
            sample_indices = list(range(len(dataset)))
        elif args.sample_idx is not None:
            if args.sample_idx >= len(dataset):
                raise ValueError(f"Sample index {args.sample_idx} out of range (max {len(dataset)-1})")
            sample_indices = [args.sample_idx]
        else:
            sample_indices = list(range(len(dataset)))

        logger.info(f"Will process {len(sample_indices)} sample(s)")
        for idx in sample_indices[:10]:  # Show first 10
            logger.info(f"  [{idx}] {dataset.samples[idx]['sample_id']}")
        if len(sample_indices) > 10:
            logger.info(f"  ... and {len(sample_indices) - 10} more")

        # Process each sample
        for i, sample_idx in enumerate(sample_indices):
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing sample {i+1}/{len(sample_indices)}: "
                        f"{dataset.samples[sample_idx]['sample_id']} (idx={sample_idx})")
            logger.info(f"{'='*60}")

            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()

            batch = load_sample(dataset, sample_idx, model)

            generated, raw_data = run_inference_single(
                model=model,
                batch=batch,
                guidance=args.guidance,
                num_steps=args.num_steps,
            )

            if is_rank0:
                save_per_view_results(
                    generated=generated,
                    raw_data=raw_data,
                    batch=batch,
                    output_dir=args.output_dir,
                    sample_idx=sample_idx,
                    camera_names=camera_names,
                    fps=args.fps,
                )

            del batch, generated, raw_data
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()

            if dist.is_initialized():
                dist.barrier()

            logger.info(f"Sample {dataset.samples[sample_idx]['sample_id']} completed")

        logger.info("=" * 60)
        logger.info(f"SUCCESS! All {len(sample_indices)} sample(s) completed.")
        logger.info(f"Results saved to: {args.output_dir}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Inference failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
