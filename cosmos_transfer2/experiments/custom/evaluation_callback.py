# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Evaluation Callback for Multi-Control Post Training
# Runs inference on fixed test samples and saves videos for comparison across iterations.

import os
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision
import wandb
from einops import rearrange, repeat
from torch.utils.data import DataLoader

from cosmos_transfer2._src.imaginaire.utils.callback import Callback
from cosmos_transfer2._src.imaginaire.utils import log, misc
from cosmos_transfer2._src.imaginaire.utils.parallel_state_helper import is_tp_cp_pp_rank0
from cosmos_transfer2._src.predict2.callbacks.every_n_draw_sample import resize_image
from cosmos_transfer2._src.predict2.models.video2world_model import NUM_CONDITIONAL_FRAMES_KEY
from cosmos_transfer2._src.predict2_multiview.callbacks.every_n_draw_sample_multiviewvideo import (
    visualization_view_index_order,
    camera_to_view_id,
)

try:
    import imageio
except Exception:
    imageio = None

import cv2
import numpy as np

CONTROL_WEIGHT_KEY = "control_weight"


class EveryNEvalMultiviewVideo(Callback):
    """
    Evaluation callback that runs inference on fixed test samples.

    This callback:
    1. Loads a fixed batch of samples from the test dataset
    2. Runs inference every N iterations
    3. Saves videos with control inputs, ground truth, and predictions stacked vertically
    4. Logs to wandb for comparison across iterations
    """

    def __init__(
        self,
        eval_dataset,
        eval_sample_indices: List[int] = [0, 1],  # Which samples to use from test set
        every_n: int = 2000,
        num_sampling_step: int = 35,
        guidance: List[float] = [7.0],
        fps: int = 10,
        ctrl_hint_keys: List[str] = None,
        control_weights: List[float] = [1.0],
        num_cond_frames: List[int] = [1],  # Use 1 conditioning frame for evaluation
        save_local: bool = True,
        local_dir: str = None,
        name: str = "eval",
    ):
        """
        Args:
            eval_dataset: The evaluation dataset (test set)
            eval_sample_indices: Indices of samples to use from test set
            every_n: Run evaluation every N iterations
            num_sampling_step: Number of sampling steps for inference
            guidance: CFG guidance scale(s)
            fps: FPS for saved videos
            ctrl_hint_keys: Control input keys to visualize
            control_weights: Control weights to test
            num_cond_frames: Number of conditioning frames
            save_local: Whether to save videos locally
            local_dir: Local directory for saving (defaults to output_dir/eval)
            name: Name prefix for saved files
        """
        super().__init__()
        self.eval_dataset = eval_dataset
        self.eval_sample_indices = eval_sample_indices
        self.every_n = every_n
        self.num_sampling_step = num_sampling_step
        self.guidance = guidance
        self.fps = fps
        self.ctrl_hint_keys = ctrl_hint_keys or []
        self.control_weights = control_weights
        self.num_cond_frames = num_cond_frames
        self.save_local = save_local
        self.local_dir = local_dir
        self.name = name

        self.eval_batch = None  # Will be loaded on first call
        self.rank = 0
        self.data_parallel_id = 0

    def _load_eval_batch(self, model) -> Dict[str, Any]:
        """Load fixed evaluation batch from test dataset."""
        samples = []
        for idx in self.eval_sample_indices:
            if idx < len(self.eval_dataset):
                sample = self.eval_dataset[idx]
                samples.append(sample)
            else:
                log.warning(f"Eval sample index {idx} out of range, dataset has {len(self.eval_dataset)} samples")

        if not samples:
            log.error("No valid evaluation samples found!")
            return None

        # Collate samples into a batch
        from cosmos_transfer2.experiments.custom.custom_multi_control_dataset import collate_fn
        batch = collate_fn(samples)

        # Move to device
        batch = misc.to(batch, **model.tensor_kwargs)
        return batch

    def on_train_start(self, model, iteration: int = 0) -> None:
        """Initialize on training start."""
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.data_parallel_id = self.rank

        # Use self.config.job.path_local like EveryNDrawSample does
        # self.config is automatically set by CallbackManager
        if self.local_dir is None:
            if hasattr(self, 'config') and hasattr(self.config, 'job') and hasattr(self.config.job, 'path_local'):
                self.local_dir = os.path.join(self.config.job.path_local, "eval_outputs")
            else:
                # Fallback to current directory
                self.local_dir = os.path.join(os.getcwd(), "eval_outputs")

        if self.rank == 0:
            os.makedirs(self.local_dir, exist_ok=True)
            log.info(f"[EveryNEvalMultiviewVideo] Will save eval outputs to {self.local_dir}")
            log.info(f"[EveryNEvalMultiviewVideo] Using {len(self.eval_sample_indices)} fixed samples from test set")

    def on_train_batch_end(
        self,
        trainer,
        model,
        data_batch: dict,
        output_batch: dict,
        loss: torch.Tensor,
        iteration: int,
    ) -> None:
        """Run evaluation every N iterations."""
        if iteration % self.every_n != 0:
            return

        if iteration == 0:
            return  # Skip iteration 0

        log.info(f"[EveryNEvalMultiviewVideo] Running evaluation at iteration {iteration}")

        try:
            self._run_evaluation(trainer, model, iteration)
        except Exception as e:
            log.error(f"[EveryNEvalMultiviewVideo] Evaluation failed: {e}")
            import traceback
            traceback.print_exc()

    @torch.no_grad()
    def _run_evaluation(self, trainer, model, iteration: int):
        """Run inference on fixed test samples and save videos."""
        # Load eval batch if not already loaded
        if self.eval_batch is None:
            self.eval_batch = self._load_eval_batch(model)
            if self.eval_batch is None:
                return

        # Make a copy of the batch to avoid modifying the cached version
        data_batch = {k: v.clone() if isinstance(v, torch.Tensor) else v
                      for k, v in self.eval_batch.items()}

        n_views = len(data_batch["view_indices_selection"][0])

        # Get raw data and condition
        raw_data, x0, condition = model.get_data_and_condition(data_batch)
        batch_size = x0.shape[0]

        def time_to_width_dimension(mv_video):
            """Rearrange multiview video for visualization."""
            current_view_index_order = [i.item() for i in data_batch["view_indices_selection"][0]]
            expected_view_index_order = visualization_view_index_order

            # Reorder views to match expected visualization order
            if current_view_index_order != expected_view_index_order:
                reorder_indices = []
                for expected_view in expected_view_index_order:
                    if expected_view in current_view_index_order:
                        reorder_indices.append(current_view_index_order.index(expected_view))

                B, C, VT, H, W = mv_video.shape
                T = VT // n_views
                mv_video = rearrange(mv_video, "B C (V T) H W -> B C V T H W", V=n_views)
                mv_video = mv_video[:, :, reorder_indices, :, :, :]
                mv_video = rearrange(mv_video, "B C V T H W -> B C (V T) H W")

            return rearrange(mv_video, "B C (V T) H W -> B C T H (V W)", V=n_views)

        # Clear GPU cache before sampling
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        to_show = []

        # Generate samples with different configurations
        for num_cond_frames in self.num_cond_frames:
            for control_weight in self.control_weights:
                data_batch[NUM_CONDITIONAL_FRAMES_KEY] = num_cond_frames
                data_batch[CONTROL_WEIGHT_KEY] = control_weight

                for guidance in self.guidance:
                    sample = model.generate_samples_from_batch(
                        data_batch,
                        guidance=guidance,
                        state_shape=x0.shape[1:],
                        n_sample=x0.shape[0],
                        num_steps=self.num_sampling_step,
                        is_negative_prompt=False,
                    )
                    if hasattr(model, "decode"):
                        sample = model.decode(sample)
                    to_show.append(sample.float().cpu())

        # Add ground truth
        to_show.append(raw_data.float().cpu())

        # Add control inputs visualization
        if self.ctrl_hint_keys:
            for key in self.ctrl_hint_keys:
                if key in data_batch and data_batch[key] is not None:
                    hint = data_batch[key]
                    to_show.append(hint.float().cpu())

        # Rearrange for visualization (views side by side)
        if n_views > 1:
            to_show = [time_to_width_dimension(t) for t in to_show]

        # Save outputs
        if is_tp_cp_pp_rank0():
            self._save_outputs(to_show, batch_size, n_views, iteration)

    def _save_outputs(self, to_show: List[torch.Tensor], batch_size: int, n_views: int, iteration: int):
        """Save visualization outputs."""
        to_show = (1.0 + torch.stack(to_show, dim=0).clamp(-1, 1)) / 2.0  # [n, b, c, t, h, w]

        base_fp = f"{self.name}_Iter{iteration:09d}_{n_views}views"

        # Save 12-frame grid image
        _T = to_show.shape[3]
        n_frames = min(12, _T)
        frame_indices = [round(ix * (_T - 1) / (n_frames - 1)) for ix in range(n_frames)]
        to_show_frames = to_show[:, :, :, frame_indices]
        to_show_frames = rearrange(to_show_frames, "n b c t h w -> 1 c (n h) (b t w)")

        image_grid = torchvision.utils.make_grid(to_show_frames, nrow=1, padding=0, normalize=False)

        local_path_frames = f"{self.local_dir}/{base_fp}_frames.jpg"
        torchvision.utils.save_image(resize_image(image_grid, 1024), local_path_frames, nrow=1, scale_each=True)

        # Save video
        video_tensor = rearrange(to_show, "n b c t h (v w) -> t (n h) (b v w) c", v=n_views)

        # Resize if too wide
        max_w = 2048
        T, H, W, C = video_tensor.shape
        if W > max_w:
            scale = max_w / W
            new_w = max_w
            new_h = int(H * scale)
            video_tensor_f = video_tensor.permute(0, 3, 1, 2)
            video_tensor_f = F.interpolate(video_tensor_f, size=(new_h, new_w), mode="bilinear", align_corners=False)
            video_tensor = video_tensor_f.permute(0, 2, 3, 1)

        video_tensor = rearrange(video_tensor, "T H W C -> C T H W")
        video_fp = f"{self.local_dir}/{base_fp}.mp4"
        self._save_video(video_tensor.cpu().numpy(), video_fp)

        # Log to wandb
        if wandb.run:
            info = {
                "trainer/global_step": iteration,
                f"{self.name}/eval_frames": wandb.Image(local_path_frames, caption=f"Iter {iteration}"),
                f"{self.name}/eval_video": wandb.Video(video_fp, caption=f"Iter {iteration}"),
            }
            wandb.log(info, step=iteration)

        log.info(f"[EveryNEvalMultiviewVideo] Saved eval outputs to {self.local_dir}")

    def _save_video(self, grid: np.ndarray, video_name: str):
        """Save video using imageio."""
        grid = (grid * 255).astype(np.uint8)
        grid = np.transpose(grid, (1, 2, 3, 0))  # (T, H, W, C)

        # Ensure even dimensions for H.264
        processed_frames = []
        for frame in grid:
            h, w = frame.shape[:2]
            pad_h = h % 2
            pad_w = w % 2
            if pad_h or pad_w:
                frame = cv2.copyMakeBorder(frame, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)
            processed_frames.append(frame)

        try:
            if imageio is not None:
                kwargs = {
                    "fps": self.fps,
                    "quality": 5,
                    "macro_block_size": 1,
                    "ffmpeg_params": ["-c:v", "libx264", "-preset", "medium"],
                }
                imageio.mimsave(video_name, processed_frames, "mp4", **kwargs)
            else:
                log.warning("imageio not available, skipping video save")
        except Exception as e:
            log.error(f"Failed to save video: {e}")
