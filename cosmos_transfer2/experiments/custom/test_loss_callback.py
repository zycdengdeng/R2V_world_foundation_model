# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Test Loss Callback for Multi-Control Post Training
# Computes loss on the entire test set and logs to wandb for checkpoint selection.

import os
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
from einops import rearrange
from torch.utils.data import DataLoader

from cosmos_transfer2._src.imaginaire.utils.callback import Callback
from cosmos_transfer2._src.imaginaire.utils import log
from cosmos_transfer2._src.imaginaire.utils.parallel_state_helper import is_tp_cp_pp_rank0

try:
    import wandb
except ImportError:
    wandb = None


class EveryNTestLoss(Callback):
    """
    Callback that computes loss on the entire test set every N iterations.

    This helps:
    1. Monitor overfitting (train loss vs test loss)
    2. Select the best checkpoint based on test loss
    3. Early stopping decisions

    The loss is computed using the same method as training loss.
    """

    def __init__(
        self,
        eval_dataset,
        every_n: int = 200,
        num_timestep_samples: int = 4,  # Number of timesteps to average over per sample
        name: str = "test_loss",
    ):
        """
        Args:
            eval_dataset: The evaluation dataset (test set)
            every_n: Compute test loss every N iterations
            num_timestep_samples: Number of random timesteps to sample per data point
                                  (averaging reduces variance)
            name: Name prefix for wandb logging
        """
        super().__init__()
        self.eval_dataset = eval_dataset
        self.every_n = every_n
        self.num_timestep_samples = num_timestep_samples
        self.name = name

        self.dataloader = None
        self.rank = 0

    def on_train_start(self, model, iteration: int = 0) -> None:
        """Initialize dataloader on training start."""
        from cosmos_transfer2.experiments.custom.custom_multi_control_dataset import collate_fn

        self.rank = dist.get_rank() if dist.is_initialized() else 0

        # Create dataloader for test set (no shuffling for reproducibility)
        self.dataloader = DataLoader(
            self.eval_dataset,
            batch_size=1,  # Process one sample at a time to avoid OOM
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=False,
        )

        if self.rank == 0:
            log.info(f"[EveryNTestLoss] Initialized with {len(self.eval_dataset)} test samples")
            log.info(f"[EveryNTestLoss] Will compute test loss every {self.every_n} iterations")

    def on_training_step_end(
        self,
        model,
        data_batch: dict,
        output_batch: dict,
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        """Compute test loss every N iterations."""
        if iteration % self.every_n != 0:
            return

        if iteration == 0:
            return  # Skip iteration 0

        log.info(f"[EveryNTestLoss] Computing test loss at iteration {iteration}")

        try:
            test_loss = self._compute_test_loss(model)

            # Log to wandb
            if is_tp_cp_pp_rank0() and wandb is not None and wandb.run:
                wandb.log({
                    "trainer/global_step": iteration,
                    f"{self.name}/loss": test_loss,
                }, step=iteration)

            log.info(f"[EveryNTestLoss] Iteration {iteration}: test_loss = {test_loss:.6f}")

        except Exception as e:
            log.error(f"[EveryNTestLoss] Failed to compute test loss: {e}")
            import traceback
            traceback.print_exc()

    @torch.no_grad()
    def _compute_test_loss(self, model) -> float:
        """Compute average loss on the entire test set."""
        model.eval()

        device = model.tensor_kwargs.get('device', 'cuda')
        uint8_keys = {'video', 'control_input_blur', 'control_input_depth', 'control_input_hdmap_bbox'}

        total_loss = 0.0
        num_samples = 0

        # Clear GPU cache before starting
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for batch_idx, batch in enumerate(self.dataloader):
            try:
                # Move batch to device
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        if key in uint8_keys:
                            batch[key] = value.to(device=device)
                        else:
                            batch[key] = value.to(**model.tensor_kwargs)

                # Compute loss for this sample
                sample_loss = self._compute_sample_loss(model, batch)

                if sample_loss is not None:
                    total_loss += sample_loss
                    num_samples += 1

                # Clear cache periodically
                if batch_idx % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                log.warning(f"[EveryNTestLoss] Error processing sample {batch_idx}: {e}")
                continue

        model.train()

        if num_samples == 0:
            log.error("[EveryNTestLoss] No valid samples processed!")
            return float('nan')

        avg_loss = total_loss / num_samples
        return avg_loss

    def _compute_sample_loss(self, model, data_batch: dict) -> Optional[float]:
        """
        Compute loss for a single sample, averaging over multiple timesteps.

        This mimics the training_step logic but for evaluation.
        """
        # Compute text embeddings online (same as training)
        if hasattr(model, 'inplace_compute_text_embeddings_online'):
            model.inplace_compute_text_embeddings_online(data_batch)
        elif model.config.text_encoder_config is not None and model.config.text_encoder_config.compute_online:
            text_embeddings = model.text_encoder.compute_text_embeddings_online(
                data_batch, model.input_caption_key
            )
            data_batch["t5_text_embeddings"] = text_embeddings
            data_batch["t5_text_mask"] = torch.ones(
                text_embeddings.shape[0], text_embeddings.shape[1], device="cuda"
            )

        # Get input data and condition
        _, x0_B_C_T_H_W, condition = model.get_data_and_condition(data_batch)

        losses = []

        for _ in range(self.num_timestep_samples):
            # Sample noise
            epsilon_B_C_T_H_W = torch.randn(x0_B_C_T_H_W.size(), **model.tensor_kwargs_fp32)

            # Sample timestep
            batch_size = x0_B_C_T_H_W.size()[0]
            t_B = model.rectified_flow.sample_train_time(batch_size).to(**model.tensor_kwargs_fp32)
            t_B = rearrange(t_B, "b -> b 1")

            # Broadcast for model parallelism
            x0_split, condition_split, epsilon_split, t_split = model.broadcast_split_for_model_parallelsim(
                x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, t_B
            )

            timesteps = model.rectified_flow.get_discrete_timestamp(t_split, model.tensor_kwargs_fp32)
            sigmas = model.rectified_flow.get_sigmas(timesteps, model.tensor_kwargs_fp32)

            timesteps = rearrange(timesteps, "b -> b 1")
            sigmas = rearrange(sigmas, "b -> b 1")

            # Get interpolation (noisy sample and target velocity)
            xt_B_C_T_H_W, vt_B_C_T_H_W = model.rectified_flow.get_interpolation(
                epsilon_split, x0_split, sigmas
            )

            # Predict velocity
            vt_pred_B_C_T_H_W = model.denoise(
                noise=epsilon_split,
                xt_B_C_T_H_W=xt_B_C_T_H_W.to(**model.tensor_kwargs),
                timesteps_B_T=timesteps,
                condition=condition_split,
            )

            # Compute loss (MSE between predicted and target velocity)
            time_weights_B = model.rectified_flow.train_time_weight(timesteps, model.tensor_kwargs_fp32)
            per_instance_loss = torch.mean(
                (vt_pred_B_C_T_H_W - vt_B_C_T_H_W) ** 2,
                dim=list(range(1, vt_pred_B_C_T_H_W.dim()))
            )
            loss = torch.mean(time_weights_B * per_instance_loss)

            losses.append(loss.item())

        # Average over timestep samples
        return sum(losses) / len(losses) if losses else None
