# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Callback to reinitialize control-related weights after checkpoint loading
# This ensures all 3 control heads start from scratch

import math
from typing import Optional

import torch
import torch.nn as nn

from cosmos_transfer2._src.imaginaire.utils.callback import Callback
from cosmos_transfer2._src.imaginaire.utils.distributed import is_rank0
from cosmos_transfer2._src.imaginaire.utils.easy_io import easy_io as log


class ReinitControlWeightsCallback(Callback):
    """
    Callback to reinitialize control-related weights after checkpoint loading.

    This ensures that all control heads (vis, depth, seg) start from random
    initialization, rather than inheriting weights from the pre-trained
    hdmap_bbox control head in Transfer2.5.

    Weights reinitialized:
    - net.control_embedder: PatchEmbed layer that embeds control latents
    - net.input_hint_block: MLP that processes control features (if exists)
    """

    def __init__(self, reinit_control_embedder: bool = True, reinit_input_hint_block: bool = True):
        """
        Args:
            reinit_control_embedder: Whether to reinitialize control_embedder weights
            reinit_input_hint_block: Whether to reinitialize input_hint_block weights
        """
        super().__init__()
        self.reinit_control_embedder = reinit_control_embedder
        self.reinit_input_hint_block = reinit_input_hint_block

    def on_load_checkpoint_end(
        self, model, iteration: int = 0, checkpoint_path: Optional[str] = None
    ) -> None:
        """Called after checkpoint is loaded. Reinitialize control weights here."""
        if is_rank0():
            log.info("ReinitControlWeightsCallback: Reinitializing control weights...")

        # Get the network module
        net = model.net

        # Reinitialize control_embedder
        if self.reinit_control_embedder and hasattr(net, 'control_embedder'):
            self._reinit_patch_embed(net.control_embedder)
            if is_rank0():
                log.info("  - Reinitialized net.control_embedder")

        # Reinitialize input_hint_block
        if self.reinit_input_hint_block and hasattr(net, 'input_hint_block'):
            self._reinit_mlp_block(net.input_hint_block)
            if is_rank0():
                log.info("  - Reinitialized net.input_hint_block")

        if is_rank0():
            log.info("ReinitControlWeightsCallback: Done reinitializing control weights")

    def _reinit_patch_embed(self, module):
        """Reinitialize a PatchEmbed module."""
        if isinstance(module, nn.ModuleList):
            # Multiple control branches
            for m in module:
                self._reinit_patch_embed_single(m)
        else:
            self._reinit_patch_embed_single(module)

    def _reinit_patch_embed_single(self, patch_embed):
        """Reinitialize a single PatchEmbed module."""
        # PatchEmbed has proj = nn.Sequential(Rearrange, nn.Linear)
        if hasattr(patch_embed, 'proj'):
            for layer in patch_embed.proj:
                if isinstance(layer, nn.Linear):
                    # Use the same initialization as in PatchEmbed.init_weights
                    dim = layer.in_features
                    std = 1.0 / math.sqrt(dim)
                    torch.nn.init.trunc_normal_(layer.weight, std=std, a=-3 * std, b=3 * std)
                    if layer.bias is not None:
                        torch.nn.init.zeros_(layer.bias)

    def _reinit_mlp_block(self, module):
        """Reinitialize an MLP block (nn.Sequential of Linear + SiLU)."""
        for layer in module.modules():
            if isinstance(layer, nn.Linear):
                # Use the same initialization as in the network's init_weights
                std = 1.0 / math.sqrt(layer.weight.shape[0])
                torch.nn.init.trunc_normal_(layer.weight, std=std, a=-3 * std, b=3 * std)
                if layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias)
