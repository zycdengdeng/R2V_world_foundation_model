# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Caches online text embeddings keyed by caption content.
#
# Motivation: with compute_online=True the 7B text encoder recomputes embeddings
# every micro-step. When the dataset uses (near-)identical captions across all
# samples (e.g. one shared scene description + fixed view prefixes), this is
# redundant work. This callback wraps model.inplace_compute_text_embeddings_online
# with a cache: for a caption set already seen, the stored embeddings are written
# into the batch and the text encoder is skipped entirely.
#
# Correctness: the wrapped function is deterministic given the captions (it only
# runs the frozen text encoder). Text dropout happens downstream in the
# conditioner and is unaffected. Cache hits hand out fresh tensors (moved from
# CPU storage), so downstream code never aliases cache state.

from typing import Any, Dict

import torch

from cosmos_transfer2._src.imaginaire.utils import log
from cosmos_transfer2._src.imaginaire.utils.callback import Callback

_CACHED_KEYS = ("t5_text_embeddings", "neg_t5_text_embeddings")


def _caption_key(ai_caption) -> tuple:
    """Nested list of caption strings -> hashable key."""

    def _freeze(x):
        if isinstance(x, (list, tuple)):
            return tuple(_freeze(v) for v in x)
        return x

    return _freeze(ai_caption)


def _to_storage(value, device: str):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device, copy=True)
    if isinstance(value, dict):
        return {k: _to_storage(v, device) for k, v in value.items()}
    return value


def _to_cuda(value):
    if isinstance(value, torch.Tensor):
        return value.to("cuda", non_blocking=True)
    if isinstance(value, dict):
        return {k: _to_cuda(v) for k, v in value.items()}
    return value


class CacheTextEmbeddings(Callback):
    """Wraps inplace_compute_text_embeddings_online with a caption-keyed cache.

    Args:
        store_on_gpu: keep cached embeddings on GPU (fastest, costs GPU memory
            roughly the size of one batch's pos+neg embeddings per unique
            caption set). Default False: store on CPU and copy to GPU on hit.
        max_entries: cap on distinct caption sets to cache (FIFO eviction).
    """

    def __init__(self, store_on_gpu: bool = False, max_entries: int = 8):
        super().__init__()
        self.store_on_gpu = store_on_gpu
        self.max_entries = max_entries
        self._cache: Dict[tuple, Dict[str, Any]] = {}
        self._hits = 0
        self._misses = 0

    def on_train_start(self, model, iteration: int = 0) -> None:
        if getattr(model, "_text_embedding_cache_installed", False):
            return

        original_fn = model.inplace_compute_text_embeddings_online
        storage_device = "cuda" if self.store_on_gpu else "cpu"
        cache = self._cache

        def cached_fn(data_batch: dict) -> None:
            captions = data_batch.get("ai_caption", None)
            if captions is None:
                return original_fn(data_batch)
            key = _caption_key(captions)

            entry = cache.get(key)
            if entry is not None:
                self._hits += 1
                for k in _CACHED_KEYS:
                    data_batch[k] = _to_cuda(entry[k])
                emb = entry[_CACHED_KEYS[0]]
                emb_tensor = emb["text_embeddings"] if isinstance(emb, dict) else emb
                data_batch["t5_text_mask"] = torch.ones(
                    emb_tensor.shape[0], emb_tensor.shape[1], device="cuda"
                )
                if self._hits in (1, 100, 1000):
                    log.info(
                        f"[CacheTextEmbeddings] cache hits={self._hits}, misses={self._misses} "
                        f"(text encoder skipped on hits)"
                    )
                return

            self._misses += 1
            original_fn(data_batch)
            if len(cache) >= self.max_entries:
                cache.pop(next(iter(cache)))
            cache[key] = {k: _to_storage(data_batch[k], storage_device) for k in _CACHED_KEYS}
            log.info(
                f"[CacheTextEmbeddings] cached new caption set "
                f"(entries={len(cache)}, storage={storage_device})"
            )

        model.inplace_compute_text_embeddings_online = cached_fn
        model._text_embedding_cache_installed = True
        log.info(f"[CacheTextEmbeddings] installed (storage={'gpu' if self.store_on_gpu else 'cpu'})")
