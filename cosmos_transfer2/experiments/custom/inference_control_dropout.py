#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Modality-dropout inference: zero out selected control modalities in LATENT space.

Reuses all machinery from inference.py; the only addition is a wrapper around
model.get_data_and_condition that zeroes the 16-channel latent slice of each
dropped modality AFTER VAE encoding. This exactly reproduces the model's own
"missing modality -> zero latent" semantics (multiview control model,
get_data_and_condition), and applies to both CFG branches since generation
re-derives the condition through the wrapped method.

Channel layout of latent_control_input (hint_keys="hdmap_blur_depth"):
    hdmap: 0-15, blur: 16-31, depth: 32-47

Usage (same env/launch as inference.py):
    torchrun --nproc_per_node=8 --master_port=12345 \
        -m cosmos_transfer2.experiments.custom.inference_control_dropout \
        --ckpt_path .../iter_000006600 \
        --output_dir .../inference_dropout/no_hdmap \
        --context_parallel_size 8 --num_views 7 \
        --scene_ids 033 053 056 076 077 088 089 \
        --drop_controls hdmap

    --drop_controls accepts any of: hdmap blur depth (one or more).
    Omitting it runs the full-condition baseline (identical to inference.py).
"""

import argparse
import gc

import torch
import torch.distributed as dist
from loguru import logger

# Reuse everything from the proven inference script (also sets NVTE_FUSED_ATTN=0)
from cosmos_transfer2.experiments.custom.inference import (
    cleanup_distributed,
    create_test_dataset,
    get_sample_indices_for_scenes,
    init_distributed,
    load_model_and_config,
    load_sample,
    run_inference_single,
    save_per_view_results,
)

# Must match the model's hint_keys order ("hdmap_blur_depth")
MODALITY_ORDER = ["hdmap", "blur", "depth"]


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-control inference with modality dropout")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--experiment", type=str, default="custom_multi_control_post_train")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--drop_controls", type=str, nargs="+", default=[],
                        choices=MODALITY_ORDER,
                        help="Modalities to zero out in latent space (e.g. --drop_controls hdmap)")
    parser.add_argument("--scene_ids", type=str, nargs="+", default=None)
    parser.add_argument("--all_samples", action="store_true", default=False)
    parser.add_argument("--sample_idx", type=int, default=None)
    parser.add_argument("--use_train_set", action="store_true", default=False)
    parser.add_argument("--context_parallel_size", type=int, default=1)
    parser.add_argument("--num_views", type=int, default=None)
    parser.add_argument("--guidance", type=float, default=7.0)
    parser.add_argument("--num_steps", type=int, default=35)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--load_ema", action="store_true", default=True)
    return parser.parse_args()


def install_control_dropout(model, drop_modalities):
    """Wrap model.get_data_and_condition to zero dropped modalities' latent slices.

    The wrapper takes effect for every condition derivation, including the ones
    inside generate_samples_from_batch / get_velocity_fn_from_batch (both CFG
    branches), because they call self.get_data_and_condition.
    """
    drop_indices = sorted(MODALITY_ORDER.index(m) for m in set(drop_modalities))
    num_modalities = len(model.hint_keys)
    original_fn = model.get_data_and_condition

    def wrapped(data_batch, *args, **kwargs):
        raw, latent, condition = original_fn(data_batch, *args, **kwargs)
        lci = getattr(condition, "latent_control_input", None)
        if lci is not None:
            ch_per_mod = lci.shape[1] // num_modalities
            lci = lci.clone()
            for mi in drop_indices:
                lci[:, mi * ch_per_mod:(mi + 1) * ch_per_mod] = 0
            condition = condition.set_control_condition(
                latent_control_input=lci,
                control_weight=getattr(condition, "control_context_scale", 1.0),
            )
        return raw, latent, condition

    model.get_data_and_condition = wrapped
    logger.info(f"Control dropout installed: zeroing {drop_modalities} "
                f"(latent channel groups {drop_indices}, {num_modalities} modalities total)")


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    num_views = args.num_views if args.num_views is not None else args.context_parallel_size

    from cosmos_transfer2.experiments.custom.custom_multi_control_experiment import TRAINING_CAMERAS
    camera_names = list(TRAINING_CAMERAS[:num_views])

    logger.info("=" * 60)
    logger.info("Multi-Control Inference with Modality Dropout")
    logger.info("=" * 60)
    logger.info(f"Checkpoint: {args.ckpt_path}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"DROPPED MODALITIES: {args.drop_controls or 'none (full baseline)'}")
    logger.info(f"Scene IDs: {args.scene_ids or ('all' if args.all_samples else args.sample_idx)}")
    logger.info(f"Views: {num_views}, Guidance: {args.guidance}, Steps: {args.num_steps}, Seed: {args.seed}")
    logger.info("=" * 60)

    if num_views > args.context_parallel_size:
        raise ValueError(f"num_views ({num_views}) must be <= context_parallel_size ({args.context_parallel_size})")

    process_group, is_rank0 = init_distributed(args.context_parallel_size)

    try:
        model, config = load_model_and_config(
            experiment_name=args.experiment,
            ckpt_path=args.ckpt_path,
            context_parallel_size=args.context_parallel_size,
            load_ema=args.load_ema,
            process_group=process_group,
        )

        if args.drop_controls:
            install_control_dropout(model, args.drop_controls)

        test_dataset = create_test_dataset(num_views=num_views, use_train_set=args.use_train_set)

        if args.scene_ids:
            sample_indices = get_sample_indices_for_scenes(test_dataset, args.scene_ids)
            if not sample_indices:
                raise ValueError(f"No samples found for scene IDs: {args.scene_ids}")
        elif args.all_samples:
            sample_indices = list(range(len(test_dataset)))
        elif args.sample_idx is not None:
            sample_indices = [args.sample_idx]
        else:
            sample_indices = [0]

        logger.info(f"Will process {len(sample_indices)} sample(s)")

        for i, sample_idx in enumerate(sample_indices):
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Sample {i + 1}/{len(sample_indices)}: {test_dataset.samples[sample_idx]}")
            logger.info(f"{'=' * 60}")

            torch.cuda.empty_cache()
            gc.collect()

            batch = load_sample(test_dataset, sample_idx, model)

            generated, raw_data = run_inference_single(
                model=model,
                batch=batch,
                guidance=args.guidance,
                num_steps=args.num_steps,
                num_conditional_frames=0,
                control_weight=1.0,
            )

            if is_rank0:
                save_per_view_results(
                    generated=generated,
                    batch=batch,
                    output_dir=args.output_dir,
                    sample_idx=sample_idx,
                    camera_names=camera_names,
                    fps=args.fps,
                )

            del batch, generated, raw_data
            torch.cuda.empty_cache()
            gc.collect()

            if dist.is_initialized():
                dist.barrier()

        logger.info("=" * 60)
        logger.info(f"SUCCESS! {len(sample_indices)} sample(s) done. Dropped: {args.drop_controls or 'none'}")
        logger.info(f"Results: {args.output_dir}")
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
