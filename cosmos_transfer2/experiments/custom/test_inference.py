#!/usr/bin/env python3
"""
Minimal inference test script.
First verify that model loading and inference work correctly.

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m cosmos_transfer2.experiments.custom.test_inference \
        --ckpt_path /mnt/zihanw/Output_R2V_world_foundation_model_v1/cosmos_transfer_custom/multi_control/2b_custom_multi_control_20251226_155343/checkpoints/iter_000005000
"""

import argparse
import os
import torch
from loguru import logger

# Disable fused attention for compatibility
os.environ["NVTE_FUSED_ATTN"] = "0"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--experiment", type=str, default="custom_multi_control_post_train")
    parser.add_argument("--output_dir", type=str, default="./test_inference_output")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Step 1: Import modules")
    logger.info("=" * 60)

    from cosmos_transfer2.experiments.custom.custom_multi_control_experiment import (
        BLUR_DATASET_DIR,
        DEPTH_DATASET_DIR,
        HDMAP_DATASET_DIR,
        TRAINING_CAMERAS,
        TRAIN_SCENE_IDS,
    )
    from cosmos_transfer2.experiments.custom.custom_multi_control_dataset import (
        MultiControlMultiviewDataset,
        collate_fn,
    )
    logger.info("Imports successful")

    logger.info("=" * 60)
    logger.info("Step 2: Load model from checkpoint (DCP format)")
    logger.info(f"Checkpoint: {args.ckpt_path}")
    logger.info(f"Experiment: {args.experiment}")
    logger.info("=" * 60)

    try:
        import importlib
        from pathlib import Path
        from cosmos_transfer2._src.imaginaire.lazy_config import instantiate
        from cosmos_transfer2._src.imaginaire.utils.config_helper import get_config_module, override
        from cosmos_transfer2._src.predict2.checkpointer.dcp import (
            DefaultLoadPlanner,
            ModelWrapper,
            dcp_load_state_dict,
        )
        from torch.distributed.checkpoint import FileSystemReader

        # Step 2a: Load config
        logger.info("Step 2a: Loading config...")
        config_file = "cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/config.py"
        config_module = get_config_module(config_file)
        config = importlib.import_module(config_module).make_config()
        config = override(config, ["--", f"experiment={args.experiment}"])

        # Disable EMA (we'll load EMA weights to regular model)
        config.model.config.ema.enabled = False
        # Disable FSDP for single GPU
        config.model.config.fsdp_shard_size = 1

        config.validate()
        config.freeze()
        logger.info("Config loaded")

        # Step 2b: Instantiate model
        logger.info("Step 2b: Instantiating model...")
        model = instantiate(config.model).cuda()
        model.on_train_start()
        logger.info(f"Model instantiated: {type(model)}")

        # Step 2c: Load DCP checkpoint
        logger.info("Step 2c: Loading DCP checkpoint...")
        ckpt_path = Path(args.ckpt_path)
        model_ckpt_path = ckpt_path / "model"

        if not model_ckpt_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_ckpt_path}")

        # Create model wrapper for loading EMA weights to regular model
        model_wrapper = ModelWrapper(model, load_ema_to_reg=True)
        state_dict = model_wrapper.state_dict()

        # Load using FileSystemReader
        storage_reader = FileSystemReader(str(model_ckpt_path))
        load_planner = DefaultLoadPlanner(allow_partial_load=True)
        dcp_load_state_dict(state_dict, storage_reader, load_planner)
        model_wrapper.load_state_dict(state_dict)

        logger.info("DCP checkpoint loaded successfully!")
        logger.info(f"Model type: {type(model)}")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return

    logger.info("=" * 60)
    logger.info("Step 3: Load one test sample")
    logger.info("=" * 60)

    try:
        test_dataset = MultiControlMultiviewDataset(
            blur_dataset_dir=BLUR_DATASET_DIR,
            depth_dataset_dir=DEPTH_DATASET_DIR,
            hdmap_dataset_dir=HDMAP_DATASET_DIR,
            camera_views=TRAINING_CAMERAS,
            num_video_frames=29,
            fps_downsample_factor=3,
            height=720,
            width=1280,
            exclude_scene_ids=TRAIN_SCENE_IDS,
        )
        logger.info(f"Test dataset has {len(test_dataset)} samples")

        if len(test_dataset) == 0:
            logger.error("No test samples found!")
            return

        # Get first sample
        sample = test_dataset[0]
        batch = collate_fn([sample])
        logger.info(f"Loaded sample, batch keys: {list(batch.keys())}")
    except Exception as e:
        logger.error(f"Failed to load test data: {e}")
        import traceback
        traceback.print_exc()
        return

    logger.info("=" * 60)
    logger.info("Step 4: Move data to GPU")
    logger.info("=" * 60)

    try:
        device = torch.device("cuda")
        uint8_keys = {'video', 'control_input_blur', 'control_input_depth', 'control_input_hdmap_bbox'}

        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                if key in uint8_keys:
                    batch[key] = value.to(device=device)
                else:
                    batch[key] = value.to(**model.tensor_kwargs)
        logger.info("Data moved to GPU")
    except Exception as e:
        logger.error(f"Failed to move data to GPU: {e}")
        import traceback
        traceback.print_exc()
        return

    logger.info("=" * 60)
    logger.info("Step 5: Compute text embeddings")
    logger.info("=" * 60)

    try:
        if hasattr(model, 'inplace_compute_text_embeddings_online'):
            model.inplace_compute_text_embeddings_online(batch)
            logger.info("Text embeddings computed")
        else:
            logger.warning("Model does not have inplace_compute_text_embeddings_online")
    except Exception as e:
        logger.error(f"Failed to compute text embeddings: {e}")
        import traceback
        traceback.print_exc()
        return

    logger.info("=" * 60)
    logger.info("Step 6: Get data and condition")
    logger.info("=" * 60)

    try:
        batch["num_conditional_frames"] = 0
        batch["control_weight"] = 1.0
        raw_data, x0, condition = model.get_data_and_condition(batch)
        logger.info(f"raw_data shape: {raw_data.shape}")
        logger.info(f"x0 shape: {x0.shape}")
    except Exception as e:
        logger.error(f"Failed to get data and condition: {e}")
        import traceback
        traceback.print_exc()
        return

    logger.info("=" * 60)
    logger.info("Step 7: Generate samples")
    logger.info("=" * 60)

    try:
        model.eval()
        with torch.no_grad():
            sample = model.generate_samples_from_batch(
                batch,
                guidance=7.0,
                state_shape=x0.shape[1:],
                n_sample=x0.shape[0],
                num_steps=35,
                is_negative_prompt=False,
            )
        logger.info(f"Generated sample shape: {sample.shape}")
    except Exception as e:
        logger.error(f"Failed to generate samples: {e}")
        import traceback
        traceback.print_exc()
        return

    logger.info("=" * 60)
    logger.info("Step 8: Decode")
    logger.info("=" * 60)

    try:
        if hasattr(model, "decode"):
            generated = model.decode(sample)
            logger.info(f"Decoded video shape: {generated.shape}")
        else:
            generated = sample
            logger.warning("Model does not have decode method")
    except Exception as e:
        logger.error(f"Failed to decode: {e}")
        import traceback
        traceback.print_exc()
        return

    logger.info("=" * 60)
    logger.info("Step 9: Save output")
    logger.info("=" * 60)

    try:
        os.makedirs(args.output_dir, exist_ok=True)

        # Convert to [0, 1] range
        generated_01 = ((generated + 1.0) / 2.0).clamp(0, 1)
        gt_01 = ((raw_data + 1.0) / 2.0).clamp(0, 1)

        from cosmos_transfer2._src.imaginaire.visualize.video import save_img_or_video

        # Save generated video
        save_img_or_video(generated_01[0], f"{args.output_dir}/generated", fps=10)
        logger.info(f"Saved generated video to {args.output_dir}/generated.mp4")

        # Save ground truth video
        save_img_or_video(gt_01[0], f"{args.output_dir}/ground_truth", fps=10)
        logger.info(f"Saved ground truth video to {args.output_dir}/ground_truth.mp4")

        # Save side-by-side comparison
        comparison = torch.cat([gt_01, generated_01], dim=-1)
        save_img_or_video(comparison[0], f"{args.output_dir}/comparison", fps=10)
        logger.info(f"Saved comparison video to {args.output_dir}/comparison.mp4")

    except Exception as e:
        logger.error(f"Failed to save output: {e}")
        import traceback
        traceback.print_exc()
        return

    logger.info("=" * 60)
    logger.info("SUCCESS! Inference test completed.")
    logger.info("=" * 60)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_grad_enabled(False)
    main()
