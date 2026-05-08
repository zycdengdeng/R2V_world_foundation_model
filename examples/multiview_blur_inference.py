# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Per-camera blur ControlNet inference for multi-camera scenes.

Cosmos-Transfer2.5-2B publishes a multi-view ControlNet head only for hdmap_bbox
(``nvidia/Cosmos-Transfer2.5-2B/auto/multiview``); the blur ControlNet
(``nvidia/Cosmos-Transfer2.5-2B/general/blur``) is single-view. This script
runs the single-view blur head once per camera defined in a multi-camera spec
JSON, loading the model only once. Cross-camera consistency is not enforced by
the model.

Spec JSON schema (paths in the spec are resolved relative to the spec file)::

    {
      "name": "scene_001",
      "prompt": "A driving scene captured from a car-mounted camera ...",
      "cameras": {
        "front_wide": {
          "video_path": "input_videos/front_wide.mp4",
          "control_path": "blur_videos/front_wide_blur.mp4"
        },
        "cross_right": { "video_path": "input_videos/cross_right.mp4" }
      },
      "blur": { "control_weight": 1.0, "preset_blur_strength": "medium" },
      "num_conditional_frames": 1,
      "guidance": 3,
      "num_steps": 35,
      "seed": 2025
    }

If a camera entry omits ``control_path`` the blur map is generated on-the-fly
from ``video_path`` using the bilateral Gaussian blur filter (preset strength
controlled by ``blur.preset_blur_strength``).

Output: one MP4 per camera at ``<output_dir>/<scene_name>__<camera>.mp4``.
"""

import json
import os
from pathlib import Path
from typing import Annotated, Literal

import pydantic
import tyro
from cosmos_oss.init import cleanup_environment, init_environment, init_output_dir

from cosmos_transfer2._src.imaginaire.utils import log
from cosmos_transfer2.config import (
    DEFAULT_NEGATIVE_PROMPT,
    BlurConfig,
    CommonSetupArguments,
    Guidance,
    InferenceArguments,
    ModelKey,
    ModelVariant,
    ResolvedFilePath,
    get_model_literal,
    handle_tyro_exception,
    is_rank0,
)

_BLUR_MODEL_KEY = ModelKey(variant=ModelVariant.VIS)


class BlurSetupArguments(CommonSetupArguments):
    """Setup arguments locked to the single-view blur ControlNet."""

    # pyrefly: ignore  # invalid-annotation
    model: get_model_literal([ModelVariant.VIS]) = _BLUR_MODEL_KEY.name


class _CameraEntry(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")

    video_path: ResolvedFilePath
    """Path to the input RGB video for this camera."""
    control_path: ResolvedFilePath | None = None
    """Optional path to a pre-computed blur video. If omitted the blur is
    generated on-the-fly from ``video_path``."""


class MultiViewBlurSpec(pydantic.BaseModel):
    """Schema for a multi-camera blur inference sample."""

    model_config = pydantic.ConfigDict(extra="forbid")

    name: str
    prompt: str | None = None
    prompt_path: ResolvedFilePath | None = None
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT
    cameras: dict[str, _CameraEntry]
    blur: BlurConfig = pydantic.Field(default_factory=BlurConfig)
    num_conditional_frames: Literal[0, 1, 2] = 1
    guidance: Guidance = 3
    num_steps: pydantic.PositiveInt = 35
    seed: int = 2025
    resolution: str = "720"
    show_control_condition: bool = False
    show_input: bool = False
    keep_input_resolution: bool = True

    @pydantic.model_validator(mode="after")
    def _validate(self) -> "MultiViewBlurSpec":
        if self.prompt is None and self.prompt_path is not None:
            self.prompt = Path(self.prompt_path).read_text().strip()
        if not self.prompt:
            raise ValueError("Either 'prompt' or 'prompt_path' must be provided.")
        if not self.cameras:
            raise ValueError("'cameras' must be a non-empty mapping.")
        return self

    @classmethod
    def from_file(cls, path: Path) -> "MultiViewBlurSpec":
        # Resolve any relative paths inside the spec relative to the spec file.
        cwd = os.getcwd()
        os.chdir(path.parent)
        try:
            return cls.model_validate(json.loads(path.read_text()))
        finally:
            os.chdir(cwd)

    def to_per_camera_inference_args(self) -> list[InferenceArguments]:
        samples: list[InferenceArguments] = []
        for camera_name, entry in self.cameras.items():
            # control_path is per-camera by construction; if a top-level
            # blur.control_path was supplied it would be wrong to reuse it
            # across cameras, so we always replace it with the per-camera value.
            blur = self.blur.model_copy(update={"control_path": entry.control_path})
            samples.append(
                InferenceArguments(
                    name=f"{self.name}__{camera_name}",
                    prompt=self.prompt,
                    negative_prompt=self.negative_prompt,
                    video_path=entry.video_path,
                    vis=blur,
                    num_conditional_frames=self.num_conditional_frames,
                    guidance=self.guidance,
                    num_steps=self.num_steps,
                    seed=self.seed,
                    resolution=self.resolution,
                    show_control_condition=self.show_control_condition,
                    show_input=self.show_input,
                    keep_input_resolution=self.keep_input_resolution,
                )
            )
        return samples


class Args(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    input_files: Annotated[list[Path], tyro.conf.arg(aliases=("-i",))]
    """One or more multi-view blur spec JSON files. The model is loaded once
    and all cameras across all files are run sequentially."""
    setup: BlurSetupArguments
    """Setup arguments. These can only be provided via CLI."""


def main(args: Args) -> None:
    inference_samples: list[InferenceArguments] = []
    for path in args.input_files:
        spec = MultiViewBlurSpec.from_file(path)
        per_camera = spec.to_per_camera_inference_args()
        log.info(f"{path.name}: expanded {spec.name} into {len(per_camera)} per-camera samples.")
        inference_samples.extend(per_camera)

    if not inference_samples:
        raise SystemExit("No camera samples derived from input specs.")

    init_output_dir(args.setup.output_dir, profile=args.setup.profile)

    from cosmos_transfer2.inference import Control2WorldInference

    inference = Control2WorldInference(args.setup, batch_hint_keys=["vis"])
    inference.generate(inference_samples, output_dir=args.setup.output_dir)


if __name__ == "__main__":
    init_environment()

    try:
        args = tyro.cli(
            Args,
            description=__doc__,
            console_outputs=is_rank0(),
            config=(tyro.conf.OmitArgPrefixes,),
        )
    except Exception as e:
        handle_tyro_exception(e)
    # pyrefly: ignore  # unbound-name
    main(args)

    cleanup_environment()
