"""
PiPER policy for dual-arm robot.

Supports:
- 3 or 4 camera inputs (main + left wrist + right wrist + optional 4th image)
- 14-dim state (joint positions only)
- 14-dim action (7 per arm)
- Optional sweep_mask as 4th image for sweep tasks
"""

import dataclasses
from typing import Literal

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_piper_example(use_fourth_image: bool = False, use_sweep_mask: bool = False) -> dict:
    """Creates a random input example for the Piper policy."""
    example = {
        "observation/state": np.random.rand(14),  # 14-dim: dual-arm joint positions
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/right_wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }
    if use_fourth_image:
        example["observation/wide_top_image"] = np.random.randint(256, size=(224, 224, 3), dtype=np.uint8)
    if use_sweep_mask:
        example["observation/sweep_mask"] = np.random.randint(256, size=(224, 224, 3), dtype=np.uint8)
    return example


def _parse_image(image) -> np.ndarray:
    """Parse image to uint8 format with shape (H, W, C)."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class PiperInputs(transforms.DataTransformFn):
    """
    Transform inputs for PiPER dual-arm robot.

    For PiPER dataset:
    - observation/state: 14-dim joint positions (7 per arm: 6 DOF + 1 gripper)
    - observation/image: main camera (realsense_top)
    - observation/wrist_image: left wrist camera (fisheye_left)
    - observation/right_wrist_image: right wrist camera (fisheye_right)
    - observation/wide_top_image: (optional) wide top camera as 4th image
    - observation/sweep_mask: (optional) sweep mask as 4th image for sweep tasks
    """

    # Determines which model will be used.
    model_type: _model.ModelType

    # Whether to include wide_top_image as the 4th image input
    # If True, expects "observation/wide_top_image" in the data
    use_fourth_image: bool = False

    # Whether to include sweep_mask as the 4th image input
    # If True, expects "observation/sweep_mask" in the data
    # Note: sweep_mask and wide_top are mutually exclusive as 4th image
    use_sweep_mask: bool = False

    def __call__(self, data: dict) -> dict:
        # Parse the 3 standard camera images to uint8 (H,W,C) format
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])
        right_wrist_image = _parse_image(data["observation/right_wrist_image"])

        # Use only joint state (14-dim)
        state = data["observation/state"]

        # Create inputs dict with base 3 images
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        # Optionally add 4th image (wide_top camera)
        if self.use_fourth_image:
            wide_top_image = _parse_image(data["observation/wide_top_image"])
            inputs["image"]["wide_top_0_rgb"] = wide_top_image
            inputs["image_mask"]["wide_top_0_rgb"] = np.True_

        # Optionally add sweep_mask as 4th image (for sweep tasks)
        if self.use_sweep_mask:
            sweep_mask_image = _parse_image(data["observation/sweep_mask"])
            inputs["image"]["sweep_mask"] = sweep_mask_image
            inputs["image_mask"]["sweep_mask"] = np.True_

        # Actions are only available during training
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # Pass the prompt (language instruction) to the model
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class PiperOutputs(transforms.DataTransformFn):
    """
    Convert model outputs back to dataset format for PiPER robot.
    Used for inference only.

    For PiPER dual-arm robot, we return 14-dim actions (7 per arm).
    """

    def __call__(self, data: dict) -> dict:
        # Return the first 14 actions (dual-arm: 7 per arm)
        return {"actions": np.asarray(data["actions"][:, :14])}
