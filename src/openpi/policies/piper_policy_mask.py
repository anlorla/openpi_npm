"""
PiPER policy with sweep_mask image support for sweep blocks training.

This extends the original piper_policy.py to support sweep_mask as a 4th image input.
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_piper_sweep_example() -> dict:
    """Creates a random input example for the Piper policy with sweep_mask."""
    return {
        "observation/state": np.random.rand(14),  # 14-dim: dual-arm joint positions
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/right_wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/sweep_mask": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),  # sweep_mask as 4th image
        "prompt": "sweep blocks to target",
    }


def _parse_image(image) -> np.ndarray:
    """Parse image to uint8 format with shape (H, W, C)."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class PiperSweepInputs(transforms.DataTransformFn):
    """
    Transform inputs for PiPER dual-arm robot with sweep_mask support.

    This extends PiperInputs to add sweep_mask as the 4th image input for sweep blocks task.
    The sweep_mask is treated as an RGB image and processed by the vision encoder (196 tokens).

    For PiPER sweep blocks dataset:
    - observation/state: 14-dim joint positions (7 per arm: 6 DOF + 1 gripper)
    - observation/image: main camera image
    - observation/wrist_image: left wrist camera image
    - observation/right_wrist_image: right wrist camera image
    - observation/sweep_mask: spatial mask for sweep task (4th image)

    Note: This version does NOT use end-effector pose (ee_pose).
    """

    # Determines which model will be used.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Parse all 4 images to uint8 (H,W,C) format
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])
        right_wrist_image = _parse_image(data["observation/right_wrist_image"])

        # Parse sweep_mask - treat as 4th camera view
        # This provides spatial guidance for the sweep task
        sweep_mask_image = _parse_image(data["observation/sweep_mask"])

        # Use joint state only (no ee_pose for sweep blocks task)
        state = data["observation/state"]

        # Create inputs dict with 4 images including sweep_mask
        # The sweep_mask will be encoded by Vision Encoder into 196 tokens
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": right_wrist_image,
                "sweep_mask": sweep_mask_image,  # Add sweep_mask as 4th image
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
                "sweep_mask": np.True_,  # Always use sweep_mask when provided
            },
        }

        # Actions are only available during training
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # Pass the prompt (language instruction) to the model
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class PiperSweepOutputs(transforms.DataTransformFn):
    """
    Convert model outputs back to dataset format for PiPER sweep blocks task.
    Used for inference only.

    For PiPER dual-arm robot, we return 14-dim actions (7 per arm).
    """

    def __call__(self, data: dict) -> dict:
        # Return the first 14 actions (dual-arm: 7 per arm)
        return {"actions": np.asarray(data["actions"][:, :14])}
