import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_piper_example() -> dict:
    """Creates a random input example for the Piper policy."""
    return {
        "observation/state": np.random.rand(14),  # 14-dim: dual-arm joint positions
        "observation/ee_pose": np.random.rand(14),  # 14-dim: dual-arm end effector poses
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/right_wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/sweep_mask": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),  # sweep mask as 4th image
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class PiperInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format for PiPER dual-arm robot.
    It supports concatenating end-effector pose with joint state, and optionally includes sweep_mask as 4th image.

    For PiPER dataset:
    - observation/state: 14-dim joint positions (7 per arm: 6 DOF + 1 gripper)
    - observation/ee_pose: 14-dim end effector poses (7 per arm: x,y,z,qx,qy,qz,qw)
    - observation/sweep_mask: (optional) RGB image mask for sweep blocks task
    """

    # Determines which model will be used.
    model_type: _model.ModelType

    # Whether to concatenate end-effector pose with state
    # If True: state will be [joint_state, ee_pose] (28-dim)
    # If False: state will be only joint_state (14-dim)
    concat_ee_pose: bool = True

    # Whether to use only ee_pose (ignoring joint state)
    # If True: state will be only ee_pose (14-dim)
    # Note: concat_ee_pose must be False if this is True
    use_only_ee_pose: bool = False

    # Whether to include sweep_mask as the 4th image input
    # If True, expects "observation/sweep_mask" in the data
    # The mask will be treated as an RGB image and processed by the vision encoder
    # Named with "sweep_" prefix to avoid confusion with attention masks
    use_sweep_mask: bool = False

    def __call__(self, data: dict) -> dict:
        # Parse the 3 standard camera images to uint8 (H,W,C) format
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])
        right_wrist_image = _parse_image(data["observation/right_wrist_image"])

        # Prepare state vector based on configuration
        if self.use_only_ee_pose:
            # Use only end-effector pose
            state = data["observation/ee_pose"]
        elif self.concat_ee_pose:
            # Concatenate joint state and end-effector pose
            joint_state = data["observation/state"]
            ee_pose = data["observation/ee_pose"]
            state = np.concatenate([joint_state, ee_pose], axis=-1)
        else:
            # Use only joint state
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

        # Optionally add sweep_mask as 4th image
        # This is used for sweep blocks task where the mask provides spatial guidance
        # The mask is treated as an RGB image and processed by the vision encoder (196 tokens)
        if self.use_sweep_mask:
            sweep_mask_image = _parse_image(data["observation/sweep_mask"])
            inputs["image"]["sweep_mask"] = sweep_mask_image
            # Always use sweep_mask when provided (set mask to True)
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
    This class is used to convert outputs from the model back to the dataset specific format.
    Used for inference only.

    For PiPER dual-arm robot, we return 14-dim actions (7 per arm).
    """

    def __call__(self, data: dict) -> dict:
        # Return the first 14 actions (dual-arm: 7 per arm)
        return {"actions": np.asarray(data["actions"][:, :14])}