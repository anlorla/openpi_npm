"""
Configuration for PiPER with mask image support.

This file contains the data config factory for training pi05 with mask as 4th image.
Import this in your training script or add to the main config.py file.
"""

import dataclasses
import pathlib
from collections.abc import Sequence
from typing_extensions import override

import openpi.models.model as _model
import openpi.policies.piper_policy_mask as piper_policy_mask
import openpi.training.config as _base_config
import openpi.transforms as _transforms


@dataclasses.dataclass(frozen=True)
class LeRobotPiperMaskDataConfig(_base_config.DataConfigFactory):
    """
    Config for PiPER dual-arm robot dataset with end-effector pose and mask image.

    This config extends LeRobotPiperDataConfig to support mask as a 4th image input.

    Supports:
    - observation/state: 14-dim joint positions (7 per arm: 6 DOF + 1 gripper)
    - observation/ee_pose: 14-dim end effector poses (7 per arm: x,y,z,qx,qy,qz,qw)
    - action: 14-dim actions (7 per arm)
    - 4 cameras (main + left wrist + right wrist + mask)
    """

    extra_delta_transform: bool = False
    # Whether to concatenate end-effector pose with joint state (28-dim) or only use joint state (14-dim)
    concat_ee_pose: bool = True

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> _base_config.DataConfig:
        # Repack transform maps dataset keys to expected keys
        # Added mask_image mapping
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "observation.images.main",
                        "observation/wrist_image": "observation.images.secondary_0",
                        "observation/right_wrist_image": "observation.images.secondary_1",
                        "observation/mask_image": "observation.images.mask",  # Add mask image
                        "observation/state": "observation.state",
                        "observation/ee_pose": "observation.ee_pose",
                        "actions": "action",
                        "prompt": "task",
                    }
                )
            ]
        )

        # Use PiperMaskInputs to handle joint state + end-effector pose + mask image
        # PiperMaskOutputs handles dual-arm 14-dim actions
        data_transforms = _transforms.Group(
            inputs=[piper_policy_mask.PiperMaskInputs(
                model_type=model_config.model_type,
                concat_ee_pose=self.concat_ee_pose,
            )],
            outputs=[piper_policy_mask.PiperMaskOutputs()],
        )

        # For dual-arm robot: apply delta transform to all joint actions (14 dims total)
        # Each arm has 7 joints: 6 regular joints + 1 gripper
        # We apply delta to joints but keep gripper absolute for both arms
        if self.extra_delta_transform:
            # Left arm: joints 0-5 delta, joint 6 (gripper) absolute
            # Right arm: joints 7-12 delta, joint 13 (gripper) absolute
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = _base_config.ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


# Example training config using the mask data config
def create_pi05_npm_mask_config():
    """
    Create a training config for pi05 with mask image support.

    Usage:
        from openpi.training.config_mask import create_pi05_npm_mask_config
        config = create_pi05_npm_mask_config()
    """
    from openpi.models import pi0_config
    from openpi.training import optimizer as _optimizer
    from openpi.training import weight_loaders

    return _base_config.TrainConfig(
        name="pi05_npm_mask",
        # Dual-arm robot with 14-dim actions (7 per arm) and potentially 28-dim state if concat_ee_pose=True
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=25,
            discrete_state_input=False,
            max_token_len=180,
        ),
        data=LeRobotPiperMaskDataConfig(
            repo_id="your_hf_username/your_dataset_with_mask",  # Replace with your dataset
            base_config=_base_config.DataConfig(
                prompt_from_task=True,
                action_sequence_keys=("action",),
            ),
            extra_delta_transform=False,
            concat_ee_pose=True,  # Use both joint state and ee_pose (28-dim state)
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path=None,
        num_train_steps=30_000,
    )
