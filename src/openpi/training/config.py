"""
Training configurations for PiPER dual-arm robot.

Available configs:
- pi0_piper: Pi0 model with 3 cameras
- pi05_piper: Pi0.5 model with 3 cameras
- pi05_piper_4cam: Pi0.5 model with 4 cameras (wide_top as 4th)
- pi05_piper_mask: Pi0.5 model with sweep_mask as 4th image
- pi05_piper_lora: Pi0.5 LoRA fine-tuning
"""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.piper_policy as piper_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats)."""
    assets_dir: str | None = None
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    """Base data configuration."""
    repo_id: str | None = None
    asset_id: str | None = None
    norm_stats: dict[str, _transforms.NormStats] | None = None
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    use_quantile_norm: bool = False
    action_sequence_keys: Sequence[str] = ("actions",)
    prompt_from_task: bool = False
    rlds_data_dir: str | None = None
    action_space: Any | None = None
    filter_dict_path: str | None = None


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    """Base class for data config factories."""
    repo_id: str = tyro.MISSING
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    """Fake data config for debugging."""
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


# =============================================================================
# PiPER Data Config - Unified configuration for dual-arm robot
# =============================================================================
@dataclasses.dataclass(frozen=True)
class LeRobotPiperDataConfig(DataConfigFactory):
    """
    Unified config for PiPER dual-arm robot dataset.

    Supports:
    - 14-dim state (7 per arm: 6 DOF + 1 gripper)
    - 14-dim action (7 per arm)
    - 3 or 4 cameras

    Flags:
    - use_fourth_image: Use wide_top camera as 4th image
    - use_sweep_mask: Use sweep_mask as 4th image (for sweep tasks)
    - extra_delta_transform: Apply delta transform to joint actions
    """

    extra_delta_transform: bool = False
    # Whether to use wide_top as 4th camera
    use_fourth_image: bool = False
    # Whether to use sweep_mask as 4th image (for sweep tasks)
    use_sweep_mask: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Build repack transform based on configuration
        repack_dict = {
            "observation/image": "observation.images.main",
            "observation/wrist_image": "observation.images.secondary_0",
            "observation/right_wrist_image": "observation.images.secondary_1",
            "observation/state": "observation.state",
            "actions": "action",
            "prompt": "task",
        }

        # Add 4th image mapping if needed
        if self.use_fourth_image:
            repack_dict["observation/wide_top_image"] = "observation.images.secondary_2"
        if self.use_sweep_mask:
            repack_dict["observation/sweep_mask"] = "observation.images.sweep_mask"

        repack_transform = _transforms.Group(
            inputs=[_transforms.RepackTransform(repack_dict)]
        )

        # Use PiperInputs with appropriate flags
        data_transforms = _transforms.Group(
            inputs=[piper_policy.PiperInputs(
                model_type=model_config.model_type,
                use_fourth_image=self.use_fourth_image,
                use_sweep_mask=self.use_sweep_mask,
            )],
            outputs=[piper_policy.PiperOutputs()],
        )

        # Optional delta transform for joint actions
        if self.extra_delta_transform:
            # Left arm: joints 0-5 delta, joint 6 (gripper) absolute
            # Right arm: joints 7-12 delta, joint 13 (gripper) absolute
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


# =============================================================================
# Train Config
# =============================================================================
@dataclasses.dataclass(frozen=True)
class TrainConfig:
    """Training configuration."""
    name: tyro.conf.Suppress[str]
    project_name: str = "openpi"
    exp_name: str = tyro.MISSING

    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)
    pytorch_weight_path: str | None = None
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    assets_base_dir: str = "./assets"
    checkpoint_base_dir: str = "./checkpoints"

    seed: int = 42
    batch_size: int = 32
    num_workers: int = 2
    num_train_steps: int = 30_000

    log_interval: int = 100
    save_interval: int = 1000
    keep_period: int | None = 5000

    overwrite: bool = False
    resume: bool = False
    wandb_enabled: bool = True
    policy_metadata: dict[str, Any] | None = None
    fsdp_devices: int = 4

    @property
    def assets_dirs(self) -> pathlib.Path:
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# =============================================================================
# PiPER Training Configs
# =============================================================================
_CONFIGS = [
    # ------------------------------------------------------------------
    # Pi0 - 3 cameras (basic)
    # ------------------------------------------------------------------
    TrainConfig(
        name="pi0_piper",
        model=pi0_config.Pi0Config(
            pi05=False,
            action_horizon=10,
            discrete_state_input=False,
        ),
        data=LeRobotPiperDataConfig(
            repo_id="zeno/piper_dataset",
            base_config=DataConfig(
                prompt_from_task=True,
                action_sequence_keys=("action",),
            ),
            extra_delta_transform=False,
            use_fourth_image=False,
            use_sweep_mask=False,
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
    ),

    # ------------------------------------------------------------------
    # Pi0.5 - 3 cameras (basic)
    # ------------------------------------------------------------------
    TrainConfig(
        name="pi05_piper",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=25,
            discrete_state_input=False,
            max_token_len=180,
        ),
        data=LeRobotPiperDataConfig(
            repo_id="zeno/piper_dataset",
            base_config=DataConfig(
                prompt_from_task=True,
                action_sequence_keys=("action",),
            ),
            extra_delta_transform=False,
            use_fourth_image=False,
            use_sweep_mask=False,
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
        num_train_steps=30_000,
    ),

    # ------------------------------------------------------------------
    # Pi0.5 - 4 cameras (with wide_top)
    # ------------------------------------------------------------------
    TrainConfig(
        name="pi05_piper_4cam",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=25,
            discrete_state_input=False,
            max_token_len=200,  
        ),
        data=LeRobotPiperDataConfig(
            repo_id="Anlorla/sweep_to_E",
            base_config=DataConfig(
                prompt_from_task=True,
                action_sequence_keys=("action",),
            ),
            extra_delta_transform=False,
            use_fourth_image=True,  # Enable 4th camera
            use_sweep_mask=False,
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1800,
            peak_lr=5e-5,
            decay_steps=3000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=6000,
    ),

    # ------------------------------------------------------------------
    # Pi0.5 - with sweep_mask (for sweep tasks)
    # ------------------------------------------------------------------
    TrainConfig(
        name="pi05_piper_mask",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=25,
            discrete_state_input=False,
            max_token_len=200,  # Increased for 4 images
        ),
        data=LeRobotPiperDataConfig(
            repo_id="zeno/sweep_dataset",
            base_config=DataConfig(
                prompt_from_task=True,
                action_sequence_keys=("action",),
            ),
            extra_delta_transform=False,
            use_fourth_image=False,
            use_sweep_mask=True,  # Enable sweep_mask
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
        num_train_steps=30_000,
    ),

    # ------------------------------------------------------------------
    # Pi0.5 - LoRA fine-tuning (low memory)
    # ------------------------------------------------------------------
    TrainConfig(
        name="pi05_piper_lora",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotPiperDataConfig(
            repo_id="zeno/piper_dataset",
            base_config=DataConfig(
                prompt_from_task=True,
                action_sequence_keys=("action",),
            ),
            extra_delta_transform=False,
            use_fourth_image=False,
            use_sweep_mask=False,
        ),
        batch_size=16,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=8_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=14,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,  # No EMA for LoRA
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=10_000,
    ),

    # ------------------------------------------------------------------
    # Debug config
    # ------------------------------------------------------------------
    TrainConfig(
        name="debug",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        save_interval=100,
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
]

# Validate config names are unique
if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")
    return _CONFIGS_DICT[config_name]
