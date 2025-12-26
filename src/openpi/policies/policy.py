from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
            # RTC support for PyTorch models
            if hasattr(model, 'realtime_action'):
                self._realtime_action = model.realtime_action
            else:
                self._realtime_action = None
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)
            # RTC support for JAX models
            if hasattr(model, 'realtime_action'):
                # Mark string parameters as static for JIT compilation
                self._realtime_action = nnx_utils.module_jit(
                    model.realtime_action,
                    static_argnames=['prefix_attention_schedule']
                )
            else:
                self._realtime_action = None

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    def realtime_action(
        self,
        obs: dict,
        *,
        num_flow_steps: int,
        prev_action_chunk: np.ndarray,
        inference_delay: int,
        execute_horizon: int,
        prefix_attention_schedule: str = "exp",
        max_guidance_weight: float = 5.0,
    ) -> dict:
        """
        Real-time action generation with guided inference for action chunking.

        Implements RTC (Real-Time Chunking) from "Real-Time Execution of Action Chunking
        Flow Policies" (Black et al., 2025). Uses VJP-based guidance to align new action
        chunks with previously executed actions in the prefix region.

        Args:
            obs: Observation dictionary containing images, state, and prompt
            num_flow_steps: Number of denoising steps (n in paper)
            prev_action_chunk: Previous action chunk [H, action_dim]
            inference_delay: Committed prefix length (d in paper)
            execute_horizon: Number of actions executed since last inference (s in paper)
            prefix_attention_schedule: Weight decay schedule ("linear", "exp", "ones", "zeros")
            max_guidance_weight: Maximum guidance strength (β in paper)

        Returns:
            Dictionary with "actions" key containing generated action chunk [H, action_dim]
        """
        if self._realtime_action is None:
            raise NotImplementedError(
                "Model does not support realtime_action. "
                "Ensure your model (e.g., Pi0Model) implements the realtime_action method."
            )

        # Make a copy since transformations may modify the inputs in place
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)

        # Calculate prefix_attention_horizon from execute_horizon
        # In the paper: prefix_attention_horizon = H - s
        prefix_attention_horizon = prev_action_chunk.shape[0] - execute_horizon

        if not self._is_pytorch_model:
            # JAX model
            # Add batch dimension and convert to jax.Array
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            prev_action_chunk_batched = jnp.asarray(prev_action_chunk)[np.newaxis, ...]

            self._rng, sample_rng = jax.random.split(self._rng)

            observation = _model.Observation.from_dict(inputs)
            start_time = time.monotonic()

            actions = self._realtime_action(
                sample_rng,
                observation,
                num_flow_steps=num_flow_steps,
                prev_action_chunk=prev_action_chunk_batched,
                inference_delay=inference_delay,
                prefix_attention_horizon=prefix_attention_horizon,
                prefix_attention_schedule=prefix_attention_schedule,
                max_guidance_weight=max_guidance_weight,
            )

            model_time = time.monotonic() - start_time

            # Remove batch dimension
            outputs = {
                "state": inputs["state"],
                "actions": np.asarray(actions[0, ...]),  # [H, 32] → will be sliced by output_transform
            }
        else:
            # PyTorch model
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            prev_action_chunk_batched = torch.from_numpy(prev_action_chunk).to(self._pytorch_device)[None, ...]

            observation = _model.Observation.from_dict(inputs)
            start_time = time.monotonic()

            with torch.no_grad():
                actions = self._realtime_action(
                    self._pytorch_device,
                    observation,
                    num_flow_steps=num_flow_steps,
                    prev_action_chunk=prev_action_chunk_batched,
                    inference_delay=inference_delay,
                    prefix_attention_horizon=prefix_attention_horizon,
                    prefix_attention_schedule=prefix_attention_schedule,
                    max_guidance_weight=max_guidance_weight,
                )

            model_time = time.monotonic() - start_time

            # Note: Pi05 models output 32-dim actions for compatibility with various robots
            # Dimension slicing (32→14 for dual-arm) is handled by output_transform
            actions_np = np.asarray(actions[0, ...].detach().cpu())  # [H, 32]

            outputs = {
                "state": inputs["state"],
                "actions": actions_np,  # [H, 32] → will be sliced by output_transform
            }

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
