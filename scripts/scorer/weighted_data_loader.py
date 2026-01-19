#!/usr/bin/env python3
"""
weighted_data_loader.py - Data loader with episode index support for weighted training.

This module extends the standard data loader to include episode_index in each batch,
enabling per-trajectory weighting during training.
"""

from collections.abc import Iterator
import logging
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as _transforms

logger = logging.getLogger(__name__)


class DatasetWithEpisodeIndex(Dataset):
    """Wrapper that adds episode_index to each sample."""

    def __init__(self, base_dataset: Dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        sample = self.base_dataset[idx]

        # Try to get episode_index from sample
        # LeRobot datasets typically have 'episode_index' field
        if isinstance(sample, dict):
            episode_idx = sample.get("episode_index", idx)
            if hasattr(episode_idx, "item"):
                episode_idx = episode_idx.item()
            sample["_episode_index"] = int(episode_idx)

        return sample


class TransformedDatasetWithMeta(Dataset):
    """Transform dataset while preserving episode metadata."""

    def __init__(
        self,
        dataset: Dataset,
        transforms: list[_transforms.DataTransformFn],
    ):
        self.dataset = dataset
        self.transform = _transforms.compose(transforms)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        # Extract episode index before transform
        episode_index = sample.pop("_episode_index", idx)

        # Apply transforms
        transformed = self.transform(sample)

        # Add back episode index
        transformed["_episode_index"] = episode_index

        return transformed


def create_weighted_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
) -> Iterator[Tuple[_model.Observation, _model.Actions, jnp.ndarray]]:
    """Create a data loader that returns (observation, actions, episode_indices).

    This is used for weighted training where we need to look up per-trajectory weights.
    """
    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

    data_config = config.data.create(config.assets_dirs, config.model)
    logger.info(f"data_config: {data_config}")

    # Get repo_ids
    repo_ids = data_config.repo_ids
    if repo_ids is None:
        repo_id = data_config.repo_id
        if repo_id is None or repo_id == "fake":
            raise ValueError("Weighted training requires a real dataset")
        repo_ids = [repo_id]

    # Load datasets
    datasets = []
    all_tasks = {}

    for repo_id in repo_ids:
        logger.info(f"Loading dataset from: {repo_id}")
        dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
        dataset = lerobot_dataset.LeRobotDataset(
            repo_id,
            delta_timestamps={
                key: [t / dataset_meta.fps for t in range(config.model.action_horizon)]
                for key in data_config.action_sequence_keys
            },
        )
        all_tasks.update(dataset_meta.tasks)

        # Wrap to include episode index
        dataset = DatasetWithEpisodeIndex(dataset)
        datasets.append(dataset)
        logger.info(f"  - Loaded {len(dataset)} samples from {repo_id}")

    # Concatenate if multiple
    if len(datasets) == 1:
        combined_dataset = datasets[0]
    else:
        combined_dataset = torch.utils.data.ConcatDataset(datasets)
        logger.info(f"Combined {len(repo_ids)} datasets, total samples: {len(combined_dataset)}")

    # Apply prompt transform if needed
    if data_config.prompt_from_task:
        combined_dataset = _data_loader.TransformedDataset(
            combined_dataset, [_transforms.PromptFromLeRobotTask(all_tasks)]
        )

    # Apply all transforms while preserving episode index
    norm_stats = data_config.norm_stats or {}
    combined_dataset = TransformedDatasetWithMeta(
        combined_dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )

    # Create data loader
    local_batch_size = config.batch_size // jax.process_count()

    def collate_fn(items):
        """Collate batch and separate episode indices."""
        episode_indices = [item.pop("_episode_index") for item in items]
        batch = jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)
        batch["_episode_indices"] = np.array(episode_indices)
        return batch

    torch_loader = DataLoader(
        combined_dataset,
        batch_size=local_batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        drop_last=True,
    )

    # Set up sharding
    if sharding is None:
        sharding = jax.sharding.NamedSharding(
            jax.sharding.Mesh(jax.devices(), ("B",)),
            jax.sharding.PartitionSpec("B"),
        )

    # Iterate and yield (observation, actions, episode_indices)
    while True:
        for batch in torch_loader:
            episode_indices = batch.pop("_episode_indices")
            sharded_batch = jax.tree.map(
                lambda x: jax.make_array_from_process_local_data(sharding, x), batch
            )
            observation = _model.Observation.from_dict(sharded_batch)
            actions = sharded_batch["actions"]
            episode_indices = jax.make_array_from_process_local_data(sharding, episode_indices)

            yield observation, actions, episode_indices


class TrajectoryWeightLookup:
    """Lookup table for trajectory weights."""

    def __init__(self, weights_path: str):
        """Load weights from npz file."""
        data = np.load(weights_path, allow_pickle=True)
        self.episode_ids = data["episode_ids"]
        self.weights = data["weights"]

        # Create lookup dict: episode_id (int or str) -> weight
        self.lookup = {}
        for eid, w in zip(self.episode_ids, self.weights):
            self.lookup[int(eid) if str(eid).isdigit() else str(eid)] = float(w)
            self.lookup[str(eid)] = float(w)

        logger.info(f"Loaded {len(self.lookup)} trajectory weights")

    def get_weights(self, episode_indices: np.ndarray) -> np.ndarray:
        """Get weights for a batch of episode indices."""
        return np.array([self.lookup.get(int(idx), 1.0) for idx in episode_indices])

    def get_weights_jax(
        self,
        episode_indices: jnp.ndarray,
        step: int = 0,
        warmup_steps: int = 0,
    ) -> jnp.ndarray:
        """Get JAX array of weights with optional warmup."""
        # Convert to numpy for lookup
        indices_np = np.asarray(episode_indices)
        weights = self.get_weights(indices_np)

        # Apply warmup
        if warmup_steps > 0 and step < warmup_steps:
            warmup_ratio = step / warmup_steps
            weights = 1.0 + warmup_ratio * (weights - 1.0)

        return jnp.array(weights)
