#!/usr/bin/env python3
"""
dump_episode_features.py - Export episode features for tiny scorer training.

This script extracts observation embeddings and actions from LeRobot dataset,
grouped by episode, for training the tiny trajectory scorer.

Usage:
    python scripts/scorer/dump_episode_features.py \
        --repo-id zeno/piper_dataset \
        --output-dir ./scorer_data \
        --checkpoint-path ./checkpoints/pi05_piper/exp_name

Output format (per episode):
    episode_{idx}.npz:
        - obs_emb: [T, D_emb]  # Frozen encoder embeddings
        - actions: [T, H*action_dim]  # Flattened action chunks
        - state: [T, state_dim]  # Joint states
        - episode_id: str
        - task: str
"""

import argparse
import logging
import os
from pathlib import Path
from collections import defaultdict

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Export episode features for scorer training")
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="LeRobot dataset repo_id (e.g., zeno/piper_dataset)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./scorer_data",
        help="Output directory for episode .npz files"
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Path to pi0/pi05 checkpoint for extracting embeddings (optional, uses random init if not provided)"
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="pi05_piper",
        help="Config name for model initialization"
    )
    parser.add_argument(
        "--action-horizon",
        type=int,
        default=10,
        help="Action horizon (chunk size)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding extraction"
    )
    parser.add_argument(
        "--use-simple-features",
        action="store_true",
        help="Use simple features (state + flattened images) instead of model embeddings"
    )
    return parser.parse_args()


def load_dataset_by_episode(repo_id: str, action_horizon: int):
    """Load LeRobot dataset and group samples by episode."""
    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

    logger.info(f"Loading dataset: {repo_id}")
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    dataset = lerobot_dataset.LeRobotDataset(
        repo_id,
        delta_timestamps={
            "action": [t / dataset_meta.fps for t in range(action_horizon)]
        },
    )

    logger.info(f"Dataset has {len(dataset)} samples, {dataset_meta.total_episodes} episodes")
    logger.info(f"Tasks: {dataset_meta.tasks}")

    # Group samples by episode
    episodes = defaultdict(list)
    for idx in tqdm(range(len(dataset)), desc="Grouping by episode"):
        sample = dataset[idx]
        ep_idx = int(sample.get("episode_index", sample.get("index", idx)))
        episodes[ep_idx].append((idx, sample))

    # Sort samples within each episode by index
    for ep_idx in episodes:
        episodes[ep_idx].sort(key=lambda x: x[0])

    return dataset, episodes, dataset_meta


def extract_simple_features(sample: dict) -> np.ndarray:
    """Extract simple features without using the model encoder.

    Features: flattened images (downsampled) + state
    """
    features = []

    # Get state
    if "observation.state" in sample:
        state = np.asarray(sample["observation.state"]).flatten()
        features.append(state)

    # Get flattened/downsampled image features
    for key in sample:
        if "images" in key and "observation" in key:
            img = np.asarray(sample[key])
            # Downsample to 32x32 and flatten
            if img.ndim == 3:
                from skimage.transform import resize
                img_small = resize(img, (32, 32, 3), anti_aliasing=True)
                features.append(img_small.flatten())

    return np.concatenate(features) if features else np.zeros(128)


def create_model_and_extract_embeddings(
    config_name: str,
    checkpoint_path: str | None,
    samples: list[dict],
    batch_size: int = 32,
) -> np.ndarray:
    """Create model and extract prefix embeddings for samples.

    Uses the frozen embed_prefix() from pi0/pi05 model.
    """
    from openpi.training import config as _config
    from openpi.models import model as _model
    from openpi.training import checkpoints as _checkpoints
    import openpi.transforms as _transforms
    from openpi.policies import piper_policy

    # Load config
    config = _config.get_config(config_name)
    model_config = config.model

    # Create data transforms
    data_config = config.data.create(config.assets_dirs, model_config)
    transforms = _transforms.compose([
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,
        *data_config.model_transforms.inputs,
    ])

    # Initialize model
    rng = jax.random.key(0)
    model = model_config.create(rng)

    # Load checkpoint if provided
    if checkpoint_path:
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        params = _model.restore_params(checkpoint_path)
        model = model_config.load(params)

    model.eval()

    # Extract embeddings in batches
    all_embeddings = []

    for i in tqdm(range(0, len(samples), batch_size), desc="Extracting embeddings"):
        batch_samples = samples[i:i+batch_size]

        # Transform samples
        transformed = [transforms(s) for s in batch_samples]

        # Stack into batch
        batch = jax.tree.map(
            lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0),
            *transformed
        )

        # Convert to Observation
        obs = _model.Observation.from_dict(batch)

        # Get prefix embeddings (frozen)
        prefix_tokens, prefix_mask, _ = model.embed_prefix(obs)

        # Pool over sequence dimension (mean pooling)
        # prefix_tokens: [B, seq_len, emb_dim]
        # prefix_mask: [B, seq_len]
        masked_tokens = prefix_tokens * prefix_mask[:, :, None]
        emb = jnp.sum(masked_tokens, axis=1) / (jnp.sum(prefix_mask, axis=1, keepdims=True) + 1e-8)

        all_embeddings.append(np.asarray(emb))

    return np.concatenate(all_embeddings, axis=0)


def main():
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset grouped by episode
    dataset, episodes, dataset_meta = load_dataset_by_episode(
        args.repo_id,
        args.action_horizon
    )

    logger.info(f"Found {len(episodes)} episodes")

    # Process each episode
    for ep_idx, ep_samples in tqdm(episodes.items(), desc="Processing episodes"):
        indices, samples = zip(*ep_samples)
        samples = list(samples)

        # Get task info
        task_idx = samples[0].get("task_index", 0)
        task = dataset_meta.tasks.get(int(task_idx), "unknown")

        # Extract features
        if args.use_simple_features:
            # Simple approach: just use state and downsampled images
            obs_emb = np.stack([extract_simple_features(s) for s in samples])
        else:
            # Use model encoder for embeddings
            try:
                obs_emb = create_model_and_extract_embeddings(
                    args.config_name,
                    args.checkpoint_path,
                    samples,
                    args.batch_size
                )
            except Exception as e:
                logger.warning(f"Failed to extract model embeddings: {e}, using simple features")
                obs_emb = np.stack([extract_simple_features(s) for s in samples])

        # Extract actions (flatten horizon)
        actions_list = []
        for s in samples:
            action = np.asarray(s["action"])
            if action.ndim == 2:  # [H, action_dim]
                action = action.flatten()  # [H * action_dim]
            actions_list.append(action)
        actions = np.stack(actions_list)

        # Extract states
        states = np.stack([np.asarray(s["observation.state"]) for s in samples])

        # Save episode data
        output_path = output_dir / f"episode_{ep_idx:04d}.npz"
        np.savez(
            output_path,
            obs_emb=obs_emb.astype(np.float32),
            actions=actions.astype(np.float32),
            state=states.astype(np.float32),
            episode_id=str(ep_idx),
            task=task,
            indices=np.array(indices),
        )

        logger.debug(f"Saved episode {ep_idx}: obs_emb={obs_emb.shape}, actions={actions.shape}")

    logger.info(f"Exported {len(episodes)} episodes to {output_dir}")

    # Save metadata
    meta_path = output_dir / "metadata.npz"
    np.savez(
        meta_path,
        num_episodes=len(episodes),
        action_horizon=args.action_horizon,
        tasks=list(dataset_meta.tasks.values()),
        repo_id=args.repo_id,
    )
    logger.info(f"Saved metadata to {meta_path}")


if __name__ == "__main__":
    main()
