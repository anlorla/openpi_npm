#!/usr/bin/env python3
"""
tiny_scorer.py - Train an ensemble of small MLPs to score trajectory quality.

The scorer predicts actions from observation embeddings. Trajectories where
the scorer can easily predict actions (low BC loss) are considered higher quality.

Features:
- Ensemble of K small MLPs for uncertainty estimation
- BC loss (action prediction error)
- Jerk penalty (action smoothness)
- Saturation penalty (actions near limits)

Usage:
    python scripts/scorer/tiny_scorer.py \
        --data-dir ./scorer_data \
        --output-dir ./scorer_checkpoints \
        --ensemble-size 5

Output:
    - scorer_ensemble.pt: Trained ensemble model
    - scorer_config.json: Model configuration
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TrajDataset(Dataset):
    """Dataset for trajectory scoring."""

    def __init__(self, data_dir: str, use_state: bool = True):
        self.data_dir = Path(data_dir)
        self.use_state = use_state

        # Load all episode files
        self.episodes = []
        self.samples = []  # (episode_idx, timestep_idx)

        npz_files = sorted(self.data_dir.glob("episode_*.npz"))
        logger.info(f"Found {len(npz_files)} episode files")

        for ep_path in npz_files:
            data = np.load(ep_path, allow_pickle=True)
            ep_data = {
                "obs_emb": data["obs_emb"],
                "actions": data["actions"],
                "state": data.get("state", None),
                "episode_id": str(data["episode_id"]),
            }
            ep_idx = len(self.episodes)
            self.episodes.append(ep_data)

            # Add all timesteps as samples
            T = len(ep_data["obs_emb"])
            for t in range(T):
                self.samples.append((ep_idx, t))

        logger.info(f"Loaded {len(self.episodes)} episodes, {len(self.samples)} samples")

        # Compute input/output dimensions
        sample_ep = self.episodes[0]
        self.obs_dim = sample_ep["obs_emb"].shape[-1]
        self.action_dim = sample_ep["actions"].shape[-1]
        self.state_dim = sample_ep["state"].shape[-1] if sample_ep["state"] is not None else 0

        self.input_dim = self.obs_dim
        if use_state and self.state_dim > 0:
            self.input_dim += self.state_dim

        logger.info(f"Input dim: {self.input_dim}, Action dim: {self.action_dim}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ep_idx, t = self.samples[idx]
        ep = self.episodes[ep_idx]

        obs_emb = torch.from_numpy(ep["obs_emb"][t]).float()

        # Optionally concatenate state
        if self.use_state and ep["state"] is not None:
            state = torch.from_numpy(ep["state"][t]).float()
            features = torch.cat([obs_emb, state], dim=-1)
        else:
            features = obs_emb

        actions = torch.from_numpy(ep["actions"][t]).float()

        return features, actions, ep_idx, t


class MLP(nn.Module):
    """Simple MLP for action prediction."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [256, 256],
        dropout: float = 0.1,
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class EnsembleScorer(nn.Module):
    """Ensemble of MLPs for scoring with uncertainty estimation."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        ensemble_size: int = 5,
        hidden_dims: List[int] = [256, 256],
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ensemble_size = ensemble_size

        self.models = nn.ModuleList([
            MLP(input_dim, output_dim, hidden_dims, dropout)
            for _ in range(ensemble_size)
        ])

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning mean and variance.

        Returns:
            mean: [B, action_dim] - Mean prediction
            var: [B, action_dim] - Variance (epistemic uncertainty)
        """
        preds = torch.stack([m(x) for m in self.models], dim=0)  # [K, B, action_dim]
        mean = preds.mean(dim=0)
        var = preds.var(dim=0)
        return mean, var

    def predict_all(self, x) -> torch.Tensor:
        """Get predictions from all ensemble members.

        Returns:
            preds: [K, B, action_dim]
        """
        return torch.stack([m(x) for m in self.models], dim=0)


def compute_jerk(actions: np.ndarray) -> float:
    """Compute jerk (second derivative) of action sequence.

    Args:
        actions: [T, action_dim]

    Returns:
        Mean jerk across all dimensions and timesteps
    """
    if len(actions) < 3:
        return 0.0

    # First derivative (velocity)
    vel = np.diff(actions, axis=0)
    # Second derivative (acceleration)
    acc = np.diff(vel, axis=0)
    # Jerk is the magnitude of acceleration changes
    jerk = np.mean(np.abs(acc))
    return float(jerk)


def compute_saturation(actions: np.ndarray, threshold: float = 0.95) -> float:
    """Compute saturation ratio (actions near limits).

    Args:
        actions: [T, action_dim] - Normalized to roughly [-1, 1]
        threshold: Actions with |a| > threshold are considered saturated

    Returns:
        Ratio of saturated actions
    """
    saturated = np.abs(actions) > threshold
    return float(np.mean(saturated))


def train_epoch(
    model: EnsembleScorer,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    for features, actions, _, _ in dataloader:
        features = features.to(device)
        actions = actions.to(device)

        optimizer.zero_grad()

        # Train each ensemble member with different subsets
        loss = 0.0
        for i, mlp in enumerate(model.models):
            # Bootstrap: each model sees a random subset
            mask = torch.rand(len(features), device=device) > 0.2
            if mask.sum() == 0:
                mask[0] = True

            pred = mlp(features[mask])
            target = actions[mask]
            loss += nn.functional.mse_loss(pred, target)

        loss = loss / model.ensemble_size
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(
    model: EnsembleScorer,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate model, returning mean loss and mean variance."""
    model.eval()
    total_loss = 0.0
    total_var = 0.0

    with torch.no_grad():
        for features, actions, _, _ in dataloader:
            features = features.to(device)
            actions = actions.to(device)

            mean, var = model(features)
            loss = nn.functional.mse_loss(mean, actions)

            total_loss += loss.item()
            total_var += var.mean().item()

    n = len(dataloader)
    return total_loss / n, total_var / n


def parse_args():
    parser = argparse.ArgumentParser(description="Train tiny scorer ensemble")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing episode .npz files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./scorer_checkpoints",
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--ensemble-size",
        type=int,
        default=5,
        help="Number of ensemble members"
    )
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[256, 256],
        help="Hidden layer dimensions"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--use-state",
        action="store_true",
        help="Include state in input features"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataset = TrajDataset(args.data_dir, use_state=args.use_state)

    # Split into train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )

    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Create model
    device = torch.device(args.device)
    model = EnsembleScorer(
        input_dim=dataset.input_dim,
        output_dim=dataset.action_dim,
        ensemble_size=args.ensemble_size,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
    ).to(device)

    logger.info(f"Model: {sum(p.numel() for p in model.parameters())} parameters")

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_var = evaluate(model, val_loader, device)
        scheduler.step()

        logger.info(
            f"Epoch {epoch+1}/{args.epochs} - "
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Var: {val_var:.4f}"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
            }, output_dir / "scorer_ensemble_best.pt")

    # Save final model
    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch": args.epochs,
    }, output_dir / "scorer_ensemble_final.pt")

    # Save config
    config = {
        "input_dim": dataset.input_dim,
        "output_dim": dataset.action_dim,
        "ensemble_size": args.ensemble_size,
        "hidden_dims": args.hidden_dims,
        "dropout": args.dropout,
        "use_state": args.use_state,
        "obs_dim": dataset.obs_dim,
        "state_dim": dataset.state_dim,
        "action_dim": dataset.action_dim,
    }
    with open(output_dir / "scorer_config.json", "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")
    logger.info(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
