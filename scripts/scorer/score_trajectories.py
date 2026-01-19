#!/usr/bin/env python3
"""
score_trajectories.py - Score trajectories and generate training weights.

Scoring components:
1. BC Loss: How well can the scorer predict actions (lower = better quality)
2. Jerk: Action smoothness (lower = smoother, better quality)
3. Saturation: Actions near limits (lower = better control)
4. Disagreement: Ensemble variance (lower = more certain, better quality)

Weight mapping: score -> weight using robust z-score + exponential

Usage:
    python scripts/scorer/score_trajectories.py \
        --data-dir ./scorer_data \
        --scorer-path ./scorer_checkpoints/scorer_ensemble_best.pt \
        --output ./trajectory_weights.npz

Output:
    trajectory_weights.npz:
        - episode_ids: [N] episode identifiers
        - weights: [N] per-trajectory weights
        - scores: [N] raw scores (before mapping)
        - bc_losses: [N] BC loss per trajectory
        - jerks: [N] jerk per trajectory
        - saturations: [N] saturation per trajectory
        - disagreements: [N] ensemble disagreement per trajectory
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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
        preds = torch.stack([m(x) for m in self.models], dim=0)
        mean = preds.mean(dim=0)
        var = preds.var(dim=0)
        return mean, var


def load_scorer(scorer_path: str, config_path: str, device: torch.device) -> EnsembleScorer:
    """Load trained scorer model."""
    with open(config_path, "r") as f:
        config = json.load(f)

    model = EnsembleScorer(
        input_dim=config["input_dim"],
        output_dim=config["output_dim"],
        ensemble_size=config["ensemble_size"],
        hidden_dims=config["hidden_dims"],
        dropout=config["dropout"],
    ).to(device)

    checkpoint = torch.load(scorer_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, config


def compute_jerk(actions: np.ndarray) -> float:
    """Compute jerk (second derivative) of action sequence."""
    if len(actions) < 3:
        return 0.0
    vel = np.diff(actions, axis=0)
    acc = np.diff(vel, axis=0)
    return float(np.mean(np.abs(acc)))


def compute_saturation(actions: np.ndarray, threshold: float = 0.95) -> float:
    """Compute saturation ratio."""
    # Normalize actions if needed (assuming roughly [-1, 1] range)
    action_range = np.max(np.abs(actions), axis=0, keepdims=True)
    action_range = np.maximum(action_range, 1e-6)
    normalized = actions / action_range
    saturated = np.abs(normalized) > threshold
    return float(np.mean(saturated))


def score_episode(
    model: EnsembleScorer,
    obs_emb: np.ndarray,
    actions: np.ndarray,
    state: np.ndarray | None,
    config: dict,
    device: torch.device,
    lambda_jerk: float = 0.1,
    lambda_sat: float = 0.1,
    beta_disagree: float = 0.5,
) -> Dict[str, float]:
    """Score a single episode.

    Returns dict with:
        - score: Total score (lower = better)
        - bc_loss: Behavior cloning loss
        - jerk: Action jerk
        - saturation: Saturation ratio
        - disagreement: Ensemble disagreement
    """
    # Prepare features
    obs_emb_t = torch.from_numpy(obs_emb).float().to(device)
    if config.get("use_state", False) and state is not None:
        state_t = torch.from_numpy(state).float().to(device)
        features = torch.cat([obs_emb_t, state_t], dim=-1)
    else:
        features = obs_emb_t

    actions_t = torch.from_numpy(actions).float().to(device)

    # BC loss and disagreement
    with torch.no_grad():
        mean_pred, variance = model(features)
        bc_loss = nn.functional.mse_loss(mean_pred, actions_t).item()
        disagreement = variance.mean().item()

    # Jerk (computed on raw actions)
    jerk = compute_jerk(actions)

    # Saturation
    saturation = compute_saturation(actions)

    # Total score (lower = better quality)
    score = bc_loss + lambda_jerk * jerk + lambda_sat * saturation + beta_disagree * disagreement

    return {
        "score": score,
        "bc_loss": bc_loss,
        "jerk": jerk,
        "saturation": saturation,
        "disagreement": disagreement,
    }


def score_to_weight(
    scores: np.ndarray,
    alpha: float = 1.0,
    w_min: float = 0.1,
    w_max: float = 10.0,
    per_task_normalize: bool = False,
    task_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Convert scores to training weights using robust z-score + exponential.

    Formula:
        z = (score - median) / (MAD + eps)
        weight = clip(exp(-alpha * z), w_min, w_max)

    Lower score -> higher weight
    """
    if per_task_normalize and task_ids is not None:
        # Normalize within each task
        weights = np.zeros_like(scores)
        unique_tasks = np.unique(task_ids)

        for task in unique_tasks:
            mask = task_ids == task
            task_scores = scores[mask]

            median = np.median(task_scores)
            mad = np.median(np.abs(task_scores - median))

            z = (task_scores - median) / (mad + 1e-8)
            task_weights = np.clip(np.exp(-alpha * z), w_min, w_max)
            weights[mask] = task_weights
    else:
        # Global normalization
        median = np.median(scores)
        mad = np.median(np.abs(scores - median))

        z = (scores - median) / (mad + 1e-8)
        weights = np.clip(np.exp(-alpha * z), w_min, w_max)

    # Normalize weights to have mean 1 (so total gradient magnitude is preserved)
    weights = weights / np.mean(weights)

    return weights


def parse_args():
    parser = argparse.ArgumentParser(description="Score trajectories and generate weights")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing episode .npz files"
    )
    parser.add_argument(
        "--scorer-path",
        type=str,
        required=True,
        help="Path to trained scorer checkpoint"
    )
    parser.add_argument(
        "--scorer-config",
        type=str,
        default=None,
        help="Path to scorer config (default: scorer_config.json in same dir)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./trajectory_weights.npz",
        help="Output path for weights"
    )
    # Scoring coefficients
    parser.add_argument("--lambda-jerk", type=float, default=0.1, help="Jerk penalty weight")
    parser.add_argument("--lambda-sat", type=float, default=0.1, help="Saturation penalty weight")
    parser.add_argument("--beta-disagree", type=float, default=0.5, help="Disagreement penalty weight")
    # Weight mapping
    parser.add_argument("--alpha", type=float, default=1.0, help="Exponential mapping steepness")
    parser.add_argument("--w-min", type=float, default=0.1, help="Minimum weight")
    parser.add_argument("--w-max", type=float, default=10.0, help="Maximum weight")
    parser.add_argument("--per-task-normalize", action="store_true", help="Normalize scores per task")
    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device)
    data_dir = Path(args.data_dir)

    # Load scorer
    scorer_config = args.scorer_config
    if scorer_config is None:
        scorer_config = Path(args.scorer_path).parent / "scorer_config.json"

    model, config = load_scorer(args.scorer_path, scorer_config, device)
    logger.info(f"Loaded scorer from {args.scorer_path}")

    # Load episodes
    npz_files = sorted(data_dir.glob("episode_*.npz"))
    logger.info(f"Found {len(npz_files)} episodes to score")

    # Score each episode
    results = {
        "episode_ids": [],
        "tasks": [],
        "scores": [],
        "bc_losses": [],
        "jerks": [],
        "saturations": [],
        "disagreements": [],
    }

    for ep_path in tqdm(npz_files, desc="Scoring episodes"):
        data = np.load(ep_path, allow_pickle=True)

        obs_emb = data["obs_emb"]
        actions = data["actions"]
        state = data.get("state", None)
        episode_id = str(data["episode_id"])
        task = str(data.get("task", "unknown"))

        scores = score_episode(
            model,
            obs_emb,
            actions,
            state,
            config,
            device,
            lambda_jerk=args.lambda_jerk,
            lambda_sat=args.lambda_sat,
            beta_disagree=args.beta_disagree,
        )

        results["episode_ids"].append(episode_id)
        results["tasks"].append(task)
        results["scores"].append(scores["score"])
        results["bc_losses"].append(scores["bc_loss"])
        results["jerks"].append(scores["jerk"])
        results["saturations"].append(scores["saturation"])
        results["disagreements"].append(scores["disagreement"])

    # Convert to arrays
    for key in results:
        results[key] = np.array(results[key])

    # Compute weights
    task_ids = results["tasks"] if args.per_task_normalize else None
    weights = score_to_weight(
        results["scores"],
        alpha=args.alpha,
        w_min=args.w_min,
        w_max=args.w_max,
        per_task_normalize=args.per_task_normalize,
        task_ids=task_ids,
    )
    results["weights"] = weights

    # Log statistics
    logger.info("\n=== Scoring Statistics ===")
    logger.info(f"Total episodes: {len(results['episode_ids'])}")
    logger.info(f"BC Loss:      mean={np.mean(results['bc_losses']):.4f}, std={np.std(results['bc_losses']):.4f}")
    logger.info(f"Jerk:         mean={np.mean(results['jerks']):.4f}, std={np.std(results['jerks']):.4f}")
    logger.info(f"Saturation:   mean={np.mean(results['saturations']):.4f}, std={np.std(results['saturations']):.4f}")
    logger.info(f"Disagreement: mean={np.mean(results['disagreements']):.4f}, std={np.std(results['disagreements']):.4f}")
    logger.info(f"Total Score:  mean={np.mean(results['scores']):.4f}, std={np.std(results['scores']):.4f}")
    logger.info(f"Weights:      mean={np.mean(weights):.4f}, min={np.min(weights):.4f}, max={np.max(weights):.4f}")

    # Per-task statistics
    unique_tasks = np.unique(results["tasks"])
    if len(unique_tasks) > 1:
        logger.info("\n=== Per-Task Statistics ===")
        for task in unique_tasks:
            mask = results["tasks"] == task
            logger.info(f"Task '{task}': n={np.sum(mask)}, "
                        f"score={np.mean(results['scores'][mask]):.4f}, "
                        f"weight={np.mean(weights[mask]):.4f}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_path,
        **results,
        # Also save mapping parameters for reference
        alpha=args.alpha,
        w_min=args.w_min,
        w_max=args.w_max,
        lambda_jerk=args.lambda_jerk,
        lambda_sat=args.lambda_sat,
        beta_disagree=args.beta_disagree,
    )

    logger.info(f"\nSaved weights to {output_path}")

    # Also save a simple lookup dict for easy loading
    weight_dict = {eid: w for eid, w in zip(results["episode_ids"], weights)}
    np.save(output_path.with_suffix(".dict.npy"), weight_dict)
    logger.info(f"Saved weight dict to {output_path.with_suffix('.dict.npy')}")


if __name__ == "__main__":
    main()
