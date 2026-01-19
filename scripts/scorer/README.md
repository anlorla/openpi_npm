# Tiny Trajectory Scorer

This module provides a system for scoring trajectory quality and using those scores as training weights.

## Overview

The system consists of three main steps:

1. **Feature Extraction** (`dump_episode_features.py`): Export observation embeddings and actions from LeRobot dataset
2. **Scorer Training** (`tiny_scorer.py`): Train an ensemble of small MLPs to predict actions from embeddings
3. **Trajectory Scoring** (`score_trajectories.py`): Score each trajectory and convert scores to training weights

## Quick Start

### Step 1: Export Episode Features

```bash
python scripts/scorer/dump_episode_features.py \
    --repo-id zeno/piper_sweep_dataset \
    --output-dir ./scorer_data \
    --use-simple-features  # Use this if you don't have a trained model yet
```

### Step 2: Train Tiny Scorer

```bash
python scripts/scorer/tiny_scorer.py \
    --data-dir ./scorer_data \
    --output-dir ./scorer_checkpoints \
    --ensemble-size 5 \
    --epochs 100
```

### Step 3: Score Trajectories

```bash
python scripts/scorer/score_trajectories.py \
    --data-dir ./scorer_data \
    --scorer-path ./scorer_checkpoints/scorer_ensemble_best.pt \
    --output ./trajectory_weights.npz \
    --lambda-jerk 0.1 \
    --lambda-sat 0.1 \
    --beta-disagree 0.5
```

### Step 4: Train with Weights (Optional)

```bash
python scripts/train_weighted.py \
    --config-name pi05_piper \
    --exp_name weighted_training \
    --weights-path ./trajectory_weights.npz \
    --weight-warmup-steps 1000
```

## Scoring Components

The total score for each trajectory is computed as:

```
score = BC_loss + λ_jerk * jerk + λ_sat * saturation + β * disagreement
```

Where:
- **BC Loss**: How well the scorer can predict actions (lower = more predictable = better)
- **Jerk**: Action smoothness (second derivative magnitude, lower = smoother)
- **Saturation**: Fraction of actions near limits (lower = better control)
- **Disagreement**: Ensemble variance (lower = more certain = better)

## Weight Mapping

Scores are converted to weights using robust z-score + exponential:

```python
z = (score - median(scores)) / (MAD(scores) + eps)
weight = clip(exp(-alpha * z), w_min, w_max)
```

This ensures:
- Lower scores get higher weights
- Robust to outliers (uses median/MAD instead of mean/std)
- Weights are bounded to prevent extreme values

## Configuration Options

### Feature Extraction
- `--use-simple-features`: Use downsampled images + state instead of model embeddings
- `--checkpoint-path`: Path to trained pi0/pi05 for embedding extraction

### Scorer Training
- `--ensemble-size`: Number of ensemble members (default: 5)
- `--hidden-dims`: MLP hidden layer sizes (default: [256, 256])
- `--use-state`: Include robot state in features

### Scoring
- `--lambda-jerk`: Jerk penalty weight (default: 0.1)
- `--lambda-sat`: Saturation penalty weight (default: 0.1)
- `--beta-disagree`: Disagreement penalty weight (default: 0.5)
- `--alpha`: Exponential mapping steepness (default: 1.0)
- `--per-task-normalize`: Normalize scores within each task

## Output Files

After running the full pipeline:

```
scorer_data/
├── episode_0000.npz    # Per-episode features
├── episode_0001.npz
├── ...
└── metadata.npz        # Dataset metadata

scorer_checkpoints/
├── scorer_ensemble_best.pt    # Best model
├── scorer_ensemble_final.pt   # Final model
└── scorer_config.json         # Model config

trajectory_weights.npz   # Final weights
trajectory_weights.dict.npy  # Weight lookup dict
```

## Tips

1. **Start simple**: Use `--use-simple-features` for initial testing
2. **Per-task normalization**: Use `--per-task-normalize` if you have multiple task types
3. **Warmup**: Use weight warmup during training to avoid early instability
4. **Tuning**: Adjust `--alpha` to control weight distribution spread
