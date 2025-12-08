#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
检查：LeRobot parquet 里的 state / action
  --(用 norm_stats.json 里的 q01/q99 归一化)--> z
  --(再用同样的 q01/q99 反归一化)--> x_recon
是否能还原回原始值。

用法示例：
    python test_norm.py  --parquet "D:\\DESKTOP\\episode_000000.parquet" --norm-json "D:\\DESKTOP\\norm_stats.json"
"""

import argparse
import json
import numpy as np
import pandas as pd


def load_parquet(parquet_path: str):
    df = pd.read_parquet(parquet_path)
    print(f"[parquet] columns: {list(df.columns)}")

    states_col = df["observation.state"].to_numpy()
    actions_col = df["action"].to_numpy()

    states = np.stack([np.array(x, dtype=np.float32) for x in states_col], axis=0)
    actions = np.stack([np.array(x, dtype=np.float32) for x in actions_col], axis=0)

    print(f"[parquet] states shape : {states.shape}")
    print(f"[parquet] actions shape: {actions.shape}")
    return states, actions


def load_norm_stats(json_path: str):
    with open(json_path, "r") as f:
        data = json.load(f)["norm_stats"]

    state_stats = data["state"]
    action_stats = data["actions"]

    # 转成 numpy，方便广播
    for k in ["mean", "std", "q01", "q99"]:
        state_stats[k] = np.asarray(state_stats[k], dtype=np.float32)
        action_stats[k] = np.asarray(action_stats[k], dtype=np.float32)

    print("\n=== norm_stats 维度检查 ===")
    print(f"state_stats.mean.shape  = {state_stats['mean'].shape}")
    print(f"action_stats.mean.shape = {action_stats['mean'].shape}")

    print("\n=== state_stats 数值范围 ===")
    print(f"state q01: min={state_stats['q01'].min():.6f}, max={state_stats['q01'].max():.6f}")
    print(f"state q99: min={state_stats['q99'].min():.6f}, max={state_stats['q99'].max():.6f}")
    print(f"state mean: min={state_stats['mean'].min():.6f}, max={state_stats['mean'].max():.6f}")
    print(f"state std: min={state_stats['std'].min():.6f}, max={state_stats['std'].max():.6f}")

    print("\n=== action_stats 数值范围 ===")
    print(f"action q01: min={action_stats['q01'].min():.6f}, max={action_stats['q01'].max():.6f}")
    print(f"action q99: min={action_stats['q99'].min():.6f}, max={action_stats['q99'].max():.6f}")
    print(f"action mean: min={action_stats['mean'].min():.6f}, max={action_stats['mean'].max():.6f}")
    print(f"action std: min={action_stats['std'].min():.6f}, max={action_stats['std'].max():.6f}")

    # 检查是否有 q01 >= q99 的维度（这会导致归一化失败）
    print("\n=== 检查异常维度 ===")
    state_bad_dims = np.where(state_stats['q01'] >= state_stats['q99'])[0]
    if len(state_bad_dims) > 0:
        print(f"WARNING: state 有 {len(state_bad_dims)} 个维度的 q01 >= q99:")
        for dim in state_bad_dims[:10]:  # 只显示前10个
            print(f"  dim {dim}: q01={state_stats['q01'][dim]:.6f}, q99={state_stats['q99'][dim]:.6f}")
    else:
        print("✓ state 所有维度的 q01 < q99")

    action_bad_dims = np.where(action_stats['q01'] >= action_stats['q99'])[0]
    if len(action_bad_dims) > 0:
        print(f"WARNING: action 有 {len(action_bad_dims)} 个维度的 q01 >= q99:")
        for dim in action_bad_dims[:10]:  # 只显示前10个
            print(f"  dim {dim}: q01={action_stats['q01'][dim]:.6f}, q99={action_stats['q99'][dim]:.6f}")
    else:
        print("✓ action 所有维度的 q01 < q99")

    # 检查是否有非常小的范围（可能导致数值不稳定）
    state_range = state_stats['q99'] - state_stats['q01']
    action_range = action_stats['q99'] - action_stats['q01']

    print(f"\n=== 检查数值稳定性 ===")
    state_small_range = np.where(state_range < 1e-5)[0]
    if len(state_small_range) > 0:
        print(f"WARNING: state 有 {len(state_small_range)} 个维度的 q99-q01 < 1e-5")
    else:
        print(f"✓ state 范围正常: min={state_range.min():.6f}, max={state_range.max():.6f}")

    action_small_range = np.where(action_range < 1e-5)[0]
    if len(action_small_range) > 0:
        print(f"WARNING: action 有 {len(action_small_range)} 个维度的 q99-q01 < 1e-5")
    else:
        print(f"✓ action 范围正常: min={action_range.min():.6f}, max={action_range.max():.6f}")

    return state_stats, action_stats


def normalize_zscore(x: np.ndarray, stats: dict) -> np.ndarray:
    """
    使用 mean / std 做 z-score 归一化（与 transforms.py:_normalize 一致）：
        z = (x - mean) / (std + 1e-6)

    标准正态分布归一化，大部分数据会落在 [-3, 3] 范围内。
    """
    mean = stats["mean"]  # (D,)
    std = stats["std"]    # (D,)

    z = (x - mean) / (std + 1e-6)  # (N, D)
    return z


def denormalize_zscore(z: np.ndarray, stats: dict) -> np.ndarray:
    """
    normalize_zscore 的反操作（与 transforms.py:_unnormalize 一致）：
        x = z * (std + 1e-6) + mean
    """
    mean = stats["mean"]
    std = stats["std"]

    x = z * (std + 1e-6) + mean
    return x


def quantile_normalize(x: np.ndarray, stats: dict) -> np.ndarray:
    """
    使用 q01 / q99 做线性归一化（与 transforms.py:_normalize_quantile 一致）：
        z = (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0

    将 [q01, q99] 映射到 [-1, 1]，大部分数据会落在 [-1, 1]，少量 outlier 会稍微超过。
    """
    low = stats["q01"]   # (D,)
    high = stats["q99"]  # (D,)

    z = (x - low) / (high - low + 1e-6) * 2.0 - 1.0  # (N, D)
    return z


def quantile_denormalize(z: np.ndarray, stats: dict) -> np.ndarray:
    """
    quantile_normalize 的反操作（与 transforms.py:_unnormalize_quantile 一致）：
        x = (z + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
    """
    low = stats["q01"]
    high = stats["q99"]

    x = (z + 1.0) / 2.0 * (high - low + 1e-6) + low
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True, help="某一条 episode_xxxxxx.parquet 路径")
    parser.add_argument("--norm-json", required=True, help="norm_stats.json 路径")
    args = parser.parse_args()

    # 1) 读取一条 episode 的原始 state / action
    states_raw, actions_raw = load_parquet(args.parquet)

    print("\n=== 原始数据的数值范围 ===")
    print(
        f"states_raw:  min={states_raw.min():.6f}, "
        f"max={states_raw.max():.6f}, mean={states_raw.mean():.6f}, std={states_raw.std():.6f}"
    )
    print(
        f"actions_raw: min={actions_raw.min():.6f}, "
        f"max={actions_raw.max():.6f}, mean={actions_raw.mean():.6f}, std={actions_raw.std():.6f}"
    )

    # 2) 从 json 里加载 norm_stats
    state_stats, action_stats = load_norm_stats(args.norm_json)

    # 2.5) 检查问题维度的实际值
    print("\n=== 检查 action 维度 6 和 13 的实际值 ===")
    print(f"维度 6 (左臂 gripper) 在当前 episode 中:")
    print(f"  min={actions_raw[:, 6].min():.6f}, max={actions_raw[:, 6].max():.6f}")
    print(f"  mean={actions_raw[:, 6].mean():.6f}, std={actions_raw[:, 6].std():.6f}")
    print(f"  unique values: {np.unique(actions_raw[:, 6])}")
    print(f"norm_stats 中:")
    print(f"  mean={action_stats['mean'][6]:.6f}, std={action_stats['std'][6]:.6f}")
    print(f"  q01={action_stats['q01'][6]:.6f}, q99={action_stats['q99'][6]:.6f}")

    print(f"\n维度 13 (右臂 gripper) 在当前 episode 中:")
    print(f"  min={actions_raw[:, 13].min():.6f}, max={actions_raw[:, 13].max():.6f}")
    print(f"  mean={actions_raw[:, 13].mean():.6f}, std={actions_raw[:, 13].std():.6f}")
    print(f"  unique values (前10个): {np.unique(actions_raw[:, 13])[:10]}")
    print(f"norm_stats 中:")
    print(f"  mean={action_stats['mean'][13]:.6f}, std={action_stats['std'][13]:.6f}")
    print(f"  q01={action_stats['q01'][13]:.6f}, q99={action_stats['q99'][13]:.6f}")

    # 3) 测试两种归一化方法
    print("\n" + "=" * 80)
    print("方法 1: Z-Score Normalization (mean/std)")
    print("=" * 80)

    states_norm_z = normalize_zscore(states_raw, state_stats)
    actions_norm_z = normalize_zscore(actions_raw, action_stats)

    print("归一化后的数值范围:")
    print(
        f"  states:  min={states_norm_z.min():.3f}, "
        f"max={states_norm_z.max():.3f}, mean={states_norm_z.mean():.3f}, std={states_norm_z.std():.3f}"
    )
    print(
        f"  actions: min={actions_norm_z.min():.3f}, "
        f"max={actions_norm_z.max():.3f}, mean={actions_norm_z.mean():.3f}, std={actions_norm_z.std():.3f}"
    )

    states_recon_z = denormalize_zscore(states_norm_z, state_stats)
    actions_recon_z = denormalize_zscore(actions_norm_z, action_stats)

    state_err_z = np.linalg.norm(states_recon_z - states_raw, axis=1)
    action_err_z = np.linalg.norm(actions_recon_z - actions_raw, axis=1)

    print("Round-trip 误差:")
    print(
        f"  state:  mean L2={state_err_z.mean():.6e}, max L2={state_err_z.max():.6e}"
    )
    print(
        f"  action: mean L2={action_err_z.mean():.6e}, max L2={action_err_z.max():.6e}"
    )

    # 显示每个维度的归一化范围
    print("\n每个维度的归一化范围 (前3帧平均):")
    print("  Action 维度:")
    for dim in range(actions_norm_z.shape[1]):
        dim_min = actions_norm_z[:3, dim].min()
        dim_max = actions_norm_z[:3, dim].max()
        dim_mean = actions_norm_z[:3, dim].mean()
        std_val = action_stats['std'][dim]
        print(f"    dim {dim:2d}: [{dim_min:8.3f}, {dim_max:8.3f}], mean={dim_mean:8.3f}, std={std_val:.6f}")

    print("\n" + "=" * 80)
    print("方法 2: Quantile Normalization (q01/q99)")
    print("=" * 80)

    states_norm_q = quantile_normalize(states_raw, state_stats)
    actions_norm_q = quantile_normalize(actions_raw, action_stats)

    print("归一化后的数值范围:")
    print(
        f"  states:  min={states_norm_q.min():.3f}, "
        f"max={states_norm_q.max():.3f}, mean={states_norm_q.mean():.3f}, std={states_norm_q.std():.3f}"
    )
    print(
        f"  actions: min={actions_norm_q.min():.3f}, "
        f"max={actions_norm_q.max():.3f}, mean={actions_norm_q.mean():.3f}, std={actions_norm_q.std():.3f}"
    )

    states_recon_q = quantile_denormalize(states_norm_q, state_stats)
    actions_recon_q = quantile_denormalize(actions_norm_q, action_stats)

    state_err_q = np.linalg.norm(states_recon_q - states_raw, axis=1)
    action_err_q = np.linalg.norm(actions_recon_q - actions_raw, axis=1)

    print("Round-trip 误差:")
    print(
        f"  state:  mean L2={state_err_q.mean():.6e}, max L2={state_err_q.max():.6e}"
    )
    print(
        f"  action: mean L2={action_err_q.mean():.6e}, max L2={action_err_q.max():.6e}"
    )

    print("\n" + "=" * 80)
    print("结论")
    print("=" * 80)
    print("\n如果方法 1 (Z-Score) 的归一化范围合理（大部分在 [-3, 3] 附近），")
    print("那么训练时可能使用的是 mean/std normalization (use_quantiles=False)")
    print("\n如果方法 2 (Quantile) 的归一化范围合理（大部分在 [-1, 1] 附近），")
    print("那么训练时使用的是 quantile normalization (use_quantiles=True)")

if __name__ == "__main__":
    main()
