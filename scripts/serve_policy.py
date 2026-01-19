import dataclasses
import enum
import logging
import socket
from pathlib import Path

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config

# 注意力可视化
from openpi.models.gemma import (
    enable_attention_capture,
    get_captured_attention,
    disable_attention_capture,
    clear_captured_attention,
)

# 注意力输出目录
ATTENTION_OUTPUT_DIR = Path(r"D:\RESERACH\Zeno\NPM-VLA\openpi\attention_output")
ATTENTION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)

    # 注意力可视化选项
    visualize_attention: bool = False  # 是否启用注意力可视化
    attention_interval: int = 1  # 每隔多少步保存一次注意力图


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)


class AttentionVisualizingPolicy:
    """包装 Policy，在推理时捕获并保存注意力可视化图。"""

    def __init__(self, policy, interval: int = 1):
        self._policy = policy
        self._interval = interval
        self._step = 0
        self._last_obs = None

    @property
    def metadata(self):
        return self._policy.metadata

    def infer(self, obs: dict) -> dict:
        # 保存 obs 用于可视化
        self._last_obs = obs

        # 启用注意力捕获
        enable_attention_capture()
        clear_captured_attention()

        # 推理
        result = self._policy.infer(obs)

        # 可视化并保存
        if self._step % self._interval == 0:
            self._save_attention_visualization(obs)

        # 禁用捕获
        disable_attention_capture()

        self._step += 1
        return result

    def _save_attention_visualization(self, obs: dict):
        """保存注意力可视化图"""
        attention_list = get_captured_attention()
        if not attention_list:
            logging.warning("No attention captured")
            return

        try:
            import numpy as np
            import cv2
            import matplotlib.pyplot as plt

            # 获取最后一层注意力
            attn = np.array(attention_list[-1])

            # 处理 Gemma 格式 [B, K, G, T, S]
            if attn.ndim == 5:
                attn = attn[0]  # [K, G, T, S]
                attn = attn.reshape(-1, attn.shape[-2], attn.shape[-1])  # [heads, T, S]

            # wide_top 是第4个相机，token 范围 768:1024
            # action tokens 在最后
            wide_top_start, wide_top_end = 768, 1024
            action_start = -50  # 假设 action horizon = 50

            # 提取 action → wide_top 的注意力
            attn_to_wide_top = attn[:, action_start:, wide_top_start:wide_top_end]
            attn_map = attn_to_wide_top.mean(axis=(0, 1))  # [256]

            # reshape 成 16x16
            attn_2d = attn_map.reshape(16, 16)

            # 归一化
            attn_min, attn_max = attn_2d.min(), attn_2d.max()
            if attn_max > attn_min:
                attn_2d = (attn_2d - attn_min) / (attn_max - attn_min)

            # 获取 wide_top 图像
            img_key = "observation/wide_top_image"
            if img_key not in obs:
                img_key = "observation/image"  # fallback

            image = np.array(obs.get(img_key, np.zeros((224, 224, 3), dtype=np.uint8)))
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            h, w = image.shape[:2]

            # 上采样注意力图
            attn_resized = cv2.resize(attn_2d.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)

            # 应用 colormap
            cmap = plt.get_cmap("jet")
            attn_colored = (cmap(attn_resized)[:, :, :3] * 255).astype(np.uint8)

            # 叠加
            blended = cv2.addWeighted(image, 0.5, attn_colored, 0.5, 0)

            # 保存
            output_path = ATTENTION_OUTPUT_DIR / f"attention_{self._step:06d}.png"

            # 创建并排显示的图像
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(image)
            axes[0].set_title("Image")
            axes[0].axis("off")

            axes[1].imshow(attn_2d, cmap="jet")
            axes[1].set_title("Attention")
            axes[1].axis("off")

            axes[2].imshow(blended)
            axes[2].set_title("Overlay")
            axes[2].axis("off")

            plt.tight_layout()
            plt.savefig(output_path, dpi=100, bbox_inches="tight")
            plt.close()

            logging.info(f"Saved attention: {output_path}")

        except Exception as e:
            logging.error(f"Failed to save attention: {e}")


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    # 注意力可视化包装
    if args.visualize_attention:
        policy = AttentionVisualizingPolicy(policy, interval=args.attention_interval)
        logging.info(f"Attention visualization enabled, output: {ATTENTION_OUTPUT_DIR}")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
