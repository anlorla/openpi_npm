"""
简化版注意力可视化 - 只针对 wide_top 相机

使用方法:
    from openpi.models.gemma import enable_attention_capture, get_captured_attention, disable_attention_capture
    from visualize_attention_simple import visualize_wide_top_attention

    enable_attention_capture()
    actions = policy.infer(obs)
    attention_list = get_captured_attention()

    visualize_wide_top_attention(
        image=obs["image"]["wide_top"][0],  # wide_top 图像
        attention=attention_list[-1],        # 最后一层注意力
        output_path="attention.png"
    )
    disable_attention_capture()
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np


def visualize_wide_top_attention(
    image: np.ndarray,
    attention: np.ndarray,
    output_path: str | None = None,
    alpha: float = 0.5,
    colormap: str = "jet",
    wide_top_token_start: int = 768,   # 第4个相机: 3*256 = 768
    wide_top_token_end: int = 1024,    # 768 + 256 = 1024
    action_token_start: int = -50,
    show: bool = True,
) -> np.ndarray:
    """
    可视化 wide_top 相机的注意力热力图

    Args:
        image: wide_top 图像 [H, W, 3]
        attention: 注意力权重，Gemma格式 [B, K, G, T, S] 或已处理的 [heads, T, S]
        output_path: 保存路径
        alpha: 热力图透明度
        colormap: 颜色映射
        wide_top_token_start: wide_top 图像 tokens 的起始位置
        wide_top_token_end: wide_top 图像 tokens 的结束位置 (256 = 16x16 patches)
        action_token_start: action tokens 的起始位置 (负数表示从末尾算)
        show: 是否显示图像

    Returns:
        叠加后的图像
    """
    attention = np.array(attention)

    # 处理 Gemma 格式 [B, K, G, T, S]
    if attention.ndim == 5:
        attention = attention[0]  # 去掉 batch: [K, G, T, S]
        attention = attention.reshape(-1, attention.shape[-2], attention.shape[-1])  # [heads, T, S]

    # 提取 action tokens 对 wide_top image tokens 的注意力
    # attention: [heads, query_len, key_len]
    attn_to_wide_top = attention[:, action_token_start:, wide_top_token_start:wide_top_token_end]
    # [heads, num_action_tokens, 256]

    # 聚合：对所有 heads 和 action tokens 取平均
    attn_map = attn_to_wide_top.mean(axis=(0, 1))  # [256]

    # reshape 成 16x16
    grid_size = int(np.sqrt(len(attn_map)))
    attn_2d = attn_map.reshape(grid_size, grid_size)

    # 归一化到 [0, 1]
    attn_min, attn_max = attn_2d.min(), attn_2d.max()
    if attn_max > attn_min:
        attn_2d = (attn_2d - attn_min) / (attn_max - attn_min)

    # 处理图像
    image = np.array(image)
    if image.max() <= 1.0:
        if image.min() < 0:  # [-1, 1]
            image = ((image + 1) / 2 * 255).astype(np.uint8)
        else:  # [0, 1]
            image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)

    h, w = image.shape[:2]

    # 上采样注意力图到图像尺寸
    attn_resized = cv2.resize(attn_2d.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)

    # 应用 colormap
    cmap = plt.get_cmap(colormap)
    attn_colored = (cmap(attn_resized)[:, :, :3] * 255).astype(np.uint8)

    # 叠加
    blended = cv2.addWeighted(image, 1 - alpha, attn_colored, alpha, 0)

    # 显示
    if show or output_path:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes[0].imshow(image)
        axes[0].set_title("wide_top")
        axes[0].axis("off")

        im = axes[1].imshow(attn_2d, cmap=colormap)
        axes[1].set_title("Attention (16x16)")
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1], fraction=0.046)

        axes[2].imshow(blended)
        axes[2].set_title("Overlay")
        axes[2].axis("off")

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"Saved to {output_path}")

        if show:
            plt.show()
        else:
            plt.close()

    return blended


if __name__ == "__main__":
    # Demo
    print("Demo: 生成随机注意力可视化")

    image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    # 模拟注意力: [B, K, G, T, S] = [1, 1, 8, 306, 306]
    # 假设 256 image tokens + 50 action tokens = 306
    fake_attention = np.random.rand(1, 1, 8, 306, 306).astype(np.float32)

    visualize_wide_top_attention(
        image,
        fake_attention,
        output_path="demo_wide_top_attention.png",
    )
