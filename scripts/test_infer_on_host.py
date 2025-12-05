from pathlib import Path
import numpy as np

from openpi.training import config as training_config
from openpi.policies import policy_config

def main():
    # 1. Load the training config name you used during training
    cfg = training_config.get_config("pi05_npm_lora")

    # 2. Load checkpoint directory (use run directory, script will auto-find latest step)
    ckpt_dir = Path("checkpoints/pi05_npm_lora/sweep2E/2999")

    policy = policy_config.create_trained_policy(cfg, ckpt_dir)

    # 3. Create dummy observations with correct keys and shapes
    # Note: Keys must match what libero_policy expects
    # libero_policy expects: observation/image, observation/wrist_image, observation/right_wrist_image, observation/state
    H, W = 224, 224  # libero_policy default image size is 224x224
    fake_img = np.zeros((H, W, 3), dtype=np.uint8)
    fake_q_left = np.zeros(7, dtype=np.float32)
    fake_q_right = np.zeros(7, dtype=np.float32)
    # State is concatenation of left and right joint positions (7+7=14 dims)
    fake_state = np.concatenate([fake_q_left, fake_q_right], axis=0)

    example = {
        "observation/image": fake_img,  # Third-person view
        "observation/wrist_image": fake_img,  # Left wrist camera
        "observation/right_wrist_image": fake_img,  # Right wrist camera
        "observation/state": fake_state,  # State (14-dim: left 7 + right 7)
        "prompt": "sweep the blocks into an E shape",
    }

    out = policy.infer(example)
    actions = out["actions"]
    print("actions shape:", actions.shape)
    print("first action:", actions[0])

if __name__ == "__main__":
    main()
