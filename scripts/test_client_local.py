import numpy as np

from openpi_client import websocket_client_policy, image_tools 

def main():
    client = websocket_client_policy.WebsocketClientPolicy(
        host="127.0.0.1", 
        port=8000,
    )

    H, W = 256, 256
    fake_img = np.zeros((H, W, 3), dtype=np.uint8)
    fake_q_left = np.zeros(7, dtype=np.float32)
    fake_q_right = np.zeros(7, dtype=np.float32)

    obs = {
        "observation/image": fake_img,
        "observation/wrist_image":      fake_img,
        "observation/right_wrist_image":     fake_img,
        "observation/state":  np.concatenate([fake_q_left, fake_q_right], axis=0),
        "prompt": "sweep the blocks into an E shape",
    }

    result = client.infer(obs)
    actions = np.array(result["actions"])
    print("actions shape:", actions.shape)
    print("first action:", actions[0])

if __name__ == "__main__":
    main()
