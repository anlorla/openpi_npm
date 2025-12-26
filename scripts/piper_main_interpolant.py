#!/usr/bin/env python3
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import CompressedImage, JointState
from cv_bridge import CvBridge

from openpi_client import websocket_client_policy, image_tools

bridge = CvBridge()

# Store latest sensor data (3 cameras + 2 arms joint states)
latest_imgs = {
    "main": None,
    "wrist_l": None,
    "wrist_r": None,
}
latest_q = {
    "left": None,
    "right": None,
}


# ====== ROS callbacks ======


def cb_main(msg):
    """Top camera callback."""
    latest_imgs["main"] = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
    rospy.logdebug("Received main camera image")


def cb_wrist_l(msg):
    """Left wrist camera callback."""
    latest_imgs["wrist_l"] = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
    rospy.logdebug("Received left wrist image")


def cb_wrist_r(msg):
    """Right wrist camera callback."""
    latest_imgs["wrist_r"] = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
    rospy.logdebug("Received right wrist image")


def cb_joints_left(msg):
    """Left arm joint state callback."""
    latest_q["left"] = np.array(msg.position, dtype=np.float32)
    rospy.logdebug(f"Received left arm joint states: {latest_q['left'][:3]}...")


def cb_joints_right(msg):
    """Right arm joint state callback."""
    latest_q["right"] = np.array(msg.position, dtype=np.float32)
    rospy.logdebug(f"Received right arm joint states: {latest_q['right'][:3]}...")

def main():
    rospy.init_node("pi05_zeno_main")

    # ====== Subscriptions / publications ======
    rospy.Subscriber(
        "/realsense_top/color/image_raw/compressed",
        CompressedImage,
        cb_main,
        queue_size=1,
    )
    rospy.Subscriber(
        "/realsense_left/color/image_raw/compressed",
        CompressedImage,
        cb_wrist_l,
        queue_size=1,
    )
    rospy.Subscriber(
        "/realsense_right/color/image_raw/compressed",
        CompressedImage,
        cb_wrist_r,
        queue_size=1,
    )

    rospy.Subscriber(
        "/robot/arm_left/joint_states_single",
        JointState,
        cb_joints_left,
        queue_size=1,
    )
    rospy.Subscriber(
        "/robot/arm_right/joint_states_single",
        JointState,
        cb_joints_right,
        queue_size=1,
    )

    pub_left = rospy.Publisher(
        "/robot/arm_left/vla_joint_cmd", JointState, queue_size=1
    )
    pub_right = rospy.Publisher(
        "/robot/arm_right/vla_joint_cmd", JointState, queue_size=1
    )

    rospy.loginfo("Robot arm command publishers initialized")

    # ====== OpenPI policy client ======
    client = websocket_client_policy.WebsocketClientPolicy(
        host="127.0.0.1",
        port=8000,
    )

    # ====== Control / chunking configuration ======
    # Control frequency matches your training data collection
    rate = rospy.Rate(50)

    # Action horizon: Number of actions predicted per chunk (training config)
    action_horizon = 20

    # Number of actions to execute from each chunk before querying new actions
    num_actions_to_execute = 15
    assert num_actions_to_execute <= action_horizon, \
        "num_actions_to_execute must be <= action_horizon"

    # Linear interpolation steps for smoother motion between actions
    interpolation_steps = 5

    # Image preprocessing size (224 is default in OpenPI docs)
    image_size = 224

    # Prompt matches your fine-tuning / dataset format
    prompt = "Sweep lego blocks to yellow cross marker."

    # ====== Chunk execution state ======
    current_chunk = None  # np.ndarray [H, 14]
    chunk_step = 0  # index of the next action in the chunk
    prev_action_left = None  # for interpolation
    prev_action_right = None

    rospy.loginfo("Waiting for sensor data from robot arms...")
    data_ready_logged = False

    while not rospy.is_shutdown():
        # ------------------------------------------------------------------
        # 1) Wait until we have all required observations
        # ------------------------------------------------------------------
        if any(v is None for v in latest_imgs.values()) or any(
            v is None for v in latest_q.values()
        ):
            rate.sleep()
            continue

        if not data_ready_logged:
            rospy.loginfo(
                "✓ Successfully receiving observations from robot arms "
                "(cameras + joint states)"
            )
            data_ready_logged = True

        # ------------------------------------------------------------------
        # 2) If we have exhausted the current chunk, query a new one
        # ------------------------------------------------------------------
        if current_chunk is None or chunk_step >= num_actions_to_execute:
            # Convert images from BGR (ROS) to RGB
            rgb_main = cv2.cvtColor(latest_imgs["main"], cv2.COLOR_BGR2RGB)
            rgb_l = cv2.cvtColor(latest_imgs["wrist_l"], cv2.COLOR_BGR2RGB)
            rgb_r = cv2.cvtColor(latest_imgs["wrist_r"], cv2.COLOR_BGR2RGB)

            # Resize + uint8 conversion 
            img_main = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(rgb_main, image_size, image_size)
            )
            img_l = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(rgb_l, image_size, image_size)
            )
            img_r = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(rgb_r, image_size, image_size)
            )

            # Use first 7 joints per arm
            q_left = latest_q["left"][:7].astype(np.float32)
            q_right = latest_q["right"][:7].astype(np.float32)
            state = np.concatenate([q_left, q_right], axis=0)

            obs = {
                "observation/image": img_main,
                "observation/wrist_image": img_l,
                "observation/right_wrist_image": img_r,
                "observation/state": state,
                "prompt": prompt,
            }

            rospy.logdebug("Sending observation to policy server...")
            result = client.infer(obs)
            rospy.loginfo_throttle(
                5.0, "✓ Successfully communicated with policy server"
            )

            actions = np.array(result["actions"], dtype=np.float32)
            if actions.ndim != 2 or actions.shape[1] != 14:
                rospy.logwarn(
                    f"[SAFETY] Unexpected action shape: {actions.shape}, "
                    "expected (H, 14)"
                )
                rate.sleep()
                continue

            # Truncate to action_horizon if the server returns a longer chunk
            current_chunk = actions[:action_horizon]
            chunk_step = 0

            rospy.loginfo_throttle(
                5.0,
                f"✓ Received action chunk from policy server, "
                f"shape={current_chunk.shape}, "
                f"will execute first {num_actions_to_execute} actions",
            )

        # ------------------------------------------------------------------
        # 3) Take the next action from the chunk
        # ------------------------------------------------------------------
        # Safety check: if we somehow have no chunk, skip this cycle
        if current_chunk is None or chunk_step >= current_chunk.shape[0]:
            rospy.logwarn_throttle(
                2.0, "[SAFETY] No valid action chunk available, skipping step"
            )
            rate.sleep()
            continue

        action = current_chunk[chunk_step]
        chunk_step += 1

        if len(action) != 14:
            rospy.logwarn(
                f"[SAFETY] Invalid action dimension: expected 14, got {len(action)}. "
                "Skipping this action."
            )
            rate.sleep()
            continue

        # Split into left / right arm joint targets (absolute joint positions)
        action_left = action[:7]
        action_right = action[7:14]

        # Initialize previous action from current joint state on the first step
        if prev_action_left is None:
            prev_action_left = latest_q["left"][:7].astype(np.float32)
        if prev_action_right is None:
            prev_action_right = latest_q["right"][:7].astype(np.float32)

        # ------------------------------------------------------------------
        # 4) Execute this action with joint-space interpolation
        # ------------------------------------------------------------------
        for step in range(interpolation_steps):
            if rospy.is_shutdown():
                break

            alpha = float(step + 1) / float(interpolation_steps)
            interp_left = prev_action_left + alpha * (action_left - prev_action_left)
            interp_right = prev_action_right + alpha * (
                action_right - prev_action_right
            )

            cmd_left = JointState()
            cmd_left.header.stamp = rospy.Time.now()
            cmd_left.position = interp_left.tolist()

            cmd_right = JointState()
            cmd_right.header.stamp = rospy.Time.now()
            cmd_right.position = interp_right.tolist()

            pub_left.publish(cmd_left)
            pub_right.publish(cmd_right)

            if step == interpolation_steps - 1:
                rospy.loginfo_throttle(
                    2.0,
                    f"✓ Executed action {chunk_step}/"
                    f"{num_actions_to_execute} with {interpolation_steps} "
                    f"interpolation steps",
                )

            rate.sleep()

        # Update previous action to the latest target for the next interpolation
        prev_action_left = action_left
        prev_action_right = action_right

    rospy.loginfo("Shutting down pi05_zeno_main")

if __name__ == "__main__":
    main()
