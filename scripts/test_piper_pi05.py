#!/usr/bin/env python3
import numpy as np
import rospy
from sensor_msgs.msg import CompressedImage, JointState
from cv_bridge import CvBridge

from openpi_client import websocket_client_policy, image_tools

bridge = CvBridge()

# Store latest sensor data
latest_imgs = {
    "main": None,
    "wrist_l": None,
    "wrist_r": None,
}
latest_q = {
    "left": None,
    "right": None,
}

# Callback for main (top) camera - CompressedImage format
def cb_main(msg):
    latest_imgs["main"] = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")

# Callback for left wrist camera - CompressedImage format
def cb_wrist_l(msg):
    latest_imgs["wrist_l"] = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")

# Callback for right wrist camera - CompressedImage format
def cb_wrist_r(msg):
    latest_imgs["wrist_r"] = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")

# Callback for left arm joint states (8 joints)
def cb_joints_left(msg):
    latest_q["left"] = np.array(msg.position, dtype=np.float32)

# Callback for right arm joint states (8 joints)
def cb_joints_right(msg):
    latest_q["right"] = np.array(msg.position, dtype=np.float32)

def main():
    rospy.init_node("pi05_zeno_ghost")

    # Subscribe to camera topics (CompressedImage format as per README)
    rospy.Subscriber("/realsense_top/color/image_raw/compressed",   CompressedImage, cb_main,     queue_size=1)
    rospy.Subscriber("/realsense_left/color/image_raw/compressed",  CompressedImage, cb_wrist_l,  queue_size=1)
    rospy.Subscriber("/realsense_right/color/image_raw/compressed", CompressedImage, cb_wrist_r,  queue_size=1)

    # Subscribe to joint state topics (8 joints per arm)
    rospy.Subscriber("/robot/arm_left/joint_states_single",  JointState, cb_joints_left,  queue_size=1)
    rospy.Subscriber("/robot/arm_right/joint_states_single", JointState, cb_joints_right, queue_size=1)

    # Initialize websocket client to connect to policy server
    client = websocket_client_policy.WebsocketClientPolicy(
        host="127.0.0.1",
        port=8000,
    )

    rate = rospy.Rate(2)  # Run at 2 Hz
    prompt = "sweep the blocks into an E shape"

    while not rospy.is_shutdown():
        # Wait until all sensors have data
        if any(v is None for v in latest_imgs.values()) or any(v is None for v in latest_q.values()):
            rate.sleep()
            continue

        # Resize and convert images to 256x256 uint8 format
        img_main = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(latest_imgs["main"], 256, 256)
        )
        img_l = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(latest_imgs["wrist_l"], 256, 256)
        )
        img_r = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(latest_imgs["wrist_r"], 256, 256)
        )

        # Concatenate left and right joint positions to create state vector
        # State: 16-dim (left 8 joints + right 8 joints)
        state = np.concatenate([latest_q["left"], latest_q["right"]], axis=0)

        # Create observation dictionary matching libero_policy.py expectations
        # Keys: observation/image, observation/wrist_image, observation/right_wrist_image, observation/state
        obs = {
            "observation/image": img_main,  # Third-person (top) camera view
            "observation/wrist_image": img_l,  # Left wrist camera
            "observation/right_wrist_image": img_r,  # Right wrist camera
            "observation/state": state,  # State vector (16-dim: left 8 + right 8)
            "prompt": prompt,
        }

        # Send observation to policy server and get action prediction
        result = client.infer(obs)
        actions = np.array(result["actions"])
        a0 = actions[0]  # First action in the predicted action chunk

        # GHOST MODE: Only log the action, do not send to robot
        rospy.loginfo_throttle(1.0, f"[pi05 ghost] first action: {a0}")

        rate.sleep()

if __name__ == "__main__":
    main()
