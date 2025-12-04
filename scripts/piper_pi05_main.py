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

# Smoothing: Store previous smoothed actions for EMA filter
smoothed_action = {
    "left": None,
    "right": None,
}

# Callback for main (top) camera
def cb_main(msg):
    latest_imgs["main"] = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
    rospy.logdebug("Received main camera image")

# Callback for left wrist camera
def cb_wrist_l(msg):
    latest_imgs["wrist_l"] = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
    rospy.logdebug("Received left wrist camera image")

# Callback for right wrist camera
def cb_wrist_r(msg):
    latest_imgs["wrist_r"] = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
    rospy.logdebug("Received right wrist camera image")


def cb_joints_left(msg):
    latest_q["left"] = np.array(msg.position, dtype=np.float32)
    rospy.logdebug(f"Received left arm joint states: {latest_q['left'][:3]}...")

# Callback for right arm joint states
def cb_joints_right(msg):
    latest_q["right"] = np.array(msg.position, dtype=np.float32)
    rospy.logdebug(f"Received right arm joint states: {latest_q['right'][:3]}...")

def main():
    rospy.init_node("pi05_zeno_main")

    # Subscribe to camera topics
    rospy.Subscriber("/realsense_top/color/image_raw/compressed",   CompressedImage, cb_main,     queue_size=1)
    rospy.Subscriber("/realsense_left/color/image_raw/compressed",  CompressedImage, cb_wrist_l,  queue_size=1)
    rospy.Subscriber("/realsense_right/color/image_raw/compressed", CompressedImage, cb_wrist_r,  queue_size=1)

    # Subscribe to joint state topics
    rospy.Subscriber("/robot/arm_left/joint_states_single",  JointState, cb_joints_left,  queue_size=1)
    rospy.Subscriber("/robot/arm_right/joint_states_single", JointState, cb_joints_right, queue_size=1)

    # Create publishers to send actions to robot arms
    pub_left = rospy.Publisher("/robot/arm_left/vla_pos_cmd", JointState, queue_size=1)
    pub_right = rospy.Publisher("/robot/arm_right/vla_pos_cmd", JointState, queue_size=1)

    rospy.loginfo("Robot arm command publishers initialized")

    # Initialize websocket client to connect to policy server
    client = websocket_client_policy.WebsocketClientPolicy(
        host="127.0.0.1",
        port=8000,
    )

    # SAFETY: Use low control frequency for initial testing
    rate = rospy.Rate(10)  # Run at 1 Hz (once every second) for safety
    prompt = "pass cucumber from left to right"

    # Safety parameters
    MAX_JOINT_DELTA = 0.15 # Maximum joint position change per step (radians)
    ENABLE_ACTION_CLIPPING = False  # Clip large action deltas for safety

    # Smoothing parameters
    ENABLE_SMOOTHING = False  # Enable EMA smoothing for smoother motion
    SMOOTHING_ALPHA = 0.3  # EMA smoothing factor (0-1): lower=smoother but slower, higher=more responsive but less smooth

    rospy.loginfo("Waiting for sensor data from robot arms...")
    data_ready_logged = False

    while not rospy.is_shutdown():
        # Wait until all sensors have data
        if any(v is None for v in latest_imgs.values()) or any(v is None for v in latest_q.values()):
            rate.sleep()
            continue

        if not data_ready_logged:
            rospy.loginfo("✓ Successfully receiving observations from robot arms (cameras + joint states)")
            data_ready_logged = True

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
        state = np.concatenate([latest_q["left"], latest_q["right"]], axis=0)

        # Keys: observation/image, observation/wrist_image, observation/right_wrist_image, observation/state
        obs = {
            "observation/image": img_main,  
            "observation/wrist_image": img_l,  
            "observation/right_wrist_image": img_r,  
            "observation/state": state,  
            "prompt": prompt,
        }

        # Send observation to policy server and get action prediction
        rospy.logdebug("Sending observation to policy server...")
        result = client.infer(obs)
        rospy.loginfo_throttle(5.0, "✓ Successfully communicated with policy server")

        actions = np.array(result["actions"])
        a0 = actions[0]  # First action in the predicted action chunk
        rospy.loginfo_throttle(5.0, f"✓ Successfully received action from policy server, shape: {actions.shape}")

        # Validate action dimension
        if len(a0) != 14:
            rospy.logwarn(f"[SAFETY] Invalid action dimension: expected 14, got {len(a0)}. Skipping this action.")
            rate.sleep()
            continue

        # Split action into left and right arm commands (14-dim total: 7 joints per arm)
        # Assuming action is 14-dim: [left_7_joints, right_7_joints]
        action_left = a0[:7]
        action_right = a0[7:14]

        # SMOOTHING: Apply EMA (Exponential Moving Average) filter for smoother motion
        if ENABLE_SMOOTHING:
            if smoothed_action["left"] is None:
                # Initialize with current action on first run
                smoothed_action["left"] = action_left.copy()
                smoothed_action["right"] = action_right.copy()
            else:
                # Apply EMA: smoothed = alpha * new + (1-alpha) * previous
                smoothed_action["left"] = SMOOTHING_ALPHA * action_left + (1 - SMOOTHING_ALPHA) * smoothed_action["left"]
                smoothed_action["right"] = SMOOTHING_ALPHA * action_right + (1 - SMOOTHING_ALPHA) * smoothed_action["right"]

                action_left = smoothed_action["left"]
                action_right = smoothed_action["right"]
                rospy.logdebug(f"Applied EMA smoothing with alpha={SMOOTHING_ALPHA}")

        # SAFETY: Check and clip action deltas if needed
        if ENABLE_ACTION_CLIPPING:
            delta_left = action_left - latest_q["left"][:7]
            delta_right = action_right - latest_q["right"][:7]

            max_delta_left = np.abs(delta_left).max()
            max_delta_right = np.abs(delta_right).max()

            if max_delta_left > MAX_JOINT_DELTA:
                rospy.logwarn(f"[SAFETY] Left arm delta too large ({max_delta_left:.3f}), clipping to {MAX_JOINT_DELTA}")
                delta_left = np.clip(delta_left, -MAX_JOINT_DELTA, MAX_JOINT_DELTA)
                action_left = latest_q["left"][:7] + delta_left

            if max_delta_right > MAX_JOINT_DELTA:
                rospy.logwarn(f"[SAFETY] Right arm delta too large ({max_delta_right:.3f}), clipping to {MAX_JOINT_DELTA}")
                delta_right = np.clip(delta_right, -MAX_JOINT_DELTA, MAX_JOINT_DELTA)
                action_right = latest_q["right"][:7] + delta_right

        # Create JointState messages for both arms
        cmd_left = JointState()
        cmd_left.header.stamp = rospy.Time.now()
        cmd_left.position = action_left.tolist()

        cmd_right = JointState()
        cmd_right.header.stamp = rospy.Time.now()
        cmd_right.position = action_right.tolist()

        # Publish commands to robot arms
        pub_left.publish(cmd_left)
        pub_right.publish(cmd_right)
        rospy.loginfo(f"✓ Successfully sent actions to robot arms")
        rospy.loginfo(f"  Left arm:  [{', '.join([f'{x:.3f}' for x in action_left])}]")
        rospy.loginfo(f"  Right arm: [{', '.join([f'{x:.3f}' for x in action_right])}]")

        rate.sleep()

if __name__ == "__main__":
    main()
