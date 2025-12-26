#!/usr/bin/env python3
import cv2
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
    pub_left = rospy.Publisher("/robot/arm_left/vla_joint_cmd", JointState, queue_size=1)
    pub_right = rospy.Publisher("/robot/arm_right/vla_joint_cmd", JointState, queue_size=1)

    rospy.loginfo("Robot arm command publishers initialized")

    # Initialize websocket client to connect to policy server
    client = websocket_client_policy.WebsocketClientPolicy(
        host="127.0.0.1",
        port=8000,
    )

    # SAFETY: Use low control frequency for initial testing
    rate = rospy.Rate(10)  # Run at 10 Hz to match training data collection frequency
    prompt = "Push the block to the right and then move both arms back to the home pose."

    # Sliding window action buffer configuration
    action_buffer = None  # Stores the current predicted action chunk
    action_index = 0      # Current position in the action buffer
    replan_threshold = 4  # Re-predict when fewer than this many actions remain

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

        # Convert images from BGR (ROS default) to RGB
        rgb_main = cv2.cvtColor(latest_imgs["main"], cv2.COLOR_BGR2RGB)
        rgb_l = cv2.cvtColor(latest_imgs["wrist_l"], cv2.COLOR_BGR2RGB)
        rgb_r = cv2.cvtColor(latest_imgs["wrist_r"], cv2.COLOR_BGR2RGB)

        # Resize and convert images to 256x256 uint8 format
        img_main = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(rgb_main, 256, 256)
        )
        img_l = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(rgb_l, 256, 256)
        )
        img_r = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(rgb_r, 256, 256)
        )

        # Only use first 7 joints per arm (aligned with LeRobot dataset)
        q_left = latest_q["left"][:7].astype(np.float32)
        q_right = latest_q["right"][:7].astype(np.float32)

        # Concatenate left and right joint positions to create state vector
        state = np.concatenate([q_left, q_right], axis=0)

        # Check if we need to re-predict (buffer empty or running low on actions)
        if action_buffer is None or action_index >= len(action_buffer) - replan_threshold:
            # Keys: observation/image, observation/wrist_image, observation/right_wrist_image, observation/state
            obs = {
                "observation/image": img_main,
                "observation/wrist_image": img_l,
                "observation/right_wrist_image": img_r,
                "observation/state": state,
                "prompt": prompt,
            }

            # Send observation to policy server and get action prediction
            rospy.loginfo("Requesting new action chunk from policy server...")
            result = client.infer(obs)
            rospy.loginfo("✓ Successfully cxxommunicated with policy server")

            action_buffer = np.array(result["actions"])
            action_index = 0
            rospy.loginfo(f"✓ Received new action chunk with {len(action_buffer)} actions")

        # Get the current action from the buffer
        action = action_buffer[action_index]

        # Validate action dimension
        if len(action) != 14:
            rospy.logwarn(f"[SAFETY] Invalid action dimension: expected 14, got {len(action)}. Skipping this action.")
            action_index += 1
            rate.sleep()
            continue

        # Split action into left and right arm commands (14-dim total: 7 joints per arm)
        # Actions are absolute joint positions from the policy
        action_left = action[:7]
        action_right = action[7:14]

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

        remaining = len(action_buffer) - action_index
        rospy.loginfo(f"✓ Executed action {action_index+1}/{len(action_buffer)} (remaining: {remaining-1})")
        rospy.logdebug(f"  Left arm:  [{', '.join([f'{x:.3f}' for x in action_left])}]")
        rospy.logdebug(f"  Right arm: [{', '.join([f'{x:.3f}' for x in action_right])}]")

        # Move to next action in buffer
        action_index += 1

        rate.sleep() 

if __name__ == "__main__":
    main()
