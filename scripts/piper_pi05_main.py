#!/usr/bin/env python3
import cv2
import numpy as np
from collections import deque
import rospy
from sensor_msgs.msg import CompressedImage, JointState
from cv_bridge import CvBridge

from openpi_client import websocket_client_policy, image_tools
import threading
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

    # ========== RTC Configuration ==========
    # Based on latency test: 195ms @ different frequencies
    # RTC Constraint: d ≤ s ≤ H - d 
    #
    # Option 1: 25 Hz → d=5, s∈[5,15], choose s=8
    # Option 2: 20 Hz → d=4, s∈[4,16], choose s=8
    # Option 3: 10 Hz → d=2, s∈[2,18], choose s=10

    CONTROL_FREQ = 25     # Hz - reduced from 50Hz to satisfy RTC constraint
    ACTION_HORIZON = 20   # steps - from training config (model predicts 20 steps)

    # Calculate inference_delay from latency test
    LATENCY_MS = 198      # 95th percentile from test
    INFERENCE_DELAY = int(np.ceil(LATENCY_MS / (1000.0 / CONTROL_FREQ)))

    # Choose execute_horizon within valid range [d, H-d]
    MIN_EXECUTE_HORIZON = INFERENCE_DELAY
    MAX_EXECUTE_HORIZON = ACTION_HORIZON - INFERENCE_DELAY
    EXECUTE_HORIZON = 8   # Conservative choice within valid range

    # Validate RTC constraint
    assert INFERENCE_DELAY <= EXECUTE_HORIZON <= ACTION_HORIZON - INFERENCE_DELAY, \
        f"RTC constraint violated! d={INFERENCE_DELAY}, s={EXECUTE_HORIZON}, H-d={ACTION_HORIZON-INFERENCE_DELAY}"
        
    rtc_lock = threading.Lock()
    rtc_cond = threading.Condition(rtc_lock)

    rtc_state = {
        "cur_chunk": None,          # np.ndarray [H, 14]
        "t": 0,                     # Current time step in episode
        "last_obs": None,           # Latest observation dict
        "delay_steps": deque(maxlen=10),  # Recent delay measurements
        "inference_running": False,
        "shutdown": False,
    }
    
    

    rospy.loginfo("="*60)
    rospy.loginfo("RTC Configuration (Corrected):")
    rospy.loginfo(f"  Control frequency:  {CONTROL_FREQ} Hz (period: {1000/CONTROL_FREQ:.1f}ms)")
    rospy.loginfo(f"  Latency (95th):     {LATENCY_MS} ms")
    rospy.loginfo(f"  Inference delay:    {INFERENCE_DELAY} steps")
    rospy.loginfo(f"  Execute horizon:    {EXECUTE_HORIZON} steps")
    rospy.loginfo(f"  Action horizon:     {ACTION_HORIZON} steps")
    rospy.loginfo(f"  Valid s range:      [{MIN_EXECUTE_HORIZON}, {MAX_EXECUTE_HORIZON}]")
    rospy.loginfo(f"  RTC constraint:     {INFERENCE_DELAY} ≤ {EXECUTE_HORIZON} ≤ {ACTION_HORIZON - INFERENCE_DELAY} ✓")
    rospy.loginfo("="*60)
    rospy.loginfo("NOTE: Current implementation is 'Simplified RTC'")
    rospy.loginfo("      - Has action chunking strategy ✓")
    rospy.loginfo("      - Missing prefix attention guidance (soft-mask) ✗")
    rospy.loginfo("      - For full RTC, need to modify model inference")
    rospy.loginfo("="*60)

    rate = rospy.Rate(CONTROL_FREQ)
    prompt = "<Clear> <Box> <0.5, 0.5, 0.7, 0.7>"  # Training format prompt

    # ========== RTC State Variables ==========
    action_plan = deque()           # Queue of actions to execute
    old_chunk = None                 # Previous chunk (for RTC during inference)

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

        # ========== Step 3: RTC with Inference Delay Handling ==========
        # Check if we need to request a new action chunk
        if not action_plan:
            rospy.loginfo(f"[RTC] Action plan empty, requesting new chunk...")

            # Prepare observation
            rgb_main = cv2.cvtColor(latest_imgs["main"], cv2.COLOR_BGR2RGB)
            rgb_l = cv2.cvtColor(latest_imgs["wrist_l"], cv2.COLOR_BGR2RGB)
            rgb_r = cv2.cvtColor(latest_imgs["wrist_r"], cv2.COLOR_BGR2RGB)

            img_main = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(rgb_main, 224, 224)
            )
            img_l = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(rgb_l, 224, 224)
            )
            img_r = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(rgb_r, 224, 224)
            )

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

            # CRITICAL: During inference (195ms = 10 steps), we need to execute old actions
            # This is the core of RTC!

            # Request new action chunk from policy server (this will block for ~195ms)
            result = client.infer(obs)
            new_chunk = np.array(result["actions"])  # Shape: [ACTION_HORIZON, 14]

            rospy.loginfo(f"[RTC] Received new chunk: shape={new_chunk.shape}")

            # ========== RTC Logic: Build action_plan ==========
            # RTC Strategy from real-time-chunking-kinetix (eval_flow.py:120-135):
            #
            # action_chunk_to_execute = concatenate([
            #     old_chunk[:inference_delay],               # Old actions (during inference)
            #     new_chunk[inference_delay:execute_horizon] # New actions (after inference)
            # ])
            #
            # In our implementation:
            # - Old actions were already added to action_plan before this inference
            # - During inference (195ms), they were being executed
            # - Now we only add the NEW actions from the new chunk

            if old_chunk is not None:
                # We have an old chunk - RTC active!
                rospy.loginfo(f"[RTC] Old chunk exists, applying RTC strategy")

                # During the inference just completed (~195ms = 10 steps):
                # - Steps 0-9 of old_chunk were executing
                # - We acknowledge this by not re-adding them

                # Now add the NEW part: new_chunk[INFERENCE_DELAY:EXECUTE_HORIZON]
                start_idx = INFERENCE_DELAY   # Start from step 10
                end_idx = EXECUTE_HORIZON      # End at step 13

                for i in range(start_idx, min(end_idx, len(new_chunk))):
                    action_plan.append(new_chunk[i])

                rospy.loginfo(f"[RTC] Added {len(action_plan)} actions from new_chunk[{start_idx}:{end_idx}]")
                rospy.loginfo(f"[RTC] Total to execute: {EXECUTE_HORIZON} steps "
                             f"(inference_delay={INFERENCE_DELAY} already done + {end_idx-start_idx} new)")
            else:
                # First chunk - cold start, no RTC yet
                rospy.loginfo(f"[RTC] First chunk (cold start) - taking first {EXECUTE_HORIZON} actions")
                for i in range(min(EXECUTE_HORIZON, len(new_chunk))):
                    action_plan.append(new_chunk[i])

            # Update old_chunk for next iteration
            # Shift the chunk: discard executed portion, keep unexecuted portion
            # This simulates: next_chunk = concat([new_chunk[execute_horizon:], zeros])
            if len(new_chunk) > EXECUTE_HORIZON:
                old_chunk = new_chunk[EXECUTE_HORIZON:]  # Keep unexecuted portion
            else:
                old_chunk = np.zeros((0, 14))  # All used up

            rospy.loginfo(f"[RTC] Saved {len(old_chunk)} unexecuted actions for next cycle")

        # ========== Execute next action from plan ==========
        if action_plan:
            action = action_plan.popleft()

            # Validate action dimension
            if len(action) != 14:
                rospy.logwarn(f"[SAFETY] Invalid action dimension: expected 14, got {len(action)}. Skipping.")
                rate.sleep()
                continue

            # Split action into left and right arm commands
            action_left = action[:7]
            action_right = action[7:14]

            # Create and publish JointState messages
            cmd_left = JointState()
            cmd_left.header.stamp = rospy.Time.now()
            cmd_left.position = action_left.tolist()

            cmd_right = JointState()
            cmd_right.header.stamp = rospy.Time.now()
            cmd_right.position = action_right.tolist()

            pub_left.publish(cmd_left)
            pub_right.publish(cmd_right)

            rospy.loginfo_throttle(2.0, f"[RTC] Executing action, {len(action_plan)} remaining in plan")

        rate.sleep()

if __name__ == "__main__":
    main()
