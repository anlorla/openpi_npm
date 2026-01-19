#!/usr/bin/env python3
import cv2
import numpy as np
import rospy
import argparse
from sensor_msgs.msg import CompressedImage, JointState
from cv_bridge import CvBridge

from openpi_client import websocket_client_policy, image_tools

bridge = CvBridge()

# ============================================================
# PREDEFINED SKILLS / PROMPTS
# ============================================================
SKILL_PROMPTS = {
    "sweep": "<skill>sweep<skill> Sweep red beads into letter 'U' shape inside the masked squre.",
    "recover": "<skill>recover<skill> Gather the red beads into a dense, contiguous pile inside the marked square with minimal gaps.",
}

# ============================================================
# ROS TOPICS (aligned with convert_bag2lerobot21_dualarm.py)
# ============================================================
# Camera topics (updated for fisheye cameras)
TOPIC_CAM_MAIN = "/realsense_top/color/image_raw/compressed"
TOPIC_CAM_WRIST_LEFT = "/fisheye_left/image_raw/compressed"
TOPIC_CAM_WRIST_RIGHT = "/fisheye_right/image_raw/compressed"
TOPIC_CAM_WIDE_TOP = "/wide_top/image_raw/compressed"

# Joint state topics
TOPIC_STATE_LEFT = "/robot/arm_left/joint_states_single"
TOPIC_STATE_RIGHT = "/robot/arm_right/joint_states_single"

# Action command topics
TOPIC_CMD_LEFT = "/robot/arm_left/vla_joint_cmd"
TOPIC_CMD_RIGHT = "/robot/arm_right/vla_joint_cmd"

# Store latest sensor data
latest_imgs = {
    "main": None,
    "wrist_l": None,
    "wrist_r": None,
    "wide_top": None,
}
latest_q = {
    "left": None,
    "right": None,
}

# Optional sweep mask (loaded from file if provided)
sweep_mask_image = None

# ============================================================
# ROS CALLBACKS
# ============================================================
def cb_main(msg):
    """Callback for main (top) camera - realsense_top"""
    latest_imgs["main"] = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
    rospy.logdebug("Received main camera image")

def cb_wrist_l(msg):
    """Callback for left wrist camera - fisheye_left"""
    latest_imgs["wrist_l"] = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
    rospy.logdebug("Received left wrist camera image")

def cb_wrist_r(msg):
    """Callback for right wrist camera - fisheye_right"""
    latest_imgs["wrist_r"] = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
    rospy.logdebug("Received right wrist camera image")

def cb_wide_top(msg):
    """Callback for wide top camera - wide_top"""
    latest_imgs["wide_top"] = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
    rospy.logdebug("Received wide top camera image")

def cb_joints_left(msg):
    """Callback for left arm joint states"""
    latest_q["left"] = np.array(msg.position, dtype=np.float32)
    rospy.logdebug(f"Received left arm joint states: {latest_q['left'][:3]}...")

def cb_joints_right(msg):
    """Callback for right arm joint states"""
    latest_q["right"] = np.array(msg.position, dtype=np.float32)
    rospy.logdebug(f"Received right arm joint states: {latest_q['right'][:3]}...")

def select_prompt_interactive():
    """Interactive prompt selection menu."""
    print("\n" + "=" * 60)
    print("SELECT A SKILL/PROMPT:")
    print("=" * 60)

    skills = list(SKILL_PROMPTS.keys())
    for i, skill in enumerate(skills, 1):
        print(f"  [{i}] {skill}: {SKILL_PROMPTS[skill][:60]}...")
    print(f"  [c] Custom prompt (enter your own)")
    print("=" * 60)

    while True:
        choice = input("Enter your choice: ").strip().lower()
        if choice == 'c':
            custom = input("Enter your custom prompt: ").strip()
            if custom:
                return custom
            print("Prompt cannot be empty. Try again.")
        elif choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(skills):
                return SKILL_PROMPTS[skills[idx]]
            print(f"Invalid choice. Enter 1-{len(skills)} or 'c'.")
        else:
            print(f"Invalid input. Enter 1-{len(skills)} or 'c'.")

def load_sweep_mask(mask_path):
    """Load sweep mask image from file."""
    global sweep_mask_image
    if mask_path and mask_path != "":
        mask_img = cv2.imread(mask_path)
        if mask_img is not None:
            mask_rgb = cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB)
            sweep_mask_image = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(mask_rgb, 256, 256)
            )
            rospy.loginfo(f"Loaded sweep mask from: {mask_path}")
        else:
            rospy.logwarn(f"Failed to load sweep mask from: {mask_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="PiPER dual-arm robot deployment")
    parser.add_argument(
        "--skill", "-s",
        type=str,
        choices=list(SKILL_PROMPTS.keys()),
        help="Select a predefined skill (sweep/recover)"
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        help="Custom prompt string (overrides --skill)"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Interactive prompt selection mode"
    )
    parser.add_argument(
        "--sweep-mask",
        type=str,
        default="",
        help="Path to sweep mask image file (optional)"
    )
    parser.add_argument(
        "--use-wide-top",
        action="store_true",
        help="Enable wide top camera as 4th image input"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Policy server host (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Policy server port (default: 8000)"
    )
    parser.add_argument(
        "--hz",
        type=int,
        default=10,
        help="Control frequency in Hz (default: 10)"
    )
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()

    rospy.init_node("pi05_piper_main")

    # ============================================================
    # PROMPT SELECTION
    # ============================================================
    if args.prompt:
        prompt = args.prompt
        rospy.loginfo(f"Using custom prompt: {prompt}")
    elif args.skill:
        prompt = SKILL_PROMPTS[args.skill]
        rospy.loginfo(f"Using skill [{args.skill}]: {prompt}")
    elif args.interactive:
        prompt = select_prompt_interactive()
        rospy.loginfo(f"Selected prompt: {prompt}")
    else:
        # Default to interactive mode if no prompt specified
        rospy.loginfo("No prompt specified, entering interactive selection mode...")
        prompt = select_prompt_interactive()
        rospy.loginfo(f"Selected prompt: {prompt}")

    # Load sweep mask if provided
    if args.sweep_mask:
        load_sweep_mask(args.sweep_mask)

    # ============================================================
    # ROS SUBSCRIBERS
    # ============================================================
    rospy.Subscriber(TOPIC_CAM_MAIN, CompressedImage, cb_main, queue_size=1)
    rospy.Subscriber(TOPIC_CAM_WRIST_LEFT, CompressedImage, cb_wrist_l, queue_size=1)
    rospy.Subscriber(TOPIC_CAM_WRIST_RIGHT, CompressedImage, cb_wrist_r, queue_size=1)

    if args.use_wide_top:
        rospy.Subscriber(TOPIC_CAM_WIDE_TOP, CompressedImage, cb_wide_top, queue_size=1)
        rospy.loginfo("Wide top camera enabled")

    rospy.Subscriber(TOPIC_STATE_LEFT, JointState, cb_joints_left, queue_size=1)
    rospy.Subscriber(TOPIC_STATE_RIGHT, JointState, cb_joints_right, queue_size=1)

    # ============================================================
    # ROS PUBLISHERS
    # ============================================================
    pub_left = rospy.Publisher(TOPIC_CMD_LEFT, JointState, queue_size=1)
    pub_right = rospy.Publisher(TOPIC_CMD_RIGHT, JointState, queue_size=1)

    rospy.loginfo("Robot arm command publishers initialized")

    # ============================================================
    # POLICY CLIENT
    # ============================================================
    client = websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )
    rospy.loginfo(f"Connected to policy server at {args.host}:{args.port}")

    # Control loop settings
    rate = rospy.Rate(args.hz)
    rospy.loginfo(f"Control frequency: {args.hz} Hz")

    # Sliding window action buffer configuration
    action_buffer = None  # Stores the current predicted action chunk
    action_index = 0      # Current position in the action buffer
    replan_threshold = 4  # Re-predict when fewer than this many actions remain

    rospy.loginfo("Waiting for sensor data from robot arms...")
    rospy.loginfo(f"  - Main camera: {TOPIC_CAM_MAIN}")
    rospy.loginfo(f"  - Left wrist camera: {TOPIC_CAM_WRIST_LEFT}")
    rospy.loginfo(f"  - Right wrist camera: {TOPIC_CAM_WRIST_RIGHT}")
    if args.use_wide_top:
        rospy.loginfo(f"  - Wide top camera: {TOPIC_CAM_WIDE_TOP}")
    rospy.loginfo(f"  - Left arm joints: {TOPIC_STATE_LEFT}")
    rospy.loginfo(f"  - Right arm joints: {TOPIC_STATE_RIGHT}")

    data_ready_logged = False

    while not rospy.is_shutdown():
        # Determine which images are required
        required_imgs = ["main", "wrist_l", "wrist_r"]
        if args.use_wide_top:
            required_imgs.append("wide_top")

        # Wait until all required sensors have data
        if any(latest_imgs[k] is None for k in required_imgs) or any(v is None for v in latest_q.values()):
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

        # Process wide_top camera if enabled
        img_wide_top = None
        if args.use_wide_top and latest_imgs["wide_top"] is not None:
            rgb_wide_top = cv2.cvtColor(latest_imgs["wide_top"], cv2.COLOR_BGR2RGB)
            img_wide_top = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(rgb_wide_top, 256, 256)
            )

        # Only use first 7 joints per arm (aligned with LeRobot dataset)
        q_left = latest_q["left"][:7].astype(np.float32)
        q_right = latest_q["right"][:7].astype(np.float32)

        # Concatenate left and right joint positions to create state vector
        state = np.concatenate([q_left, q_right], axis=0)

        # Check if we need to re-predict (buffer empty or running low on actions)
        if action_buffer is None or action_index >= len(action_buffer) - replan_threshold:
            # Build observation dictionary
            # Keys match piper_policy.py: observation/image, observation/wrist_image,
            # observation/right_wrist_image, observation/state
            obs = {
                "observation/image": img_main,
                "observation/wrist_image": img_l,
                "observation/right_wrist_image": img_r,
                "observation/state": state,
                "prompt": prompt,
            }

            # Optionally add wide_top image (4th image input)
            if img_wide_top is not None:
                obs["observation/wide_top_image"] = img_wide_top

            # Optionally add sweep_mask (loaded from file)
            if sweep_mask_image is not None:
                obs["observation/sweep_mask"] = sweep_mask_image

            # Send observation to policy server and get action prediction
            rospy.loginfo("Requesting new action chunk from policy server...")
            result = client.infer(obs)
            rospy.loginfo("✓ Successfully communicated with policy server")

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
