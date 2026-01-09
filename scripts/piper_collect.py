#!/usr/bin/env python3
"""
piper_collect.py - Policy deployment + automatic data collection script

Features:
- Run policy inference and execute actions
- Automatically control rosbag recording
- Support sweep/recover task alternation
- Interactive control: ENTER to stop recording, n/r/q to choose action

Usage:
  # Collect data for letter E, starting with sweep task
  python piper_collect.py --letter E --start-task sweep

  # Collect data for letter A, starting with recover task
  python piper_collect.py --letter A --start-task recover
"""

import cv2
import numpy as np
import rospy
import argparse
import subprocess
import signal
import threading
import os
import sys
import time
from sensor_msgs.msg import CompressedImage, JointState
from cv_bridge import CvBridge

from openpi_client import websocket_client_policy, image_tools

bridge = CvBridge()

# ============================================================
# ROS TOPICS (aligned with convert_bag2lerobot21_dualarm.py)
# ============================================================
# Camera topics
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

# Topics to record (same as record_2task.sh)
RECORD_TOPICS = [
    "/robot/arm_left/end_pose",
    "/robot/arm_right/end_pose",
    "/robot/arm_left/joint_states_single",
    "/robot/arm_right/joint_states_single",
    "/robot/arm_left/pos_cmd",
    "/robot/arm_right/pos_cmd",
    "/teleop/arm_left/joint_states_single",
    "/teleop/arm_right/joint_states_single",
    # Fisheye cameras
    "/fisheye_left/image_raw/compressed",
    "/fisheye_right/image_raw/compressed",
    "/fisheye_left/camera_info",
    "/fisheye_right/camera_info",
    # Realsense top camera
    "/realsense_top/color/image_raw/compressed",
    "/realsense_top/aligned_depth_to_color/image_raw/compressed",
    "/realsense_top/color/camera_info",
    "/realsense_top/aligned_depth_to_color/camera_info",
    # Wide top camera
    "/wide_top/image_raw/compressed",
    "/wide_top/camera_info",
]

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

# Optional sweep mask
sweep_mask_image = None


# ============================================================
# ROS CALLBACKS
# ============================================================
def cb_main(msg):
    latest_imgs["main"] = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")

def cb_wrist_l(msg):
    latest_imgs["wrist_l"] = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")

def cb_wrist_r(msg):
    latest_imgs["wrist_r"] = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")

def cb_wide_top(msg):
    latest_imgs["wide_top"] = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")

def cb_joints_left(msg):
    latest_q["left"] = np.array(msg.position, dtype=np.float32)

def cb_joints_right(msg):
    latest_q["right"] = np.array(msg.position, dtype=np.float32)


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


# Base directory for data collection
DATA_COLLECT_DIR = "/home/zeno-yifan/NPM-Project/NPM-Ros/piper_ros/data_collect"


class DataCollector:
    """Data collector: run policy and automatically record rosbag"""

    def __init__(self, args):
        self.args = args
        self.letter = args.letter

        # Task configuration (dynamically generate prompt based on letter)
        self.tasks = {
            "sweep": f"<skill>sweep<skill> Sweep red beads into letter '{self.letter}' shape inside the masked square.",
            "recover": f"<skill>recover<skill> Gather the red beads into a dense, contiguous pile inside the marked square with minimal gaps.",
        }

        # Recording directory and prefix (under DATA_COLLECT_DIR)
        sweep_dir = os.path.join(DATA_COLLECT_DIR, f"sweep_to_{self.letter}_AC")
        recover_dir = os.path.join(DATA_COLLECT_DIR, f"recover_from_{self.letter}_AC")
        self.sweep_prefix = os.path.join(sweep_dir, f"sweep_to_{self.letter}_AC_")
        self.recover_prefix = os.path.join(recover_dir, f"recover_from_{self.letter}_AC_")

        # Create directories
        os.makedirs(sweep_dir, exist_ok=True)
        os.makedirs(recover_dir, exist_ok=True)

        # Current task
        self.current_task = args.start_task

        # Recording index (auto-find next available index)
        self.sweep_idx = self._find_next_idx(self.sweep_prefix)
        self.recover_idx = self._find_next_idx(self.recover_prefix)

        # Control flags
        self.stop_episode = threading.Event()
        self.rosbag_proc = None

        # Policy client
        self.client = websocket_client_policy.WebsocketClientPolicy(
            host=args.host,
            port=args.port,
        )
        rospy.loginfo(f"Connected to policy server at {args.host}:{args.port}")

        # Action buffer
        self.action_buffer = None
        self.action_index = 0
        self.replan_threshold = 4

        # ROS publishers
        self.pub_left = rospy.Publisher(TOPIC_CMD_LEFT, JointState, queue_size=1)
        self.pub_right = rospy.Publisher(TOPIC_CMD_RIGHT, JointState, queue_size=1)

    def _find_next_idx(self, prefix):
        """Find next available index for bag files"""
        i = 0
        while os.path.exists(f"{prefix}{i:03d}.bag"):
            i += 1
        return i

    def start_recording(self, bag_path):
        """Start rosbag recording"""
        cmd = ["rosbag", "record", "-O", bag_path, "--bz2", "-b", "4096"] + RECORD_TOPICS
        self.rosbag_proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        rospy.loginfo(f"[REC] Started recording to {bag_path} (PID: {self.rosbag_proc.pid})")

    def stop_recording(self):
        """Stop rosbag recording"""
        if self.rosbag_proc:
            rospy.loginfo("[REC] Stopping rosbag (SIGINT)...")
            self.rosbag_proc.send_signal(signal.SIGINT)
            self.rosbag_proc.wait()
            self.rosbag_proc = None
            # Ensure disk write completes
            time.sleep(1)
            rospy.loginfo("[REC] Recording stopped")

    def _wait_for_enter(self):
        """Wait for user to press ENTER (runs in separate thread)"""
        try:
            input()
            self.stop_episode.set()
        except EOFError:
            pass

    def _get_observation(self):
        """Get current observation"""
        # Check if data is ready
        required_imgs = ["main", "wrist_l", "wrist_r"]
        if self.args.use_wide_top:
            required_imgs.append("wide_top")

        if any(latest_imgs[k] is None for k in required_imgs):
            return None
        if any(v is None for v in latest_q.values()):
            return None

        # Convert images
        rgb_main = cv2.cvtColor(latest_imgs["main"], cv2.COLOR_BGR2RGB)
        rgb_l = cv2.cvtColor(latest_imgs["wrist_l"], cv2.COLOR_BGR2RGB)
        rgb_r = cv2.cvtColor(latest_imgs["wrist_r"], cv2.COLOR_BGR2RGB)

        img_main = image_tools.convert_to_uint8(image_tools.resize_with_pad(rgb_main, 256, 256))
        img_l = image_tools.convert_to_uint8(image_tools.resize_with_pad(rgb_l, 256, 256))
        img_r = image_tools.convert_to_uint8(image_tools.resize_with_pad(rgb_r, 256, 256))

        img_wide_top = None
        if self.args.use_wide_top and latest_imgs["wide_top"] is not None:
            rgb_wide_top = cv2.cvtColor(latest_imgs["wide_top"], cv2.COLOR_BGR2RGB)
            img_wide_top = image_tools.convert_to_uint8(image_tools.resize_with_pad(rgb_wide_top, 256, 256))

        # Joint states
        q_left = latest_q["left"][:7].astype(np.float32)
        q_right = latest_q["right"][:7].astype(np.float32)
        state = np.concatenate([q_left, q_right], axis=0)

        return {
            "img_main": img_main,
            "img_l": img_l,
            "img_r": img_r,
            "img_wide_top": img_wide_top,
            "state": state,
        }

    def _execute_action(self, action):
        """Execute a single action"""
        if len(action) != 14:
            rospy.logwarn(f"[SAFETY] Invalid action dimension: {len(action)}, skipping")
            return False

        action_left = action[:7]
        action_right = action[7:14]

        cmd_left = JointState()
        cmd_left.header.stamp = rospy.Time.now()
        cmd_left.position = action_left.tolist()

        cmd_right = JointState()
        cmd_right.header.stamp = rospy.Time.now()
        cmd_right.position = action_right.tolist()

        self.pub_left.publish(cmd_left)
        self.pub_right.publish(cmd_right)
        return True

    def run_episode(self):
        """Run one episode, returns (bag_path, success)"""
        # Get current task configuration
        if self.current_task == "sweep":
            prefix = self.sweep_prefix
            idx = self.sweep_idx
            task_name = "SWEEP"
        else:
            prefix = self.recover_prefix
            idx = self.recover_idx
            task_name = "RECOVER"

        bag_path = f"{prefix}{idx:03d}.bag"
        prompt = self.tasks[self.current_task]

        print("\n" + "=" * 60)
        print(f"[EPISODE] Task: {task_name} | Index: {idx}")
        print(f"[EPISODE] Bag: {bag_path}")
        print(f"[EPISODE] Prompt: {prompt[:60]}...")
        print("=" * 60)

        # Wait for sensor data
        print("[WAIT] Waiting for sensor data...")
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            obs = self._get_observation()
            if obs is not None:
                break
            rate.sleep()
        print("[WAIT] Sensor data ready")

        # Pre-fetch first action chunk (warm up policy before recording)
        print("[WARM] Pre-fetching first action chunk...")
        obs = self._get_observation()
        obs_dict = {
            "observation/image": obs["img_main"],
            "observation/wrist_image": obs["img_l"],
            "observation/right_wrist_image": obs["img_r"],
            "observation/state": obs["state"],
            "prompt": prompt,
        }
        if obs["img_wide_top"] is not None:
            obs_dict["observation/wide_top_image"] = obs["img_wide_top"]
        if sweep_mask_image is not None:
            obs_dict["observation/sweep_mask"] = sweep_mask_image

        result = self.client.infer(obs_dict)
        self.action_buffer = np.array(result["actions"])
        self.action_index = 0
        print(f"[WARM] Got {len(self.action_buffer)} actions, ready to start")

        # User confirmation to start
        input("\nPress ENTER to start recording (press ENTER again to stop)...")

        # Start recording
        self.start_recording(bag_path)

        # Start input listener thread
        self.stop_episode.clear()
        input_thread = threading.Thread(target=self._wait_for_enter)
        input_thread.daemon = True
        input_thread.start()

        print("[RUN] Episode running... Press ENTER to stop")

        # Run policy inference loop
        rate = rospy.Rate(self.args.hz)
        step_count = 0

        while not rospy.is_shutdown() and not self.stop_episode.is_set():
            obs = self._get_observation()
            if obs is None:
                rate.sleep()
                continue

            # Check if need to re-predict
            if self.action_buffer is None or self.action_index >= len(self.action_buffer) - self.replan_threshold:
                # Build observation dict
                obs_dict = {
                    "observation/image": obs["img_main"],
                    "observation/wrist_image": obs["img_l"],
                    "observation/right_wrist_image": obs["img_r"],
                    "observation/state": obs["state"],
                    "prompt": prompt,
                }

                if obs["img_wide_top"] is not None:
                    obs_dict["observation/wide_top_image"] = obs["img_wide_top"]

                if sweep_mask_image is not None:
                    obs_dict["observation/sweep_mask"] = sweep_mask_image

                # Request new action chunk
                result = self.client.infer(obs_dict)
                self.action_buffer = np.array(result["actions"])
                self.action_index = 0
                rospy.logdebug(f"Got new action chunk with {len(self.action_buffer)} actions")

            # Execute current action
            action = self.action_buffer[self.action_index]
            if self._execute_action(action):
                step_count += 1
                self.action_index += 1

            rate.sleep()

        # Stop recording
        self.stop_recording()

        print(f"[DONE] Episode finished, {step_count} steps executed")

        # Check if file exists
        if not os.path.exists(bag_path):
            rospy.logwarn(f"[WARN] Bag file not found: {bag_path}")
            return bag_path, False

        return bag_path, True

    def run(self):
        """Main loop"""
        print("\n" + "=" * 60)
        print("  PiPER Auto Data Collection")
        print("=" * 60)
        print(f"  Letter: {self.letter}")
        print(f"  SWEEP dir: {os.path.dirname(self.sweep_prefix)}/")
        print(f"  RECOVER dir: {os.path.dirname(self.recover_prefix)}/")
        print(f"  SWEEP start index: {self.sweep_idx}")
        print(f"  RECOVER start index: {self.recover_idx}")
        print("=" * 60)
        print("\nControls:")
        print("  ENTER: Start/stop recording")
        print("  n: Keep recording, switch to next task")
        print("  r: Delete recording, re-record current task")
        print("  q: Quit")
        print()

        while not rospy.is_shutdown():
            # Run one episode
            bag_path, success = self.run_episode()

            if not success:
                print("[WARN] Recording may have failed, will retry...")
                continue

            # Ask user for action
            next_task = "RECOVER" if self.current_task == "sweep" else "SWEEP"
            print(f"\nRecording complete: {bag_path}")
            print(f"  [n] Keep and switch to {next_task}")
            print(f"  [r] Delete and re-record current task")
            print(f"  [q] Quit")

            choice = input("Choice [n/r/q]: ").strip().lower()

            if choice == 'r':
                try:
                    os.remove(bag_path)
                    rospy.loginfo(f"[DEL] Deleted {bag_path}")
                except Exception as e:
                    rospy.logwarn(f"[DEL] Failed to delete: {e}")
            elif choice == 'q':
                print("Exiting collection.")
                break
            else:
                # Keep and switch task (default to n)
                if self.current_task == "sweep":
                    self.sweep_idx += 1
                    self.current_task = "recover"
                else:
                    self.recover_idx += 1
                    self.current_task = "sweep"


def parse_args():
    parser = argparse.ArgumentParser(description="PiPER Auto Data Collection")
    parser.add_argument(
        "--letter", "-l",
        type=str,
        default="E",
        help="Target letter (default: E)"
    )
    parser.add_argument(
        "--start-task",
        type=str,
        choices=["sweep", "recover"],
        default="sweep",
        help="Starting task (default: sweep)"
    )
    parser.add_argument(
        "--sweep-mask",
        type=str,
        default="",
        help="Path to sweep mask image (optional)"
    )
    parser.add_argument(
        "--use-wide-top",
        action="store_true",
        help="Enable wide top camera"
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
    args = parse_args()

    rospy.init_node("pi05_piper_collect")

    # Load sweep mask
    if args.sweep_mask:
        load_sweep_mask(args.sweep_mask)

    # Setup ROS subscribers
    rospy.Subscriber(TOPIC_CAM_MAIN, CompressedImage, cb_main, queue_size=1)
    rospy.Subscriber(TOPIC_CAM_WRIST_LEFT, CompressedImage, cb_wrist_l, queue_size=1)
    rospy.Subscriber(TOPIC_CAM_WRIST_RIGHT, CompressedImage, cb_wrist_r, queue_size=1)

    if args.use_wide_top:
        rospy.Subscriber(TOPIC_CAM_WIDE_TOP, CompressedImage, cb_wide_top, queue_size=1)

    rospy.Subscriber(TOPIC_STATE_LEFT, JointState, cb_joints_left, queue_size=1)
    rospy.Subscriber(TOPIC_STATE_RIGHT, JointState, cb_joints_right, queue_size=1)

    # Create and run collector
    collector = DataCollector(args)

    try:
        collector.run()
    except KeyboardInterrupt:
        print("\n[EXIT] Interrupted by user")
        collector.stop_recording()
    except Exception as e:
        rospy.logerr(f"[ERROR] {e}")
        collector.stop_recording()
        raise


if __name__ == "__main__":
    main()