#!/usr/bin/env python3
import cv2
import numpy as np
import time
from typing import List, Dict
import statistics

import rospy
from sensor_msgs.msg import CompressedImage, JointState
from cv_bridge import CvBridge

from openpi_client import websocket_client_policy, image_tools

# Global variables to store latest sensor data
bridge = CvBridge()
latest_imgs = {
    "main": None,
    "wrist_l": None,
    "wrist_r": None,
}
latest_q = {
    "left": None,
    "right": None,
}

# Callback functions
def cb_main(msg):
    latest_imgs["main"] = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")

def cb_wrist_l(msg):
    latest_imgs["wrist_l"] = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")

def cb_wrist_r(msg):
    latest_imgs["wrist_r"] = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")

def cb_joints_left(msg):
    latest_q["left"] = np.array(msg.position, dtype=np.float32)

def cb_joints_right(msg):
    latest_q["right"] = np.array(msg.position, dtype=np.float32)


def prepare_observation(prompt: str) -> Dict:
    """
    Prepare observation from latest sensor data.
    Includes image processing time in the measurement.
    """
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

    # Only use first 7 joints per arm
    q_left = latest_q["left"][:7].astype(np.float32)
    q_right = latest_q["right"][:7].astype(np.float32)

    # Concatenate left and right joint positions
    state = np.concatenate([q_left, q_right], axis=0)

    return {
        "observation/image": img_main,
        "observation/wrist_image": img_l,
        "observation/right_wrist_image": img_r,
        "observation/state": state,
        "prompt": prompt,
    }


def measure_inference_latency_with_real_data(
    client: websocket_client_policy.WebsocketClientPolicy,
    prompt: str,
    num_trials: int = 10,
    warmup_trials: int = 3,
) -> Dict[str, float]:
    """
    Measure COMPLETE end-to-end inference latency including:
    - Image reading and processing
    - Network communication
    - Model inference

    This measures the TRUE latency your robot will experience.

    Args:
        client: Policy client
        prompt: Task prompt
        num_trials: Number of measurement trials
        warmup_trials: Number of warmup trials (excluded from statistics)

    Returns:
        Dictionary with latency statistics
    """
    latencies: List[float] = []
    processing_times: List[float] = []

    print(f"\n{'='*60}")
    print(f"Starting REAL DATA latency measurement...")
    print(f"Warmup trials: {warmup_trials}, Measurement trials: {num_trials}")
    print(f"{'='*60}\n")

    # Warmup phase
    print("Warmup phase...")
    for i in range(warmup_trials):
        start = time.perf_counter()
        obs = prepare_observation(prompt)
        process_end = time.perf_counter()
        result = client.infer(obs)
        end = time.perf_counter()

        warmup_latency = (end - start) * 1000
        warmup_process = (process_end - start) * 1000
        print(f"  Warmup {i+1}/{warmup_trials}: {warmup_latency:.2f} ms "
              f"(processing: {warmup_process:.2f} ms)")

    print("\nMeasurement phase...")
    # Measurement phase
    for i in range(num_trials):
        # Measure COMPLETE end-to-end time
        start = time.perf_counter()

        # Image processing (part of real latency)
        obs = prepare_observation(prompt)
        process_end = time.perf_counter()

        # Inference
        result = client.infer(obs)
        end = time.perf_counter()

        total_latency_ms = (end - start) * 1000
        processing_time_ms = (process_end - start) * 1000

        latencies.append(total_latency_ms)
        processing_times.append(processing_time_ms)

        print(f"  Trial {i+1}/{num_trials}: {total_latency_ms:.2f} ms "
              f"(processing: {processing_time_ms:.2f} ms, "
              f"network+inference: {total_latency_ms - processing_time_ms:.2f} ms)")

    # Compute statistics
    stats = {
        "mean_ms": statistics.mean(latencies),
        "std_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "median_ms": statistics.median(latencies),
        "p95_ms": np.percentile(latencies, 95),
        "p99_ms": np.percentile(latencies, 99),
        "mean_processing_ms": statistics.mean(processing_times),
        "mean_network_inference_ms": statistics.mean([t - p for t, p in zip(latencies, processing_times)]),
    }

    return stats, result


def compute_inference_delay(latency_ms: float, control_freq_hz: float) -> int:
    """
    Compute inference delay in number of control steps.

    Args:
        latency_ms: Inference latency in milliseconds
        control_freq_hz: Control frequency in Hz

    Returns:
        Number of control steps delayed
    """
    latency_sec = latency_ms / 1000.0 
    delay_steps = int(np.ceil(latency_sec * control_freq_hz))
    return delay_steps


def print_latency_report(stats: Dict[str, float], control_freq_hz: float):
    """Print a comprehensive latency report."""
    print(f"\n{'='*60}")
    print("INFERENCE LATENCY STATISTICS")
    print(f"{'='*60}")
    print(f"Mean:       {stats['mean_ms']:.2f} ms  (± {stats['std_ms']:.2f} ms)")
    print(f"Median:     {stats['median_ms']:.2f} ms")
    print(f"Min:        {stats['min_ms']:.2f} ms")
    print(f"Max:        {stats['max_ms']:.2f} ms")
    print(f"95th %ile:  {stats['p95_ms']:.2f} ms")
    print(f"99th %ile:  {stats['p99_ms']:.2f} ms")
    print(f"{'='*60}")

    # Compute delay in control steps
    mean_delay = compute_inference_delay(stats['mean_ms'], control_freq_hz)
    max_delay = compute_inference_delay(stats['max_ms'], control_freq_hz)
    p95_delay = compute_inference_delay(stats['p95_ms'], control_freq_hz)

    print(f"\nINFERENCE DELAY ANALYSIS (at {control_freq_hz} Hz control frequency)")
    print(f"{'='*60}")
    print(f"Mean latency delay:  {mean_delay} control steps")
    print(f"95th percentile:     {p95_delay} control steps")
    print(f"Worst case (max):    {max_delay} control steps")
    print(f"{'='*60}")

    # RTC recommendations
    print(f"\nREAL-TIME CHUNKING (RTC) RECOMMENDATIONS")
    print(f"{'='*60}")
    print(f"Recommended inference_delay:  {p95_delay} steps")
    print(f"  - Use 95th percentile for robustness")
    print(f"  - Accounts for 95% of inference times")
    print(f"\nRecommended execute_horizon:  {p95_delay + 3} to {p95_delay + 5} steps")
    print(f"  - Should be > inference_delay for stability")
    print(f"  - Smaller = more responsive, larger = more stable")
    print(f"{'='*60}\n")


def main():
    """Main function to test inference latency with REAL robot data."""
    # Configuration
    HOST = "127.0.0.1"
    PORT = 8000
    CONTROL_FREQ = 50  # Hz (matching your robot control frequency)
    NUM_TRIALS = 20    # Number of measurement trials
    WARMUP_TRIALS = 5  # Number of warmup trials
    PROMPT = "<Clear> <Box> <0.5, 0.5, 0.7, 0.7>"  # Training format prompt

    print(f"\n{'='*60}")
    print("OpenPi Policy Server Latency Benchmark (REAL DATA)")
    print(f"{'='*60}")
    print(f"Host:               {HOST}:{PORT}")
    print(f"Control frequency:  {CONTROL_FREQ} Hz")
    print(f"Measurement trials: {NUM_TRIALS}")
    print(f"Warmup trials:      {WARMUP_TRIALS}")
    print(f"{'='*60}\n")

    # Initialize ROS node
    rospy.init_node("test_client_latency", anonymous=True)
    print("✓ ROS node initialized\n")

    # Subscribe to camera topics
    print("Subscribing to camera topics...")
    rospy.Subscriber("/realsense_top/color/image_raw/compressed",
                     CompressedImage, cb_main, queue_size=1)
    rospy.Subscriber("/realsense_left/color/image_raw/compressed",
                     CompressedImage, cb_wrist_l, queue_size=1)
    rospy.Subscriber("/realsense_right/color/image_raw/compressed",
                     CompressedImage, cb_wrist_r, queue_size=1)
    print("✓ Camera subscriptions created\n")

    # Subscribe to joint state topics
    print("Subscribing to joint state topics...")
    rospy.Subscriber("/robot/arm_left/joint_states_single",
                     JointState, cb_joints_left, queue_size=1)
    rospy.Subscriber("/robot/arm_right/joint_states_single",
                     JointState, cb_joints_right, queue_size=1)
    print("✓ Joint state subscriptions created\n")

    # Wait for sensor data
    print("Waiting for sensor data from robot...")
    rate = rospy.Rate(10)  # 10 Hz
    while not rospy.is_shutdown():
        if all(v is not None for v in latest_imgs.values()) and \
           all(v is not None for v in latest_q.values()):
            print("✓ All sensor data received!\n")
            break
        rate.sleep()

    if rospy.is_shutdown():
        print("ERROR: ROS shutdown before receiving data")
        return

    # Initialize policy client
    print("Connecting to policy server...")
    client = websocket_client_policy.WebsocketClientPolicy(
        host=HOST,
        port=PORT,
    )
    print("✓ Connected to policy server!\n")

    # Measure latency with REAL data
    print("="*60)
    print("IMPORTANT: This measures COMPLETE end-to-end latency:")
    print("  1. Reading images from ROS topics")
    print("  2. Image processing (resize, color conversion)")
    print("  3. Network communication")
    print("  4. Model inference on server")
    print("  5. Receiving results")
    print("="*60)

    stats, result = measure_inference_latency_with_real_data(
        client,
        PROMPT,
        num_trials=NUM_TRIALS,
        warmup_trials=WARMUP_TRIALS
    )

    # Print comprehensive report
    print_latency_report(stats, CONTROL_FREQ)

    # Print breakdown
    print(f"LATENCY BREAKDOWN")
    print(f"{'='*60}")
    print(f"Image processing:         {stats['mean_processing_ms']:.2f} ms")
    print(f"Network + Inference:      {stats['mean_network_inference_ms']:.2f} ms")
    print(f"Total (end-to-end):       {stats['mean_ms']:.2f} ms")
    print(f"{'='*60}\n")

    # Print action information
    actions = np.array(result["actions"])
    print(f"ACTION OUTPUT INFORMATION")
    print(f"{'='*60}")
    print(f"Action shape:  {actions.shape}")
    print(f"  - Chunk size (horizon): {actions.shape[0]} steps")
    print(f"  - Action dimension:     {actions.shape[1]}")
    print(f"\nFirst action preview:")
    print(f"  Left arm  (0-6):   {actions[0, :7]}")
    print(f"  Right arm (7-13):  {actions[0, 7:14]}")
    print(f"{'='*60}\n")

    print("Latency test completed successfully!")
    print("Use the RTC recommendations above to configure your deployment.\n")


if __name__ == "__main__":
    main()