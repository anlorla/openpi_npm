#!/usr/bin/env python3
"""
RTC evaluation client for a real robot (ROS1) aligned with:

- Algorithm 1 (GETACTION + background INFERENCELOOP) from
  "Real-Time Execution of Action Chunking Flow Policies" (Black et al., 2025).
- The chunk boundary execution rule used in the official eval_flow.py:
  execute old[:d] + new[d:s], then shift new by s for the next cycle.
- Uses fixed delay parameters (d, s) matching the official eval_flow.py approach.

This script assumes your policy server is the OpenPI websocket server, extended to support
an "rtc" field in the request (see serve_policy_rtc.py).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
import time
import numpy as np
import cv2
import rospy
from sensor_msgs.msg import CompressedImage, JointState
from cv_bridge import CvBridge

from openpi_client import websocket_client_policy, image_tools


# -----------------------------
# ROS data buffers (callbacks)
# -----------------------------
bridge = CvBridge()

_latest_imgs = {"main": None, "wrist_l": None, "wrist_r": None}
_latest_q = {"left": None, "right": None}

def _cb_main(msg: CompressedImage):
    _latest_imgs["main"] = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")

def _cb_wrist_l(msg: CompressedImage):
    _latest_imgs["wrist_l"] = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")

def _cb_wrist_r(msg: CompressedImage):
    _latest_imgs["wrist_r"] = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")

def _cb_joints_left(msg: JointState):
    _latest_q["left"] = np.array(msg.position, dtype=np.float32)

def _cb_joints_right(msg: JointState):
    _latest_q["right"] = np.array(msg.position, dtype=np.float32)


# -----------------------------
# RTC controller (Algorithm 1)
# -----------------------------
@dataclass
class RTCConfig:
    control_freq: int = 25          # Δt = 1/control_freq
    action_horizon: int = 20        # H (must match server/model)
    s_min: int = 8                  # minimum execution horizon smin
    # Fixed delay parameters (matching official eval_flow.py approach)
    fixed_inference_delay: int = 5  # d (fixed at ~200ms / 40ms)
    num_flow_steps: int = 5         # n (denoising / flow steps)
    max_guidance_weight: float = 5.0  # β
    prefix_attention_schedule: str = "exp"  # "exp" matches official default
    image_size: int = 224
    # Safety: publish hold if we don't have a valid action
    hold_on_invalid: bool = True

class RTCController:
    """
    Implements Algorithm 1 at the *client/controller* side, with fixed delay parameters
    (matching the official eval_flow.py approach).

    Shared state:
      - t: steps since last inference started (indexes into Acur)
      - Acur: current chunk [H, action_dim]
      - ocur: latest observation payload (raw images + joint state)
    """

    def __init__(self, client: websocket_client_policy.WebsocketClientPolicy, cfg: RTCConfig, prompt: str):
        self.client = client
        self.cfg = cfg
        self.prompt = prompt

        self._M = threading.Lock()
        self._C = threading.Condition(self._M)

        self._shutdown = False

        # Shared variables (Algorithm 1 initialization)
        self._t = 0
        self._Acur = None            # np.ndarray [H, 14]
        self._ocur_raw = None        # (bgr_main, bgr_l, bgr_r, q14) snapshots

        # Diagnostics
        self.last_s = 0
        self.last_server_latency_ms = 0.0

        self._infer_thread = threading.Thread(target=self._inference_loop, daemon=True)

    # ---- Observation handling ----
    def update_observation_from_ros(self) -> bool:
        """
        Called every control tick. Stores the *raw* observation into shared state
        (cheap). Preprocessing is done inside inference thread to keep control loop stable.
        """
        if any(v is None for v in _latest_imgs.values()) or any(v is None for v in _latest_q.values()):
            return False

        # Shallow copies of arrays to avoid mutation during inference
        bgr_main = _latest_imgs["main"].copy()
        bgr_l = _latest_imgs["wrist_l"].copy()
        bgr_r = _latest_imgs["wrist_r"].copy()

        q_left = _latest_q["left"][:7].astype(np.float32).copy()
        q_right = _latest_q["right"][:7].astype(np.float32).copy()
        q14 = np.concatenate([q_left, q_right], axis=0)

        with self._M:
            self._ocur_raw = (bgr_main, bgr_l, bgr_r, q14)
            # notify inference loop that ocur is updated
            self._C.notify_all()
        return True

    # ---- GETACTION (Algorithm 1) ----
    def get_action(self) -> np.ndarray:
        """
        Called each control tick after update_observation_from_ros().

        Returns:
          a_t: np.ndarray [14] action at index (t-1) from Acur.
        """
        with self._M:
            self._t += 1
            self._C.notify_all()

            if self._Acur is None:
                # Not initialized yet; hold current joint state if available
                if self.cfg.hold_on_invalid and self._ocur_raw is not None:
                    return self._ocur_raw[3]
                return np.zeros((14,), dtype=np.float32)

            idx = self._t - 1
            if idx < 0 or idx >= self._Acur.shape[0]:
                # Out-of-range: hold last valid (or current state)
                if self.cfg.hold_on_invalid and self._ocur_raw is not None:
                    return self._ocur_raw[3]
                return self._Acur[-1].copy()

            return self._Acur[idx].copy()

    # ---- Initialization ----
    def warm_start(self):
        """
        Obtain the initial chunk Ainit synchronously (one-time).
        This may pause once at startup; after this, all inference is async.
        """
        # Wait for first observation
        while not rospy.is_shutdown():
            ok = self.update_observation_from_ros()
            if ok:
                break
            rospy.sleep(0.01)

        with self._M:
            raw = self._ocur_raw

        obs = self._preprocess_obs(raw)
        t0 = time.time()
        out = self.client.infer(obs)
        t1 = time.time()

        actions = np.array(out["actions"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 14:
            raise RuntimeError(f"Server returned invalid actions shape: {actions.shape}")

        # Truncate/Pad to H
        H = self.cfg.action_horizon
        if actions.shape[0] >= H:
            actions = actions[:H]
        else:
            pad = np.zeros((H - actions.shape[0], 14), dtype=np.float32)
            actions = np.concatenate([actions, pad], axis=0)

        with self._M:
            self._Acur = actions
            self._t = 0  # per Algorithm 1
            self.last_server_latency_ms = (t1 - t0) * 1000.0

        rospy.loginfo(f"[RTC] Warm start done. Ainit shape={actions.shape}, latency={self.last_server_latency_ms:.1f}ms")

    def start(self):
        self._infer_thread.start()

    def stop(self):
        with self._M:
            self._shutdown = True
            self._C.notify_all()
        self._infer_thread.join(timeout=1.0)

    # ---- Inference Loop (Algorithm 1) ----
    def _inference_loop(self):
        """
        Background inference loop with fixed delay parameters (matching official eval_flow.py).

        Key operations:
          - wait until t >= max(smin, d)
          - s = max(smin, d)  (ensure s >= d)
          - Aprev = right-pad(Acur[s:], H)
          - d = fixed_inference_delay (no dynamic estimation)
          - Anew = GUIDEDINFERENCE(o, Aprev, d, s)
          - swap Acur = Anew; t = t - s
        """
        cfg = self.cfg
        H = cfg.action_horizon
        d = cfg.fixed_inference_delay  # Fixed delay parameter

        while not rospy.is_shutdown():
            with self._C:
                # Wait until we have observation + chunk, and executed at least max(smin, d) steps
                self._C.wait_for(lambda: self._shutdown or (
                    self._Acur is not None and
                    self._ocur_raw is not None and
                    self._t >= max(cfg.s_min, d)
                ))
                if self._shutdown:
                    return

                # Execute horizon: ensure s >= d (matching official eval_flow.py constraint)
                s = max(cfg.s_min, d)
                Acur = self._Acur
                raw = self._ocur_raw

                # Check constraint d ≤ s ≤ H - d for logging only
                if s > (H - d):
                    rospy.logwarn_throttle(1.0, f"[RTC] Constraint s({s}) > H-d({H-d}) violated. May need larger H.")

                self.last_s = s

                # Build Aprev: remove first s actions, right-pad with zeros to length H
                suffix = Acur[s:].copy()  # [remaining_steps, 14]

                # Pi05 models use 32-dim actions internally for compatibility
                # Pad our 14-dim actions to 32-dim (zeros for unused dims 14-31)
                if suffix.shape[1] == 14:
                    pad_action_dim = np.zeros((suffix.shape[0], 18), dtype=np.float32)
                    suffix = np.concatenate([suffix, pad_action_dim], axis=1)  # [remaining_steps, 32]

                # Pad horizon dimension to H if needed
                if suffix.shape[0] < H:
                    pad_horizon = np.zeros((H - suffix.shape[0], suffix.shape[1]), dtype=np.float32)
                    Aprev = np.concatenate([suffix, pad_horizon], axis=0)
                else:
                    Aprev = suffix[:H]  # [H, 32]

            # Run guided inference WITHOUT holding the mutex (Algorithm 1 line 18-19)
            obs = self._preprocess_obs(raw)

            rtc_payload = {
                "enabled": True,
                "prev_action_chunk": Aprev.tolist(),  # [H,14]
                "inference_delay": int(d),
                "execute_horizon": int(s),
                "num_flow_steps": int(cfg.num_flow_steps),
                "prefix_attention_schedule": str(cfg.prefix_attention_schedule),
                "max_guidance_weight": float(cfg.max_guidance_weight),
            }

            obs_with_rtc = dict(obs)
            obs_with_rtc["rtc"] = rtc_payload

            t0 = time.time()
            out = self.client.infer(obs_with_rtc)
            t1 = time.time()
            latency_ms = (t1 - t0) * 1000.0

            actions = np.array(out["actions"], dtype=np.float32)
            if actions.ndim != 2 or actions.shape[1] != 14:
                rospy.logwarn(f"[RTC] Server returned invalid actions shape: {actions.shape}. Skip swap.")
                continue

            # Truncate/Pad to H
            if actions.shape[0] >= H:
                Anew = actions[:H]
            else:
                pad = np.zeros((H - actions.shape[0], 14), dtype=np.float32)
                Anew = np.concatenate([actions, pad], axis=0)

            # Swap to new chunk as soon as it is available (Algorithm 1 line 20)
            with self._C:
                self._Acur = Anew

                # t = t - s (Algorithm 1 line 21)
                # Note: while inference ran, GETACTION continued increasing self._t.
                self._t = int(self._t - s)

                self.last_server_latency_ms = latency_ms

                # notify in case someone waits on updated Acur
                self._C.notify_all()

            rospy.loginfo_throttle(
                1.0,
                f"[RTC] swap chunk | s={s} d={d} lat={latency_ms:.1f}ms t_after_swap={self._t}"
            )

    # ---- Preprocessing (done only in inference thread) ----
    def _preprocess_obs(self, raw):
        bgr_main, bgr_l, bgr_r, q14 = raw

        rgb_main = cv2.cvtColor(bgr_main, cv2.COLOR_BGR2RGB)
        rgb_l = cv2.cvtColor(bgr_l, cv2.COLOR_BGR2RGB)
        rgb_r = cv2.cvtColor(bgr_r, cv2.COLOR_BGR2RGB)

        img_main = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(rgb_main, self.cfg.image_size, self.cfg.image_size)
        )
        img_l = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(rgb_l, self.cfg.image_size, self.cfg.image_size)
        )
        img_r = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(rgb_r, self.cfg.image_size, self.cfg.image_size)
        )

        return {
            "observation/image": img_main,
            "observation/wrist_image": img_l,
            "observation/right_wrist_image": img_r,
            "observation/state": q14.astype(np.float32),
            "prompt": self.prompt,
        }


# -----------------------------
# Main ROS node
# -----------------------------
def main():
    rospy.init_node("piper_rtc")

    # Subscriptions
    rospy.Subscriber("/realsense_top/color/image_raw/compressed",   CompressedImage, _cb_main,    queue_size=1)
    rospy.Subscriber("/realsense_left/color/image_raw/compressed",  CompressedImage, _cb_wrist_l, queue_size=1)
    rospy.Subscriber("/realsense_right/color/image_raw/compressed", CompressedImage, _cb_wrist_r, queue_size=1)

    rospy.Subscriber("/robot/arm_left/joint_states_single",  JointState, _cb_joints_left,  queue_size=1)
    rospy.Subscriber("/robot/arm_right/joint_states_single", JointState, _cb_joints_right, queue_size=1)

    pub_left  = rospy.Publisher("/robot/arm_left/vla_joint_cmd",  JointState, queue_size=1)
    pub_right = rospy.Publisher("/robot/arm_right/vla_joint_cmd", JointState, queue_size=1)

    # Policy client
    client = websocket_client_policy.WebsocketClientPolicy(host="127.0.0.1", port=8000)

    # RTC config (edit to match your setup)
    cfg = RTCConfig(
        control_freq=25,
        action_horizon=20,
        s_min=8,
        fixed_inference_delay=5,  # Fixed at ~200ms / 40ms per step
        num_flow_steps=5,
        max_guidance_weight=5.0,
        prefix_attention_schedule="exp",
        image_size=224,
    )
    # Yifan 
    prompt = "<Sweep> <Box> <0.349, 0.500, 0.537, 0.500> <to> <Position> <0.693, 0.549>"

    rtc = RTCController(client, cfg, prompt)

    rospy.loginfo("[RTC] Waiting for first observations, then warm starting...")
    rtc.warm_start()
    rtc.start()

    rate = rospy.Rate(cfg.control_freq)

    rospy.loginfo("[RTC] Control loop running.")
    while not rospy.is_shutdown():
        ok = rtc.update_observation_from_ros()
        if not ok:
            rate.sleep()
            continue

        action14 = rtc.get_action()

        # Split and publish
        cmd_left = JointState()
        cmd_left.header.stamp = rospy.Time.now()
        cmd_left.position = action14[:7].tolist()

        cmd_right = JointState()
        cmd_right.header.stamp = rospy.Time.now()
        cmd_right.position = action14[7:14].tolist()

        pub_left.publish(cmd_left)
        pub_right.publish(cmd_right)

        rate.sleep()

    rtc.stop()


if __name__ == "__main__":
    main()
