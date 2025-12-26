#!/usr/bin/env python3
"""
RTC evaluation client for a real robot (ROS1), aligned with:

- Algorithm 1 (GETACTION + background INFERENCELOOP + delay queue)
  from "Real-Time Execution of Action Chunking Flow Policies" (Black et al., 2025).
- The boundary rule in official eval_flow.py is implicitly satisfied by:
    - continuing to execute old Acur during inference
    - swapping to Anew when it arrives
    - setting t := t - s so the next executed index becomes ~d (delay steps)

This script assumes your policy server is an OpenPI websocket server that supports
an optional "rtc" dict in the request (see serve_policy_rtc.py).
"""

from __future__ import annotations

import threading
from collections import deque
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
    control_freq: int = 25          # controller frequency (Hz)
    action_horizon: int = 20        # H (must match server/model)
    s_min: int = 8                  # minimum execution horizon smin
    delay_buf_size: int = 10        # b (history length for delay queue)
    num_flow_steps: int = 5         # n (denoising / flow steps)
    max_guidance_weight: float = 5.0  # β
    prefix_attention_schedule: str = "exp"
    image_size: int = 224

    # Safety
    hold_on_invalid: bool = True

    # ---- Fix A: prewarm RTC JIT on server then reset counters ----
    prewarm_rtc: bool = True
    # Choose a realistic (d,s) pair you will actually use at runtime.
    # Example: 25Hz => 40ms/step; if your measured latency is ~200-250ms, d≈5-7.
    prewarm_inference_delay: int = 6  # d_pre
    prewarm_execute_horizon: int = 8  # s_pre (must satisfy d_pre <= s_pre <= H - d_pre)


class RTCController:
    """
    Implements Algorithm 1 (client-side):

    Shared state:
      - _t: steps since last inference started (indexes into Acur)
      - _Acur: current chunk [H, 14] (already sliced for robot control)
      - _ocur_raw: latest raw observation snapshot (BGR imgs + q14)
      - _Q: deque of past observed delays (in steps), maxlen=b
    """

    def __init__(self, client: websocket_client_policy.WebsocketClientPolicy, cfg: RTCConfig, prompt: str):
        self.client = client
        self.cfg = cfg
        self.prompt = prompt

        self._M = threading.Lock()
        self._C = threading.Condition(self._M)
        self._shutdown = False

        # Algorithm 1 init
        self._t = 0
        self._Acur: np.ndarray | None = None            # [H, 14]
        self._ocur_raw = None                           # (bgr_main, bgr_l, bgr_r, q14)
        self._Q = deque([0], maxlen=cfg.delay_buf_size)  # d_init = 0

        # Diagnostics
        self.last_observed_delay = 0
        self.last_predicted_delay = 0
        self.last_s = 0
        self.last_server_latency_ms = 0.0

        self._infer_thread = threading.Thread(target=self._inference_loop, daemon=True)

    # ---- Observation handling ----
    def update_observation_from_ros(self) -> bool:
        """Called every control tick. Stores raw obs into shared state (cheap)."""
        if any(v is None for v in _latest_imgs.values()) or any(v is None for v in _latest_q.values()):
            return False

        bgr_main = _latest_imgs["main"].copy()
        bgr_l = _latest_imgs["wrist_l"].copy()
        bgr_r = _latest_imgs["wrist_r"].copy()

        q_left = _latest_q["left"][:7].astype(np.float32).copy()
        q_right = _latest_q["right"][:7].astype(np.float32).copy()
        q14 = np.concatenate([q_left, q_right], axis=0)

        with self._C:
            self._ocur_raw = (bgr_main, bgr_l, bgr_r, q14)
            self._C.notify_all()
        return True

    # ---- GETACTION (Algorithm 1) ----
    def get_action(self) -> np.ndarray:
        """Return a_t = Acur[t-1], with holds if not ready."""
        with self._C:
            self._t += 1
            self._C.notify_all()

            if self._Acur is None:
                if self.cfg.hold_on_invalid and self._ocur_raw is not None:
                    return self._ocur_raw[3]
                return np.zeros((14,), dtype=np.float32)

            idx = self._t - 1
            if idx < 0 or idx >= self._Acur.shape[0]:
                if self.cfg.hold_on_invalid and self._ocur_raw is not None:
                    return self._ocur_raw[3]
                return self._Acur[-1].copy()

            return self._Acur[idx].copy()

    # ---- Startup: warm start ----
    def warm_start(self):
        """One-time synchronous call to get initial Acur via normal infer (non-RTC)."""
        while not rospy.is_shutdown():
            if self.update_observation_from_ros():
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

        H = self.cfg.action_horizon
        if actions.shape[0] >= H:
            actions = actions[:H]
        else:
            pad = np.zeros((H - actions.shape[0], 14), dtype=np.float32)
            actions = np.concatenate([actions, pad], axis=0)

        with self._C:
            self._Acur = actions
            self._t = 0
            self._Q = deque([0], maxlen=self.cfg.delay_buf_size)
            self.last_server_latency_ms = (t1 - t0) * 1000.0
            self._C.notify_all()

        rospy.loginfo(f"[RTC] Warm start done. Ainit shape={actions.shape}, latency={self.last_server_latency_ms:.1f}ms")

    # ---- Fix A: prewarm RTC JIT on server, then reset counters ----
    def prewarm_rtc_jit_and_reset(self):
        """
        Send ONE RTC request before starting control/inference threads to:
          - trigger server-side JIT compilation for realtime_action path
          - avoid polluting Q with compile time
        Then reset (_t, Q) back to clean state.
        """
        cfg = self.cfg
        if not cfg.prewarm_rtc:
            return

        # Ensure we have Acur + observation
        with self._C:
            if self._Acur is None or self._ocur_raw is None:
                rospy.logwarn("[RTC] prewarm skipped: Acur/observation not ready.")
                return
            Acur = self._Acur.copy()
            raw = self._ocur_raw

        H = cfg.action_horizon
        d = int(cfg.prewarm_inference_delay)
        s = int(cfg.prewarm_execute_horizon)

        # Clamp into feasible range
        d = max(0, min(d, H - 1))
        s = max(1, s)
        s = max(s, d)
        s = min(s, H - d)

        # Build Aprev = right-pad(Acur[s:], H), and pad action-dim 14->32 for server/model
        suffix14 = Acur[s:].copy()  # [H-s, 14]
        suffix32 = np.concatenate([suffix14, np.zeros((suffix14.shape[0], 18), dtype=np.float32)], axis=1)  # -> [*,32]
        if suffix32.shape[0] < H:
            Aprev32 = np.concatenate([suffix32, np.zeros((H - suffix32.shape[0], 32), dtype=np.float32)], axis=0)
        else:
            Aprev32 = suffix32[:H]

        obs = self._preprocess_obs(raw)
        rtc_payload = {
            "enabled": True,
            "prev_action_chunk": Aprev32.astype(np.float32).tolist(),  # [H, 32]
            "inference_delay": int(d),
            "execute_horizon": int(s),
            "num_flow_steps": int(cfg.num_flow_steps),
            "prefix_attention_schedule": str(cfg.prefix_attention_schedule),
            "max_guidance_weight": float(cfg.max_guidance_weight),
        }
        obs_with_rtc = dict(obs)
        obs_with_rtc["rtc"] = rtc_payload

        rospy.loginfo(f"[RTC] Prewarming RTC JIT on server with (d={d}, s={s}) ...")
        t0 = time.time()
        _ = self.client.infer(obs_with_rtc)  # discard output; purpose is to JIT-compile server path
        t1 = time.time()
        prewarm_ms = (t1 - t0) * 1000.0
        rospy.loginfo(f"[RTC] Prewarm finished. rtc_call_latency={prewarm_ms:.1f}ms")

        # Critical: reset counters so compile time is not treated as runtime delay
        with self._C:
            self._t = 0
            self._Q = deque([0], maxlen=cfg.delay_buf_size)
            self.last_observed_delay = 0
            self.last_predicted_delay = 0
            self._C.notify_all()

        rospy.loginfo("[RTC] Counters reset after prewarm (t=0, Q=[0]).")

    def start(self):
        self._infer_thread.start()

    def stop(self):
        with self._C:
            self._shutdown = True
            self._C.notify_all()
        self._infer_thread.join(timeout=1.0)

    # ---- Background inference loop (Algorithm 1) ----
    def _inference_loop(self):
        cfg = self.cfg
        H = cfg.action_horizon

        while not rospy.is_shutdown():
            with self._C:
                # Wait until: have Acur + obs and executed at least s_min steps
                self._C.wait_for(lambda: self._shutdown or (
                    self._Acur is not None and self._ocur_raw is not None and self._t >= cfg.s_min
                ))
                if self._shutdown:
                    return

                s = int(self._t)  # executed steps since inference started
                Acur = self._Acur
                raw = self._ocur_raw

                # Conservative delay estimate
                d = int(max(self._Q)) if len(self._Q) > 0 else 0
                d = max(0, min(d, H - 1))  # avoid d==H -> H-d==0 corner

                # Enforce feasibility: d ≤ s ≤ H - d
                if s < d:
                    rospy.logwarn(f"[RTC] Constraint violated: s({s}) < d({d}). Clamping s=d.")
                    s = d
                if s > (H - d):
                    rospy.logwarn(f"[RTC] Constraint violated: s({s}) > H-d({H-d}). Clamping s=H-d.")
                    s = max(d, H - d)

                self.last_s = s
                self.last_predicted_delay = d

                # Build Aprev = right-pad(Acur[s:], H), and pad action-dim 14->32
                suffix14 = Acur[s:].copy()  # [H-s,14] (or smaller if s close to H)
                suffix32 = np.concatenate([suffix14, np.zeros((suffix14.shape[0], 18), dtype=np.float32)], axis=1)

                if suffix32.shape[0] < H:
                    Aprev32 = np.concatenate([suffix32, np.zeros((H - suffix32.shape[0], 32), dtype=np.float32)], axis=0)
                else:
                    Aprev32 = suffix32[:H]

            # Run guided inference WITHOUT holding the mutex
            obs = self._preprocess_obs(raw)

            rtc_payload = {
                "enabled": True,
                "prev_action_chunk": Aprev32.astype(np.float32).tolist(),  # [H, 32]
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

            # Truncate/Pad to H (controller uses 14-dim)
            if actions.shape[0] >= H:
                Anew = actions[:H]
            else:
                pad = np.zeros((H - actions.shape[0], 14), dtype=np.float32)
                Anew = np.concatenate([actions, pad], axis=0)

            # Swap to new chunk and update counters
            with self._C:
                self._Acur = Anew

                # t := t - s (Algorithm 1 line 21)
                self._t = int(self._t - s)

                # observed_delay := t (steps executed while inference was running beyond s)
                observed_delay = int(self._t)
                observed_delay = max(0, min(observed_delay, H))
                self._Q.append(observed_delay)

                self.last_observed_delay = observed_delay
                self.last_server_latency_ms = latency_ms

                self._C.notify_all()

            rospy.loginfo_throttle(
                1.0,
                f"[RTC] swap chunk | s={s} d_pred={d} d_obs={observed_delay} "
                f"lat={latency_ms:.1f}ms Q={list(self._Q)}"
            )

    # ---- Preprocessing (do it only in inference thread or warmup calls) ----
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

    # RTC config (match your setup)
    cfg = RTCConfig(
        control_freq=25,
        action_horizon=20,
        s_min=8,
        delay_buf_size=10,
        num_flow_steps=5,
        max_guidance_weight=5.0,
        prefix_attention_schedule="exp",
        image_size=224,
        prewarm_rtc=True,
        prewarm_inference_delay=6,
        prewarm_execute_horizon=8,
    )

    prompt = "<Sweep> <Box> <0.349, 0.500, 0.537, 0.500> <to> <Position> <0.693, 0.549>"

    rtc = RTCController(client, cfg, prompt)

    rospy.loginfo("[RTC] Waiting for observations, warm starting...")
    rtc.warm_start()

    # ---- Fix A here ----
    rtc.prewarm_rtc_jit_and_reset()

    rospy.loginfo("[RTC] Starting background inference thread...")
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
