#!/usr/bin/env python3
"""
RTC-capable policy server wrapper.

This extends the default OpenPI serve_policy.py by allowing clients to send an optional
'rtc' dict in the request. If present, the server attempts to call the policy's realtime
chunking method with soft-masking guidance, matching the official eval_flow.py logic.

Important:
- This file assumes the underlying policy object exposes a method compatible with:
    policy.realtime_action(obs, num_flow_steps, prev_action_chunk, inference_delay, execute_horizon,
                           prefix_attention_schedule, max_guidance_weight)
  OR provides an equivalent hook. If your OpenPI policy does not expose this yet, you must
  implement it inside openpi.policies (see notes in comments).
"""

import dataclasses
import enum
import logging
import socket
from typing import Any

import numpy as np
import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    config: str
    dir: str


@dataclasses.dataclass
class Default:
    pass


@dataclasses.dataclass
class Args:
    env: EnvMode = EnvMode.ALOHA_SIM
    default_prompt: str | None = None
    port: int = 8000
    record: bool = False
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(config="pi05_aloha", dir="gs://openpi-assets/checkpoints/pi05_base"),
    EnvMode.ALOHA_SIM: Checkpoint(config="pi0_aloha_sim", dir="gs://openpi-assets/checkpoints/pi0_aloha_sim"),
    EnvMode.DROID: Checkpoint(config="pi05_droid", dir="gs://openpi-assets/checkpoints/pi05_droid"),
    EnvMode.LIBERO: Checkpoint(config="pi05_libero", dir="gs://openpi-assets/checkpoints/pi05_libero"),
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)


class RTCAwarePolicy:
    """
    Wrapper to add RTC support without changing websocket_policy_server.

    websocket_policy_server will call the underlying policy to produce outputs.
    We implement a conservative duck-typed interface by providing:
      - infer(payload) if base policy has infer
      - __call__(payload) fallback
    """

    def __init__(self, base: _policy.Policy):
        self._base = base
        self.metadata = getattr(base, "metadata", None)

    def _base_infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        if hasattr(self._base, "infer"):
            return self._base.infer(payload)
        if callable(self._base):
            return self._base(payload)
        raise AttributeError("Base policy has neither .infer() nor __call__()")

    def infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        rtc = payload.get("rtc", None)

        # If no RTC requested, behave exactly like the original server.
        if not rtc or not isinstance(rtc, dict) or not rtc.get("enabled", False):
            return self._base_infer(payload)

        # Extract RTC fields
        prev_action_chunk = np.array(rtc["prev_action_chunk"], dtype=np.float32)   # [H, D]
        inference_delay = int(rtc["inference_delay"])                              # d
        execute_horizon = int(rtc["execute_horizon"])                              # s
        num_flow_steps = int(rtc.get("num_flow_steps", 5))
        prefix_attention_schedule = str(rtc.get("prefix_attention_schedule", "exp"))
        max_guidance_weight = float(rtc.get("max_guidance_weight", 5.0))

        # Remove the 'rtc' key before passing obs onward
        obs = dict(payload)
        obs.pop("rtc", None)

        # ----------- Critical alignment to official code -----------
        # In eval_flow.py, prefix_attention_horizon = H - execute_horizon
        # and realtime_action is called with (prev_action_chunk, inference_delay, prefix_attention_horizon, schedule, beta).
        #
        # Your OpenPI policy must expose an equivalent guided inference hook.
        # -----------------------------------------------------------

        # Preferred: base policy directly provides realtime_action with the official signature.
        if hasattr(self._base, "realtime_action"):
            # Call policy's realtime_action method
            result = self._base.realtime_action(
                obs=obs,
                num_flow_steps=num_flow_steps,
                prev_action_chunk=prev_action_chunk,
                inference_delay=inference_delay,
                execute_horizon=execute_horizon,
                prefix_attention_schedule=prefix_attention_schedule,
                max_guidance_weight=max_guidance_weight,
            )
            # result is a dict with "actions" key
            if isinstance(result, dict) and "actions" in result:
                actions = np.array(result["actions"], dtype=np.float32)
                return {"actions": actions.tolist()}
            else:
                # Backward compatibility: if result is just the actions array
                actions = np.array(result, dtype=np.float32)
                return {"actions": actions.tolist()}

        # If not available, fail loudly with a precise message so you can implement it at the right layer.
        raise NotImplementedError(
            "RTC request received but base policy does not implement realtime_action(). "
            "To fully align with the official RTC implementation, you must add a guided inpainting "
            "inference path to the OpenPI policy (soft masking + vJP correction)."
        )

    def __call__(self, payload: dict[str, Any]) -> dict[str, Any]:
        # websocket_policy_server might call __call__ instead of infer
        return self.infer(payload)


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata

    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    # Wrap with RTC support
    policy = RTCAwarePolicy(policy)

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating RTC-capable server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
