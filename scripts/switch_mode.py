#!/usr/bin/env python3
"""
Mode switching script for piper_dual.yaml configuration.

Modes:
  1. VLA Testing with Gripper    - VLA controls both arm and gripper
  2. VLA Testing without Gripper - VLA controls arm, gripper controlled separately
  3. Teleoperation Mode          - Manual teleoperation for data collection

Usage:
  python3 switch_mode.py [mode]

  mode: 1, 2, or 3 (optional, interactive menu if not provided)
"""

import argparse
import sys
from pathlib import Path

import yaml

# Configuration file path (absolute path to piper_dual.yaml)
CONFIG_FILE = Path(
    "/home/zeno-yifan/NPM-Project/NPM-Ros/piper_ros/src/zeno-wholebody-teleop/common/piper_ctrl/config/piper_dual.yaml"
)

# Mode definitions - what topics to use for each mode
MODES = {
    1: {
        "name": "VLA Testing with Gripper",
        "description": "VLA controls both arm joints and gripper",
        "left": {
            "joint_pos_cmd_to": "/robot/arm_left/vla_joint_cmd",
            "gripper_pos_cmd_to": "/robot/arm_left/vla_joint_cmd",
        },
        "right": {
            "joint_pos_cmd_to": "/robot/arm_right/vla_joint_cmd",
            "gripper_pos_cmd_to": "/robot/arm_right/vla_joint_cmd",
        },
    },
    2: {
        "name": "VLA Testing without Gripper",
        "description": "VLA controls arm, gripper controlled separately",
        "left": {
            "joint_pos_cmd_to": "/robot/arm_left/vla_joint_cmd",
            "gripper_pos_cmd_to": "/robot/arm_left/joint_pos_cmd",
        },
        "right": {
            "joint_pos_cmd_to": "/robot/arm_right/vla_joint_cmd",
            "gripper_pos_cmd_to": "/robot/arm_right/joint_pos_cmd",
        },
    },
    3: {
        "name": "Teleoperation Mode",
        "description": "Manual teleoperation for data collection",
        "left": {
            "joint_pos_cmd_to": "/teleop/arm_left/joint_states_single",
            "gripper_pos_cmd_to": "/teleop/arm_left/joint_states_single",
        },
        "right": {
            "joint_pos_cmd_to": "/teleop/arm_right/joint_states_single",
            "gripper_pos_cmd_to": "/teleop/arm_right/joint_states_single",
        },
    },
}


def detect_current_mode(config: dict) -> int | None:
    """Detect current mode based on config values."""
    left_remap = config.get("piper_ctrl_left_node", {}).get("remap", {})
    right_remap = config.get("piper_ctrl_right_node", {}).get("remap", {})

    for mode_num, mode_def in MODES.items():
        left_match = (
            left_remap.get("joint_pos_cmd_to") == mode_def["left"]["joint_pos_cmd_to"]
            and left_remap.get("gripper_pos_cmd_to")
            == mode_def["left"]["gripper_pos_cmd_to"]
        )
        right_match = (
            right_remap.get("joint_pos_cmd_to") == mode_def["right"]["joint_pos_cmd_to"]
            and right_remap.get("gripper_pos_cmd_to")
            == mode_def["right"]["gripper_pos_cmd_to"]
        )
        if left_match and right_match:
            return mode_num
    return None


def load_config() -> dict:
    """Load YAML configuration file."""
    if not CONFIG_FILE.exists():
        print(f"Error: Config file not found: {CONFIG_FILE}")
        sys.exit(1)

    with open(CONFIG_FILE, "r") as f:
        return yaml.safe_load(f)


def save_config(config: dict):
    """Save YAML configuration file with preserved formatting."""
    # Custom representer to preserve key order and formatting
    class OrderedDumper(yaml.SafeDumper):
        pass

    def dict_representer(dumper, data):
        return dumper.represent_mapping("tag:yaml.org,2002:map", data.items())

    OrderedDumper.add_representer(dict, dict_representer)

    with open(CONFIG_FILE, "w") as f:
        yaml.dump(
            config, f, Dumper=OrderedDumper, default_flow_style=False, allow_unicode=True
        )


def apply_mode(config: dict, mode: int) -> dict:
    """Apply mode settings to config."""
    mode_def = MODES[mode]

    # Update left arm robot node
    if "piper_ctrl_left_node" in config:
        config["piper_ctrl_left_node"]["remap"]["joint_pos_cmd_to"] = mode_def["left"][
            "joint_pos_cmd_to"
        ]
        config["piper_ctrl_left_node"]["remap"]["gripper_pos_cmd_to"] = mode_def["left"][
            "gripper_pos_cmd_to"
        ]

    # Update right arm robot node
    if "piper_ctrl_right_node" in config:
        config["piper_ctrl_right_node"]["remap"]["joint_pos_cmd_to"] = mode_def["right"][
            "joint_pos_cmd_to"
        ]
        config["piper_ctrl_right_node"]["remap"]["gripper_pos_cmd_to"] = mode_def[
            "right"
        ]["gripper_pos_cmd_to"]

    return config


def print_mode_menu(current_mode: int | None):
    """Print mode selection menu."""
    print("\n" + "=" * 60)
    print("  Piper Robot Mode Switcher")
    print("=" * 60)

    for mode_num, mode_def in MODES.items():
        marker = " *" if mode_num == current_mode else "  "
        print(f"\n  [{mode_num}] {mode_def['name']}{marker}")
        print(f"      {mode_def['description']}")

    if current_mode:
        print(f"\n  Current mode: {current_mode} ({MODES[current_mode]['name']})")
    else:
        print("\n  Current mode: Unknown (custom configuration)")

    print("\n" + "-" * 60)


def interactive_select() -> int:
    """Interactive mode selection."""
    while True:
        try:
            choice = input("Select mode (1/2/3) or 'q' to quit: ").strip().lower()
            if choice == "q":
                print("Cancelled.")
                sys.exit(0)
            mode = int(choice)
            if mode in MODES:
                return mode
            print("Invalid choice. Please enter 1, 2, or 3.")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            print("\nCancelled.")
            sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description="Switch piper robot control mode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  1 - VLA Testing with Gripper    (VLA controls both arm and gripper)
  2 - VLA Testing without Gripper (VLA controls arm only)
  3 - Teleoperation Mode          (Manual control)

Examples:
  python3 switch_mode.py      # Interactive menu
  python3 switch_mode.py 1    # Switch to VLA with gripper mode
  python3 switch_mode.py 3    # Switch to teleoperation mode
""",
    )
    parser.add_argument(
        "mode",
        type=int,
        nargs="?",
        choices=[1, 2, 3],
        help="Mode to switch to (1, 2, or 3)",
    )
    parser.add_argument(
        "--status", "-s", action="store_true", help="Show current mode and exit"
    )

    args = parser.parse_args()

    # Load current config
    config = load_config()
    current_mode = detect_current_mode(config)

    # Status only mode
    if args.status:
        if current_mode:
            print(f"Current mode: {current_mode} - {MODES[current_mode]['name']}")
        else:
            print("Current mode: Unknown (custom configuration)")
        sys.exit(0)

    # Interactive or direct mode selection
    if args.mode is None:
        print_mode_menu(current_mode)
        selected_mode = interactive_select()
    else:
        selected_mode = args.mode

    # Check if already in selected mode
    if selected_mode == current_mode:
        print(f"Already in mode {selected_mode}: {MODES[selected_mode]['name']}")
        sys.exit(0)

    # Apply new mode
    print(f"\nSwitching to mode {selected_mode}: {MODES[selected_mode]['name']}...")
    config = apply_mode(config, selected_mode)
    save_config(config)

    print(f"Done! Config updated: {CONFIG_FILE}")
    print("\nNew topic mappings:")
    for arm in ["left", "right"]:
        mode_def = MODES[selected_mode]
        print(f"  {arm.capitalize()} arm:")
        print(f"    joint_pos_cmd_to:   {mode_def[arm]['joint_pos_cmd_to']}")
        print(f"    gripper_pos_cmd_to: {mode_def[arm]['gripper_pos_cmd_to']}")

    print("\nRemember to restart the robot launch file for changes to take effect!")


if __name__ == "__main__":
    main()
