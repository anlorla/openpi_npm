#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Replay robot joint trajectories from a bag file, similar style to piper_pi05_main.py.

用法示例：
    uv run scripts/replay_robot_from_bag.py /home/zeno/data/push_block/demo_001.bag
    # 或者
    python scripts/replay_robot_from_bag.py /home/zeno/data/push_block/demo_001.bag
"""

import sys
import argparse

import rospy
import rosbag
from sensor_msgs.msg import JointState

# 录制时用的 robot 关节 topic
LEFT_STATE_TOPIC = "/teleop/arm_left/joint_states_single"
RIGHT_STATE_TOPIC = "/teleop/arm_right/joint_states_single"

LEFT_CMD_TOPIC = "/robot/arm_left/vla_joint_cmd"
RIGHT_CMD_TOPIC = "/robot/arm_right/vla_joint_cmd"


def replay_bag(bag_path: str):
    rospy.loginfo(f"[replay] Opening bag: {bag_path}")
    bag = rosbag.Bag(bag_path, "r")

    pub_left = rospy.Publisher(LEFT_CMD_TOPIC, JointState, queue_size=10)
    pub_right = rospy.Publisher(RIGHT_CMD_TOPIC, JointState, queue_size=10)

    # 给 publisher 一点时间建立连接
    rospy.sleep(1.0)

    last_t = None

    # 只读 robot 这两个 topic
    topics = [LEFT_STATE_TOPIC, RIGHT_STATE_TOPIC]

    for topic, msg, t in bag.read_messages(topics=topics):
        if rospy.is_shutdown():
            break

        # 按 bag 中的时间间隔重放
        if last_t is None:
            last_t = t
        else:
            dt = (t - last_t).to_sec()
            if dt > 0:
                rospy.sleep(dt)
            last_t = t

        cmd = JointState()
        cmd.header.stamp = rospy.Time.now()

        cmd.name = list(msg.name)
        cmd.position = list(msg.position)

        if topic == LEFT_STATE_TOPIC:
            pub_left.publish(cmd)
        elif topic == RIGHT_STATE_TOPIC:
            pub_right.publish(cmd)

    bag.close()
    rospy.loginfo("[replay] Finished replaying bag.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bag_path", type=str, help="Path to bag file to replay",default="/home/zeno/piper_ros/data_collect/push_block_dual/push_block_dual__000.bag")
    args = parser.parse_args()

    rospy.init_node("replay_robot_from_bag", anonymous=True)

    try:
        replay_bag(args.bag_path)
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    # 和 piper_pi05_main 一样，直接 python/uv run 即可
    if len(sys.argv) == 1:
        print("Usage: replay_robot_from_bag.py /path/to/demo.bag")
        sys.exit(1)
    main()
