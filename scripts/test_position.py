#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import JointState
import numpy as np

def main():
    rospy.init_node("test_position_publisher")

    # Create publishers to send actions to robot arms
    pub_left = rospy.Publisher("/robot/arm_left/vla_joint_cmd", JointState, queue_size=1)
    pub_right = rospy.Publisher("/robot/arm_right/vla_joint_cmd", JointState, queue_size=1)

    rospy.loginfo("Test position publisher initialized")

    # Test position: 14-dim array (7 DOF per arm)
    test_position = [
        -0.11900296807289124,
        0.9283173680305481,
        -1.0302928686141968,
        -0.04029456153512001,
        1.3130273818969727,
        -0.05948403850197792,
        0.029680000618100166,     # Left gripper
        0.13410526514053345,
        0.9375596046447754,
        -0.9353703856468201,
        0.26442471146583557,
        1.2828665971755981,
        -0.017495153471827507,
        0.04969999939203262       # Right gripper
    ]

    # Split into left and right arms (7 DOF each)
    action_left = test_position[:7]
    action_right = test_position[7:14]

    rospy.loginfo(f"Left arm target:  {action_left}")
    rospy.loginfo(f"Right arm target: {action_right}")

    # Wait for publishers to be ready
    rospy.sleep(1.0)

    rate = rospy.Rate(10)  # 10 Hz

    rospy.loginfo("Publishing test position... Press Ctrl+C to stop")

    while not rospy.is_shutdown():
        # Create JointState messages for both arms
        cmd_left = JointState()
        cmd_left.header.stamp = rospy.Time.now()
        cmd_left.position = action_left

        cmd_right = JointState()
        cmd_right.header.stamp = rospy.Time.now()
        cmd_right.position = action_right

        # Publish commands to robot arms
        pub_left.publish(cmd_left)
        pub_right.publish(cmd_right)

        rate.sleep()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.loginfo("Test position publisher stopped")