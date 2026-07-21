#!/usr/bin/env python3
"""
initial_pose_publisher : publishes a one-shot /initialpose so AMCL can
start localizing without a human clicking [2D Pose Estimate] in RViz.

By default it assumes the car is powered on at the same spot where
mapping was started (map origin (0,0,0) - which is exactly where
SLAM begins its coordinate frame), so this is a reasonable default
for a fully autonomous / driverless startup.  Override x/y/yaw params
if the car is physically placed elsewhere before starting autopilot.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy

from geometry_msgs.msg import PoseWithCovarianceStamped
import math


class InitialPosePublisher(Node):
    def __init__(self):
        super().__init__('initial_pose_publisher')

        self.declare_parameter('x', 0.0)
        self.declare_parameter('y', 0.0)
        self.declare_parameter('yaw', 0.0)
        self.declare_parameter('frame_id', 'map')

        x = self.get_parameter('x').value
        y = self.get_parameter('y').value
        yaw = self.get_parameter('yaw').value
        frame_id = self.get_parameter('frame_id').value

        qos = QoSProfile(depth=1)
        qos.reliability = QoSReliabilityPolicy.RELIABLE
        qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL

        pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', qos)

        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = frame_id
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
        msg.pose.pose.orientation.w = math.cos(yaw / 2.0)
        # modest covariance so AMCL still does a bit of its own refinement
        msg.pose.covariance[0] = 0.25   # x
        msg.pose.covariance[7] = 0.25   # y
        msg.pose.covariance[35] = 0.06853891945200942  # yaw

        # Give any late subscribers (AMCL) a moment to be discovered
        self.create_timer(1.5, lambda: self._publish_and_shutdown(pub, msg))

    def _publish_and_shutdown(self, pub, msg):
        pub.publish(msg)
        self.get_logger().info('Published initial pose (%.2f, %.2f)' %
                                (msg.pose.pose.position.x, msg.pose.pose.position.y))
        self.create_timer(1.0, lambda: rclpy.shutdown())


def main(args=None):
    rclpy.init(args=args)
    node = InitialPosePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
