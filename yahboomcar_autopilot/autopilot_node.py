#!/usr/bin/env python3
"""
autopilot_node : "Tesla mode" continuous self-driving for the Yahboom
MicroROS-Car-Pi5.

Once a map has been built (see roam_node.py / SLAM), this node keeps the
car driving indefinitely:

  1. Localizes on the existing map via AMCL (already running as part of
     the Nav2 bringup stack).
  2. Picks a random, reachable, free-space point on the known map that is
     a reasonable distance away.
  3. Sends it to Nav2 (NavigateToPose) which performs path planning,
     dynamic obstacle avoidance (via the costmap + laser scan) and
     recovery behaviours automatically.
  4. When the goal is reached (or fails/is blocked), waits briefly and
     picks a new destination - forever - giving a simple full "autopilot"
     loop: the car continuously & autonomously roams the mapped area
     while avoiding obstacles, exactly like a very small-scale
     self-driving demo.

It also subscribes to /scan as a safety net: if something is detected
extremely close in front of the car the current Nav2 goal is cancelled
immediately and a stop command is issued (belt-and-braces on top of
Nav2's own costmap avoidance), then a new goal is picked.
"""

import math
import random
import time

import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan

from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

FREE_MAX = 20   # occupancy value considered safely drivable


class AutopilotNode(Node):
    def __init__(self):
        super().__init__('autopilot_node')

        # ---------------- parameters ----------------------------------
        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('global_frame', 'map')
        self.declare_parameter('robot_frame', 'base_footprint')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('min_goal_distance', 0.8)
        self.declare_parameter('max_goal_distance', 4.0)
        self.declare_parameter('goal_timeout', 60.0)
        self.declare_parameter('replan_period', 1.0)
        self.declare_parameter('emergency_stop_distance', 0.18)
        self.declare_parameter('emergency_stop_angle_deg', 25.0)
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')

        self.map_topic = self.get_parameter('map_topic').value
        self.global_frame = self.get_parameter('global_frame').value
        self.robot_frame = self.get_parameter('robot_frame').value
        self.scan_topic = self.get_parameter('scan_topic').value
        self.min_goal_distance = self.get_parameter('min_goal_distance').value
        self.max_goal_distance = self.get_parameter('max_goal_distance').value
        self.goal_timeout = self.get_parameter('goal_timeout').value
        self.replan_period = self.get_parameter('replan_period').value
        self.emergency_stop_distance = self.get_parameter('emergency_stop_distance').value
        self.emergency_stop_angle_deg = self.get_parameter('emergency_stop_angle_deg').value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value

        # ---------------- state ----------------------------------------
        self.latest_map = None
        self.nav_busy = False
        self.goal_sent_time = None
        self._goal_handle = None
        self.emergency_active = False

        # ---------------- tf --------------------------------------------
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ---------------- map subscription ------------------------------
        map_qos = QoSProfile(depth=1)
        map_qos.reliability = QoSReliabilityPolicy.RELIABLE
        map_qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        self.create_subscription(OccupancyGrid, self.map_topic, self.map_callback, map_qos)

        # ---------------- laser safety net -------------------------------
        self.create_subscription(LaserScan, self.scan_topic, self.scan_callback, 5)
        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 5)

        # ---------------- Nav2 action client -----------------------------
        self._cb_group = ReentrantCallbackGroup()
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose',
                                        callback_group=self._cb_group)

        self.timer = self.create_timer(self.replan_period, self.control_loop,
                                        callback_group=self._cb_group)

        self.get_logger().info('autopilot_node started - entering Tesla-style autopilot loop.')

    # ------------------------------------------------------------------
    def map_callback(self, msg: OccupancyGrid):
        self.latest_map = msg

    # ------------------------------------------------------------------
    def scan_callback(self, msg: LaserScan):
        """Emergency braking safety-net independent from Nav2's own avoidance."""
        if not msg.ranges:
            return
        n = len(msg.ranges)
        angle_min = msg.angle_min
        angle_inc = msg.angle_increment
        limit_rad = math.radians(self.emergency_stop_angle_deg)

        min_front = float('inf')
        for i in range(n):
            angle = angle_min + i * angle_inc
            # normalize to [-pi, pi]
            angle = math.atan2(math.sin(angle), math.cos(angle))
            if abs(angle) <= limit_rad:
                r = msg.ranges[i]
                if not math.isnan(r) and not math.isinf(r) and r > 0.01:
                    min_front = min(min_front, r)

        if min_front < self.emergency_stop_distance:
            if not self.emergency_active:
                self.get_logger().warn(
                    'EMERGENCY STOP: obstacle at %.2fm in front!' % min_front)
            self.emergency_active = True
            stop = Twist()
            self.cmd_pub.publish(stop)
            if self._goal_handle is not None:
                self._goal_handle.cancel_goal_async()
                self.nav_busy = False
        else:
            self.emergency_active = False

    # ------------------------------------------------------------------
    def get_robot_pose(self):
        try:
            trans = self.tf_buffer.lookup_transform(
                self.global_frame, self.robot_frame, rclpy.time.Time())
            return trans.transform.translation.x, trans.transform.translation.y
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None

    # ------------------------------------------------------------------
    def pick_random_goal(self, robot_x, robot_y):
        m = self.latest_map
        arr = np.asarray(m.data, dtype=np.int16).reshape((m.info.height, m.info.width))
        free_ys, free_xs = np.nonzero((arr >= 0) & (arr <= FREE_MAX))
        if len(free_xs) == 0:
            return None

        origin_x = m.info.origin.position.x
        origin_y = m.info.origin.position.y
        res = m.info.resolution

        for _ in range(60):
            idx = random.randrange(len(free_xs))
            gx, gy = free_xs[idx], free_ys[idx]
            wx = origin_x + (gx + 0.5) * res
            wy = origin_y + (gy + 0.5) * res
            dist = math.hypot(wx - robot_x, wy - robot_y)
            if self.min_goal_distance <= dist <= self.max_goal_distance:
                return wx, wy
        return None

    # ------------------------------------------------------------------
    def control_loop(self):
        if self.emergency_active:
            return

        if self.nav_busy:
            if self.goal_sent_time and (time.time() - self.goal_sent_time) > self.goal_timeout:
                self.get_logger().warn('Autopilot goal timed out, replanning.')
                if self._goal_handle is not None:
                    self._goal_handle.cancel_goal_async()
                self.nav_busy = False
            return

        if self.latest_map is None:
            self.get_logger().info('Waiting for /map ...', throttle_duration_sec=5.0)
            return

        pose = self.get_robot_pose()
        if pose is None:
            self.get_logger().info('Waiting for tf (%s -> %s)...' %
                                    (self.global_frame, self.robot_frame),
                                    throttle_duration_sec=5.0)
            return

        robot_x, robot_y = pose
        goal = self.pick_random_goal(robot_x, robot_y)
        if goal is None:
            self.get_logger().warn('No valid random goal found this cycle, retrying...')
            return

        self.send_nav_goal(goal[0], goal[1])

    # ------------------------------------------------------------------
    def send_nav_goal(self, x, y):
        if not self.nav_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().warn('navigate_to_pose action server not available yet.')
            return

        yaw = random.uniform(-math.pi, math.pi)
        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = self.global_frame
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = x
        goal.pose.pose.position.y = y
        goal.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.pose.orientation.w = math.cos(yaw / 2.0)

        self.get_logger().info('Autopilot driving to (%.2f, %.2f)' % (x, y))
        self.nav_busy = True
        self.goal_sent_time = time.time()

        send_future = self.nav_client.send_goal_async(goal)
        send_future.add_done_callback(self._goal_response_cb)

    # ------------------------------------------------------------------
    def _goal_response_cb(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Autopilot goal rejected by Nav2.')
            self.nav_busy = False
            return

        self._goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._goal_result_cb)

    # ------------------------------------------------------------------
    def _goal_result_cb(self, future):
        status = future.result().status
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('Autopilot reached destination.')
        else:
            self.get_logger().warn('Autopilot goal ended with status=%d, picking new goal.' %
                                    status)
        self.nav_busy = False
        self._goal_handle = None


def main(args=None):
    rclpy.init(args=args)
    node = AutopilotNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
