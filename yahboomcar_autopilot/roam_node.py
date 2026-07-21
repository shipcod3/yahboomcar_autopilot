#!/usr/bin/env python3
"""
roam_node : Autonomous "self-mapping" explorer for the Yahboom MicroROS-Car-Pi5.

While a SLAM node (cartographer / slam_toolbox / gmapping) is running and
publishing /map, this node:
  1. Looks at the occupancy grid for FRONTIERS (free cells next to unknown
     cells) - i.e. the edge of what has been explored so far.
  2. Picks the best nearby & largest frontier and sends it to Nav2's
     NavigateToPose action so the car drives there on its own, avoiding
     obstacles using the normal Nav2 costmap/controller stack.
  3. Repeats until no more frontiers are found (the whole reachable area
     has been mapped), then automatically saves the map to disk and exits.

This gives the "drive around and build the map by itself" behaviour asked
for, instead of manually joysticking the car around during SLAM.
"""

import math
import subprocess
import time

import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid

from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

from yahboomcar_autopilot.frontier_utils import find_frontiers, pick_best_frontier


class RoamNode(Node):
    def __init__(self):
        super().__init__('roam_node')

        # ---- parameters -----------------------------------------------
        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('global_frame', 'map')
        self.declare_parameter('robot_frame', 'base_footprint')
        self.declare_parameter('explore_period', 3.0)          # seconds between planning cycles
        self.declare_parameter('min_frontier_cluster', 6)      # cells
        self.declare_parameter('blacklist_radius', 0.6)        # metres
        self.declare_parameter('min_travel_distance', 0.4)     # metres
        self.declare_parameter('goal_timeout', 60.0)           # seconds per nav goal
        self.declare_parameter('idle_cycles_before_done', 4)   # consecutive empty cycles -> done
        self.declare_parameter('save_map_path', '/root/autopilot_maps/yahboom_map')
        self.declare_parameter('auto_shutdown', True)

        self.map_topic = self.get_parameter('map_topic').value
        self.global_frame = self.get_parameter('global_frame').value
        self.robot_frame = self.get_parameter('robot_frame').value
        self.explore_period = self.get_parameter('explore_period').value
        self.min_cluster = self.get_parameter('min_frontier_cluster').value
        self.blacklist_radius = self.get_parameter('blacklist_radius').value
        self.min_travel = self.get_parameter('min_travel_distance').value
        self.goal_timeout = self.get_parameter('goal_timeout').value
        self.idle_cycles_before_done = self.get_parameter('idle_cycles_before_done').value
        self.save_map_path = self.get_parameter('save_map_path').value
        self.auto_shutdown = self.get_parameter('auto_shutdown').value

        # ---- state -------------------------------------------------------
        self.latest_map = None
        self.blacklist = []
        self.nav_busy = False
        self.idle_cycles = 0
        self.finished = False
        self.goal_sent_time = None
        self._goal_handle = None

        # ---- tf ------------------------------------------------------
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ---- map subscription (SLAM publishes transient-local) --------
        map_qos = QoSProfile(depth=1)
        map_qos.reliability = QoSReliabilityPolicy.RELIABLE
        map_qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        self.create_subscription(OccupancyGrid, self.map_topic, self.map_callback, map_qos)

        # ---- Nav2 action client ---------------------------------------
        self._cb_group = ReentrantCallbackGroup()
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose',
                                        callback_group=self._cb_group)

        self.timer = self.create_timer(self.explore_period, self.explore_cycle,
                                        callback_group=self._cb_group)

        self.get_logger().info('roam_node started - waiting for map & Nav2 action server...')

    # ------------------------------------------------------------------
    def map_callback(self, msg: OccupancyGrid):
        self.latest_map = msg

    # ------------------------------------------------------------------
    def get_robot_pose(self):
        try:
            trans = self.tf_buffer.lookup_transform(
                self.global_frame, self.robot_frame, rclpy.time.Time())
            return trans.transform.translation.x, trans.transform.translation.y
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None

    # ------------------------------------------------------------------
    def explore_cycle(self):
        if self.finished:
            return

        if self.nav_busy:
            # Safety timeout in case the action never reports completion
            if self.goal_sent_time and (time.time() - self.goal_sent_time) > self.goal_timeout:
                self.get_logger().warn('Nav goal timed out, cancelling.')
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
        m = self.latest_map
        clusters = find_frontiers(
            m.data, m.info.width, m.info.height,
            m.info.origin.position.x, m.info.origin.position.y,
            m.info.resolution, min_cluster_size=self.min_cluster)

        target = pick_best_frontier(clusters, robot_x, robot_y, self.blacklist,
                                     blacklist_radius=self.blacklist_radius,
                                     min_travel=self.min_travel)

        if target is None:
            self.idle_cycles += 1
            self.get_logger().info(
                'No frontier found (%d/%d idle cycles).' %
                (self.idle_cycles, self.idle_cycles_before_done))
            if self.idle_cycles >= self.idle_cycles_before_done:
                self.finish_exploration()
            return

        self.idle_cycles = 0
        self.send_nav_goal(target['x'], target['y'])

    # ------------------------------------------------------------------
    def send_nav_goal(self, x, y):
        if not self.nav_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().warn('navigate_to_pose action server not available yet.')
            return

        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = self.global_frame
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = x
        goal.pose.pose.position.y = y
        goal.pose.pose.orientation.w = 1.0

        self.get_logger().info('Exploring towards frontier (%.2f, %.2f)' % (x, y))
        self.nav_busy = True
        self.goal_sent_time = time.time()
        self._pending_goal_xy = (x, y)

        send_future = self.nav_client.send_goal_async(goal)
        send_future.add_done_callback(self._goal_response_cb)

    # ------------------------------------------------------------------
    def _goal_response_cb(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Goal rejected by Nav2.')
            self.blacklist.append(self._pending_goal_xy)
            self.nav_busy = False
            return

        self._goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._goal_result_cb)

    # ------------------------------------------------------------------
    def _goal_result_cb(self, future):
        status = future.result().status
        x, y = self._pending_goal_xy
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('Reached frontier (%.2f, %.2f)' % (x, y))
        else:
            self.get_logger().warn('Failed to reach frontier (%.2f, %.2f), status=%d' %
                                    (x, y, status))
            self.blacklist.append((x, y))
        self.nav_busy = False
        self._goal_handle = None

    # ------------------------------------------------------------------
    def finish_exploration(self):
        self.finished = True
        self.get_logger().info('Exploration complete - saving map to %s' % self.save_map_path)
        try:
            subprocess.run(
                ['ros2', 'run', 'nav2_map_server', 'map_saver_cli',
                 '-f', self.save_map_path, '--ros-args', '-p', 'save_map_timeout:=10.0'],
                check=False, timeout=30)
            self.get_logger().info('Map saved successfully.')
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error('Failed to save map: %s' % exc)

        if self.auto_shutdown:
            self.get_logger().info('roam_node shutting down.')
            self.timer.cancel()
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = RoamNode()
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
