#!/usr/bin/env python3
"""
autopilot_launch.py

"Tesla mode" - full self-driving autopilot launch file - single entry point.

Brings up:
  - yahboomcar_bringup (base driver / odom / imu / robot description)
  - Nav2 (bringup_launch.py: map_server, AMCL, planner, controller,
    behavior/recovery servers, bt_navigator) using the map saved by
    map_surroundings_launch.py
  - A one-shot /initialpose publisher so AMCL starts localized without
    needing a human to click [2D Pose Estimate] in RViz
  - autopilot_node, which continuously & autonomously picks new
    destinations on the map and drives to them, avoiding obstacles via
    the Nav2 costmap/controller stack plus an extra laser-based
    emergency-stop safety net.
  - stop_car (stock Yahboom node) so the car reliably stops on shutdown.

Usage (inside the car's docker container, after mapping is complete):
    ros2 launch yahboomcar_autopilot autopilot_launch.py

Precautions: power on the car, place it stably on the ground and let
it sit still for ~5s before launching so the gyroscope can finish
initializing, and place it at the same spot mapping was started from
(map origin) unless you override initial_x/initial_y/initial_yaw.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    nav2_bringup_dir = get_package_share_directory('nav2_bringup')
    nav_pkg = get_package_share_directory('yahboomcar_nav')
    bringup_pkg = get_package_share_directory('yahboomcar_bringup')

    map_yaml_path = LaunchConfiguration('map')
    nav2_param_path = LaunchConfiguration('params_file')
    use_sim_time = LaunchConfiguration('use_sim_time')
    initial_x = LaunchConfiguration('initial_x')
    initial_y = LaunchConfiguration('initial_y')
    initial_yaw = LaunchConfiguration('initial_yaw')

    declare_map = DeclareLaunchArgument(
        'map',
        default_value=os.path.join('/root/autopilot_maps', 'yahboom_map.yaml'),
        description='Full path to map yaml file produced by map_surroundings_launch.py')

    declare_params_file = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(nav_pkg, 'params', 'dwb_nav_params.yaml'),
        description='Full path to Nav2 parameters file')

    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time', default_value='false',
        description='Use simulation clock if true')

    declare_initial_x = DeclareLaunchArgument('initial_x', default_value='0.0')
    declare_initial_y = DeclareLaunchArgument('initial_y', default_value='0.0')
    declare_initial_yaw = DeclareLaunchArgument('initial_yaw', default_value='0.0')

    bringup_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [bringup_pkg, '/launch/yahboomcar_bringup_launch.py']),
    )

    nav2_bringup_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [nav2_bringup_dir, '/launch', '/bringup_launch.py']),
        launch_arguments={
            'map': map_yaml_path,
            'use_sim_time': use_sim_time,
            'params_file': nav2_param_path}.items(),
    )

    base_link_to_laser_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_link_to_base_laser',
        arguments=['-0.0046412', '0', '0.094079', '0', '0', '0',
                   'base_link', 'laser_frame']
    )

    stop_car_node = Node(
        package='yahboomcar_nav',
        executable='stop_car',
    )

    initial_pose_publisher = Node(
        package='yahboomcar_autopilot',
        executable='initial_pose_publisher',
        name='initial_pose_publisher',
        output='screen',
        parameters=[{
            'x': initial_x,
            'y': initial_y,
            'yaw': initial_yaw,
        }],
    )

    autopilot_node = Node(
        package='yahboomcar_autopilot',
        executable='autopilot_node',
        name='autopilot_node',
        output='screen',
    )

    # Give Nav2 + AMCL a few seconds to come up before publishing the
    # initial pose and a few more before autopilot starts sending goals.
    delayed_initial_pose = TimerAction(period=6.0, actions=[initial_pose_publisher])
    delayed_autopilot = TimerAction(period=10.0, actions=[autopilot_node])

    return LaunchDescription([
        declare_map,
        declare_params_file,
        declare_use_sim_time,
        declare_initial_x,
        declare_initial_y,
        declare_initial_yaw,
        bringup_launch,
        base_link_to_laser_tf_node,
        stop_car_node,
        nav2_bringup_launch,
        delayed_initial_pose,
        delayed_autopilot,
    ])
