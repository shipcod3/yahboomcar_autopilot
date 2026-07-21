#!/usr/bin/env python3
"""
map_surroundings_launch.py

"Map the surroundings by itself" launch file - single entry point.

Brings up:
  - yahboomcar_bringup (base driver / odom / imu / robot description)
  - Cartographer SLAM (builds the map live from lidar + odom)
  - roam_node (autonomously drives to frontiers of the known map until
    the whole reachable area has been explored, then auto-saves the map)

Usage (inside the car's docker container):
    ros2 launch yahboomcar_autopilot map_surroundings_launch.py

Precautions (see 09.Lidar course/Precautions for Radar Mapping and
Navigation.txt): power on the car, place it stably on the ground and
let it sit still for ~5s before it starts moving so the gyroscope can
finish initializing.
"""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    nav_pkg = get_package_share_directory('yahboomcar_nav')
    bringup_pkg = get_package_share_directory('yahboomcar_bringup')

    save_map_path = LaunchConfiguration('save_map_path')
    gyro_settle_time = LaunchConfiguration('gyro_settle_time')

    declare_save_map_path = DeclareLaunchArgument(
        'save_map_path', default_value='/root/autopilot_maps/yahboom_map',
        description='Path (without extension) to save the finished map to')

    declare_gyro_settle_time = DeclareLaunchArgument(
        'gyro_settle_time', default_value='5.0',
        description='Seconds to let the gyroscope settle before exploring starts')

    bringup_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [bringup_pkg, '/launch/yahboomcar_bringup_launch.py']),
    )

    cartographer_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [nav_pkg, '/launch/map_cartographer_launch.py']),
    )

    roam_node = Node(
        package='yahboomcar_autopilot',
        executable='roam_node',
        name='roam_node',
        output='screen',
        parameters=[{'save_map_path': save_map_path}],
    )

    # Delay the exploration node slightly so the gyro/EKF/cartographer have
    # time to initialize and settle before the car starts moving on its own.
    delayed_roam_node = TimerAction(period=5.0, actions=[roam_node])

    return LaunchDescription([
        declare_save_map_path,
        declare_gyro_settle_time,
        bringup_launch,
        cartographer_launch,
        delayed_roam_node,
    ])
