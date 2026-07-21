import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    camera_index = LaunchConfiguration('camera_index', default='0')
    linear_speed = LaunchConfiguration('linear_speed', default='0.18')
    max_angular_speed = LaunchConfiguration('max_angular_speed', default='0.9')
    flip_horizontal = LaunchConfiguration('flip_horizontal', default='false')

    bringup_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('yahboomcar_bringup'), 'launch'),
            '/yahboomcar_bringup_launch.py'])
    )

    vision_autopilot_node = Node(
        package='yahboomcar_autopilot',
        executable='vision_autopilot_node',
        name='vision_autopilot_node',
        output='screen',
        parameters=[{
            'camera_index': camera_index,
            'linear_speed': linear_speed,
            'max_angular_speed': max_angular_speed,
            'flip_horizontal': flip_horizontal,
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'camera_index', default_value=camera_index,
            description='OpenCV camera device index to use for the vision autopilot'),
        DeclareLaunchArgument(
            'linear_speed', default_value=linear_speed,
            description='Forward driving speed (m/s) when the path ahead is clear'),
        DeclareLaunchArgument(
            'max_angular_speed', default_value=max_angular_speed,
            description='Maximum steering angular speed (rad/s)'),
        DeclareLaunchArgument(
            'flip_horizontal', default_value=flip_horizontal,
            description='Whether to horizontally flip the camera image before processing'),
        bringup_launch,
        vision_autopilot_node,
    ])
