"""Launch the odom_tracker and motion_controller nodes together."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('turtlebot_controller')
    default_params = os.path.join(pkg_share, 'config', 'params.yaml')

    params_file = LaunchConfiguration('params_file')

    declare_params = DeclareLaunchArgument(
        'params_file',
        default_value=default_params,
        description='Full path to the ROS2 parameters file.',
    )

    odom_tracker_node = Node(
        package='turtlebot_controller',
        executable='odom_tracker',
        name='odom_tracker',
        output='screen',
        parameters=[params_file],
    )

    motion_controller_node = Node(
        package='turtlebot_controller',
        executable='motion_controller',
        name='motion_controller',
        output='screen',
        parameters=[params_file],
    )

    return LaunchDescription([
        declare_params,
        odom_tracker_node,
        motion_controller_node,
    ])
