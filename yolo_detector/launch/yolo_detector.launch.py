"""Launch the yolo_detector node with common parameters exposed."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    weights_path = LaunchConfiguration('weights_path')
    camera_device = LaunchConfiguration('camera_device')
    detect_rate_hz = LaunchConfiguration('detect_rate_hz')
    min_confidence = LaunchConfiguration('min_confidence')

    return LaunchDescription([
        DeclareLaunchArgument(
            'weights_path',
            default_value='',
            description="Absolute path to the YOLO .pt file. If empty, searches parent dirs for src/best.pt.",
        ),
        DeclareLaunchArgument(
            'camera_device',
            default_value='/dev/video0',
            description='OpenCV-compatible camera device.',
        ),
        DeclareLaunchArgument(
            'detect_rate_hz',
            default_value='5.0',
            description='Detection rate in Hz.',
        ),
        DeclareLaunchArgument(
            'min_confidence',
            default_value='0.25',
            description='Drop detections below this confidence.',
        ),
        Node(
            package='yolo_detector',
            executable='yolo_detector',
            name='yolo_detector',
            output='screen',
            parameters=[{
                'weights_path': weights_path,
                'camera_device': camera_device,
                'detect_rate_hz': detect_rate_hz,
                'min_confidence': min_confidence,
            }],
        ),
    ])
