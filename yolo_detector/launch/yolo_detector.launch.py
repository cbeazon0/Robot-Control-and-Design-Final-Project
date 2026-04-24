from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    weights_path = LaunchConfiguration('weights_path')
    camera_device = LaunchConfiguration('camera_device')
    min_confidence = LaunchConfiguration('min_confidence')

    return LaunchDescription([
        DeclareLaunchArgument(
            'weights_path',
            default_value='',
            description="path to the YOLO .pt file",
        ),
        DeclareLaunchArgument(
            'camera_device',
            default_value='/dev/video0',
            description='Camera.',
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
                'min_confidence': min_confidence,
            }],
        ),
    ])
