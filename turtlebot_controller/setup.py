from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'turtlebot_controller'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='clay',
    maintainer_email='user@example.com',
    description='Closed-loop TurtleBot motion controller with Drive and Rotate actions.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'odom_tracker = turtlebot_controller.odom_tracker:main',
            'motion_controller = turtlebot_controller.motion_controller:main',
            'drive_cli = turtlebot_controller.cli:drive_main',
            'rotate_cli = turtlebot_controller.cli:rotate_main',
            'stop_cli = turtlebot_controller.cli:stop_main',
            'sign_follower = turtlebot_controller.sign_follower:main',
        ],
    },
)
