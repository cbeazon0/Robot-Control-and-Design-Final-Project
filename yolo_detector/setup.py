from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'yolo_detector'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='clay',
    maintainer_email='user@example.com',
    description='YOLO-based red/quadrilateral detector node for TurtleBot.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yolo_detector = yolo_detector.yolo_detector:main',
            'capture_dataset = yolo_detector.capture_dataset:main',
        ],
    },
)
