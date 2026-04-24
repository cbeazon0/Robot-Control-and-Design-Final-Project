from setuptools import find_packages, setup

package_name = 'yolo_detector'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages', [f'resource/{package_name}']),
        (f'share/{package_name}', ['package.xml']),
        (f'share/{package_name}/launch', ['launch/yolo_detector.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='clay',
    maintainer_email='user@example.com',
    description='YOLO-based red/quadrilateral detector node for TurtleBot.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'yolo_detector = yolo_detector.yolo_detector:main',
        ],
    },
)
