from setuptools import setup

package_name = 'bottle_detection'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include ONNX model in the package data
        ('lib/' + package_name, [
            package_name + '/yolov8n.onnx',
            package_name + '/yolov8m.onnx',
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@example.com',
    description='ROS 2 node for bottle detection using ONNX',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'bottle_detector = bottle_detection.bottle_detector:main',
        ],
    },
    package_data={
        package_name: [
            'yolov8n.onnx',
            'yolov8m.onnx',
            'data/images/*.jpg',
        ],
    },
)