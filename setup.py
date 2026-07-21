import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'yahboomcar_autopilot'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*.py'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='pi',
    maintainer_email='pi@localhost',
    description='Autonomous mapping + Tesla-style continuous autopilot navigation',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'roam_node = yahboomcar_autopilot.roam_node:main',
            'autopilot_node = yahboomcar_autopilot.autopilot_node:main',
            'initial_pose_publisher = yahboomcar_autopilot.initial_pose_publisher:main',
            'vision_autopilot_node = yahboomcar_autopilot.vision_autopilot_node:main',

        ],
    },
)
