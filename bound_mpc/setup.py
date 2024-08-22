from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'bound_mpc'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*.xacro')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*.sdf')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*.urdf')),
        (os.path.join('share', package_name, 'urdf/Gripper'), glob('urdf/Gripper/*')),
        (os.path.join('share', package_name, 'urdf/body'), glob('urdf/body/*.xacro')),
        (os.path.join('share', package_name, 'urdf/body/meshes/iiwa14/collision'), glob('urdf/body/meshes/iiwa14/collision/*')),
        (os.path.join('share', package_name, 'urdf/body/meshes/iiwa14/visual'), glob('urdf/body/meshes/iiwa14/visual/*')),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
    ],
    include_package_data=True,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Thies Oelerich',
    maintainer_email='thies.oelerich@tuwien.ac.at',
    description='BoundMPC Package',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'bound_mpc = nodes.bound_mpc_node:main',
            'run_experiment1 = nodes.experiment1_runner:main',
            'run_experiment2 = nodes.experiment2_runner:main',
        ],
    },
)
