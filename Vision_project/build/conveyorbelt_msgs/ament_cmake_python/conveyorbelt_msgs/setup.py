from setuptools import find_packages
from setuptools import setup

setup(
    name='conveyorbelt_msgs',
    version='0.0.0',
    packages=find_packages(
        include=('conveyorbelt_msgs', 'conveyorbelt_msgs.*')),
)
