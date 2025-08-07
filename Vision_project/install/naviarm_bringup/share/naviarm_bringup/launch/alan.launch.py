import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource

def launch_setup(context, *args, **kwargs):
    naviarm_gazebo_dir = get_package_share_directory("naviarm_gazebo")
    gazebo_ros_dir = get_package_share_directory("gazebo_ros")

    # if "GAZEBO_MODEL_PATH" in os.environ:
    #     gazebo_models_path = os.path.join(naviarm_gazebo_dir, "models/")
    #     os.environ["GAZEBO_MODEL_PATH"] = gazebo_models_path
    # else:
    #     os.environ["GAZEBO_MODEL_PATH"] = gazebo_models_path + "/models"

    # env_var = SetEnvironmentVariable("GAZEBO_MODEL_PATH", combined_description_share)

    world = os.path.join(naviarm_gazebo_dir, "worlds", "alan_warehouse.world")

    start_gazebo_server = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros_dir, "launch", "gzserver.launch.py"),
        ),
        launch_arguments={"world": world, "verbose": "true"}.items(),
    )

    start_gazebo_client = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros_dir, "launch", "gzclient.launch.py")
        )
    )

    return [
        # env_var,
        start_gazebo_server,
        start_gazebo_client,
    ]


def generate_launch_description():
    return LaunchDescription([OpaqueFunction(function=launch_setup)])
