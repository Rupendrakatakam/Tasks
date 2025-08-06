import os
from ament_index_python.packages import get_package_share_directory, get_package_prefix

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.substitutions import (
    Command,
    FindExecutable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    naviarm_gazebo_dir = get_package_share_directory("naviarm_gazebo")
    gazebo_ros_dir = get_package_share_directory("gazebo_ros")
    autoserve_description_share = os.path.join(
        get_package_prefix("autoserve_description"), "share"
    )
    xarm_description_share = os.path.join(
        get_package_prefix("xarm_description"), "share"
    )

    combined_description_share = f"{autoserve_description_share}:{xarm_description_share}"

    if 'GAZEBO_MODEL_PATH' in os.environ:
        gazebo_models_path = os.path.join(naviarm_gazebo_dir, 'models/')
        os.environ['GAZEBO_MODEL_PATH'] = gazebo_models_path
    else:
        os.environ['GAZEBO_MODEL_PATH'] =  gazebo_models_path + "/models"

    env_var = SetEnvironmentVariable("GAZEBO_MODEL_PATH", combined_description_share)

    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution(
                [
                    FindPackageShare("naviarm_description"),
                    "urdf",
                    "naviarm.xacro",
                ]
            ),
        ]
    )
    robot_description = {"robot_description": robot_description_content}

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[robot_description],
    )

    world = os.path.join(
        naviarm_gazebo_dir,
        'worlds',
        'cafe.world'
    )

    start_gazebo_server = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros_dir, "launch", "gzserver.launch.py"),
        ),
        launch_arguments={'world': world}.items()
    )

    start_gazebo_client = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros_dir, "launch", "gzclient.launch.py")
        )
    )

    spawn_robot = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=[
            "-entity",
            "naviarm",
            "-topic",
            "robot_description",
        ],
        output="screen",
    )

    delayed_spawner = TimerAction(period=2.0, actions=[spawn_robot])

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "-c", "controller_manager"],
    )

    xarm7_traj_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["xarm7_traj_controller", "-c", "controller_manager"],
    )

    xarm7_posi_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["xarm7_posi_controller", "-c", "controller_manager"],
    )

    autoserve_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["autoserve_controller", "-c", "controller_manager"],
    )

    delayed_joint_state_broadcaster_spawner = TimerAction(period=5.0, actions=[joint_state_broadcaster_spawner])

    delayed_xarm7_traj_controller_spawner = TimerAction(period=5.0, actions=[xarm7_traj_controller_spawner])

    delayed_xarm7_posi_controller_spawner = TimerAction(period=5.0, actions=[xarm7_posi_controller_spawner])

    delayed_autoserve_controller_spawner = TimerAction(period=5.0, actions=[autoserve_controller_spawner])




    return LaunchDescription(
        [
            env_var,
            start_gazebo_server,
            start_gazebo_client,
            robot_state_publisher_node,
            delayed_spawner,
            delayed_joint_state_broadcaster_spawner,
            delayed_xarm7_traj_controller_spawner,
            # delayed_xarm7_posi_controller_spawner,
            # delayed_autoserve_controller_spawner
        ]
    )
