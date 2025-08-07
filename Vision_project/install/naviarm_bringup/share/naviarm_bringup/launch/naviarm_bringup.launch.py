import os
import yaml
from pathlib import Path

from ament_index_python.packages import get_package_share_directory, get_package_prefix

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
    TimerAction,
    OpaqueFunction,
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
from uf_ros_lib.moveit_configs_builder import MoveItConfigsBuilder
from uf_ros_lib.uf_robot_utils import generate_ros2_control_params_temp_file
from uf_ros_lib.uf_robot_utils import (
    get_xacro_content,
    generate_ros2_control_params_temp_file,
)


def launch_setup(context, *args, **kwargs):
    naviarm_gazebo_dir = get_package_share_directory("naviarm_gazebo")
    gazebo_ros_dir = get_package_share_directory("gazebo_ros")
    autoserve_navigation_dir = get_package_share_directory("autoserve_navigation")
    autoserve_description_share = os.path.join(
        get_package_prefix("autoserve_description"), "share"
    )
    xarm_description_share = os.path.join(
        get_package_prefix("xarm_description"), "share"
    )

    combined_description_share = (
        f"{autoserve_description_share}:{xarm_description_share}"
    )

    if "GAZEBO_MODEL_PATH" in os.environ:
        gazebo_models_path = os.path.join(naviarm_gazebo_dir, "models/")
        os.environ["GAZEBO_MODEL_PATH"] = gazebo_models_path
    else:
        os.environ["GAZEBO_MODEL_PATH"] = gazebo_models_path + "/models"

    env_var = SetEnvironmentVariable("GAZEBO_MODEL_PATH", combined_description_share)

    dof = LaunchConfiguration("dof", default=7)
    robot_type = LaunchConfiguration("robot_type", default="xarm")
    prefix = LaunchConfiguration("prefix", default="")
    hw_ns = LaunchConfiguration("hw_ns", default="xarm")
    limited = LaunchConfiguration("limited", default=False)
    effort_control = LaunchConfiguration("effort_control", default=False)
    velocity_control = LaunchConfiguration("velocity_control", default=False)
    model1300 = LaunchConfiguration("model1300", default=True)
    robot_sn = LaunchConfiguration("robot_sn", default="")
    attach_to = LaunchConfiguration("attach_to", default="base_footprint")
    attach_xyz = LaunchConfiguration("attach_xyz", default='"0.075 0 0.803"')
    attach_rpy = LaunchConfiguration("attach_rpy", default='"0 0 0"')
    mesh_suffix = LaunchConfiguration("mesh_suffix", default="stl")
    kinematics_suffix = LaunchConfiguration("kinematics_suffix", default="")
    add_gripper = LaunchConfiguration("add_gripper", default=False)
    add_vacuum_gripper = LaunchConfiguration("add_vacuum_gripper", default=True)
    add_bio_gripper = LaunchConfiguration("add_bio_gripper", default=False)
    add_realsense_d435i = LaunchConfiguration("add_realsense_d435i", default=True)
    add_d435i_links = LaunchConfiguration("add_d435i_links", default=True)
    add_other_geometry = LaunchConfiguration("add_other_geometry", default=False)
    geometry_type = LaunchConfiguration("geometry_type", default="box")
    geometry_mass = LaunchConfiguration("geometry_mass", default=0.1)
    geometry_height = LaunchConfiguration("geometry_height", default=0.1)
    geometry_radius = LaunchConfiguration("geometry_radius", default=0.1)
    geometry_length = LaunchConfiguration("geometry_length", default=0.1)
    geometry_width = LaunchConfiguration("geometry_width", default=0.1)
    geometry_mesh_filename = LaunchConfiguration("geometry_mesh_filename", default="")
    geometry_mesh_origin_xyz = LaunchConfiguration(
        "geometry_mesh_origin_xyz", default='"0 0 0"'
    )
    geometry_mesh_origin_rpy = LaunchConfiguration(
        "geometry_mesh_origin_rpy", default='"0 0 0"'
    )
    geometry_mesh_tcp_xyz = LaunchConfiguration(
        "geometry_mesh_tcp_xyz", default='"0 0 0"'
    )
    geometry_mesh_tcp_rpy = LaunchConfiguration(
        "geometry_mesh_tcp_rpy", default='"0 0 0"'
    )

    no_gui_ctrl = LaunchConfiguration("no_gui_ctrl", default=False)
    ros_namespace = LaunchConfiguration("ros_namespace", default="").perform(context)

    ros2_control_plugin = "gazebo_ros2_control/GazeboSystem"
    controllers_name = "controllers"

    ros2_control_params = generate_ros2_control_params_temp_file(
        os.path.join(
            get_package_share_directory("xarm_controller"),
            "config",
            "{}{}_controllers.yaml".format(
                robot_type.perform(context),
                (
                    dof.perform(context)
                    if robot_type.perform(context) in ("xarm", "lite")
                    else ""
                ),
            ),
        ),
        prefix=prefix.perform(context),
        add_gripper=add_gripper.perform(context) in ("True", "true"),
        add_bio_gripper=add_bio_gripper.perform(context) in ("True", "true"),
        ros_namespace=ros_namespace,
        update_rate=1000,
        use_sim_time=True,
        robot_type=robot_type.perform(context),
    )

    moveit_config = MoveItConfigsBuilder(
        context=context,
        controllers_name=controllers_name,
        dof=dof,
        robot_type=robot_type,
        prefix=prefix,
        hw_ns=hw_ns,
        limited=limited,
        effort_control=effort_control,
        velocity_control=velocity_control,
        model1300=model1300,
        robot_sn=robot_sn,
        attach_to=attach_to,
        attach_xyz=attach_xyz,
        attach_rpy=attach_rpy,
        mesh_suffix=mesh_suffix,
        kinematics_suffix=kinematics_suffix,
        ros2_control_plugin=ros2_control_plugin,
        ros2_control_params=ros2_control_params,
        add_gripper=add_gripper,
        add_vacuum_gripper=add_vacuum_gripper,
        add_bio_gripper=add_bio_gripper,
        add_realsense_d435i=add_realsense_d435i,
        add_d435i_links=add_d435i_links,
        add_other_geometry=add_other_geometry,
        geometry_type=geometry_type,
        geometry_mass=geometry_mass,
        geometry_height=geometry_height,
        geometry_radius=geometry_radius,
        geometry_length=geometry_length,
        geometry_width=geometry_width,
        geometry_mesh_filename=geometry_mesh_filename,
        geometry_mesh_origin_xyz=geometry_mesh_origin_xyz,
        geometry_mesh_origin_rpy=geometry_mesh_origin_rpy,
        geometry_mesh_tcp_xyz=geometry_mesh_tcp_xyz,
        geometry_mesh_tcp_rpy=geometry_mesh_tcp_rpy,
    ).to_moveit_configs()

    moveit_config_dump = yaml.dump(moveit_config.to_dict())

    robot_moveit_common_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("xarm_moveit_config"),
                    "launch",
                    "_robot_moveit_common2.launch.py",
                ]
            )
        ),
        launch_arguments={
            "prefix": prefix,
            "attach_to": attach_to,
            "attach_xyz": attach_xyz,
            "attach_rpy": attach_rpy,
            "no_gui_ctrl": no_gui_ctrl,
            "show_rviz": "false",
            "use_sim_time": "true",
            "moveit_config_dump": moveit_config_dump,
        }.items(),
    )

    # robot_description = {
    #     "robot_description": moveit_config.to_dict()["robot_description"]
    # }

    robot_description = {
        "robot_description": get_xacro_content(
            context,
            xacro_file=Path(get_package_share_directory("naviarm_description"))
            / "urdf"
            / "naviarm.xacro",
            dof=dof,
            robot_type=robot_type,
            prefix=prefix,
            hw_ns=hw_ns,
            limited=limited,
            effort_control=effort_control,
            velocity_control=velocity_control,
            model1300=model1300,
            robot_sn=robot_sn,
            attach_to=attach_to,
            attach_xyz=attach_xyz,
            attach_rpy=attach_rpy,
            kinematics_suffix=kinematics_suffix,
            ros2_control_plugin=ros2_control_plugin,
            add_gripper=add_gripper,
            add_vacuum_gripper=add_vacuum_gripper,
            add_bio_gripper=add_bio_gripper,
            add_realsense_d435i=add_realsense_d435i,
            add_d435i_links=add_d435i_links,
            add_other_geometry=add_other_geometry,
            geometry_type=geometry_type,
            geometry_mass=geometry_mass,
            geometry_height=geometry_height,
            geometry_radius=geometry_radius,
            geometry_length=geometry_length,
            geometry_width=geometry_width,
            geometry_mesh_filename=geometry_mesh_filename,
            geometry_mesh_origin_xyz=geometry_mesh_origin_xyz,
            geometry_mesh_origin_rpy=geometry_mesh_origin_rpy,
            geometry_mesh_tcp_xyz=geometry_mesh_tcp_xyz,
            geometry_mesh_tcp_rpy=geometry_mesh_tcp_rpy,
        )
    }
    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{"use_sim_time": True}, robot_description],
    )

    rviz_config_file = PathJoinSubstitution(
        [FindPackageShare("naviarm_bringup"), "rviz", "naviarm.rviz"]
    )
    rviz2_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config_file],
        parameters=[
            {
                "robot_description": moveit_config.to_dict()["robot_description"],
                "robot_description_semantic": moveit_config.to_dict()[
                    "robot_description_semantic"
                ],
                "robot_description_kinematics": moveit_config.to_dict()[
                    "robot_description_kinematics"
                ],
                "robot_description_planning": moveit_config.to_dict()[
                    "robot_description_planning"
                ],
                "planning_pipelines": moveit_config.to_dict()["planning_pipelines"],
                "use_sim_time": True,
            }
        ],
    )

    world = os.path.join(naviarm_gazebo_dir, "worlds", "warehouse.world")

    start_gazebo_server = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros_dir, "launch", "gzserver.launch.py"),
        ),
        launch_arguments={"world": world, "verbose": "false"}.items(),
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
            "-x",
            "3.9",
            "-y",
            "0.0",
            "-z",
            "0.05",
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

    # autoserve_controller_spawner = Node(
    #     package="controller_manager",
    #     executable="spawner",
    #     arguments=["autoserve_controller", "-c", "controller_manager"],
    # )

    delayed_joint_state_broadcaster_spawner = TimerAction(
        period=3.0, actions=[joint_state_broadcaster_spawner]
    )

    delayed_xarm7_traj_controller_spawner = TimerAction(
        period=3.0, actions=[xarm7_traj_controller_spawner]
    )

    # delayed_autoserve_controller_spawner = TimerAction(
    #     period=3.0, actions=[autoserve_controller_spawner]
    # )

    navigation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(autoserve_navigation_dir, "launch", "navigation.launch.py")
        ),
        launch_arguments={
            "sim": "true",
            "rviz": "false",
        }.items(),
    )

    return [
        env_var,
        start_gazebo_server,
        start_gazebo_client,
        robot_state_publisher_node,
        delayed_spawner,
        delayed_joint_state_broadcaster_spawner,
        delayed_xarm7_traj_controller_spawner,
        # delayed_autoserve_controller_spawner
        # robot_moveit_common_launch,
        # navigation_launch,
        # rviz2_node,
    ]


def generate_launch_description():
    return LaunchDescription([OpaqueFunction(function=launch_setup)])
