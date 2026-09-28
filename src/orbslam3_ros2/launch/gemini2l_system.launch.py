from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    # find the share directory of the orbslam3_ros2 package
    share = Path(get_package_share_directory("orbslam3_ros2"))
    print("-----------------------------------------------------------------------")
    print(f"load the share directory of the orbslam3_ros2 package: {share}")

    # when a launch argument is provided, store it in a LaunchConfiguration variable
    # LaunchConfiguration variable means Parameters are only loaded when starting up 
    dense_mapper = LaunchConfiguration("dense_mapper")
    rviz = LaunchConfiguration("rviz")
    viewer = LaunchConfiguration("viewer")

    return LaunchDescription([
        # declare some options startup parameters
        DeclareLaunchArgument(
            "dense_mapper",
            default_value="false",      # it is not start by default
            description="Start the global dense mapper.",
        ),
        DeclareLaunchArgument(
            "rviz",
            default_value="false",
            description="Start RViz with the project configuration.",
        ),
        DeclareLaunchArgument(
            "viewer",
            default_value="false",
            description="Enable the ORB-SLAM3 Pangolin viewer.",
        ),

        # Load the corresponding launch files in order. corresponding means 对应的
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(share / "launch" / "gemini2l_camera.launch.py")),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(share / "launch" / "orb_slam.launch.py")),
            launch_arguments={"viewer": viewer}.items(),        # pass the parameter "viewer" down to the slam launch file
        ),

        # Load the dense mapper launch file if the "dense_mapper" option is enabled
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(share / "launch" / "dense_mapper.launch.py")),
            condition=IfCondition(dense_mapper),
        ),
        # Load the RViz launch file if the "rviz" option is enabled
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(share / "launch" / "rviz.launch.py")),
            condition=IfCondition(rviz),
        ),
    ])
