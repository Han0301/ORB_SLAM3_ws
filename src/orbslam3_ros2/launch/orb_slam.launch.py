from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = Path(get_package_share_directory("orbslam3_ros2"))
    workspace = share.parents[2]
    # load the yaml to override the parameters for the ORB-SLAM3 node
    common_overrides = {
        "vocabulary": str(workspace / "third_party" / "ORB_SLAM3" / "Vocabulary" / "ORBvoc.txt"),
        "settings": str(share / "config" / "gemini2l_rgbd_imu.yaml"),
    }
    viewer = LaunchConfiguration("viewer")

    def slam_node(parameters, condition):
        return Node(
            package="orbslam3_ros2",    # go to the package directory of orbslam3_ros2 and find the node in it 
            executable="rgbd_node",     # find the executable node and start it 
            namespace="orb_slam3",
            name="rgbd",                # so the node's name is /orb_slam3/rgbd
            output="screen",
            parameters=[parameters, common_overrides],
            condition=condition,
        )

    return LaunchDescription([
        DeclareLaunchArgument(
            "viewer",
            default_value="false",
            description="Enable the ORB-SLAM3 Pangolin viewer.",
        ),
        slam_node(
            str(share / "config" / "orbslam3_imu.params.yaml"),
            UnlessCondition(viewer),
        ),
        slam_node(
            str(share / "config" / "orbslam3_imu_viewer.params.yaml"),
            IfCondition(viewer),
        ),
    ])
