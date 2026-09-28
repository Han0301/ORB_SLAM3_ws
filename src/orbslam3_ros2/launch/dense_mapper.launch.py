from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    share = Path(get_package_share_directory("orbslam3_ros2"))

    return LaunchDescription([
        Node(
            package="orbslam3_ros2",
            executable="dense_mapper",
            name="dense_mapper",
            output="screen",
            parameters=[str(share / "config" / "dense_mapper.params.yaml")],
        ),
    ])
