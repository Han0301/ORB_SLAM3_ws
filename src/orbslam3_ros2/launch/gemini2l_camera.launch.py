import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    share = Path(get_package_share_directory("orbslam3_ros2"))
    workspace = share.parents[2]
    runtime_library_dirs = [
        workspace / "sysroot" / "opt" / "ros" / "jazzy" / "lib",
        workspace / "sysroot" / "usr" / "lib" / "x86_64-linux-gnu",
        workspace / "local" / "lib",
    ]
    library_path = ":".join(map(str, runtime_library_dirs))
    if os.environ.get("LD_LIBRARY_PATH"):
        library_path += ":" + os.environ["LD_LIBRARY_PATH"]

    return LaunchDescription([
        Node(
            package="orbbec_camera",
            executable="orbbec_camera_node",
            namespace="camera",
            name="camera",
            output="screen",
            parameters=[str(share / "config" / "camera_imu.params.yaml")],
            additional_env={"LD_LIBRARY_PATH": library_path},
        ),
    ])
