import os
from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    share = Path(get_package_share_directory("orbslam3_ros2"))
    workspace = share.parents[2]

    config_path = share / "config" / "gemini2l_camera.yaml"
    with config_path.open("r", encoding="utf-8") as config_file:
        configuration = yaml.safe_load(config_file)

    camera_parameters = dict(configuration["/camera/camera"]["ros__parameters"])
    use_imu = bool(camera_parameters.pop("use_imu"))
    camera_width = int(camera_parameters.pop("camera_width"))
    camera_height = int(camera_parameters.pop("camera_height"))
    camera_fps = int(camera_parameters.pop("camera_fps"))

    camera_parameters["color_width"] = camera_width
    camera_parameters["color_height"] = camera_height
    camera_parameters["color_fps"] = camera_fps

    camera_parameters["depth_width"] = camera_width
    camera_parameters["depth_height"] = camera_height
    camera_parameters["depth_fps"] = camera_fps

    camera_parameters["enable_accel"] = use_imu
    camera_parameters["enable_gyro"] = use_imu
    camera_parameters["enable_sync_output_accel_gyro"] = use_imu

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
            parameters=[camera_parameters],
            additional_env={"LD_LIBRARY_PATH": library_path},
        ),
    ])
