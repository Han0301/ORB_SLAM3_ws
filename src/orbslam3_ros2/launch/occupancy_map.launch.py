from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    share = Path(get_package_share_directory("orbslam3_ros2"))

    camera_config_path = share / "config" / "gemini2l_camera.yaml"
    with camera_config_path.open("r", encoding="utf-8") as config_file:
        camera_configuration = yaml.safe_load(config_file)

    camera_parameters = camera_configuration["/camera/camera"]["ros__parameters"]
    use_imu = bool(camera_parameters["use_imu"])

    return LaunchDescription([
        Node(
            package="orbslam3_ros2",
            executable="occupancy_mapper",
            name="occupancy_mapper",
            output="screen",
            parameters=[
                str(share / "config" / "occupancy_map.yaml"),
                {"use_imu": use_imu},
            ],
        ),
    ])
