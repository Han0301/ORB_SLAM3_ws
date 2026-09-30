import os
from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import SetEnvironmentVariable
from launch_ros.actions import Node


def generate_launch_description():
    share = Path(get_package_share_directory("orbslam3_ros2"))
    workspace = share.parents[2]

    camera_config_path = share / "config" / "gemini2l_camera.yaml"
    with camera_config_path.open("r", encoding="utf-8") as config_file:
        camera_configuration = yaml.safe_load(config_file)

    camera_parameters = camera_configuration["/camera/camera"]["ros__parameters"]
    use_imu = bool(camera_parameters["use_imu"])
    image_width = int(camera_parameters["camera_width"])
    image_height = int(camera_parameters["camera_height"])
    image_fps = int(camera_parameters["camera_fps"])

    local_library_paths = [
        str(workspace / "third_party" / "ORB_SLAM3" / "lib"),
        str(workspace / "local" / "lib"),
    ]
    inherited_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    if inherited_library_path:
        local_library_paths.append(inherited_library_path)

    return LaunchDescription([
        SetEnvironmentVariable(
            name="LD_LIBRARY_PATH",
            value=os.pathsep.join(local_library_paths),
        ),
        Node(
            package="orbslam3_ros2",
            executable="rgbd_node",
            namespace="orb_slam3",
            name="rgbd",
            output="screen",
            parameters=[{
                "vocabulary": str(
                    workspace / "third_party" / "ORB_SLAM3" / "Vocabulary" / "ORBvoc.txt"
                ),
                "settings": str(share / "config" / "orb_slam3.yaml"),
                "use_imu": use_imu,
                "image_width": image_width,
                "image_height": image_height,
                "image_fps": image_fps,
            }],
        ),
    ])
