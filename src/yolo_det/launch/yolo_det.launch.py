from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    share = Path(get_package_share_directory('yolo_det'))

    return LaunchDescription([
        Node(
            package='yolo_det',
            executable='yolo_node',
            name='yolo_node',
            output='screen',
            parameters=[str(share / 'config' / 'yolo.yaml')],
        ),
    ])
