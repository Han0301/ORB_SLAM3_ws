from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    share = Path(get_package_share_directory('multi_object_tracker'))

    return LaunchDescription([
        Node(
            package='multi_object_tracker',
            executable='tracker_node',
            name='tracker_node',
            output='screen',
            parameters=[str(share / 'config' / 'tracker.yaml')],
        ),
    ])
