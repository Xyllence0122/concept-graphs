from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    config = (
        Path(get_package_share_directory("cg_depth_projection"))
        / "config"
        / "depth_roi.yaml"
    )
    return LaunchDescription(
        [
            Node(
                package="cg_depth_projection",
                executable="depth_roi_node",
                name="depth_roi_node",
                output="screen",
                parameters=[str(config)],
            )
        ]
    )
