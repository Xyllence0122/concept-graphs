import os
from pathlib import Path
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchContext, LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _default_ai_site_packages() -> str:
    return os.environ.get(
        "CG_AI_SITE_PACKAGES",
        str(
            Path.home()
            / "Concept Graph"
            / ".venvs"
            / "cg-ai"
            / "lib"
            / f"python{sys.version_info.major}.{sys.version_info.minor}"
            / "site-packages"
        ),
    )


def _launch_node(context: LaunchContext) -> list[Node]:
    config = (
        Path(get_package_share_directory("cg_object_segmentation"))
        / "config"
        / "grounded_sam_segmentation.yaml"
    )
    ai_site_packages = LaunchConfiguration("ai_site_packages").perform(context)
    python_path = os.pathsep.join(
        value
        for value in (ai_site_packages, os.environ.get("PYTHONPATH", ""))
        if value
    )
    return [
        Node(
            package="cg_object_segmentation",
            executable="grounded_sam_segmentation_node",
            name="grounded_sam_segmentation_node",
            output="screen",
            parameters=[
                str(config),
                {
                    "text_prompt": LaunchConfiguration("text_prompt").perform(
                        context
                    )
                },
            ],
            additional_env={"PYTHONPATH": python_path},
        )
    ]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "ai_site_packages", default_value=_default_ai_site_packages()
            ),
            DeclareLaunchArgument(
                "text_prompt", default_value="box. eraser."
            ),
            OpaqueFunction(function=_launch_node),
        ]
    )
