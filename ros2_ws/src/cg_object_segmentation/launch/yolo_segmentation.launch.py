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


def _default_model_path() -> str:
    return os.environ.get(
        "CG_MODEL_PATH",
        str(Path.home() / "Concept Graph" / "models" / "yolo26n-seg.pt"),
    )


def _launch_yolo(context: LaunchContext) -> list[Node]:
    config = (
        Path(get_package_share_directory("cg_object_segmentation"))
        / "config"
        / "yolo_segmentation.yaml"
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
            executable="yolo_segmentation_node",
            name="yolo_segmentation_node",
            output="screen",
            parameters=[
                str(config),
                {
                    "model_path": LaunchConfiguration("model_path").perform(
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
                "model_path",
                default_value=_default_model_path(),
                description="Absolute path to the YOLO segmentation weights.",
            ),
            DeclareLaunchArgument(
                "ai_site_packages",
                default_value=_default_ai_site_packages(),
                description=(
                    "Python site-packages directory containing torch and "
                    "ultralytics. Set CG_AI_SITE_PACKAGES to avoid passing it."
                ),
            ),
            OpaqueFunction(function=_launch_yolo),
        ]
    )
