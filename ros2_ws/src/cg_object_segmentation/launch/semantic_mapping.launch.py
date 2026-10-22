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


def _launch_nodes(context: LaunchContext) -> list[Node]:
    yolo_config = (
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
    model_path = LaunchConfiguration("model_path").perform(context)
    target_frame = LaunchConfiguration("target_frame").perform(context)
    odom_frame = LaunchConfiguration("odom_frame").perform(context)

    return [
        Node(
            package="cg_object_segmentation",
            executable="yolo_segmentation_node",
            name="yolo_segmentation_node",
            output="screen",
            parameters=[str(yolo_config), {"model_path": model_path}],
            additional_env={"PYTHONPATH": python_path},
        ),
        Node(
            package="cg_object_segmentation",
            executable="object_3d_projection_node",
            name="object_3d_projection_node",
            output="screen",
        ),
        Node(
            package="cg_object_segmentation",
            executable="object_map_transform_node",
            name="object_map_transform_node",
            output="screen",
            parameters=[
                {"target_frame": target_frame, "odom_frame": odom_frame}
            ],
        ),
        Node(
            package="cg_object_segmentation",
            executable="object_graph_node",
            name="object_graph_node",
            output="screen",
            parameters=[
                {
                    "expected_frame_id": target_frame,
                    "max_missed_frames": -1,
                }
            ],
        ),
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
            DeclareLaunchArgument("target_frame", default_value="map"),
            DeclareLaunchArgument("odom_frame", default_value="odom"),
            OpaqueFunction(function=_launch_nodes),
        ]
    )
