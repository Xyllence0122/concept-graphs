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
    detector_backend = LaunchConfiguration("detector_backend").perform(context)
    if detector_backend not in {"grounded_sam", "yolo"}:
        raise ValueError(
            "detector_backend must be 'grounded_sam' or 'yolo', got "
            f"{detector_backend!r}"
        )
    detector_config = (
        Path(get_package_share_directory("cg_object_segmentation"))
        / "config"
        / (
            "grounded_sam_segmentation.yaml"
            if detector_backend == "grounded_sam"
            else "yolo_segmentation.yaml"
        )
    )
    graph_config = (
        Path(get_package_share_directory("cg_object_segmentation"))
        / "config"
        / "object_graph.yaml"
    )
    ai_site_packages = LaunchConfiguration("ai_site_packages").perform(context)
    python_path = os.pathsep.join(
        value
        for value in (ai_site_packages, os.environ.get("PYTHONPATH", ""))
        if value
    )
    target_frame = LaunchConfiguration("target_frame").perform(context)
    odom_frame = LaunchConfiguration("odom_frame").perform(context)

    if detector_backend == "grounded_sam":
        detector_node = Node(
            package="cg_object_segmentation",
            executable="grounded_sam_segmentation_node",
            name="grounded_sam_segmentation_node",
            output="screen",
            parameters=[
                str(detector_config),
                {
                    "grounding_model_id": LaunchConfiguration(
                        "grounding_model_id"
                    ).perform(context),
                    "grounding_model_revision": LaunchConfiguration(
                        "grounding_model_revision"
                    ).perform(context),
                    "sam_model_id": LaunchConfiguration("sam_model_id").perform(
                        context
                    ),
                    "sam_model_revision": LaunchConfiguration(
                        "sam_model_revision"
                    ).perform(context),
                    "text_prompt": LaunchConfiguration("text_prompt").perform(
                        context
                    ),
                },
            ],
            additional_env={"PYTHONPATH": python_path},
        )
    else:
        detector_node = Node(
            package="cg_object_segmentation",
            executable="yolo_segmentation_node",
            name="yolo_segmentation_node",
            output="screen",
            parameters=[
                str(detector_config),
                {
                    "model_path": LaunchConfiguration("model_path").perform(
                        context
                    )
                },
            ],
            additional_env={"PYTHONPATH": python_path},
        )

    return [
        detector_node,
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
                str(graph_config),
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
            DeclareLaunchArgument(
                "detector_backend", default_value="grounded_sam"
            ),
            DeclareLaunchArgument(
                "text_prompt", default_value="box. eraser."
            ),
            DeclareLaunchArgument(
                "grounding_model_id",
                default_value="IDEA-Research/grounding-dino-tiny",
            ),
            DeclareLaunchArgument(
                "grounding_model_revision",
                default_value="a2bb814dd30d776dcf7e30523b00659f4f141c71",
            ),
            DeclareLaunchArgument(
                "sam_model_id", default_value="facebook/sam-vit-base"
            ),
            DeclareLaunchArgument(
                "sam_model_revision",
                default_value="70c1a07f894ebb5b307fd9eaaee97b9dfc16068f",
            ),
            DeclareLaunchArgument("target_frame", default_value="map"),
            DeclareLaunchArgument("odom_frame", default_value="odom"),
            OpaqueFunction(function=_launch_nodes),
        ]
    )
