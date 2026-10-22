import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchContext, LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def _launch_file(package: str, name: str) -> PythonLaunchDescriptionSource:
    return PythonLaunchDescriptionSource(
        str(Path(get_package_share_directory(package)) / "launch" / name)
    )


def _semantic_nodes(context: LaunchContext) -> list[Node]:
    config = (
        Path(get_package_share_directory("cg_object_segmentation"))
        / "config"
        / "yolo_segmentation.yaml"
    )
    ai_site_packages = LaunchConfiguration("yolo_ai_site_packages").perform(
        context
    )
    python_path = os.pathsep.join(
        value
        for value in (ai_site_packages, os.environ.get("PYTHONPATH", ""))
        if value
    )
    model_path = LaunchConfiguration("yolo_model_path").perform(context)

    return [
        Node(
            package="cg_object_segmentation",
            executable="yolo_segmentation_node",
            name="yolo_segmentation_node",
            output="screen",
            parameters=[str(config), {"model_path": model_path}],
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
            parameters=[{"target_frame": "map", "odom_frame": "odom"}],
        ),
        Node(
            package="cg_object_segmentation",
            executable="object_graph_node",
            name="object_graph_node",
            output="screen",
            parameters=[
                {"expected_frame_id": "map", "max_missed_frames": -1}
            ],
        ),
    ]


def generate_launch_description() -> LaunchDescription:
    database_path = LaunchConfiguration("database_path")
    delete_db_on_start = LaunchConfiguration("delete_db_on_start")

    realsense = IncludeLaunchDescription(
        _launch_file("realsense2_camera", "rs_launch.py"),
        launch_arguments={
            "camera_namespace": "cg",
            "camera_name": "d455",
            "device_type": "d455",
            "enable_color": "true",
            "enable_depth": "true",
            "enable_gyro": "true",
            "enable_accel": "true",
            "unite_imu_method": "2",
            "enable_sync": "true",
            "align_depth.enable": "true",
            "enable_rgbd": "true",
            "pointcloud.enable": "false",
            "depth_module.depth_profile": "640x480x30",
            "rgb_camera.color_profile": "640x480x30",
            "diagnostics_period": "1.0",
        }.items(),
        condition=IfCondition(LaunchConfiguration("start_camera")),
    )

    rtabmap = IncludeLaunchDescription(
        _launch_file("rtabmap_launch", "rtabmap.launch.py"),
        launch_arguments={
            "rgb_topic": "/cg/d455/color/image_raw",
            "depth_topic": "/cg/d455/aligned_depth_to_color/image_raw",
            "camera_info_topic": "/cg/d455/color/camera_info",
            "frame_id": "d455_link",
            "map_frame_id": "map",
            "rgbd_sync": "true",
            "approx_rgbd_sync": "true",
            "approx_sync": "true",
            "qos": "2",
            "sync_queue_size": "30",
            "topic_queue_size": "30",
            "wait_for_transform": "1.0",
            "visual_odometry": "true",
            "rtabmap_viz": LaunchConfiguration("rtabmap_viz"),
            "rviz": LaunchConfiguration("rviz"),
            "database_path": database_path,
            "args": PythonExpression(
                [
                    "'--delete_db_on_start' if '",
                    delete_db_on_start,
                    "'.lower() == 'true' else ''",
                ]
            ),
        }.items(),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("start_camera", default_value="true"),
            DeclareLaunchArgument("rtabmap_viz", default_value="false"),
            DeclareLaunchArgument("rviz", default_value="false"),
            DeclareLaunchArgument(
                "database_path", default_value="~/.ros/concept_graph_d455.db"
            ),
            DeclareLaunchArgument("delete_db_on_start", default_value="false"),
            DeclareLaunchArgument(
                "yolo_model_path",
                default_value=os.environ.get(
                    "CG_MODEL_PATH", "~/concept_graph_models/yolo26n-seg.pt"
                ),
            ),
            DeclareLaunchArgument(
                "yolo_ai_site_packages",
                default_value=os.environ.get("CG_AI_SITE_PACKAGES", ""),
            ),
            realsense,
            rtabmap,
            OpaqueFunction(function=_semantic_nodes),
        ]
    )
