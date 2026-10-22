from __future__ import annotations

import colorsys
import json
import struct
from typing import Any

import cv2
import message_filters
import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

from .projection3d import ObjectGeometry, project_mask_to_3d


StampKey = tuple[int, int]


def _stamp_key(msg: Image) -> StampKey:
    return int(msg.header.stamp.sec), int(msg.header.stamp.nanosec)


def _object_color(instance_id: int) -> tuple[float, float, float]:
    hue = (instance_id * 0.618033988749895) % 1.0
    return colorsys.hsv_to_rgb(hue, 0.80, 1.0)


def _packed_rgb(red: float, green: float, blue: float) -> float:
    value = (
        (int(np.clip(red, 0.0, 1.0) * 255.0) << 16)
        | (int(np.clip(green, 0.0, 1.0) * 255.0) << 8)
        | int(np.clip(blue, 0.0, 1.0) * 255.0)
    )
    return struct.unpack("f", struct.pack("I", value))[0]


class Object3DProjectionNode(Node):
    """Fuse instance masks and aligned depth into camera-frame 3D objects."""

    def __init__(self) -> None:
        super().__init__("object_3d_projection_node")

        self.declare_parameter(
            "instance_map_topic", "/concept_graph/segmentation/instance_map"
        )
        self.declare_parameter(
            "detections_topic", "/concept_graph/segmentation/detections_json"
        )
        self.declare_parameter(
            "depth_topic", "/cg/d455/aligned_depth_to_color/image_raw"
        )
        self.declare_parameter(
            "camera_info_topic", "/cg/d455/aligned_depth_to_color/camera_info"
        )
        self.declare_parameter(
            "objects_3d_topic", "/concept_graph/objects/detections_3d_json"
        )
        self.declare_parameter(
            "object_cloud_topic", "/concept_graph/debug/object_points"
        )
        self.declare_parameter(
            "markers_topic", "/concept_graph/debug/object_markers"
        )
        self.declare_parameter("depth_scale", 0.001)
        self.declare_parameter("min_depth_m", 0.20)
        self.declare_parameter("max_depth_m", 4.00)
        self.declare_parameter("point_stride", 2)
        self.declare_parameter("min_points", 30)
        self.declare_parameter("mask_erosion_px", 1)
        self.declare_parameter("sync_slop_seconds", 0.04)

        self._bridge = CvBridge()
        self._camera_matrix: np.ndarray | None = None
        self._camera_size: tuple[int, int] | None = None
        self._metadata: dict[StampKey, dict[str, Any]] = {}
        self._pending: dict[StampKey, tuple[Image, Image]] = {}

        self._json_pub = self.create_publisher(
            String, str(self.get_parameter("objects_3d_topic").value), 10
        )
        self._cloud_pub = self.create_publisher(
            PointCloud2, str(self.get_parameter("object_cloud_topic").value), 5
        )
        self._markers_pub = self.create_publisher(
            MarkerArray, str(self.get_parameter("markers_topic").value), 5
        )

        self._info_sub = self.create_subscription(
            CameraInfo,
            str(self.get_parameter("camera_info_topic").value),
            self._camera_info_callback,
            qos_profile_sensor_data,
        )
        self._metadata_sub = self.create_subscription(
            String,
            str(self.get_parameter("detections_topic").value),
            self._metadata_callback,
            10,
        )
        self._mask_sub = message_filters.Subscriber(
            self,
            Image,
            str(self.get_parameter("instance_map_topic").value),
            qos_profile=qos_profile_sensor_data,
        )
        self._depth_sub = message_filters.Subscriber(
            self,
            Image,
            str(self.get_parameter("depth_topic").value),
            qos_profile=qos_profile_sensor_data,
        )
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [self._mask_sub, self._depth_sub],
            queue_size=90,
            slop=float(self.get_parameter("sync_slop_seconds").value),
        )
        self._sync.registerCallback(self._synchronized_callback)

        self.get_logger().info("Waiting for instance maps, aligned depth, and metadata")

    def _camera_info_callback(self, msg: CameraInfo) -> None:
        matrix = np.asarray(msg.k, dtype=np.float64).reshape(3, 3)
        if matrix[0, 0] <= 0.0 or matrix[1, 1] <= 0.0:
            self.get_logger().error("Received invalid camera intrinsics")
            return
        self._camera_matrix = matrix
        self._camera_size = (int(msg.width), int(msg.height))

    def _metadata_callback(self, msg: String) -> None:
        try:
            metadata = json.loads(msg.data)
            stamp = metadata["stamp"]
            key = int(stamp["sec"]), int(stamp["nanosec"])
            self._metadata[key] = metadata
            self._try_process(key)
            self._trim_caches()
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            self.get_logger().error(f"Invalid detection metadata: {error}")

    def _synchronized_callback(self, instance_msg: Image, depth_msg: Image) -> None:
        key = _stamp_key(instance_msg)
        self._pending[key] = (instance_msg, depth_msg)
        self._try_process(key)
        self._trim_caches()

    def _try_process(self, key: StampKey) -> None:
        if self._camera_matrix is None:
            return
        if key not in self._pending or key not in self._metadata:
            return
        instance_msg, depth_msg = self._pending.pop(key)
        metadata = self._metadata.pop(key)
        try:
            self._process_frame(instance_msg, depth_msg, metadata)
        except Exception as error:  # keep processing future frames after bad input
            self.get_logger().error(f"3D projection failed: {error}")

    def _trim_caches(self) -> None:
        for cache in (self._metadata, self._pending):
            while len(cache) > 120:
                cache.pop(next(iter(cache)))

    def _depth_in_metres(self, msg: Image) -> np.ndarray:
        depth = np.asarray(
            self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        )
        if msg.encoding in ("16UC1", "mono16"):
            return depth.astype(np.float32) * float(
                self.get_parameter("depth_scale").value
            )
        if msg.encoding == "32FC1":
            return depth.astype(np.float32, copy=False)
        raise ValueError(f"unsupported depth encoding: {msg.encoding}")

    def _process_frame(
        self, instance_msg: Image, depth_msg: Image, metadata: dict[str, Any]
    ) -> None:
        instance_map = np.asarray(
            self._bridge.imgmsg_to_cv2(instance_msg, desired_encoding="passthrough"),
            dtype=np.uint16,
        )
        depth_m = self._depth_in_metres(depth_msg)
        if instance_map.shape != depth_m.shape:
            raise ValueError(
                f"instance/depth dimensions differ: {instance_map.shape} vs {depth_m.shape}"
            )
        height, width = depth_m.shape
        if self._camera_size != (width, height):
            raise ValueError(
                f"CameraInfo dimensions {self._camera_size} do not match {width}x{height}"
            )

        objects_3d: list[dict[str, Any]] = []
        cloud_rows: list[tuple[float, float, float, float]] = []
        geometries: list[tuple[dict[str, Any], ObjectGeometry, tuple[float, float, float]]] = []
        erosion_px = max(0, int(self.get_parameter("mask_erosion_px").value))

        for detected in metadata.get("objects", []):
            instance_id = int(detected["instance_id"])
            mask = instance_map == instance_id
            if erosion_px > 0 and np.any(mask):
                kernel_size = erosion_px * 2 + 1
                kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
                mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=1) > 0

            geometry = project_mask_to_3d(
                mask,
                depth_m,
                self._camera_matrix,
                stride=max(1, int(self.get_parameter("point_stride").value)),
                min_depth_m=float(self.get_parameter("min_depth_m").value),
                max_depth_m=float(self.get_parameter("max_depth_m").value),
            )
            if geometry is None or len(geometry.points) < int(
                self.get_parameter("min_points").value
            ):
                continue

            color = _object_color(instance_id)
            packed_color = _packed_rgb(*color)
            cloud_rows.extend(
                (float(point[0]), float(point[1]), float(point[2]), packed_color)
                for point in geometry.points
            )

            object_record = {
                **detected,
                "frame_id": instance_msg.header.frame_id,
                "centroid_m": {
                    "x": round(float(geometry.centroid[0]), 5),
                    "y": round(float(geometry.centroid[1]), 5),
                    "z": round(float(geometry.centroid[2]), 5),
                },
                "bbox_3d_min_m": [round(float(v), 5) for v in geometry.bbox_min],
                "bbox_3d_max_m": [round(float(v), 5) for v in geometry.bbox_max],
                "valid_depth_points": int(len(geometry.points)),
            }
            objects_3d.append(object_record)
            geometries.append((object_record, geometry, color))

        output = {
            "stamp": metadata["stamp"],
            "frame_id": instance_msg.header.frame_id,
            "objects": objects_3d,
        }
        json_msg = String()
        json_msg.data = json.dumps(output, ensure_ascii=False)
        self._json_pub.publish(json_msg)
        self._publish_cloud(instance_msg, cloud_rows)
        self._publish_markers(instance_msg, geometries)
        self.get_logger().info(f"Published {len(objects_3d)} camera-frame 3D objects")

    def _publish_cloud(
        self, source_msg: Image, rows: list[tuple[float, float, float, float]]
    ) -> None:
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        cloud = point_cloud2.create_cloud(source_msg.header, fields, rows)
        self._cloud_pub.publish(cloud)

    def _publish_markers(
        self,
        source_msg: Image,
        geometries: list[
            tuple[dict[str, Any], ObjectGeometry, tuple[float, float, float]]
        ],
    ) -> None:
        marker_array = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        marker_array.markers.append(clear)

        for index, (record, geometry, color) in enumerate(geometries):
            center = (geometry.bbox_min + geometry.bbox_max) / 2.0
            size = np.maximum(geometry.bbox_max - geometry.bbox_min, 0.01)

            cube = Marker()
            cube.header = source_msg.header
            cube.ns = "object_boxes"
            cube.id = index
            cube.type = Marker.CUBE
            cube.action = Marker.ADD
            cube.pose.position.x = float(center[0])
            cube.pose.position.y = float(center[1])
            cube.pose.position.z = float(center[2])
            cube.pose.orientation.w = 1.0
            cube.scale.x, cube.scale.y, cube.scale.z = map(float, size)
            cube.color.r, cube.color.g, cube.color.b = color
            cube.color.a = 0.22
            cube.lifetime = Duration(sec=1)
            marker_array.markers.append(cube)

            text = Marker()
            text.header = source_msg.header
            text.ns = "object_labels"
            text.id = index
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = float(geometry.centroid[0])
            text.pose.position.y = float(geometry.centroid[1])
            text.pose.position.z = float(geometry.centroid[2])
            text.pose.orientation.w = 1.0
            text.scale.z = 0.07
            text.color.r = text.color.g = text.color.b = 1.0
            text.color.a = 1.0
            text.text = (
                f"{record['label']} {float(record['confidence']):.2f}\n"
                f"z={float(geometry.centroid[2]):.2f} m"
            )
            text.lifetime = Duration(sec=1)
            marker_array.markers.append(text)

        self._markers_pub.publish(marker_array)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = Object3DProjectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
