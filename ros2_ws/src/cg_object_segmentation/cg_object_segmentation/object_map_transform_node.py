from __future__ import annotations

import json
from typing import Any

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener

from .transform3d import (
    compose_transforms,
    transform_axis_aligned_bbox,
    transform_points,
)


class ObjectMapTransformNode(Node):
    """Transform camera-frame 3D object detections into a fixed map frame."""

    def __init__(self) -> None:
        super().__init__("object_map_transform_node")
        self.declare_parameter(
            "input_topic", "/concept_graph/objects/detections_3d_json"
        )
        self.declare_parameter(
            "output_topic", "/concept_graph/objects/detections_map_json"
        )
        self.declare_parameter("target_frame", "map")
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("lookup_timeout_seconds", 0.25)
        self.declare_parameter("tf_cache_seconds", 30.0)

        cache_seconds = float(self.get_parameter("tf_cache_seconds").value)
        self._tf_buffer = Buffer(
            cache_time=Duration(seconds=max(cache_seconds, 1.0)), node=self
        )
        # TF messages must keep flowing while an object callback waits for the
        # transform at that image's timestamp.  A separate TF executor avoids
        # starving /tf in this node's otherwise single-threaded executor.
        self._tf_listener = TransformListener(
            self._tf_buffer, self, spin_thread=True
        )
        self._publisher = self.create_publisher(
            String, str(self.get_parameter("output_topic").value), 10
        )
        self._subscription = self.create_subscription(
            String,
            str(self.get_parameter("input_topic").value),
            self._detections_callback,
            10,
        )
        self.get_logger().info(
            "Waiting for camera-frame objects and map transforms"
        )

    def _detections_callback(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
            source_frame = str(payload["frame_id"])
            stamp = payload["stamp"]
            target_frame = str(self.get_parameter("target_frame").value)
            int(stamp["sec"])
            int(stamp["nanosec"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            self.get_logger().error(f"Invalid camera-frame object JSON: {error}")
            return

        try:
            odom_frame = str(self.get_parameter("odom_frame").value)
            map_to_odom = self._tf_buffer.lookup_transform(
                target_frame,
                odom_frame,
                Time(clock_type=self.get_clock().clock_type),
                timeout=Duration(
                    seconds=float(
                        self.get_parameter("lookup_timeout_seconds").value
                    )
                ),
            )
            odom_to_source = self._tf_buffer.lookup_transform(
                odom_frame,
                source_frame,
                Time(clock_type=self.get_clock().clock_type),
                timeout=Duration(
                    seconds=float(
                        self.get_parameter("lookup_timeout_seconds").value
                    )
                ),
            )
        except TransformException as error:
            self.get_logger().warning(
                f"Cannot transform {source_frame} through {odom_frame} "
                f"to {target_frame} at "
                f"{stamp['sec']}.{int(stamp['nanosec']):09d}: {error}",
                throttle_duration_sec=2.0,
            )
            return

        map_translation, map_quaternion = self._transform_arrays(map_to_odom)
        odom_translation, odom_quaternion = self._transform_arrays(
            odom_to_source
        )
        translation_xyz, quaternion_xyzw = compose_transforms(
            map_translation,
            map_quaternion,
            odom_translation,
            odom_quaternion,
        )

        try:
            objects = [
                self._transform_object(
                    item,
                    target_frame,
                    translation_xyz,
                    quaternion_xyzw,
                )
                for item in payload.get("objects", [])
            ]
        except (KeyError, TypeError, ValueError) as error:
            self.get_logger().error(f"Invalid 3D object geometry: {error}")
            return

        output = {
            **payload,
            "source_frame_id": source_frame,
            "frame_id": target_frame,
            "transform_strategy": "latest_map_to_odom+latest_odom_to_sensor",
            "pose_stamp": {
                "sec": int(odom_to_source.header.stamp.sec),
                "nanosec": int(odom_to_source.header.stamp.nanosec),
            },
            "objects": objects,
        }
        output_message = String()
        output_message.data = json.dumps(output, ensure_ascii=False)
        self._publisher.publish(output_message)

    @staticmethod
    def _transform_arrays(transform: Any) -> tuple[np.ndarray, np.ndarray]:
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        return (
            np.array(
                [translation.x, translation.y, translation.z],
                dtype=np.float64,
            ),
            np.array(
                [rotation.x, rotation.y, rotation.z, rotation.w],
                dtype=np.float64,
            ),
        )

    @staticmethod
    def _transform_object(
        item: dict[str, Any],
        target_frame: str,
        translation_xyz: np.ndarray,
        quaternion_xyzw: np.ndarray,
    ) -> dict[str, Any]:
        center = item["centroid_m"]
        centroid = np.array(
            [center["x"], center["y"], center["z"]], dtype=np.float64
        )
        transformed_centroid = transform_points(
            centroid, translation_xyz, quaternion_xyzw
        )
        bbox_min, bbox_max = transform_axis_aligned_bbox(
            np.asarray(item["bbox_3d_min_m"], dtype=np.float64),
            np.asarray(item["bbox_3d_max_m"], dtype=np.float64),
            translation_xyz,
            quaternion_xyzw,
        )

        return {
            **item,
            "frame_id": target_frame,
            "centroid_m": {
                "x": round(float(transformed_centroid[0]), 5),
                "y": round(float(transformed_centroid[1]), 5),
                "z": round(float(transformed_centroid[2]), 5),
            },
            "bbox_3d_min_m": [round(float(value), 5) for value in bbox_min],
            "bbox_3d_max_m": [round(float(value), 5) for value in bbox_max],
        }


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = ObjectMapTransformNode()
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
