from __future__ import annotations

import colorsys
import json
from typing import Any

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

from .tracker import Detection3D, ObjectTrack, ObjectTracker


def _node_color(node_id: int) -> tuple[float, float, float]:
    return colorsys.hsv_to_rgb((node_id * 0.618033988749895) % 1.0, 0.75, 1.0)


class ObjectGraphNode(Node):
    """Merge map-frame detections into persistent semantic graph nodes."""

    def __init__(self) -> None:
        super().__init__("object_graph_node")
        self.declare_parameter(
            "detections_3d_topic", "/concept_graph/objects/detections_map_json"
        )
        self.declare_parameter("expected_frame_id", "map")
        self.declare_parameter("graph_topic", "/concept_graph/graph/json")
        self.declare_parameter("markers_topic", "/concept_graph/graph/markers")
        self.declare_parameter("max_match_distance_m", 0.35)
        self.declare_parameter(
            "max_missed_frames", -1
        )  # negative means persistent map nodes
        self.declare_parameter("centroid_alpha", 0.25)
        self.declare_parameter("min_observations", 3)

        self._tracker = ObjectTracker(
            max_match_distance_m=float(
                self.get_parameter("max_match_distance_m").value
            ),
            max_missed_frames=int(self.get_parameter("max_missed_frames").value),
            centroid_alpha=float(self.get_parameter("centroid_alpha").value),
        )
        self._frame_id = ""

        graph_qos = QoSProfile(depth=1)
        graph_qos.reliability = ReliabilityPolicy.RELIABLE
        graph_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self._graph_pub = self.create_publisher(
            String, str(self.get_parameter("graph_topic").value), graph_qos
        )
        self._markers_pub = self.create_publisher(
            MarkerArray, str(self.get_parameter("markers_topic").value), 5
        )
        self._detections_sub = self.create_subscription(
            String,
            str(self.get_parameter("detections_3d_topic").value),
            self._detections_callback,
            10,
        )
        self.get_logger().info("Persistent map-frame semantic graph enabled")

    def _detections_callback(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
            self._frame_id = str(payload["frame_id"])
            expected_frame = str(self.get_parameter("expected_frame_id").value)
            if self._frame_id != expected_frame:
                raise ValueError(
                    f"expected frame {expected_frame!r}, received {self._frame_id!r}"
                )
            detections = [self._parse_detection(item) for item in payload["objects"]]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            self.get_logger().error(f"Invalid 3D detection JSON: {error}")
            return

        tracks = self._tracker.update(detections)
        minimum = max(1, int(self.get_parameter("min_observations").value))
        confirmed = [track for track in tracks if track.observations >= minimum]
        self._publish_graph(payload.get("stamp", {}), confirmed)
        self._publish_markers(confirmed)
        self.get_logger().info(
            f"Graph contains {len(confirmed)} confirmed nodes "
            f"({len(tracks)} including tentative)"
        )

    @staticmethod
    def _parse_detection(item: dict[str, Any]) -> Detection3D:
        center = item["centroid_m"]
        return Detection3D(
            label=str(item["label"]),
            confidence=float(item["confidence"]),
            centroid=np.array(
                [center["x"], center["y"], center["z"]], dtype=np.float64
            ),
            bbox_min=np.asarray(item["bbox_3d_min_m"], dtype=np.float64),
            bbox_max=np.asarray(item["bbox_3d_max_m"], dtype=np.float64),
            source_instance_id=int(item["instance_id"]),
        )

    def _publish_graph(self, stamp: dict[str, Any], tracks: list[ObjectTrack]) -> None:
        graph = {
            "schema_version": 1,
            "coordinate_frame": self._frame_id,
            "fixed_camera_assumption": False,
            "stamp": stamp,
            "nodes": [self._track_record(track) for track in tracks],
            "edges": [],
        }
        message = String()
        message.data = json.dumps(graph, ensure_ascii=False)
        self._graph_pub.publish(message)

    @staticmethod
    def _track_record(track: ObjectTrack) -> dict[str, Any]:
        return {
            "node_id": f"object_{track.node_id:04d}",
            "label": track.label,
            "centroid_m": {
                "x": round(float(track.centroid[0]), 5),
                "y": round(float(track.centroid[1]), 5),
                "z": round(float(track.centroid[2]), 5),
            },
            "bbox_3d_min_m": [round(float(v), 5) for v in track.bbox_min],
            "bbox_3d_max_m": [round(float(v), 5) for v in track.bbox_max],
            "confidence_mean": round(float(track.confidence_mean), 5),
            "observations": track.observations,
            "missed_frames": track.missed_frames,
            "last_source_instance_id": track.source_instance_id,
        }

    def _publish_markers(self, tracks: list[ObjectTrack]) -> None:
        output = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        output.markers.append(clear)

        for track in tracks:
            color = _node_color(track.node_id)
            marker = Marker()
            marker.header.frame_id = self._frame_id
            marker.ns = "concept_graph_nodes"
            marker.id = track.node_id
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = float(track.centroid[0])
            marker.pose.position.y = float(track.centroid[1])
            marker.pose.position.z = float(track.centroid[2])
            marker.pose.orientation.w = 1.0
            marker.scale.x = marker.scale.y = marker.scale.z = 0.10
            marker.color.r, marker.color.g, marker.color.b = color
            marker.color.a = 1.0
            marker.lifetime = Duration(sec=1)
            output.markers.append(marker)

            text = Marker()
            text.header.frame_id = self._frame_id
            text.ns = "concept_graph_labels"
            text.id = track.node_id
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = float(track.centroid[0])
            text.pose.position.y = float(track.centroid[1]) - 0.08
            text.pose.position.z = float(track.centroid[2])
            text.pose.orientation.w = 1.0
            text.scale.z = 0.08
            text.color.r = text.color.g = text.color.b = 1.0
            text.color.a = 1.0
            text.text = f"object_{track.node_id:04d}: {track.label} ({track.observations})"
            text.lifetime = Duration(sec=1)
            output.markers.append(text)

        self._markers_pub.publish(output)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = ObjectGraphNode()
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
