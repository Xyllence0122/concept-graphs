from __future__ import annotations

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from sensor_msgs_py import point_cloud2

from .projection import project_depth_roi


class DepthRoiNode(Node):
    """Turn a centered rectangular region of aligned depth into XYZ points."""

    def __init__(self) -> None:
        super().__init__("depth_roi_node")

        self.declare_parameter(
            "depth_topic", "/cg/d455/aligned_depth_to_color/image_raw"
        )
        self.declare_parameter(
            "camera_info_topic", "/cg/d455/aligned_depth_to_color/camera_info"
        )
        self.declare_parameter("output_topic", "/concept_graph/debug/roi_points")
        self.declare_parameter("roi_width_fraction", 0.30)
        self.declare_parameter("roi_height_fraction", 0.30)
        self.declare_parameter("stride", 4)
        self.declare_parameter("depth_scale", 0.001)
        self.declare_parameter("min_depth_m", 0.20)
        self.declare_parameter("max_depth_m", 4.00)

        depth_topic = str(self.get_parameter("depth_topic").value)
        camera_info_topic = str(self.get_parameter("camera_info_topic").value)
        output_topic = str(self.get_parameter("output_topic").value)

        self._bridge = CvBridge()
        self._camera_matrix: np.ndarray | None = None
        self._info_size: tuple[int, int] | None = None
        self._frame_count = 0

        self._cloud_pub = self.create_publisher(PointCloud2, output_topic, 10)
        self._info_sub = self.create_subscription(
            CameraInfo,
            camera_info_topic,
            self._camera_info_callback,
            qos_profile_sensor_data,
        )
        self._depth_sub = self.create_subscription(
            Image,
            depth_topic,
            self._depth_callback,
            qos_profile_sensor_data,
        )

        self.get_logger().info(f"Depth input: {depth_topic}")
        self.get_logger().info(f"CameraInfo input: {camera_info_topic}")
        self.get_logger().info(f"PointCloud2 output: {output_topic}")

    def _camera_info_callback(self, msg: CameraInfo) -> None:
        matrix = np.asarray(msg.k, dtype=np.float64).reshape(3, 3)
        if matrix[0, 0] <= 0.0 or matrix[1, 1] <= 0.0:
            self.get_logger().error("Received invalid camera intrinsics")
            return
        self._camera_matrix = matrix
        self._info_size = (int(msg.width), int(msg.height))

    def _depth_callback(self, msg: Image) -> None:
        if self._camera_matrix is None:
            return

        depth = np.asarray(self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough"))
        if depth.ndim != 2:
            self.get_logger().error(f"Expected one-channel depth; got shape {depth.shape}")
            return

        if msg.encoding in ("16UC1", "mono16"):
            depth_m = depth.astype(np.float32) * float(
                self.get_parameter("depth_scale").value
            )
        elif msg.encoding == "32FC1":
            depth_m = depth.astype(np.float32, copy=False)
        else:
            self.get_logger().error(f"Unsupported depth encoding: {msg.encoding}")
            return

        height, width = depth_m.shape
        if self._info_size is not None and self._info_size != (width, height):
            self.get_logger().error(
                "Depth image and CameraInfo dimensions differ: "
                f"depth={width}x{height}, info={self._info_size[0]}x{self._info_size[1]}"
            )
            return

        roi_width_fraction = float(self.get_parameter("roi_width_fraction").value)
        roi_height_fraction = float(self.get_parameter("roi_height_fraction").value)
        roi_width = max(1, round(width * roi_width_fraction))
        roi_height = max(1, round(height * roi_height_fraction))
        u_min = (width - roi_width) // 2
        v_min = (height - roi_height) // 2
        roi = (u_min, v_min, u_min + roi_width, v_min + roi_height)

        points = project_depth_roi(
            depth_m,
            self._camera_matrix,
            roi,
            stride=int(self.get_parameter("stride").value),
            min_depth_m=float(self.get_parameter("min_depth_m").value),
            max_depth_m=float(self.get_parameter("max_depth_m").value),
        )

        cloud = point_cloud2.create_cloud_xyz32(msg.header, points.tolist())
        self._cloud_pub.publish(cloud)

        self._frame_count += 1
        if self._frame_count % 90 == 0:
            self.get_logger().info(
                f"Published {len(points)} ROI points in frame {msg.header.frame_id}"
            )


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = DepthRoiNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
