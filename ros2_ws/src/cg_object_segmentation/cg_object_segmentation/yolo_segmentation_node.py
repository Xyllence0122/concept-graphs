from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String

from .postprocess import compose_instance_map, xyxy_to_int_box


def resolve_inference_device(requested: str, cuda_available: bool) -> str:
    """Resolve the configured inference device without hiding explicit errors."""
    normalized = requested.strip().lower()
    if normalized == "auto":
        return "0" if cuda_available else "cpu"
    if not normalized:
        raise ValueError("device cannot be empty")
    return requested


class YoloSegmentationNode(Node):
    """Run instance segmentation on ROS color images using a YOLO model."""

    def __init__(self) -> None:
        super().__init__("yolo_segmentation_node")

        self.declare_parameter("image_topic", "/cg/d455/color/image_raw")
        self.declare_parameter(
            "annotated_topic", "/concept_graph/debug/segmentation_image"
        )
        self.declare_parameter(
            "instance_map_topic", "/concept_graph/segmentation/instance_map"
        )
        self.declare_parameter(
            "detections_topic", "/concept_graph/segmentation/detections_json"
        )
        self.declare_parameter(
            "model_path",
            os.environ.get(
                "CG_MODEL_PATH", "~/concept_graph_models/yolo26n-seg.pt"
            ),
        )
        self.declare_parameter("device", "auto")
        self.declare_parameter("confidence", 0.35)
        self.declare_parameter("image_size", 640)
        self.declare_parameter("process_every_n_frames", 5)
        self.declare_parameter("max_detections", 20)

        image_topic = str(self.get_parameter("image_topic").value)
        annotated_topic = str(self.get_parameter("annotated_topic").value)
        instance_map_topic = str(self.get_parameter("instance_map_topic").value)
        detections_topic = str(self.get_parameter("detections_topic").value)

        model_path = Path(
            os.path.expandvars(
                os.path.expanduser(str(self.get_parameter("model_path").value))
            )
        )
        if not model_path.is_file():
            raise FileNotFoundError(f"Segmentation model not found: {model_path}")

        try:
            import torch
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError(
                "ultralytics is unavailable. Run this node with the cg-ai "
                "virtual environment's Python interpreter."
            ) from error

        self.get_logger().info(f"Loading segmentation model: {model_path}")
        self._model = YOLO(str(model_path))
        self._torch = torch
        self._active_device: str | None = None
        self._bridge = CvBridge()
        self._frame_count = 0
        self._busy = False

        self._annotated_pub = self.create_publisher(Image, annotated_topic, 2)
        self._instance_map_pub = self.create_publisher(Image, instance_map_topic, 2)
        self._detections_pub = self.create_publisher(String, detections_topic, 10)
        self._image_sub = self.create_subscription(
            Image, image_topic, self._image_callback, qos_profile_sensor_data
        )

        self.get_logger().info(f"Color input: {image_topic}")
        self.get_logger().info(f"Annotated output: {annotated_topic}")
        self.get_logger().info(f"Instance-map output: {instance_map_topic}")
        self.get_logger().info(f"Detection metadata: {detections_topic}")

    def _image_callback(self, msg: Image) -> None:
        self._frame_count += 1
        every_n = max(
            1, int(self.get_parameter("process_every_n_frames").value)
        )
        if self._frame_count % every_n != 0 or self._busy:
            return

        self._busy = True
        try:
            requested_device = str(self.get_parameter("device").value)
            inference_device = resolve_inference_device(
                requested_device, bool(self._torch.cuda.is_available())
            )
            if inference_device != self._active_device:
                if requested_device.strip().lower() == "auto" and inference_device == "cpu":
                    self.get_logger().warning(
                        "CUDA is unavailable; YOLO is falling back to CPU"
                    )
                else:
                    self.get_logger().info(
                        f"YOLO inference device: {inference_device}"
                    )
                self._active_device = inference_device

            image = np.asarray(
                self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            )
            height, width = image.shape[:2]
            results = self._model.predict(
                source=image,
                conf=float(self.get_parameter("confidence").value),
                imgsz=int(self.get_parameter("image_size").value),
                device=inference_device,
                max_det=int(self.get_parameter("max_detections").value),
                retina_masks=True,
                verbose=False,
            )
            result = results[0]

            annotated = result.plot()
            annotated_msg = self._bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
            annotated_msg.header = msg.header
            self._annotated_pub.publish(annotated_msg)

            objects: list[dict[str, object]] = []
            if result.masks is None or result.boxes is None:
                masks = np.zeros((0, height, width), dtype=bool)
            else:
                masks = result.masks.data.detach().cpu().numpy() > 0.5
                boxes = result.boxes.xyxy.detach().cpu().numpy()
                classes = result.boxes.cls.detach().cpu().numpy().astype(int)
                confidences = result.boxes.conf.detach().cpu().numpy()

                for index, (box, class_id, confidence) in enumerate(
                    zip(boxes, classes, confidences, strict=True), start=1
                ):
                    objects.append(
                        {
                            "instance_id": index,
                            "class_id": int(class_id),
                            "label": str(result.names[int(class_id)]),
                            "confidence": round(float(confidence), 5),
                            "bbox_xyxy": xyxy_to_int_box(box, width, height),
                            "mask_pixels": int(np.count_nonzero(masks[index - 1])),
                        }
                    )

            instance_map = compose_instance_map(masks)
            instance_msg = self._bridge.cv2_to_imgmsg(
                instance_map, encoding="mono16"
            )
            instance_msg.header = msg.header
            self._instance_map_pub.publish(instance_msg)

            metadata = {
                "stamp": {
                    "sec": int(msg.header.stamp.sec),
                    "nanosec": int(msg.header.stamp.nanosec),
                },
                "frame_id": msg.header.frame_id,
                "image_width": width,
                "image_height": height,
                "objects": objects,
            }
            metadata_msg = String()
            metadata_msg.data = json.dumps(metadata, ensure_ascii=False)
            self._detections_pub.publish(metadata_msg)

            self.get_logger().info(
                f"Frame {self._frame_count}: detected {len(objects)} objects"
            )
        except Exception as error:  # keep the ROS node alive after a bad frame
            self.get_logger().error(f"Segmentation failed: {error}")
        finally:
            self._busy = False


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node: YoloSegmentationNode | None = None
    try:
        node = YoloSegmentationNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
