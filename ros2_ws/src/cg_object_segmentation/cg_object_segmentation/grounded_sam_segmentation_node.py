from __future__ import annotations

from contextlib import nullcontext
import json
import re

from cv_bridge import CvBridge
import cv2
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String

from .postprocess import compose_instance_map, xyxy_to_int_box


def parse_text_prompt(prompt: str) -> list[str]:
    """Return normalized Grounding DINO labels from a dotted prompt."""
    labels = [
        label.strip().lower()
        for label in re.split(r"[.;\n]+", prompt)
        if label.strip()
    ]
    if not labels:
        raise ValueError("text_prompt must contain at least one object label")
    return labels


def class_id_for_label(label: str, queries: list[str]) -> int:
    """Map a decoded Grounding DINO phrase back to its prompt index."""
    normalized = label.strip().lower().strip(" .")
    for index, query in enumerate(queries):
        if normalized == query or normalized in query or query in normalized:
            return index
    return -1


def select_best_sam_masks(
    masks: np.ndarray, iou_scores: np.ndarray, threshold: float
) -> np.ndarray:
    """Select SAM's highest-IoU proposal for every input box."""
    mask_array = np.asarray(masks)
    score_array = np.asarray(iou_scores)
    if mask_array.ndim != 4:
        raise ValueError("SAM masks must have shape (N, K, H, W)")
    if score_array.shape != mask_array.shape[:2]:
        raise ValueError("SAM IoU scores must have shape (N, K)")
    if mask_array.shape[0] == 0:
        return np.zeros(
            (0, mask_array.shape[-2], mask_array.shape[-1]), dtype=bool
        )
    best = np.argmax(score_array, axis=1)
    return mask_array[np.arange(mask_array.shape[0]), best] > threshold


def non_maximum_suppression(
    boxes: np.ndarray, scores: np.ndarray, iou_threshold: float
) -> np.ndarray:
    """Return score-ordered indices after class-agnostic xyxy NMS."""
    box_array = np.asarray(boxes, dtype=np.float64)
    score_array = np.asarray(scores, dtype=np.float64)
    if box_array.ndim != 2 or box_array.shape[1:] != (4,):
        raise ValueError("boxes must have shape (N, 4)")
    if score_array.shape != (len(box_array),):
        raise ValueError("scores must have shape (N,)")
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in [0, 1]")
    if len(box_array) == 0:
        return np.zeros(0, dtype=np.int64)

    x1, y1, x2, y2 = box_array.T
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = np.argsort(score_array)[::-1]
    kept: list[int] = []
    while len(order):
        current = int(order[0])
        kept.append(current)
        if len(order) == 1:
            break
        remaining = order[1:]
        intersection_width = np.maximum(
            0.0,
            np.minimum(x2[current], x2[remaining])
            - np.maximum(x1[current], x1[remaining]),
        )
        intersection_height = np.maximum(
            0.0,
            np.minimum(y2[current], y2[remaining])
            - np.maximum(y1[current], y1[remaining]),
        )
        intersection = intersection_width * intersection_height
        union = areas[current] + areas[remaining] - intersection
        iou = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection),
            where=union > 0.0,
        )
        order = remaining[iou <= iou_threshold]
    return np.asarray(kept, dtype=np.int64)


def _move_inputs(inputs: dict, device: str):
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }


class GroundedSamSegmentationNode(Node):
    """Text-prompted Grounding DINO boxes refined into masks by SAM."""

    def __init__(self) -> None:
        super().__init__("grounded_sam_segmentation_node")

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
            "grounding_model_id", "IDEA-Research/grounding-dino-tiny"
        )
        self.declare_parameter(
            "grounding_model_revision",
            "a2bb814dd30d776dcf7e30523b00659f4f141c71",
        )
        self.declare_parameter("sam_model_id", "facebook/sam-vit-base")
        self.declare_parameter(
            "sam_model_revision",
            "70c1a07f894ebb5b307fd9eaaee97b9dfc16068f",
        )
        self.declare_parameter("text_prompt", "box. eraser.")
        self.declare_parameter("device", "auto")
        self.declare_parameter("box_threshold", 0.20)
        self.declare_parameter("text_threshold", 0.20)
        self.declare_parameter("mask_threshold", 0.0)
        self.declare_parameter("nms_iou_threshold", 0.50)
        self.declare_parameter("process_every_n_frames", 30)
        self.declare_parameter("max_detections", 10)
        self.declare_parameter("use_fp16", True)

        try:
            import torch
            from transformers import (
                AutoModelForZeroShotObjectDetection,
                AutoProcessor,
                SamModel,
                SamProcessor,
            )
        except ImportError as error:
            raise RuntimeError(
                "Grounded SAM dependencies are unavailable. Install the pinned "
                "AI requirements and expose that environment with "
                "CG_AI_SITE_PACKAGES."
            ) from error

        self._torch = torch
        requested_device = str(self.get_parameter("device").value).strip()
        if requested_device.lower() == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        elif requested_device:
            self._device = requested_device
        else:
            raise ValueError("device cannot be empty")

        self._use_fp16 = bool(self.get_parameter("use_fp16").value) and str(
            self._device
        ).startswith("cuda")
        model_dtype = torch.float16 if self._use_fp16 else torch.float32
        grounding_id = str(self.get_parameter("grounding_model_id").value)
        grounding_revision = str(
            self.get_parameter("grounding_model_revision").value
        )
        sam_id = str(self.get_parameter("sam_model_id").value)
        sam_revision = str(self.get_parameter("sam_model_revision").value)

        self.get_logger().info(
            f"Loading Grounding DINO {grounding_id} on {self._device}"
        )
        self._grounding_processor = AutoProcessor.from_pretrained(
            grounding_id, revision=grounding_revision
        )
        self._grounding_model = (
            AutoModelForZeroShotObjectDetection.from_pretrained(
                grounding_id, revision=grounding_revision, dtype=model_dtype
            )
            .to(self._device)
            .eval()
        )
        self.get_logger().info(f"Loading SAM {sam_id} on {self._device}")
        self._sam_processor = SamProcessor.from_pretrained(
            sam_id, revision=sam_revision
        )
        self._sam_model = (
            SamModel.from_pretrained(
                sam_id, revision=sam_revision, dtype=model_dtype
            )
            .to(self._device)
            .eval()
        )

        self._bridge = CvBridge()
        self._frame_count = 0
        self._busy = False
        self._queries = parse_text_prompt(
            str(self.get_parameter("text_prompt").value)
        )

        self._annotated_pub = self.create_publisher(
            Image, str(self.get_parameter("annotated_topic").value), 2
        )
        self._instance_map_pub = self.create_publisher(
            Image, str(self.get_parameter("instance_map_topic").value), 2
        )
        self._detections_pub = self.create_publisher(
            String, str(self.get_parameter("detections_topic").value), 10
        )
        self._image_sub = self.create_subscription(
            Image,
            str(self.get_parameter("image_topic").value),
            self._image_callback,
            qos_profile_sensor_data,
        )
        self.get_logger().info(
            "Grounded SAM ready; open-vocabulary queries: "
            + ", ".join(self._queries)
        )

    def _autocast(self):
        if not self._use_fp16:
            return nullcontext()
        return self._torch.autocast(device_type="cuda", dtype=self._torch.float16)

    def _detect_boxes(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str]]:
        height, width = rgb.shape[:2]
        inputs = self._grounding_processor(
            images=rgb,
            text=[self._queries],
            return_tensors="pt",
        )
        inputs = _move_inputs(inputs, self._device)
        with self._torch.inference_mode(), self._autocast():
            outputs = self._grounding_model(**inputs)
        result = self._grounding_processor.post_process_grounded_object_detection(
            outputs,
            inputs.get("input_ids"),
            threshold=float(self.get_parameter("box_threshold").value),
            text_threshold=float(self.get_parameter("text_threshold").value),
            target_sizes=[(height, width)],
            text_labels=[self._queries],
        )[0]
        boxes = result["boxes"].detach().float().cpu().numpy()
        scores = result["scores"].detach().float().cpu().numpy()
        labels = [str(value) for value in result["text_labels"]]

        order = non_maximum_suppression(
            boxes,
            scores,
            float(self.get_parameter("nms_iou_threshold").value),
        )
        max_detections = max(1, int(self.get_parameter("max_detections").value))
        order = order[:max_detections]
        return boxes[order], scores[order], [labels[index] for index in order]

    def _segment_boxes(self, rgb: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        height, width = rgb.shape[:2]
        if len(boxes) == 0:
            return np.zeros((0, height, width), dtype=bool)

        inputs = self._sam_processor(
            images=rgb,
            input_boxes=[boxes.tolist()],
            return_tensors="pt",
        )
        original_sizes = inputs["original_sizes"].clone()
        reshaped_sizes = inputs["reshaped_input_sizes"].clone()
        inputs = _move_inputs(inputs, self._device)
        with self._torch.inference_mode(), self._autocast():
            outputs = self._sam_model(**inputs, multimask_output=True)
        masks = self._sam_processor.image_processor.post_process_masks(
            outputs.pred_masks.detach().float().cpu(),
            original_sizes,
            reshaped_sizes,
        )[0].numpy()
        scores = outputs.iou_scores.detach().float().cpu().numpy()[0]
        return select_best_sam_masks(
            masks, scores, float(self.get_parameter("mask_threshold").value)
        )

    @staticmethod
    def _annotate(
        image: np.ndarray,
        masks: np.ndarray,
        boxes: np.ndarray,
        scores: np.ndarray,
        labels: list[str],
    ) -> np.ndarray:
        annotated = image.copy()
        overlay = annotated.copy()
        for index, mask in enumerate(masks):
            color = (
                int((53 * (index + 1)) % 255),
                int((97 * (index + 1)) % 255),
                int((193 * (index + 1)) % 255),
            )
            overlay[mask] = color
        annotated = cv2.addWeighted(overlay, 0.40, annotated, 0.60, 0.0)

        for index, (box, score, label) in enumerate(
            zip(boxes, scores, labels, strict=True), start=1
        ):
            x1, y1, x2, y2 = xyxy_to_int_box(
                box, image.shape[1], image.shape[0]
            )
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(
                annotated,
                f"{index}:{label} {score:.2f}",
                (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        return annotated

    def _publish_empty_or_results(
        self,
        msg: Image,
        image: np.ndarray,
        boxes: np.ndarray,
        scores: np.ndarray,
        labels: list[str],
        masks: np.ndarray,
    ) -> None:
        height, width = image.shape[:2]
        objects: list[dict[str, object]] = []
        for index, (box, score, label, mask) in enumerate(
            zip(boxes, scores, labels, masks, strict=True), start=1
        ):
            objects.append(
                {
                    "instance_id": index,
                    "class_id": class_id_for_label(label, self._queries),
                    "label": label,
                    "confidence": round(float(score), 5),
                    "bbox_xyxy": xyxy_to_int_box(box, width, height),
                    "mask_pixels": int(np.count_nonzero(mask)),
                }
            )

        instance_msg = self._bridge.cv2_to_imgmsg(
            compose_instance_map(masks), encoding="mono16"
        )
        instance_msg.header = msg.header
        self._instance_map_pub.publish(instance_msg)

        annotated = self._annotate(image, masks, boxes, scores, labels)
        annotated_msg = self._bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        annotated_msg.header = msg.header
        self._annotated_pub.publish(annotated_msg)

        metadata = {
            "stamp": {
                "sec": int(msg.header.stamp.sec),
                "nanosec": int(msg.header.stamp.nanosec),
            },
            "frame_id": msg.header.frame_id,
            "detector": "grounding_dino_sam",
            "text_queries": self._queries,
            "image_width": width,
            "image_height": height,
            "objects": objects,
        }
        metadata_msg = String()
        metadata_msg.data = json.dumps(metadata, ensure_ascii=False)
        self._detections_pub.publish(metadata_msg)
        self.get_logger().info(
            f"Frame {self._frame_count}: Grounded SAM detected {len(objects)} objects"
        )

    def _image_callback(self, msg: Image) -> None:
        self._frame_count += 1
        every_n = max(
            1, int(self.get_parameter("process_every_n_frames").value)
        )
        if self._frame_count % every_n != 0 or self._busy:
            return

        self._busy = True
        try:
            image = np.asarray(
                self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            )
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            boxes, scores, labels = self._detect_boxes(rgb)
            masks = self._segment_boxes(rgb, boxes)
            self._publish_empty_or_results(
                msg, image, boxes, scores, labels, masks
            )
        except Exception as error:  # keep processing future frames after a bad input
            self.get_logger().error(f"Grounded SAM inference failed: {error}")
        finally:
            self._busy = False


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node: GroundedSamSegmentationNode | None = None
    try:
        node = GroundedSamSegmentationNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
