from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Detection3D:
    label: str
    confidence: float
    centroid: np.ndarray
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    source_instance_id: int


@dataclass
class ObjectTrack:
    node_id: int
    label: str
    centroid: np.ndarray
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    confidence_mean: float
    observations: int
    missed_frames: int
    source_instance_id: int


class ObjectTracker:
    """Greedy label-and-position tracker for detections in one fixed frame."""

    def __init__(
        self,
        *,
        max_match_distance_m: float = 0.35,
        max_missed_frames: int = 20,
        centroid_alpha: float = 0.25,
    ) -> None:
        if max_match_distance_m <= 0.0:
            raise ValueError("max_match_distance_m must be positive")
        if not 0.0 < centroid_alpha <= 1.0:
            raise ValueError("centroid_alpha must be in (0, 1]")
        self.max_match_distance_m = max_match_distance_m
        self.max_missed_frames = max_missed_frames
        self.centroid_alpha = centroid_alpha
        self.tracks: dict[int, ObjectTrack] = {}
        self._next_node_id = 1

    def update(self, detections: list[Detection3D]) -> list[ObjectTrack]:
        for track in self.tracks.values():
            track.missed_frames += 1

        candidates: list[tuple[float, int, int]] = []
        for node_id, track in self.tracks.items():
            for detection_index, detection in enumerate(detections):
                if track.label != detection.label:
                    continue
                distance = float(np.linalg.norm(track.centroid - detection.centroid))
                if distance <= self.max_match_distance_m:
                    candidates.append((distance, node_id, detection_index))

        matched_nodes: set[int] = set()
        matched_detections: set[int] = set()
        for _, node_id, detection_index in sorted(candidates):
            if node_id in matched_nodes or detection_index in matched_detections:
                continue
            self._update_track(self.tracks[node_id], detections[detection_index])
            matched_nodes.add(node_id)
            matched_detections.add(detection_index)

        for index, detection in enumerate(detections):
            if index not in matched_detections:
                self._create_track(detection)

        expired = (
            [
                node_id
                for node_id, track in self.tracks.items()
                if track.missed_frames > self.max_missed_frames
            ]
            if self.max_missed_frames >= 0
            else []
        )
        for node_id in expired:
            del self.tracks[node_id]

        return list(self.tracks.values())

    def _update_track(self, track: ObjectTrack, detection: Detection3D) -> None:
        alpha = self.centroid_alpha
        track.centroid = (1.0 - alpha) * track.centroid + alpha * detection.centroid
        track.bbox_min = (1.0 - alpha) * track.bbox_min + alpha * detection.bbox_min
        track.bbox_max = (1.0 - alpha) * track.bbox_max + alpha * detection.bbox_max
        track.confidence_mean = (
            track.confidence_mean * track.observations + detection.confidence
        ) / (track.observations + 1)
        track.observations += 1
        track.missed_frames = 0
        track.source_instance_id = detection.source_instance_id

    def _create_track(self, detection: Detection3D) -> None:
        node_id = self._next_node_id
        self._next_node_id += 1
        self.tracks[node_id] = ObjectTrack(
            node_id=node_id,
            label=detection.label,
            centroid=detection.centroid.astype(np.float64, copy=True),
            bbox_min=detection.bbox_min.astype(np.float64, copy=True),
            bbox_max=detection.bbox_max.astype(np.float64, copy=True),
            confidence_mean=detection.confidence,
            observations=1,
            missed_frames=0,
            source_instance_id=detection.source_instance_id,
        )
