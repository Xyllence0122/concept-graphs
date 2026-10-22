import numpy as np

from cg_object_segmentation.tracker import Detection3D, ObjectTracker


def detection(label: str, x: float, instance_id: int = 1) -> Detection3D:
    center = np.array([x, 0.0, 1.0])
    return Detection3D(
        label=label,
        confidence=0.9,
        centroid=center,
        bbox_min=center - 0.1,
        bbox_max=center + 0.1,
        source_instance_id=instance_id,
    )


def test_nearby_same_label_keeps_node_id() -> None:
    tracker = ObjectTracker(max_match_distance_m=0.3)
    first = tracker.update([detection("bottle", 0.0)])[0]
    second = tracker.update([detection("bottle", 0.1, 7)])[0]

    assert first.node_id == second.node_id == 1
    assert second.observations == 2
    assert second.source_instance_id == 7


def test_different_label_creates_new_node() -> None:
    tracker = ObjectTracker(max_match_distance_m=0.3)
    tracker.update([detection("bottle", 0.0)])
    tracks = tracker.update([detection("cup", 0.0)])

    assert {track.label for track in tracks} == {"bottle", "cup"}


def test_distant_same_label_creates_new_node() -> None:
    tracker = ObjectTracker(max_match_distance_m=0.3)
    tracker.update([detection("bottle", 0.0)])
    tracks = tracker.update([detection("bottle", 1.0)])

    assert {track.node_id for track in tracks} == {1, 2}


def test_stale_node_expires() -> None:
    tracker = ObjectTracker(max_missed_frames=1)
    tracker.update([detection("bottle", 0.0)])
    tracker.update([])
    tracks = tracker.update([])

    assert tracks == []


def test_negative_missed_limit_keeps_map_node() -> None:
    tracker = ObjectTracker(max_missed_frames=-1)
    tracker.update([detection("bottle", 0.0)])
    for _ in range(100):
        tracks = tracker.update([])

    assert len(tracks) == 1
    assert tracks[0].label == "bottle"
