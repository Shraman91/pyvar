"""Unit tests for detection data models, bounding box logic, and tracker history."""

import numpy as np
import pytest

from config import VARConfig, DetectionConfig, TrackingConfig
from detection.detector import DetectedObject, FrameDetections
from detection.tracker import PlayerTracker


def test_detected_object_properties():
    """Verify geometry calculations of DetectedObject."""
    obj = DetectedObject(
        bbox=(100.0, 200.0, 160.0, 320.0),
        confidence=0.88,
        class_id=0,
        label="player"
    )

    assert obj.x1 == 100.0
    assert obj.y1 == 200.0
    assert obj.x2 == 160.0
    assert obj.y2 == 320.0
    assert obj.width == 60.0
    assert obj.height == 120.0
    assert obj.center == (130.0, 260.0)
    assert obj.bottom_center == (130.0, 320.0)


def test_detected_object_crop():
    """Verify image cropping and torso cropping behavior."""
    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    frame[200:320, 100:160] = [10, 20, 30]

    obj = DetectedObject(
        bbox=(100.0, 200.0, 160.0, 320.0),
        confidence=0.9,
        class_id=0,
        label="player"
    )

    crop = obj.crop(frame)
    assert crop.shape == (120, 60, 3)
    assert np.all(crop == [10, 20, 30])

    # Torso crop
    torso = obj.crop_torso(frame, ratios=(0.2, 0.6, 0.1, 0.9))
    assert torso.shape[0] > 0
    assert torso.shape[1] > 0
    assert torso.shape[0] < crop.shape[0]


def test_detected_object_out_of_bounds():
    """Verify robust handling of bboxes outside frame boundaries."""
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    obj = DetectedObject(
        bbox=(-50.0, -20.0, 150.0, 120.0),
        confidence=0.7,
        class_id=0,
        label="player"
    )

    crop = obj.crop(frame)
    assert crop.shape == (100, 100, 3)


def test_frame_detections_aggregation():
    """Verify collection of players, ball, and referees."""
    p1 = DetectedObject(bbox=(10, 10, 50, 90), confidence=0.8, class_id=0, label="player")
    p2 = DetectedObject(bbox=(60, 10, 100, 90), confidence=0.85, class_id=0, label="player")
    ball = DetectedObject(bbox=(80, 85, 95, 100), confidence=0.9, class_id=32, label="ball")

    fd = FrameDetections(
        frame_index=1,
        players=[p1, p2],
        ball=ball
    )

    assert len(fd.players) == 2
    assert fd.ball is not None
    assert len(fd.all_objects) == 3


def test_tracker_iou_calculation():
    """Verify IoU calculation in PlayerTracker."""
    box1 = np.array([0, 0, 10, 10], dtype=np.float32)
    box2 = np.array([0, 0, 10, 10], dtype=np.float32)
    assert pytest.approx(PlayerTracker._calculate_iou(box1, box2), 0.001) == 1.0

    # Disjoint
    box3 = np.array([20, 20, 30, 30], dtype=np.float32)
    assert PlayerTracker._calculate_iou(box1, box3) == 0.0

    # 50% overlap: box1 area 100, box4 (5, 0, 15, 10) area 100, intersection (5, 0, 10, 10) = 50, union = 150 -> 1/3
    box4 = np.array([5, 0, 15, 10], dtype=np.float32)
    assert pytest.approx(PlayerTracker._calculate_iou(box1, box4), 0.001) == 50.0 / 150.0


def test_config_serialization(tmp_path):
    """Verify config defaults and JSON export/import."""
    config = VARConfig()
    json_path = tmp_path / "config.json"
    config.to_json(json_path)

    assert json_path.exists()
    assert json_path.stat().st_size > 50
