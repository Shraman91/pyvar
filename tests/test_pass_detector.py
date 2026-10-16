"""Unit tests for pass moment detection via kinematics and player proximity."""

from config import EventConfig
from detection.detector import DetectedObject, FrameDetections
from events.pass_detector import PassDetector


def test_pass_detection_on_velocity_kick():
    """Verify pass trigger when ball suddenly accelerates in proximity to a player."""
    detector = PassDetector(EventConfig(velocity_change_threshold=5.0, ball_proximity_radius_px=50.0))

    # Player stationary at foot=(100, 200)
    player = DetectedObject(bbox=(80, 140, 120, 200), confidence=0.9, class_id=0, label="player", track_id=7)

    # Frame 1: Ball with player
    b1 = DetectedObject(bbox=(95, 190, 105, 200), confidence=0.9, class_id=32, label="ball")
    fd1 = FrameDetections(frame_index=1, players=[player], ball=b1)
    ev1 = detector.update(fd1)
    assert ev1 is None

    # Frame 2: Ball still with player
    b2 = DetectedObject(bbox=(96, 190, 106, 200), confidence=0.9, class_id=32, label="ball")
    fd2 = FrameDetections(frame_index=2, players=[player], ball=b2)
    ev2 = detector.update(fd2)
    assert ev2 is None

    # Frame 3: Kick! Ball shot forward rapidly (dx=30 px/frame)
    b3 = DetectedObject(bbox=(125, 190, 135, 200), confidence=0.9, class_id=32, label="ball")
    fd3 = FrameDetections(frame_index=3, players=[player], ball=b3)
    ev3 = detector.update(fd3)

    assert ev3 is not None
    assert ev3.frame_index == 3
    assert ev3.passer_id == 7
    assert ev3.accel_magnitude >= 5.0


def test_pass_detection_manual_override():
    """Verify manual override sets the exact pass frame and identifies nearest player."""
    detector = PassDetector()
    detector.set_manual_pass_frame(frame_index=42)

    player = DetectedObject(bbox=(200, 200, 240, 280), confidence=0.9, class_id=0, label="player", track_id=10)
    ball = DetectedObject(bbox=(215, 275, 225, 285), confidence=0.9, class_id=32, label="ball")
    fd = FrameDetections(frame_index=42, players=[player], ball=ball)

    event = detector.update(fd)
    assert event is not None
    assert event.frame_index == 42
    assert event.is_manual is True
    assert event.passer_id == 10
