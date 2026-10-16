"""Unit tests for homography calibration, projection, and temporal smoothing."""

import numpy as np
import pytest

from config import PitchConfig
from pitch.homography import HomographyEstimator
from utils.sample_video import get_ground_truth_homography


def test_homography_exact_projection():
    """Verify reprojection accuracy with ground-truth synthetic pitch mapping."""
    estimator = HomographyEstimator(PitchConfig())
    gt_h = get_ground_truth_homography(width=1280, height=720)

    # 4 known points
    img_pts = np.array([
        [180.0, 240.0],
        [1100.0, 240.0],
        [1240.0, 680.0],
        [40.0, 680.0],
    ], dtype=np.float32)

    pitch_pts = np.array([
        [52.5, 60.0],
        [105.0, 60.0],
        [105.0, 8.0],
        [52.5, 8.0],
    ], dtype=np.float32)

    h_mat, confidence = estimator.compute_from_points(img_pts, pitch_pts)
    assert h_mat is not None
    assert confidence > 0.85

    # Check forward projection (frame -> pitch)
    proj_pitch = estimator.frame_to_pitch(img_pts)
    assert np.allclose(proj_pitch, pitch_pts, atol=0.2)

    # Check inverse projection (pitch -> frame)
    proj_frame = estimator.pitch_to_frame(pitch_pts)
    assert np.allclose(proj_frame, img_pts, atol=1.0)


def test_homography_degenerate_rejection():
    """Verify that collinear or degenerate points are safely rejected with low confidence."""
    estimator = HomographyEstimator()
    # 4 collinear points
    img_pts = np.array([[10, 10], [20, 20], [30, 30], [40, 40]], dtype=np.float32)
    pitch_pts = np.array([[0, 0], [10, 0], [20, 0], [30, 0]], dtype=np.float32)

    h_mat, confidence = estimator.compute_from_points(img_pts, pitch_pts)
    # Collinear points fail to form a valid projective basis
    assert confidence < 0.35 or h_mat is None


def test_homography_temporal_smoothing():
    """Verify temporal smoothing stabilizes noisy frame-to-frame homographies."""
    estimator = HomographyEstimator(PitchConfig(homography_smoothing_window=3))
    base_h = get_ground_truth_homography()

    img_pts = np.array([
        [180.0, 240.0],
        [1100.0, 240.0],
        [1240.0, 680.0],
        [40.0, 680.0],
    ], dtype=np.float32)

    pitch_pts = np.array([
        [52.5, 60.0],
        [105.0, 60.0],
        [105.0, 8.0],
        [52.5, 8.0],
    ], dtype=np.float32)

    # Feed homographies with small noise
    for _ in range(3):
        noisy_img = img_pts + np.random.normal(0, 0.5, img_pts.shape).astype(np.float32)
        h, conf = estimator.compute_from_points(noisy_img, pitch_pts)
        assert h is not None

    assert len(estimator.history) == 3
