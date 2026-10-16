"""Homography estimation and pitch metric projection module for pyvar.

Computes 3x3 perspective transformation mapping image coordinates (px)
to standard FIFA pitch coordinates (meters), with temporal smoothing
and calibration sanity validation.
"""

from collections import deque
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from config import PitchConfig

import logging

logger = logging.getLogger("pyvar.homography")


def get_standard_pitch_landmarks(side: str = "right", pitch_length: float = 105.0, pitch_width: float = 68.0) -> Dict[str, Tuple[float, float]]:
    """Return standard FIFA pitch landmarks in meters (length x width)."""
    pb_len = 16.5
    pb_w = 40.32
    y_center = pitch_width / 2.0  # 34.0
    pb_y_top = y_center + pb_w / 2.0    # 54.16
    pb_y_bot = y_center - pb_w / 2.0    # 13.84

    if side.lower() == "right":
        goal_x = pitch_length  # 105.0
        pb_x = goal_x - pb_len # 88.5
        return {
            "pb_top_goal": (goal_x, pb_y_top),
            "pb_top_outer": (pb_x, pb_y_top),
            "pb_bot_outer": (pb_x, pb_y_bot),
            "pb_bot_goal": (goal_x, pb_y_bot),
            "penalty_spot": (goal_x - 11.0, y_center),
            "halfway_top": (pitch_length / 2.0, pitch_width),
            "halfway_bot": (pitch_length / 2.0, 0.0),
        }
    else:
        goal_x = 0.0
        pb_x = pb_len  # 16.5
        return {
            "pb_top_goal": (goal_x, pb_y_top),
            "pb_top_outer": (pb_x, pb_y_top),
            "pb_bot_outer": (pb_x, pb_y_bot),
            "pb_bot_goal": (goal_x, pb_y_bot),
            "penalty_spot": (11.0, y_center),
            "halfway_top": (pitch_length / 2.0, pitch_width),
            "halfway_bot": (pitch_length / 2.0, 0.0),
        }


def get_penalty_box_corners(side: str = "right") -> np.ndarray:
    """Return array of 4 corners of penalty box: [top_outer, top_goal, bot_goal, bot_outer]."""
    lm = get_standard_pitch_landmarks(side=side)
    return np.array([
        lm["pb_top_outer"],
        lm["pb_top_goal"],
        lm["pb_bot_goal"],
        lm["pb_bot_outer"],
    ], dtype=np.float32)


class HomographyEstimator:
    """Estimates, validates, and temporally smooths camera-to-pitch homography."""

    def __init__(self, config: Optional[PitchConfig] = None):
        self.config = config or PitchConfig()
        self.current_homography: Optional[np.ndarray] = None
        self.history: deque = deque(maxlen=self.config.homography_smoothing_window)
        self.last_confidence: float = 0.0

    def compute_from_points(
        self,
        image_points: np.ndarray,
        pitch_points: np.ndarray
    ) -> Tuple[Optional[np.ndarray], float]:
        """Compute homography from point correspondences.

        Args:
            image_points: Shape (N, 2) in image pixels.
            pitch_points: Shape (N, 2) in pitch meters.

        Returns:
            Tuple of (3x3 homography matrix or None, confidence score in [0.0, 1.0]).
        """
        if len(image_points) < 4 or len(pitch_points) < 4:
            logger.warning("At least 4 correspondence points required (got %d)", len(image_points))
            return None, 0.0

        img_pts = np.array(image_points, dtype=np.float32).reshape(-1, 1, 2)
        pit_pts = np.array(pitch_points, dtype=np.float32).reshape(-1, 1, 2)

        # Use RANSAC if more than 4 points, else standard least squares
        method = cv2.RANSAC if len(img_pts) > 4 else 0
        h_matrix, mask = cv2.findHomography(img_pts, pit_pts, method, 3.0)

        if h_matrix is None:
            logger.warning("Homography computation failed.")
            return None, 0.0

        # Validate geometric plausibility
        confidence = self.validate_homography(h_matrix, img_pts, pit_pts)
        if confidence < 0.35:
            logger.warning("Computed homography failed sanity checks (confidence: %.2f)", confidence)
            return None, confidence

        # Smooth across recent frames
        smoothed_h = self._smooth_homography(h_matrix)
        self.current_homography = smoothed_h
        self.last_confidence = confidence
        return smoothed_h, confidence

    def validate_homography(
        self,
        h_matrix: np.ndarray,
        src_pts: np.ndarray,
        dst_pts: np.ndarray
    ) -> float:
        """Evaluate geometric quality and conditioning of a homography matrix.

        Returns:
            Confidence value between 0.0 (invalid) and 1.0 (optimal).
        """
        if h_matrix.shape != (3, 3) or np.any(np.isnan(h_matrix)) or np.any(np.isinf(h_matrix)):
            return 0.0

        # Check matrix determinant and conditioning
        det = np.linalg.det(h_matrix)
        if abs(det) < 1e-12:
            return 0.0

        # Compute reprojection error
        projected = cv2.perspectiveTransform(src_pts, h_matrix)
        errors = np.linalg.norm(projected - dst_pts, axis=2).ravel()
        mean_error_m = float(np.mean(errors))

        # Reprojection score: < 0.5m error is exceptional (1.0), > 4.0m is rejected
        if mean_error_m > 4.0:
            return 0.1
        error_score = max(0.0, 1.0 - (mean_error_m / 3.0))

        # Check singular value conditioning
        _, s, _ = np.linalg.svd(h_matrix)
        cond = s[0] / max(s[-1], 1e-9)
        cond_score = 1.0 if cond < 1e5 else max(0.0, 1.0 - np.log10(cond) / 10.0)

        confidence = 0.7 * error_score + 0.3 * cond_score
        return float(np.clip(confidence, 0.0, 1.0))

    def _smooth_homography(self, new_h: np.ndarray) -> np.ndarray:
        """Temporal exponential smoothing over recent homographies."""
        # Normalize scale so h[2, 2] = 1.0
        norm_h = new_h / (new_h[2, 2] if abs(new_h[2, 2]) > 1e-9 else 1.0)
        self.history.append(norm_h)

        if len(self.history) == 1:
            return norm_h

        # Compute weighted average (exponential weighting towards latest)
        weights = [1.5 ** i for i in range(len(self.history))]
        total_w = sum(weights)
        smoothed = np.zeros((3, 3), dtype=np.float64)

        for w, h in zip(weights, self.history):
            smoothed += (w / total_w) * h

        smoothed /= (smoothed[2, 2] if abs(smoothed[2, 2]) > 1e-9 else 1.0)
        return smoothed.astype(np.float32)

    def frame_to_pitch(self, points: np.ndarray) -> np.ndarray:
        """Transform image coordinates (px) to pitch metric plane (m).

        Args:
            points: Array of shape (N, 2) or (2,).

        Returns:
            Projected coordinates in meters (N, 2).
        """
        if self.current_homography is None:
            raise ValueError("Homography has not been calibrated or computed.")

        pts = np.asarray(points, dtype=np.float32)
        single = (pts.ndim == 1)
        if single:
            pts = pts.reshape(1, 1, 2)
        elif pts.ndim == 2:
            pts = pts.reshape(-1, 1, 2)

        transformed = cv2.perspectiveTransform(pts, self.current_homography)
        res = transformed.reshape(-1, 2)
        return res[0] if single else res

    def pitch_to_frame(self, points: np.ndarray) -> np.ndarray:
        """Transform pitch coordinates (m) to image perspective pixels (px).

        Args:
            points: Array of shape (N, 2) or (2,).

        Returns:
            Projected coordinates in pixels (N, 2).
        """
        if self.current_homography is None:
            raise ValueError("Homography has not been calibrated or computed.")

        h_inv = np.linalg.inv(self.current_homography)
        pts = np.asarray(points, dtype=np.float32)
        single = (pts.ndim == 1)
        if single:
            pts = pts.reshape(1, 1, 2)
        elif pts.ndim == 2:
            pts = pts.reshape(-1, 1, 2)

        transformed = cv2.perspectiveTransform(pts, h_inv)
        res = transformed.reshape(-1, 2)
        return res[0] if single else res
