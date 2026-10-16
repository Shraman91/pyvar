"""Pitch line detection using HSV grass masking, Canny edge detection, and Hough transforms."""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import cv2
import numpy as np

from config import PitchConfig

import logging

logger = logging.getLogger("pyvar.pitch_lines")


@dataclass
class LineSegment:
    """Represents a 2D line segment in image coordinates."""
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def length(self) -> float:
        return float(np.hypot(self.x2 - self.x1, self.y2 - self.y1))

    @property
    def angle_deg(self) -> float:
        return float(np.degrees(np.arctan2(self.y2 - self.y1, self.x2 - self.x1)))

    def intersect(self, other: "LineSegment") -> Optional[Tuple[float, float]]:
        """Compute intersection point between two infinite lines containing these segments."""
        # Standard line intersection: A1*x + B1*y = C1
        a1 = self.y2 - self.y1
        b1 = self.x1 - self.x2
        c1 = a1 * self.x1 + b1 * self.y1

        a2 = other.y2 - other.y1
        b2 = other.x1 - other.x2
        c2 = a2 * other.x1 + b2 * other.y1

        det = a1 * b2 - a2 * b1
        if abs(det) < 1e-6:
            return None  # Parallel lines

        x = (c1 * b2 - c2 * b1) / det
        y = (a1 * c2 - a2 * c1) / det
        return float(x), float(y)


class PitchLineDetector:
    """Detects pitch lines and intersection keypoints using computer vision."""

    def __init__(self, config: Optional[PitchConfig] = None):
        self.config = config or PitchConfig()

    def get_grass_mask(self, frame: np.ndarray) -> np.ndarray:
        """Create a binary mask isolating the green grass playing surface."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array(self.config.grass_hsv_lower, dtype=np.uint8)
        upper = np.array(self.config.grass_hsv_upper, dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)

        # Morphological clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask

    def detect_lines(self, frame: np.ndarray) -> List[LineSegment]:
        """Detect white pitch line segments in the frame.

        Args:
            frame: Video frame (BGR).

        Returns:
            List of detected LineSegment objects.
        """
        grass_mask = self.get_grass_mask(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # White lines are bright pixels within the grass region
        # High-pass or threshold white regions on the pitch
        _, white_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        line_candidates = cv2.bitwise_and(white_mask, grass_mask)

        # Canny edge detection
        edges = cv2.Canny(line_candidates, 50, 150, apertureSize=3)

        # Probabilistic Hough transform
        raw_lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=40,
            minLineLength=50,
            maxLineGap=20
        )

        segments: List[LineSegment] = []
        if raw_lines is None:
            return segments

        for line in raw_lines:
            x1, y1, x2, y2 = line[0]
            segments.append(LineSegment(float(x1), float(y1), float(x2), float(y2)))

        return segments

    def find_line_intersections(
        self,
        lines: List[LineSegment],
        frame_shape: Tuple[int, int]
    ) -> List[Tuple[float, float]]:
        """Find intersections between non-parallel line segments within frame bounds."""
        h, w = frame_shape[:2]
        intersections = []

        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                l1, l2 = lines[i], lines[j]
                # Filter lines that have at least 25 degrees difference
                angle_diff = abs(l1.angle_deg - l2.angle_deg) % 180
                if 25.0 < angle_diff < 155.0:
                    pt = l1.intersect(l2)
                    if pt is not None:
                        x, y = pt
                        if 0 <= x < w and 0 <= y < h:
                            intersections.append((x, y))

        return intersections
