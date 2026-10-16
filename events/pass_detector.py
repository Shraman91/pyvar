"""Ball pass and kick detection module for pyvar.

Identifies the exact moment the ball is played based on sudden velocity
and direction changes while within proximity of an attacking player.
"""

from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from config import EventConfig
from detection.detector import DetectedObject, FrameDetections

import logging

logger = logging.getLogger("pyvar.pass_detector")


@dataclass
class PassEvent:
    """Represents a detected pass release moment."""
    frame_index: int
    timestamp_ms: float
    passer_id: Optional[int]
    ball_position: Tuple[float, float]
    velocity_before: Tuple[float, float]
    velocity_after: Tuple[float, float]
    accel_magnitude: float
    is_manual: bool = False


class PassDetector:
    """Detects ball release moments automatically from ball kinematics and player proximity."""

    def __init__(self, config: Optional[EventConfig] = None):
        self.config = config or EventConfig()
        self.ball_positions: deque = deque(maxlen=20)  # (frame_idx, (x, y))
        self.detected_passes: List[PassEvent] = []
        self.last_pass_frame: int = -999
        self.manual_pass_frame: Optional[int] = None

    def set_manual_pass_frame(self, frame_index: int, passer_id: Optional[int] = None) -> None:
        """Manually flag a frame as the definitive pass moment."""
        self.manual_pass_frame = frame_index
        logger.info("Manual pass moment set at frame %d", frame_index)

    def update(
        self,
        frame_detections: FrameDetections,
        homography: Optional[np.ndarray] = None
    ) -> Optional[PassEvent]:
        """Process current frame detections and check for a pass release.

        Args:
            frame_detections: Detections in current frame.
            homography: Optional homography to measure physical distances in meters.

        Returns:
            PassEvent if a pass was initiated in this frame, else None.
        """
        f_idx = frame_detections.frame_index
        ball = frame_detections.ball

        # Check manual pass override
        if self.manual_pass_frame is not None and f_idx == self.manual_pass_frame:
            ball_pos = ball.center if ball is not None else (0.0, 0.0)
            closest_p = self._find_closest_player(ball_pos, frame_detections.players)
            passer_id = closest_p.track_id if closest_p is not None else None

            event = PassEvent(
                frame_index=f_idx,
                timestamp_ms=frame_detections.timestamp_ms,
                passer_id=passer_id,
                ball_position=ball_pos,
                velocity_before=(0.0, 0.0),
                velocity_after=(0.0, 0.0),
                accel_magnitude=10.0,
                is_manual=True
            )
            self.detected_passes.append(event)
            return event

        if ball is None:
            return None

        bx, by = ball.center
        self.ball_positions.append((f_idx, (bx, by)))

        if len(self.ball_positions) < 3:
            return None

        # Cooldown check
        if f_idx - self.last_pass_frame < self.config.pass_cooldown_frames:
            return None

        # Compute velocities: v1 (t-2 to t-1), v2 (t-1 to t)
        p0 = self.ball_positions[-3][1]
        p1 = self.ball_positions[-2][1]
        p2 = self.ball_positions[-1][1]

        v1 = (p1[0] - p0[0], p1[1] - p0[1])
        v2 = (p2[0] - p1[0], p2[1] - p1[1])

        # Acceleration vector: delta_v = v2 - v1
        dv = (v2[0] - v1[0], v2[1] - v1[1])
        accel_mag = float(np.hypot(dv[0], dv[1]))

        # Check if acceleration exceeds kick threshold
        if accel_mag >= self.config.velocity_change_threshold:
            # Check player proximity to confirm ball was kicked by a player
            closest_player = self._find_closest_player((bx, by), frame_detections.players)
            if closest_player is not None:
                dist = np.hypot(
                    bx - closest_player.bottom_center[0],
                    by - closest_player.bottom_center[1]
                )

                if dist <= self.config.ball_proximity_radius_px:
                    event = PassEvent(
                        frame_index=f_idx,
                        timestamp_ms=frame_detections.timestamp_ms,
                        passer_id=closest_player.track_id,
                        ball_position=(bx, by),
                        velocity_before=v1,
                        velocity_after=v2,
                        accel_magnitude=accel_mag,
                        is_manual=False
                    )
                    self.detected_passes.append(event)
                    self.last_pass_frame = f_idx
                    logger.info("Pass event detected at frame %d by player #%s (accel: %.1f)",
                                f_idx, closest_player.track_id, accel_mag)
                    return event

        return None

    def _find_closest_player(
        self,
        point: Tuple[float, float],
        players: List[DetectedObject]
    ) -> Optional[DetectedObject]:
        """Find player closest to the given point."""
        if not players:
            return None

        px, py = point
        best_player = None
        min_dist = float("inf")

        for p in players:
            foot_x, foot_y = p.bottom_center
            dist = float(np.hypot(px - foot_x, py - foot_y))
            if dist < min_dist:
                min_dist = dist
                best_player = p

        return best_player
