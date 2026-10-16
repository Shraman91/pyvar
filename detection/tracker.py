"""Player tracking module using ByteTrack from supervision.

Assigns persistent tracking IDs across video frames and tracks trajectories.
"""

from collections import deque
from typing import Dict, List, Optional, Tuple
import logging
import numpy as np

from config import TrackingConfig
from .detector import DetectedObject, FrameDetections

logger = logging.getLogger("pyvar.tracker")


class PlayerTracker:
    """ByteTrack wrapper for robust multi-player tracking and trajectory maintenance."""

    def __init__(self, config: Optional[TrackingConfig] = None):
        self.config = config or TrackingConfig()
        self._tracker = None
        self._initialized = False
        
        # History of foot positions (or centers) per player: track_id -> deque of (frame_idx, (x, y))
        self.track_histories: Dict[int, deque] = {}
        # Ball history: deque of (frame_idx, (x, y))
        self.ball_history: deque = deque(maxlen=self.config.track_history_len)

    def _ensure_tracker(self) -> None:
        """Lazy load supervision ByteTrack."""
        if not self._initialized:
            import supervision as sv

            logger.info("Initializing ByteTrack tracker...")
            self._tracker = sv.ByteTrack(
                track_activation_threshold=self.config.track_activation_threshold,
                lost_track_buffer=self.config.lost_track_buffer,
                minimum_matching_threshold=self.config.minimum_matching_threshold,
                frame_rate=self.config.frame_rate
            )
            self._initialized = True

    def update(self, frame_detections: FrameDetections) -> FrameDetections:
        """Update tracker with detections from the current frame.

        Args:
            frame_detections: Detected players and ball in the current frame.

        Returns:
            Updated FrameDetections with assigned persistent `track_id`s.
        """
        self._ensure_tracker()
        import supervision as sv

        players = frame_detections.players
        if not players:
            # Let tracker know no detections occurred
            empty_sv = sv.Detections.empty()
            self._tracker.update_with_detections(empty_sv)
            return frame_detections

        # Prepare bounding boxes for supervision
        xyxy = np.array([p.bbox for p in players], dtype=np.float32)
        confidence = np.array([p.confidence for p in players], dtype=np.float32)
        class_id = np.array([p.class_id for p in players], dtype=int)

        sv_detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id
        )

        # Update ByteTrack
        tracked = self._tracker.update_with_detections(sv_detections)

        # Associate tracked IDs back to players by matching bounding box overlap / IoU or nearest center
        tracked_xyxy = tracked.xyxy if len(tracked) > 0 else np.empty((0, 4))
        tracker_ids = tracked.tracker_id if (len(tracked) > 0 and tracked.tracker_id is not None) else []

        if len(tracked_xyxy) > 0:
            for player in players:
                p_box = np.array(player.bbox)
                # Compute IoUs with tracked boxes
                best_iou = -1.0
                best_id = None
                for t_idx, t_box in enumerate(tracked_xyxy):
                    iou = self._calculate_iou(p_box, t_box)
                    if iou > best_iou and iou > 0.4:
                        best_iou = iou
                        best_id = int(tracker_ids[t_idx])

                if best_id is not None:
                    player.track_id = best_id
                    if best_id not in self.track_histories:
                        self.track_histories[best_id] = deque(maxlen=self.config.track_history_len)
                    self.track_histories[best_id].append((frame_detections.frame_index, player.bottom_center))

        # Update ball history
        if frame_detections.ball is not None:
            self.ball_history.append((frame_detections.frame_index, frame_detections.ball.center))

        return frame_detections

    def get_trajectory(self, track_id: int) -> List[Tuple[float, float]]:
        """Retrieve recent coordinates for a given track ID."""
        if track_id not in self.track_histories:
            return []
        return [pt for _, pt in self.track_histories[track_id]]

    def get_ball_trajectory(self) -> List[Tuple[float, float]]:
        """Retrieve recent coordinates for the ball."""
        return [pt for _, pt in self.ball_history]

    @staticmethod
    def _calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute Intersection over Union (IoU) of two boxes [x1, y1, x2, y2]."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
        area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
        union = area1 + area2 - intersection

        if union <= 0.0:
            return 0.0
        return intersection / union
