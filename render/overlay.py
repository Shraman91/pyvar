"""Broadcast-style overlay and visualization engine for pyvar.

Renders player boxes, track IDs, team markers, ball indicators,
trajectory trails, top-down minimaps, and broadcast decision graphics.
"""

from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from config import RenderConfig, PitchConfig
from detection.detector import DetectedObject, FrameDetections

# Team color mapping defaults
DEFAULT_COLORS = {
    "team_a": (60, 60, 240),      # Bright Red (BGR)
    "team_b": (240, 150, 40),     # Azure Blue (BGR)
    "referee": (40, 220, 40),     # Bright Green (BGR)
    "goalkeeper": (30, 220, 240), # Bright Yellow/Gold (BGR)
    "unclassified": (180, 180, 180), # Gray (BGR)
}


class VAROverlayRenderer:
    """Renders professional TV broadcast graphics and VAR analysis overlays onto video frames."""

    def __init__(
        self,
        render_config: Optional[RenderConfig] = None,
        pitch_config: Optional[PitchConfig] = None,
    ):
        self.config = render_config or RenderConfig()
        self.pitch_config = pitch_config or PitchConfig()

    def draw_frame(
        self,
        frame: np.ndarray,
        detections: FrameDetections,
        trajectories: Optional[Dict[int, List[Tuple[float, float]]]] = None,
        ball_trajectory: Optional[List[Tuple[float, float]]] = None,
        offside_result: Optional[dict] = None,
        homography: Optional[np.ndarray] = None,
        status_text: Optional[str] = None,
        show_minimap: bool = True,
    ) -> np.ndarray:
        """Render all overlays on a single video frame.

        Args:
            frame: Input BGR image.
            detections: FrameDetections containing players and ball.
            trajectories: Dict of track_id to list of (x, y) coordinates.
            ball_trajectory: List of recent (x, y) ball coordinates.
            offside_result: Optional structured offside evaluation dict.
            homography: Optional 3x3 homography matrix (frame -> pitch).
            status_text: Optional status string to display in header.
            show_minimap: Whether to render the top-down 2D minimap.

        Returns:
            Annotated BGR frame with broadcast graphics.
        """
        output = frame.copy()
        h, w = output.shape[:2]

        # 1. Draw motion trajectories
        if self.config.show_tracks and trajectories:
            self._draw_trajectories(output, trajectories)

        if self.config.show_tracks and ball_trajectory:
            self._draw_ball_trajectory(output, ball_trajectory)

        # 2. Draw offside perspective line if available
        if offside_result and homography is not None:
            self._draw_perspective_offside_line(output, offside_result, homography)

        # 3. Draw player bounding boxes & indicators
        if self.config.show_boxes:
            for player in detections.players:
                self._draw_player(output, player)

        # 4. Draw ball indicator
        if detections.ball is not None:
            self._draw_ball(output, detections.ball)

        # 5. Draw top-down minimap
        if show_minimap and self.config.show_minimap and homography is not None:
            self._draw_minimap(output, detections, offside_result, homography)

        # 6. Draw broadcast header banner
        self._draw_header(output, detections.frame_index, detections.timestamp_ms, status_text)

        # 7. Draw decision graphic if offside result is present
        if offside_result:
            self._draw_decision_graphic(output, offside_result)

        return output

    def _draw_player(self, frame: np.ndarray, player: DetectedObject) -> None:
        """Draw sleek broadcast box, team marker, and track ID for a player."""
        x1, y1, x2, y2 = map(int, player.bbox)
        team_id = player.team_id or "unclassified"
        color = DEFAULT_COLORS.get(team_id, (180, 180, 180))

        # Modern corner brackets instead of heavy solid box
        line_len = max(8, int(min(player.width, player.height) * 0.2))
        thickness = 2

        # Top-left
        cv2.line(frame, (x1, y1), (x1 + line_len, y1), color, thickness)
        cv2.line(frame, (x1, y1), (x1, y1 + line_len), color, thickness)
        # Top-right
        cv2.line(frame, (x2, y1), (x2 - line_len, y1), color, thickness)
        cv2.line(frame, (x2, y1), (x2, y1 + line_len), color, thickness)
        # Bottom-left
        cv2.line(frame, (x1, y2), (x1 + line_len, y2), color, thickness)
        cv2.line(frame, (x1, y2), (x1, y2 - line_len), color, thickness)
        # Bottom-right
        cv2.line(frame, (x2, y2), (x2 - line_len, y2), color, thickness)
        cv2.line(frame, (x2, y2), (x2, y2 - line_len), color, thickness)

        # Draw foot contact ellipse
        foot_x, foot_y = int(player.bottom_center[0]), int(player.bottom_center[1])
        axes = (max(6, int(player.width * 0.3)), max(3, int(player.width * 0.12)))
        cv2.ellipse(frame, (foot_x, foot_y), axes, 0, 0, 360, color, 2, cv2.LINE_AA)

        # Player ID & Team tag label pill
        label = ""
        if player.track_id is not None:
            label += f"#{player.track_id}"
        if player.team_id:
            tag = "A" if player.team_id == "team_a" else ("B" if player.team_id == "team_b" else player.team_id[:3].upper())
            label = f"{tag} {label}".strip()

        if not label:
            label = f"{player.confidence:.2f}"

        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.42
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, 1)

        pill_y1 = max(0, y1 - th - 8)
        pill_y2 = pill_y1 + th + 6
        pill_x1 = max(0, x1)
        pill_x2 = pill_x1 + tw + 10

        # Background pill
        cv2.rectangle(frame, (pill_x1, pill_y1), (pill_x2, pill_y2), (20, 20, 20), -1)
        cv2.rectangle(frame, (pill_x1, pill_y1), (pill_x2, pill_y2), color, 1)
        # Text
        cv2.putText(
            frame,
            label,
            (pill_x1 + 5, pill_y2 - 4),
            font,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    def _draw_ball(self, frame: np.ndarray, ball: DetectedObject) -> None:
        """Draw illuminated ball indicator circle with reticle."""
        cx, cy = map(int, ball.center)
        radius = max(8, int(max(ball.width, ball.height) / 1.5))
        color = self.config.ball_color

        # Outer glowing ring
        cv2.circle(frame, (cx, cy), radius + 4, (0, 140, 255), 1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), radius, color, 2, cv2.LINE_AA)
        # Reticle tick marks
        cv2.line(frame, (cx - radius - 5, cy), (cx - radius + 2, cy), color, 2)
        cv2.line(frame, (cx + radius - 2, cy), (cx + radius + 5, cy), color, 2)
        cv2.line(frame, (cx, cy - radius - 5), (cx, cy - radius + 2), color, 2)
        cv2.line(frame, (cx, cy + radius - 2), (cx, cy + radius + 5), color, 2)

        # "BALL" tag
        cv2.putText(
            frame,
            "BALL",
            (cx - 15, cy - radius - 6),
            cv2.FONT_HERSHEY_DUPLEX,
            0.35,
            (0, 255, 255),
            1,
            cv2.LINE_AA
        )

    def _draw_trajectories(
        self,
        frame: np.ndarray,
        trajectories: Dict[int, List[Tuple[float, float]]]
    ) -> None:
        """Draw motion trail for each tracked player."""
        for track_id, pts in trajectories.items():
            if len(pts) < 2:
                continue
            for i in range(1, len(pts)):
                alpha = i / len(pts)
                thickness = 1 if alpha < 0.5 else 2
                pt1 = (int(pts[i - 1][0]), int(pts[i - 1][1]))
                pt2 = (int(pts[i][0]), int(pts[i][1]))
                cv2.line(frame, pt1, pt2, (180, 180, 180), thickness, cv2.LINE_AA)

    def _draw_ball_trajectory(
        self,
        frame: np.ndarray,
        ball_trajectory: List[Tuple[float, float]]
    ) -> None:
        """Draw ball movement trail."""
        if len(ball_trajectory) < 2:
            continue_pts = ball_trajectory
        else:
            continue_pts = ball_trajectory
            for i in range(1, len(continue_pts)):
                pt1 = (int(continue_pts[i - 1][0]), int(continue_pts[i - 1][1]))
                pt2 = (int(continue_pts[i][0]), int(continue_pts[i][1]))
                cv2.line(frame, pt1, pt2, (0, 200, 255), 2, cv2.LINE_AA)

    def _draw_header(
        self,
        frame: np.ndarray,
        frame_index: int,
        timestamp_ms: float,
        status_text: Optional[str] = None
    ) -> None:
        """Render top VAR status bar."""
        h, w = frame.shape[:2]
        banner_h = 36

        # Semi-transparent dark banner
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (15, 15, 18), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Brand badge: PYVAR
        cv2.rectangle(frame, (10, 6), (90, 30), (0, 102, 204), -1)
        cv2.putText(
            frame,
            "PYVAR",
            (16, 24),
            cv2.FONT_HERSHEY_DUPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        # Title & Frame Info
        sec = timestamp_ms / 1000.0
        time_str = f"{int(sec // 60):02d}:{sec % 60:05.2f}"
        info_str = f"FRAME: {frame_index:05d}  |  TIME: {time_str}"
        cv2.putText(
            frame,
            info_str,
            (105, 23),
            cv2.FONT_HERSHEY_DUPLEX,
            0.45,
            (200, 200, 200),
            1,
            cv2.LINE_AA
        )

        # Status text on the right
        right_text = status_text or "LIVE TRACKING"
        (tw, _), _ = cv2.getTextSize(right_text, cv2.FONT_HERSHEY_DUPLEX, 0.45, 1)
        cv2.putText(
            frame,
            right_text,
            (w - tw - 15, 23),
            cv2.FONT_HERSHEY_DUPLEX,
            0.45,
            (0, 215, 255),
            1,
            cv2.LINE_AA
        )

    def _draw_decision_graphic(self, frame: np.ndarray, offside_result: dict) -> None:
        """Render the VAR Offside verdict graphic."""
        h, w = frame.shape[:2]
        is_offside = offside_result.get("is_offside", False)
        margin_m = offside_result.get("margin_m", 0.0)
        confidence = offside_result.get("confidence", 0.95)
        attacker_id = offside_result.get("attacker_id", "N/A")
        defender_id = offside_result.get("defender_id", "N/A")

        verdict_str = "OFFSIDE" if is_offside else "ONSIDE"
        box_color = (30, 30, 220) if is_offside else (30, 180, 50)  # Red for offside, Green for onside

        # Center card at bottom
        card_w, card_h = 460, 85
        cx = w // 2
        card_x1 = cx - card_w // 2
        card_x2 = cx + card_w // 2
        card_y2 = h - 15
        card_y1 = card_y2 - card_h

        # Overlay card background
        overlay = frame.copy()
        cv2.rectangle(overlay, (card_x1, card_y1), (card_x2, card_y2), (18, 18, 22), -1)
        cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)

        # Border
        cv2.rectangle(frame, (card_x1, card_y1), (card_x2, card_y2), box_color, 2)

        # Verdict badge
        badge_w = 120
        cv2.rectangle(frame, (card_x1, card_y1), (card_x1 + badge_w, card_y2), box_color, -1)
        cv2.putText(
            frame,
            verdict_str,
            (card_x1 + 10, card_y1 + 50),
            cv2.FONT_HERSHEY_DUPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # Details
        margin_cm = abs(margin_m * 100.0)
        margin_str = f"Margin: {margin_m:+.2f} m ({margin_cm:.1f} cm)"
        players_str = f"Attacker: #{attacker_id}  |  2nd Last Defender: #{defender_id}"
        disclaimer = "Assistive VAR Graphic | Margin of error: +/-5.0cm (perspective limit)"

        tx = card_x1 + badge_w + 15
        cv2.putText(frame, margin_str, (tx, card_y1 + 25), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, players_str, (tx, card_y1 + 50), cv2.FONT_HERSHEY_DUPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, disclaimer, (tx, card_y1 + 72), cv2.FONT_HERSHEY_DUPLEX, 0.32, (140, 140, 140), 1, cv2.LINE_AA)

    def _draw_perspective_offside_line(
        self,
        frame: np.ndarray,
        offside_result: dict,
        homography: np.ndarray
    ) -> None:
        """Map the offside pitch-plane line into image perspective coordinates using H_inv."""
        try:
            h_inv = np.linalg.inv(homography)
        except np.linalg.LinAlgError:
            return

        offside_x = offside_result.get("offside_line_x_pitch")
        if offside_x is None:
            return

        # Line runs across full pitch width from y=0 to y=pitch_width (in meters)
        pitch_y_start = 0.0
        pitch_y_end = self.pitch_config.pitch_width

        # Sample points along the pitch line for smooth curve/perspective mapping
        num_pts = 30
        ys = np.linspace(pitch_y_start, pitch_y_end, num_pts)
        pts_pitch = np.zeros((num_pts, 1, 2), dtype=np.float32)
        pts_pitch[:, 0, 0] = offside_x
        pts_pitch[:, 0, 1] = ys

        # Project back to image frame: [u, v, 1]^T = H_inv * [X, Y, 1]^T
        pts_image = cv2.perspectiveTransform(pts_pitch, h_inv)

        is_offside = offside_result.get("is_offside", False)
        color = self.config.offside_line_color if is_offside else self.config.onside_line_color

        img_pts = []
        for pt in pts_image:
            u, v = pt[0]
            if not (np.isnan(u) or np.isnan(v)):
                img_pts.append((int(round(u)), int(round(v))))

        if len(img_pts) >= 2:
            for i in range(1, len(img_pts)):
                cv2.line(frame, img_pts[i - 1], img_pts[i], color, 3, cv2.LINE_AA)

    def _draw_minimap(
        self,
        frame: np.ndarray,
        detections: FrameDetections,
        offside_result: Optional[dict],
        homography: np.ndarray
    ) -> None:
        """Render a top-down 2D tactical pitch minimap panel."""
        mw, mh = self.config.minimap_width, self.config.minimap_height
        h_frame, w_frame = frame.shape[:2]

        # Minimap canvas: dark green pitch
        minimap = np.zeros((mh, mw, 3), dtype=np.uint8)
        minimap[:] = (28, 75, 34)  # Pitch grass green

        # Pitch scale factors (pitch_length x pitch_width in meters)
        margin = 14
        pw = mw - 2 * margin
        ph = mh - 2 * margin
        scale_x = pw / self.pitch_config.pitch_length
        scale_y = ph / self.pitch_config.pitch_width

        def pitch_to_map(px: float, py: float) -> Tuple[int, int]:
            mx = int(round(margin + px * scale_x))
            my = int(round(margin + py * scale_y))
            return max(0, min(mw - 1, mx)), max(0, min(mh - 1, my))

        # Draw pitch markings
        # Pitch border
        p1 = pitch_to_map(0, 0)
        p2 = pitch_to_map(self.pitch_config.pitch_length, self.pitch_config.pitch_width)
        cv2.rectangle(minimap, p1, p2, (240, 240, 240), 1)

        # Halfway line
        mid_x = self.pitch_config.pitch_length / 2.0
        cv2.line(minimap, pitch_to_map(mid_x, 0), pitch_to_map(mid_x, self.pitch_config.pitch_width), (240, 240, 240), 1)

        # Center circle
        center_pt = pitch_to_map(mid_x, self.pitch_config.pitch_width / 2.0)
        radius_px = int(round(9.15 * scale_x))
        cv2.circle(minimap, center_pt, radius_px, (240, 240, 240), 1)

        # Penalty boxes
        pb_len = self.pitch_config.penalty_box_length
        pb_w = self.pitch_config.penalty_box_width
        py_start = (self.pitch_config.pitch_width - pb_w) / 2.0
        py_end = py_start + pb_w
        # Left penalty box
        cv2.rectangle(minimap, pitch_to_map(0, py_start), pitch_to_map(pb_len, py_end), (240, 240, 240), 1)
        # Right penalty box
        cv2.rectangle(minimap, pitch_to_map(self.pitch_config.pitch_length - pb_len, py_start), pitch_to_map(self.pitch_config.pitch_length, py_end), (240, 240, 240), 1)

        # Project detected players into pitch plane via homography
        player_points = []
        for p in detections.players:
            player_points.append(p.bottom_center)

        if player_points:
            src = np.array(player_points, dtype=np.float32).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(src, homography)
            for idx, p in enumerate(detections.players):
                px, py = dst[idx, 0]
                if 0 <= px <= self.pitch_config.pitch_length and 0 <= py <= self.pitch_config.pitch_width:
                    map_pt = pitch_to_map(px, py)
                    color = DEFAULT_COLORS.get(p.team_id or "unclassified", (200, 200, 200))
                    cv2.circle(minimap, map_pt, 4, color, -1)
                    cv2.circle(minimap, map_pt, 4, (0, 0, 0), 1)

        # Project ball into pitch plane
        if detections.ball is not None:
            ball_pt = np.array([[detections.ball.center]], dtype=np.float32)
            dst_ball = cv2.perspectiveTransform(ball_pt, homography)
            bx, by = dst_ball[0, 0]
            if 0 <= bx <= self.pitch_config.pitch_length and 0 <= by <= self.pitch_config.pitch_width:
                map_bpt = pitch_to_map(bx, by)
                cv2.circle(minimap, map_bpt, 3, (0, 255, 255), -1)
                cv2.circle(minimap, map_bpt, 4, (0, 0, 0), 1)

        # Draw offside line on minimap
        if offside_result and offside_result.get("offside_line_x_pitch") is not None:
            ox = offside_result["offside_line_x_pitch"]
            is_off = offside_result.get("is_offside", False)
            line_col = self.config.offside_line_color if is_off else self.config.onside_line_color
            cv2.line(minimap, pitch_to_map(ox, 0), pitch_to_map(ox, self.pitch_config.pitch_width), line_col, 2)

        # Place minimap in top-right or bottom-right corner
        pos_x = w_frame - mw - 15
        pos_y = 48  # Just under the top header banner
        
        # Border for minimap
        cv2.rectangle(frame, (pos_x - 2, pos_y - 2), (pos_x + mw + 2, pos_y + mh + 2), (20, 20, 25), -1)
        frame[pos_y:pos_y + mh, pos_x:pos_x + mw] = minimap
        cv2.rectangle(frame, (pos_x, pos_y), (pos_x + mw, pos_y + mh), (100, 100, 100), 1)
        cv2.putText(
            frame,
            "TACTICAL MINIMAP (TOP-DOWN)",
            (pos_x + 6, pos_y + 12),
            cv2.FONT_HERSHEY_DUPLEX,
            0.32,
            (220, 220, 220),
            1,
            cv2.LINE_AA
        )
