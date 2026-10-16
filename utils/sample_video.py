"""Sample match video generator for pyvar testing and demonstrations.

Generates a synthetic 720p football video with realistic pitch perspective,
moving players wearing distinct team jerseys (Red vs. Blue), a moving ball,
and a calibrated pass event.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np

# Synthetic pitch coordinates (pitch plane: 105m x 68m)
# We define a known ground-truth 3x3 homography matrix from camera frame to pitch plane
# to allow automated verification of calibration.


def get_ground_truth_homography(width: int = 1280, height: int = 720) -> np.ndarray:
    """Return ground truth homography mapping image (px) -> pitch (m)."""
    # 4 image points (perspective view of pitch near right penalty box)
    # Pitch bounds visible: x in [50, 105], y in [10, 58]
    img_pts = np.array([
        [180, 240],    # Top-left (midfield touchline area)
        [1100, 240],   # Top-right (goal line top corner)
        [1240, 680],   # Bottom-right (goal line bottom corner)
        [40, 680],     # Bottom-left (midfield near camera)
    ], dtype=np.float32)

    # Corresponding pitch coordinates in meters
    pitch_pts = np.array([
        [52.5, 60.0],  # Halfway touchline
        [105.0, 60.0], # Goal line touchline
        [105.0, 8.0],  # Goal line near touchline
        [52.5, 8.0],   # Halfway line near camera
    ], dtype=np.float32)

    h_mat, _ = cv2.findHomography(img_pts, pitch_pts)
    return h_mat


def generate_sample_match_video(
    output_path: str = "data/sample_match.mp4",
    num_frames: int = 90,
    fps: int = 30,
    width: int = 1280,
    height: int = 720,
    pass_frame: int = 40
) -> Path:
    """Generate a realistic football clip for testing VAR pipeline.

    Args:
        output_path: Destination path for the MP4 video.
        num_frames: Total number of frames (e.g. 90 = 3 seconds at 30fps).
        fps: Frame rate.
        width: Video width.
        height: Video height.
        pass_frame: Frame number at which the pass occurs.

    Returns:
        Path to the generated video file.
    """
    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    h_img_to_pitch = get_ground_truth_homography(width, height)
    h_pitch_to_img = np.linalg.inv(h_img_to_pitch)

    def pitch_to_img(px: float, py: float) -> Tuple[int, int]:
        p = np.array([[[px, py]]], dtype=np.float32)
        dst = cv2.perspectiveTransform(p, h_pitch_to_img)
        return int(round(dst[0, 0, 0])), int(round(dst[0, 0, 1]))

    # Define players with starting positions on pitch (meters)
    # Team A (Attacking towards right, Red jerseys):
    # - Passer (A1): (70, 34) moves slightly forward, kicks ball at pass_frame
    # - Receiver (A2): (88, 42) running forward
    # - Attacker (A3): (80, 20)
    # Team B (Defending right goal, Blue jerseys):
    # - Goalkeeper (GK): (103, 34)
    # - Last Defender (B1): (87, 30) (Offside line reference!)
    # - Defender 2 (B2): (85, 45)
    # - Midfielder (B3): (68, 32)
    # Referee (Ref): (75, 48) in lime green

    for f in range(num_frames):
        # Base pitch frame (green grass with mowing stripes)
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (35, 115, 45)  # Pitch grass green (BGR)

        # Draw grass lawn stripes in perspective
        for stripe_x in range(50, 105, 5):
            pt_top = pitch_to_img(float(stripe_x), 68.0)
            pt_bot = pitch_to_img(float(stripe_x), 0.0)
            pt_top_next = pitch_to_img(float(stripe_x + 5), 68.0)
            pt_bot_next = pitch_to_img(float(stripe_x + 5), 0.0)
            if (stripe_x // 5) % 2 == 0:
                stripe_poly = np.array([pt_top, pt_top_next, pt_bot_next, pt_bot], dtype=np.int32)
                cv2.fillPoly(frame, [stripe_poly], (40, 128, 52))

        # Draw pitch white lines in perspective
        white = (245, 245, 245)
        # Touchlines & Goal Line
        cv2.line(frame, pitch_to_img(52.5, 68.0), pitch_to_img(105.0, 68.0), white, 2, cv2.LINE_AA)
        cv2.line(frame, pitch_to_img(52.5, 0.0), pitch_to_img(105.0, 0.0), white, 2, cv2.LINE_AA)
        cv2.line(frame, pitch_to_img(105.0, 0.0), pitch_to_img(105.0, 68.0), white, 3, cv2.LINE_AA)
        cv2.line(frame, pitch_to_img(52.5, 0.0), pitch_to_img(52.5, 68.0), white, 2, cv2.LINE_AA)

        # Penalty box: 16.5m from goal line, width 40.32m centered around 34m (13.84 to 54.16)
        pb_x = 105.0 - 16.5
        pb_y1 = 34.0 - 20.16
        pb_y2 = 34.0 + 20.16
        cv2.line(frame, pitch_to_img(pb_x, pb_y1), pitch_to_img(105.0, pb_y1), white, 2, cv2.LINE_AA)
        cv2.line(frame, pitch_to_img(pb_x, pb_y2), pitch_to_img(105.0, pb_y2), white, 2, cv2.LINE_AA)
        cv2.line(frame, pitch_to_img(pb_x, pb_y1), pitch_to_img(pb_x, pb_y2), white, 2, cv2.LINE_AA)

        # Penalty spot (11m from goal line, center y=34)
        pen_spot = pitch_to_img(105.0 - 11.0, 34.0)
        cv2.circle(frame, pen_spot, 3, white, -1, cv2.LINE_AA)

        # Center circle arc at halfway line (radius 9.15m)
        circle_pts = []
        for deg in range(-90, 91, 5):
            rad = np.radians(deg)
            cx = 52.5 + 9.15 * np.cos(rad)
            cy = 34.0 + 9.15 * np.sin(rad)
            circle_pts.append(pitch_to_img(cx, cy))
        for i in range(1, len(circle_pts)):
            cv2.line(frame, circle_pts[i - 1], circle_pts[i], white, 2, cv2.LINE_AA)

        # Player dynamics over time
        t = f / fps
        # A1 Passer
        a1_x = 72.0 + t * 1.5
        a1_y = 33.0 + t * 0.2
        # A2 Receiver (sprinting forward towards offside position!)
        a2_x = 84.0 + t * 2.8
        a2_y = 42.0 + t * 0.4
        # A3
        a3_x = 78.0 + t * 1.0
        a3_y = 20.0
        # B1 Defender (holding offside line, moving back slightly)
        b1_x = 88.0 + t * 0.5
        b1_y = 30.0
        # B2 Defender
        b2_x = 87.0 + t * 0.6
        b2_y = 46.0
        # B3 Midfielder
        b3_x = 69.0 + t * 1.2
        b3_y = 35.0
        # Goalkeeper
        gk_x = 103.5 + np.sin(t * 2) * 0.4
        gk_y = 34.0
        # Referee
        ref_x = 76.0 + t * 1.0
        ref_y = 50.0

        # Ball trajectory: with A1 before pass_frame; travelling towards A2 after pass_frame
        if f < pass_frame:
            ball_x = a1_x + 0.4
            ball_y = a1_y + 0.2
        else:
            progress = (f - pass_frame) / max(1, (num_frames - pass_frame))
            target_x = 84.0 + (num_frames / fps) * 2.8
            target_y = 42.0 + (num_frames / fps) * 0.4
            ball_x = (a1_x + 0.4) + progress * (target_x - (a1_x + 0.4))
            ball_y = (a1_y + 0.2) + progress * (target_y - (a1_y + 0.2))

        # Helper to draw a synthetic player figure
        def draw_player_figure(
            px: float,
            py: float,
            jersey_color: Tuple[int, int, int],
            shorts_color: Tuple[int, int, int] = (240, 240, 240)
        ):
            foot_x, foot_y = pitch_to_img(px, py)
            # Perspective height scaling: players further from camera appear smaller
            height_scale = max(0.4, min(1.2, foot_y / (height * 0.85)))
            p_height = int(round(52 * height_scale))
            p_width = int(round(22 * height_scale))

            top_y = foot_y - p_height
            center_x = foot_x

            # Shadow on grass
            cv2.ellipse(frame, (center_x, foot_y), (p_width // 2, max(2, p_width // 4)), 0, 0, 360, (20, 50, 20), -1)

            # Legs / Socks
            leg_h = int(p_height * 0.35)
            cv2.line(frame, (center_x - p_width // 4, foot_y), (center_x - p_width // 4, foot_y - leg_h), (200, 200, 200), 2)
            cv2.line(frame, (center_x + p_width // 4, foot_y), (center_x + p_width // 4, foot_y - leg_h), (200, 200, 200), 2)

            # Shorts
            shorts_h = int(p_height * 0.2)
            cv2.rectangle(
                frame,
                (center_x - p_width // 2, foot_y - leg_h - shorts_h),
                (center_x + p_width // 2, foot_y - leg_h),
                shorts_color,
                -1
            )

            # Torso (Jersey)
            torso_h = int(p_height * 0.3)
            cv2.rectangle(
                frame,
                (center_x - p_width // 2, foot_y - leg_h - shorts_h - torso_h),
                (center_x + p_width // 2, foot_y - leg_h - shorts_h),
                jersey_color,
                -1
            )

            # Head
            head_r = max(3, int(p_height * 0.12))
            head_y = foot_y - leg_h - shorts_h - torso_h - head_r
            cv2.circle(frame, (center_x, head_y), head_r, (200, 170, 140), -1)

        # Team A colors: Red (BGR: 45, 45, 220)
        team_a_col = (45, 45, 220)
        # Team B colors: Blue (BGR: 220, 120, 35)
        team_b_col = (220, 120, 35)
        # GK color: Yellow/Gold (BGR: 20, 215, 235)
        gk_col = (20, 215, 235)
        # Ref color: Bright Lime (BGR: 30, 210, 45)
        ref_col = (30, 210, 45)

        # Draw players sorted by y position for natural depth ordering
        players = [
            (a3_x, a3_y, team_a_col),
            (b1_x, b1_y, team_b_col),
            (b3_x, b3_y, team_b_col),
            (a1_x, a1_y, team_a_col),
            (gk_x, gk_y, gk_col),
            (a2_x, a2_y, team_a_col),
            (b2_x, b2_y, team_b_col),
            (ref_x, ref_y, ref_col),
        ]
        players.sort(key=lambda p: pitch_to_img(p[0], p[1])[1])

        for p_x, p_y, col in players:
            draw_player_figure(p_x, p_y, col)

        # Draw ball
        bx_img, by_img = pitch_to_img(ball_x, ball_y)
        cv2.circle(frame, (bx_img, by_img), 4, (245, 245, 245), -1, cv2.LINE_AA)
        cv2.circle(frame, (bx_img, by_img), 4, (20, 20, 20), 1, cv2.LINE_AA)

        writer.write(frame)

    writer.release()
    return Path(output_path)
