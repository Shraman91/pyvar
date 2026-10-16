"""Review UI module for pyvar.

Provides both an interactive Streamlit web dashboard and an OpenCV
desktop scrubber for frame-by-frame VAR analysis, manual pitch
calibration, and decision graphic export.
"""

import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Ensure project root is in sys.path when executed directly (e.g. streamlit run ui/review.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from config import VARConfig, setup_logger
from detection.detector import YOLODetector, FrameDetections
from detection.tracker import PlayerTracker
from events.offside import OffsideDetector, PlayerPitchPosition, OffsideResult
from events.pass_detector import PassDetector
from pitch.homography import HomographyEstimator
from render.overlay import VAROverlayRenderer
from teams.classifier import TeamClassifier
from utils.sample_video import get_ground_truth_homography

logger = setup_logger()


def launch_opencv_review(video_path: str, config: Optional[VARConfig] = None) -> None:
    """Launch OpenCV desktop frame-by-frame review interface with keyboard controls."""
    cfg = config or VARConfig()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video '{video_path}'")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    detector = YOLODetector(cfg.detection)
    tracker = PlayerTracker(cfg.tracking)
    classifier = TeamClassifier(cfg.teams, cfg.pitch)
    homography_est = HomographyEstimator(cfg.pitch)
    offside_det = OffsideDetector(cfg.offside, cfg.pitch)
    renderer = VAROverlayRenderer(cfg.render, cfg.pitch)

    # Initialize default homography for synthetic clips or calibrate
    h_gt = get_ground_truth_homography()
    homography_est.current_homography = h_gt

    curr_frame_idx = 0
    pass_frame_idx: Optional[int] = None
    offside_result: Optional[OffsideResult] = None
    paused = True

    clicked_points: List[Tuple[float, float]] = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_points.append((float(x), float(y)))
            print(f"Point registered: ({x}, {y}). Total clicked: {len(clicked_points)}")

    window_name = "pyvar Review Interface [Left/Right: Scrub | P: Pass | S: Export | Q: Exit]"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, curr_frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_ms = (curr_frame_idx / fps) * 1000.0

        # Inference
        raw_det = detector.detect(frame, frame_index=curr_frame_idx, timestamp_ms=timestamp_ms)
        tracked = tracker.update(raw_det)
        classified_players = classifier.classify_frame(tracked.players, frame)

        # Offside evaluation if this is the pass frame
        res_dict = None
        if pass_frame_idx is not None and curr_frame_idx == pass_frame_idx and homography_est.current_homography is not None:
            # Project players
            attackers, defenders = [], []
            for p in classified_players:
                if p.track_id is None:
                    continue
                pitch_pt = homography_est.frame_to_pitch(np.array(p.bottom_center))
                pos = PlayerPitchPosition(
                    track_id=p.track_id,
                    team_id=p.team_id or "unclassified",
                    pitch_x=float(pitch_pt[0]),
                    pitch_y=float(pitch_pt[1]),
                    image_coords=p.bottom_center
                )
                if p.team_id == "team_a":
                    attackers.append(pos)
                else:
                    defenders.append(pos)

            ball_pos = (52.5, 34.0)
            if tracked.ball is not None:
                b_pitch = homography_est.frame_to_pitch(np.array(tracked.ball.center))
                ball_pos = (float(b_pitch[0]), float(b_pitch[1]))

            if attackers and len(defenders) >= 2:
                offside_result = offside_det.evaluate_offside(
                    attackers=attackers,
                    defenders=defenders,
                    ball_pos=ball_pos,
                    direction=cfg.offside.attack_direction
                )
                res_dict = offside_result.to_dict()

        trajectories = {
            p.track_id: tracker.get_trajectory(p.track_id)
            for p in tracked.players
            if p.track_id is not None
        }

        status = f"FRAME {curr_frame_idx}/{total_frames}"
        if pass_frame_idx == curr_frame_idx:
            status += " [PASS MOMENT]"

        annotated = renderer.draw_frame(
            frame=frame,
            detections=tracked,
            trajectories=trajectories,
            offside_result=res_dict,
            homography=homography_est.current_homography,
            status_text=status
        )

        # Draw clicked points if in calibration mode
        for pt in clicked_points:
            cv2.circle(annotated, (int(pt[0]), int(pt[1])), 5, (0, 0, 255), -1)

        cv2.imshow(window_name, annotated)

        wait_time = 30 if not paused else 0
        key = cv2.waitKey(wait_time) & 0xFF

        if key in [ord("q"), 27]:  # Q or Esc
            break
        elif key == 32:  # Space
            paused = not paused
        elif key in [81, ord("a")]:  # Left arrow or 'A'
            curr_frame_idx = max(0, curr_frame_idx - 1)
        elif key in [83, ord("d")]:  # Right arrow or 'D'
            curr_frame_idx = min(total_frames - 1, curr_frame_idx + 1)
        elif key == ord("c"):  # Calibrate from 4 clicked penalty box corners
            if len(clicked_points) >= 4:
                from pitch.homography import get_penalty_box_corners
                ref_pts = get_penalty_box_corners(side=cfg.offside.attack_direction)
                img_pts = np.array(clicked_points[:4], dtype=np.float32)
                h_new, conf = homography_est.compute_from_points(img_pts, ref_pts)
                if h_new is not None:
                    homography_est.current_homography = h_new
                    print(f"Calibration successful! Confidence: {conf:.2f}")
                else:
                    print(f"Calibration confidence low ({conf:.2f}). Check point ordering: [top-outer, top-goal, bot-goal, bot-outer].")
            else:
                print(f"Need 4 points to calibrate (currently have {len(clicked_points)}). Click 4 penalty box corners!")
        elif key == ord("r"):  # Reset clicked points
            clicked_points.clear()
            print("Cleared clicked calibration points.")
        elif key == ord("s"):  # Save snapshot
            out_png = f"output/var_frame_{curr_frame_idx:04d}.png"
            Path("output").mkdir(exist_ok=True)
            cv2.imwrite(out_png, annotated)
            print(f"Decision graphic saved to '{out_png}'")
        elif not paused:
            curr_frame_idx = min(total_frames - 1, curr_frame_idx + 1)
            if curr_frame_idx >= total_frames - 1:
                paused = True

    cap.release()
    cv2.destroyAllWindows()


def launch_streamlit_review():
    """Streamlit web dashboard entry point."""
    import streamlit as st

    st.set_page_config(
        page_title="pyvar - Football VAR Assistant",
        page_icon="⚽",
        layout="wide"
    )

    st.title("⚽ pyvar: Semi-Automated Football VAR Assistant")
    st.markdown("High-precision computer vision offside decision system and broadcast graphics renderer.")

    # Sidebar parameters
    st.sidebar.header("VAR Analysis Controls")
    video_path = st.sidebar.text_input("Match Video Path", value="data/sample_match.mp4")
    attack_direction = st.sidebar.selectbox("Attack Direction", options=["right", "left"], index=0)
    stride = st.sidebar.number_input("Frame Stride", min_value=1, max_value=5, value=1)

    show_boxes = st.sidebar.checkbox("Show Bounding Boxes & IDs", value=True)
    show_minimap = st.sidebar.checkbox("Show Tactical Minimap", value=True)
    show_tracks = st.sidebar.checkbox("Show Trajectory Trails", value=True)

    if not Path(video_path).exists():
        st.warning(f"Video file '{video_path}' not found. Please click below to generate synthetic sample.")
        if st.button("Generate Sample Match Clip"):
            from utils.sample_video import generate_sample_match_video
            generate_sample_match_video(output_path="data/sample_match.mp4")
            st.success("Sample match video generated! Reload page.")
            st.rerun()
        return

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    frame_slider = st.slider("Scrub Video Frame", min_value=0, max_value=max(0, total_frames - 1), value=40)

    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
    with col_btn1:
        if st.button("Mark as Pass Moment"):
            st.session_state["pass_frame"] = frame_slider
            st.success(f"Frame {frame_slider} marked as pass release moment!")

    pass_frame = st.session_state.get("pass_frame", 40)
    st.info(f"Active Pass Moment Frame: **Frame {pass_frame}**")

    # Read selected frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_slider)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        st.error("Failed to read video frame.")
        return

    # Process frame
    cfg = VARConfig()
    cfg.offside.attack_direction = attack_direction
    cfg.render.show_boxes = show_boxes
    cfg.render.show_minimap = show_minimap
    cfg.render.show_tracks = show_tracks

    detector = YOLODetector(cfg.detection)
    tracker = PlayerTracker(cfg.tracking)
    classifier = TeamClassifier(cfg.teams, cfg.pitch)
    homography_est = HomographyEstimator(cfg.pitch)
    offside_det = OffsideDetector(cfg.offside, cfg.pitch)
    renderer = VAROverlayRenderer(cfg.render, cfg.pitch)

    # Camera calibration options
    with st.sidebar.expander("Camera Calibration (Pitch Homography)", expanded=False):
        calib_mode = st.radio("Calibration Source", ["Synthetic / Demo Preset", "Manual 4-Point Calibration"], index=0)
        if calib_mode == "Manual 4-Point Calibration":
            st.caption("Enter 4 image pixel coordinates [X, Y] corresponding to the 18-yard penalty box corners:")
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                c1_x = st.number_input("Top Outer X", value=180, key="c1x")
                c1_y = st.number_input("Top Outer Y", value=240, key="c1y")
                c4_x = st.number_input("Bottom Outer X", value=40, key="c4x")
                c4_y = st.number_input("Bottom Outer Y", value=680, key="c4y")
            with col_p2:
                c2_x = st.number_input("Top Goal X", value=1100, key="c2x")
                c2_y = st.number_input("Top Goal Y", value=240, key="c2y")
                c3_x = st.number_input("Bottom Goal X", value=1240, key="c3x")
                c3_y = st.number_input("Bottom Goal Y", value=680, key="c3y")

            from pitch.homography import get_penalty_box_corners
            img_pts = np.array([[c1_x, c1_y], [c2_x, c2_y], [c3_x, c3_y], [c4_x, c4_y]], dtype=np.float32)
            ref_pts = get_penalty_box_corners(side=attack_direction)
            h_calc, conf = homography_est.compute_from_points(img_pts, ref_pts)
            if h_calc is not None:
                homography_est.current_homography = h_calc
                st.success(f"Calibration Active! Confidence: {conf:.2f}")
            else:
                st.warning("Invalid points configuration. Using default preset.")
                homography_est.current_homography = get_ground_truth_homography()
        else:
            homography_est.current_homography = get_ground_truth_homography()

    # Detections
    raw_det = detector.detect(frame, frame_index=frame_slider)
    tracked = tracker.update(raw_det)
    classified_players = classifier.classify_frame(tracked.players, frame)

    offside_result_dict = None
    if frame_slider == pass_frame and homography_est.current_homography is not None:
        attackers, defenders = [], []
        for p in classified_players:
            pitch_pt = homography_est.frame_to_pitch(np.array(p.bottom_center))
            pos = PlayerPitchPosition(
                track_id=p.track_id or 0,
                team_id=p.team_id or "unclassified",
                pitch_x=float(pitch_pt[0]),
                pitch_y=float(pitch_pt[1]),
                image_coords=p.bottom_center
            )
            if p.team_id == "team_a":
                attackers.append(pos)
            else:
                defenders.append(pos)

        ball_pos = (52.5, 34.0)
        if tracked.ball is not None:
            b_pitch = homography_est.frame_to_pitch(np.array(tracked.ball.center))
            ball_pos = (float(b_pitch[0]), float(b_pitch[1]))

        if attackers and len(defenders) >= 2:
            res = offside_det.evaluate_offside(
                attackers=attackers,
                defenders=defenders,
                ball_pos=ball_pos,
                direction=attack_direction
            )
            offside_result_dict = res.to_dict()

    annotated = renderer.draw_frame(
        frame=frame,
        detections=tracked,
        offside_result=offside_result_dict,
        homography=homography_est.current_homography,
        status_text=f"FRAME {frame_slider}/{total_frames}"
    )

    # Convert BGR to RGB for Streamlit display
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    try:
        st.image(annotated_rgb, width="stretch")
    except TypeError:
        st.image(annotated_rgb, use_container_width=True)

    if offside_result_dict:
        st.subheader("Official VAR Decision Card")
        verdict = offside_result_dict["verdict"]
        is_off = offside_result_dict["is_offside"]
        color = "red" if is_off else "green"

        st.markdown(f"### Verdict: :{color}[{verdict}]")
        st.write(f"**Margin:** {offside_result_dict['margin_m']:+.2f} m ({offside_result_dict['margin_cm']:+.1f} cm)")
        st.write(f"**Attacker ID:** #{offside_result_dict['attacker_id']} | **2nd Last Defender:** #{offside_result_dict['defender_id']}")
        st.caption(offside_result_dict["margin_of_error_note"])


if __name__ == "__main__":
    launch_streamlit_review()
