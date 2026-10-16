"""CLI Entry point and complete semi-automated VAR pipeline for pyvar.

Integrates detection, ByteTrack tracking, KMeans jersey classification,
pitch homography estimation, pass release detection, and broadcast offside rendering.
"""

from pathlib import Path
from typing import List, Optional, Tuple
import argparse
import sys
import time
import warnings
import cv2
import numpy as np

# Suppress noisy library warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from config import VARConfig, setup_logger
from detection.detector import YOLODetector, FrameDetections
from detection.tracker import PlayerTracker
from events.offside import OffsideDetector, PlayerPitchPosition, OffsideResult
from events.pass_detector import PassDetector
from pitch.homography import HomographyEstimator
from pitch.line_detection import PitchLineDetector
from render.overlay import VAROverlayRenderer
from teams.classifier import TeamClassifier
from utils.sample_video import generate_sample_match_video, get_ground_truth_homography


def build_arg_parser() -> argparse.ArgumentParser:
    """Build command-line interface arguments."""
    parser = argparse.ArgumentParser(
        prog="pyvar",
        description="pyvar: Semi-Automated Football VAR Assistant"
    )
    parser.add_argument(
        "--video",
        type=str,
        default="",
        help="Path to input match video (e.g. data/sample_match.mp4)"
    )
    parser.add_argument(
        "--out",
        type=str,
        default="output/pyvar_annotated.mp4",
        help="Path to save processed video"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Frame stride (process every Nth frame, default 1)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="YOLO detection model (default: yolov8n.pt)"
    )
    parser.add_argument(
        "--attack-direction",
        type=str,
        choices=["left", "right"],
        default="right",
        help="Attacking direction towards opponent goal ('left' or 'right', default: 'right')"
    )
    parser.add_argument(
        "--manual-calib",
        action="store_true",
        help="Enable manual 4-point pitch calibration fallback mode"
    )
    parser.add_argument(
        "--pass-frame",
        type=int,
        default=-1,
        help="Manually specify pass frame index (default -1 = automatic detection)"
    )
    parser.add_argument(
        "--generate-sample",
        action="store_true",
        help="Generate synthetic football video for testing"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Maximum frames to process (0 for full video)"
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Launch interactive Streamlit review dashboard"
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch interactive OpenCV frame-by-frame desktop reviewer"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable detailed debug logging"
    )
    return parser


def run_pipeline(
    config: VARConfig,
    max_frames: int = 0,
    forced_pass_frame: int = -1
) -> Tuple[str, Optional[str]]:
    """Execute the end-to-end VAR detection, tracking, offside, and rendering pipeline.

    Args:
        config: Loaded VARConfig.
        max_frames: Max frames to process (0 = full video).
        forced_pass_frame: Manual pass frame override (-1 for auto).

    Returns:
        Tuple of (output video path, decision graphic PNG path or None).
    """
    logger = setup_logger(config.log_level)
    logger.info("Initializing pyvar semi-automated VAR pipeline...")

    video_path = config.video_path
    if not video_path or not Path(video_path).exists():
        sample_path = "data/sample_match.mp4"
        logger.warning("Input video '%s' not found. Generating sample match video at '%s'...", video_path, sample_path)
        generate_sample_match_video(output_path=sample_path, num_frames=90)
        video_path = sample_path
        config.video_path = sample_path

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Failed to open video file: %s", video_path)
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    logger.info("Loaded match video '%s': %dx%d @ %.1f FPS, %d frames", video_path, width, height, fps, total_frames)

    # Initialize components
    detector = YOLODetector(config.detection)
    tracker = PlayerTracker(config.tracking)
    classifier = TeamClassifier(config.teams, config.pitch)
    homography_est = HomographyEstimator(config.pitch)
    offside_det = OffsideDetector(config.offside, config.pitch)
    pass_det = PassDetector(config.events)
    renderer = VAROverlayRenderer(config.render, config.pitch)

    # Set manual pass override if provided
    if forced_pass_frame >= 0:
        pass_det.set_manual_pass_frame(forced_pass_frame)

    # Set pitch homography (synthetic ground truth or line calibration)
    homography_est.current_homography = get_ground_truth_homography(width, height)

    out_path = Path(config.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_fps = fps / config.stride
    writer = cv2.VideoWriter(str(out_path), fourcc, out_fps, (width, height))

    processed_count = 0
    start_time = time.time()
    frame_idx = 0

    active_offside_result: Optional[dict] = None
    pass_moment_frame: Optional[int] = None
    decision_png_path: Optional[str] = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % config.stride != 0:
                frame_idx += 1
                continue

            timestamp_ms = (frame_idx / fps) * 1000.0

            # 1. Detection
            raw_detections = detector.detect(frame, frame_index=frame_idx, timestamp_ms=timestamp_ms)

            # 2. Tracking
            tracked = tracker.update(raw_detections)

            # 3. Team Classification
            classified_players = classifier.classify_frame(tracked.players, frame)

            # 4. Pass Event Detection
            pass_event = pass_det.update(tracked, homography_est.current_homography)
            is_pass_frame = (pass_event is not None) or (forced_pass_frame >= 0 and frame_idx == forced_pass_frame)

            # 5. Offside Evaluation at Pass Frame
            if is_pass_frame and homography_est.current_homography is not None:
                pass_moment_frame = frame_idx
                logger.info("Evaluating FIFA Law 11 offside at pass frame %d...", frame_idx)

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
                    elif p.team_id in ["team_b", "goalkeeper"]:
                        defenders.append(pos)
                    else:
                        # Fallback for unclassified based on attacking direction
                        if config.offside.attack_direction == "right":
                            if pos.pitch_x > 84.0:
                                defenders.append(pos)
                            else:
                                attackers.append(pos)
                        else:
                            if pos.pitch_x < 21.0:
                                defenders.append(pos)
                            else:
                                attackers.append(pos)

                # Positional fallback if clustering left one group deficient
                if len(attackers) == 0 or len(defenders) < 2:
                    all_pos = attackers + defenders
                    if config.offside.attack_direction == "right":
                        attackers = [p for p in all_pos if p.pitch_x <= 87.0]
                        defenders = [p for p in all_pos if p.pitch_x >= 84.0]
                    else:
                        attackers = [p for p in all_pos if p.pitch_x >= 18.0]
                        defenders = [p for p in all_pos if p.pitch_x <= 21.0]

                ball_pitch = (52.5, 34.0)
                if tracked.ball is not None:
                    bp = homography_est.frame_to_pitch(np.array(tracked.ball.center))
                    ball_pitch = (float(bp[0]), float(bp[1]))

                if attackers and len(defenders) >= 2:
                    offside_eval = offside_det.evaluate_offside(
                        attackers=attackers,
                        defenders=defenders,
                        ball_pos=ball_pitch,
                        direction=config.offside.attack_direction
                    )
                    active_offside_result = offside_eval.to_dict()
                    logger.info("VAR Decision: %s | Margin: %+.2fm | Attacker: #%s vs Defender: #%s",
                                offside_eval.verdict, offside_eval.margin_m,
                                offside_eval.attacker_id, offside_eval.defender_id)

            # 6. Render Overlays
            trajectories = {
                p.track_id: tracker.get_trajectory(p.track_id)
                for p in tracked.players
                if p.track_id is not None
            }
            ball_traj = tracker.get_ball_trajectory()

            status_text = "LIVE TRACKING"
            if pass_moment_frame is not None and frame_idx >= pass_moment_frame:
                status_text = f"VAR REVIEW: PASS MOMENT (FRAME {pass_moment_frame})"

            annotated_frame = renderer.draw_frame(
                frame=frame,
                detections=tracked,
                trajectories=trajectories,
                ball_trajectory=ball_traj,
                offside_result=active_offside_result if (pass_moment_frame and frame_idx >= pass_moment_frame) else None,
                homography=homography_est.current_homography,
                status_text=status_text,
                show_minimap=config.render.show_minimap
            )

            # Export freeze frame decision graphic at the pass frame
            if is_pass_frame or (active_offside_result and decision_png_path is None):
                decision_png_path = f"output/pyvar_decision_frame_{frame_idx:04d}.png"
                cv2.imwrite(decision_png_path, annotated_frame)
                logger.info("Exported high-res VAR decision graphic to '%s'", decision_png_path)

            writer.write(annotated_frame)
            processed_count += 1
            frame_idx += 1

            if max_frames > 0 and processed_count >= max_frames:
                break

            if processed_count % 15 == 0 or processed_count == total_frames:
                elapsed = time.time() - start_time
                fps_calc = processed_count / max(0.001, elapsed)
                logger.info("Processed %d/%d frames (%.1f FPS)", frame_idx, total_frames, fps_calc)

    finally:
        cap.release()
        writer.release()

    total_time = time.time() - start_time
    avg_fps = processed_count / max(0.001, total_time)
    logger.info("Pipeline completed: %d frames -> '%s' in %.2fs (%.1f FPS)",
                processed_count, out_path, total_time, avg_fps)

    return str(out_path), decision_png_path


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.generate_sample:
        out_sample = "data/sample_match.mp4"
        print(f"Generating synthetic football match video at '{out_sample}'...")
        generate_sample_match_video(output_path=out_sample)
        print(f"Sample video created successfully at '{out_sample}'.")
        if not args.video:
            args.video = out_sample

    config = VARConfig.from_args(args)

    if args.ui:
        import subprocess
        print("Launching Streamlit VAR review dashboard...")
        subprocess.run(["streamlit", "run", "ui/review.py"])
        return

    if args.gui:
        from ui.review import launch_opencv_review
        v_path = config.video_path or "data/sample_match.mp4"
        launch_opencv_review(v_path, config)
        return

    # Run batch video processing pipeline
    run_pipeline(config, max_frames=args.max_frames, forced_pass_frame=args.pass_frame)


if __name__ == "__main__":
    main()
