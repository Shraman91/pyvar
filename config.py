"""Configuration module for pyvar VAR assistant.

All thresholds, pitch dimensions, model parameters, and UI settings
are encapsulated in structured dataclasses with default values.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple
import argparse
import json
import logging


@dataclass
class DetectionConfig:
    """YOLO detection thresholds and model configuration."""
    model_name: str = "yolov8n.pt"
    player_conf: float = 0.35
    ball_conf: float = 0.15
    iou_thresh: float = 0.45
    device: str = "auto"  # 'cpu', 'cuda', 'mps', or 'auto'
    person_class_id: int = 0
    ball_class_id: int = 32  # Sports ball in COCO


@dataclass
class TrackingConfig:
    """ByteTrack tracker configuration."""
    track_activation_threshold: float = 0.25
    lost_track_buffer: int = 30
    minimum_matching_threshold: float = 0.8
    frame_rate: int = 30
    track_history_len: int = 60


@dataclass
class PitchConfig:
    """Standard football pitch dimensions (FIFA regulation, in meters)."""
    pitch_length: float = 105.0
    pitch_width: float = 68.0
    penalty_box_length: float = 16.5
    penalty_box_width: float = 40.32
    goal_box_length: float = 5.5
    goal_box_width: float = 18.32
    min_calibration_points: int = 4
    homography_smoothing_window: int = 5
    grass_hsv_lower: Tuple[int, int, int] = (25, 40, 40)
    grass_hsv_upper: Tuple[int, int, int] = (85, 255, 255)


@dataclass
class TeamConfig:
    """Team jersey classification configuration."""
    n_teams: int = 2
    torso_crop: Tuple[float, float, float, float] = (0.15, 0.55, 0.15, 0.85)
    sample_size_per_player: int = 30
    kmeans_random_state: int = 42


@dataclass
class EventConfig:
    """Pass and kick detection thresholds."""
    ball_proximity_radius_m: float = 3.0
    ball_proximity_radius_px: float = 80.0
    velocity_change_threshold: float = 4.0  # significant change in velocity vector
    pass_cooldown_frames: int = 15


@dataclass
class OffsideConfig:
    """Offside geometric evaluation configuration."""
    attack_direction: str = "right"  # 'left' or 'right'
    margin_of_error_cm: float = 5.0  # Camera perspective/sampling limit
    foot_point_mode: str = "bottom_center"  # 'bottom_center' or 'lowest_point'


@dataclass
class RenderConfig:
    """Broadcast-style overlay and minimap rendering configuration."""
    show_boxes: bool = True
    show_tracks: bool = True
    show_minimap: bool = True
    minimap_width: int = 340
    minimap_height: int = 220
    team_a_color: Tuple[int, int, int] = (235, 80, 52)     # BGR: Red/Coral
    team_b_color: Tuple[int, int, int] = (66, 135, 245)    # BGR: Sky Blue
    referee_color: Tuple[int, int, int] = (50, 205, 50)    # BGR: Lime Green
    goalkeeper_color: Tuple[int, int, int] = (20, 220, 240)# BGR: Yellow/Gold
    ball_color: Tuple[int, int, int] = (0, 255, 255)       # BGR: Bright Yellow
    offside_line_color: Tuple[int, int, int] = (0, 0, 255) # BGR: Pure Red
    onside_line_color: Tuple[int, int, int] = (0, 255, 0)  # BGR: Bright Green


@dataclass
class VARConfig:
    """Master configuration encompassing all components."""
    video_path: str = ""
    output_path: str = "output/pyvar_review.mp4"
    stride: int = 1
    manual_calib: bool = False
    debug: bool = False
    log_level: str = "INFO"

    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    pitch: PitchConfig = field(default_factory=PitchConfig)
    teams: TeamConfig = field(default_factory=TeamConfig)
    events: EventConfig = field(default_factory=EventConfig)
    offside: OffsideConfig = field(default_factory=OffsideConfig)
    render: RenderConfig = field(default_factory=RenderConfig)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "VARConfig":
        """Instantiate or override configuration from parsed CLI arguments."""
        config = cls()
        if hasattr(args, "video") and args.video:
            config.video_path = args.video
        if hasattr(args, "out") and args.out:
            config.output_path = args.out
        if hasattr(args, "stride") and args.stride:
            config.stride = args.stride
        if hasattr(args, "attack_direction") and args.attack_direction:
            config.offside.attack_direction = args.attack_direction
        if hasattr(args, "manual_calib") and args.manual_calib:
            config.manual_calib = args.manual_calib
        if hasattr(args, "model") and args.model:
            config.detection.model_name = args.model
        if hasattr(args, "debug") and args.debug:
            config.debug = args.debug
            config.log_level = "DEBUG"
        return config

    def to_json(self, path: Path | str) -> None:
        """Export config to a JSON file."""
        import dataclasses
        with open(path, "w", encoding="utf-8") as f:
            json.dump(dataclasses.asdict(self), f, indent=2)


def setup_logger(log_level: str = "INFO") -> logging.Logger:
    """Configure structured logging for pyvar."""
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logger = logging.getLogger("pyvar")
    logger.setLevel(numeric_level)
    
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%H:%M:%S"
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger
