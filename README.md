# pyvar: Semi-Automated Football VAR Assistant

`pyvar` is an intelligent, computer-vision-powered football Video Assistant Referee (VAR) system designed to detect offside situations from single-camera broadcast match footage and render broadcast-style offside graphics and minimaps.

---

## Key Features

- **Object Detection**: Leverages YOLOv8 to detect players, referees, goalkeepers, and the football.
- **Robust Tracking**: Integrates ByteTrack (`supervision`) for persistent player IDs, movement trails, and trajectory histories.
- **Jersey Color Team Classification**: Uses KMeans clustering in HSV/LAB space on torso crops to categorize players into Team A, Team B, Referee, and Goalkeeper.
- **Pitch Homography Calibration**:
  - Automatic pitch-line detection via green color masking, Canny edge detection, and probabilistic Hough transforms.
  - Manual 4+ point calibration fallback (e.g. penalty box corners, touchline intersections) for maximum precision.
  - Temporal smoothing of homography matrices across frames.
- **Pass Moment Detection**: Analyzes ball velocity and direction changes in proximity to players to identify the exact kick/pass release frame.
- **FIFA Law 11 Offside Geometry**:
  - Projects player foot/body points into metric pitch coordinates (FIFA regulation 105m x 68m).
  - Determines 2nd-last defender line and receiving attacker position relative to ball and goal line.
  - Computes margin in centimeters/meters with an assistive margin-of-error disclaimer (single-camera optical perspective limits).
- **Broadcast-Grade Overlays**:
  - Perspective-mapped offside line across the grass pitch.
  - Tactical 2D top-down minimap with real-time player and ball positioning.
  - TV-style VAR decision card with verdict, margin, and involved players.
- **Interactive Review UI**: Frame-by-frame scrubbing, pass frame tagging, manual calibration, and decision graphic export.

---

## Architecture & Project Structure

```text
pyvar/
  main.py                 # CLI entry point and batch pipeline runner
  config.py               # Comprehensive dataclass configuration and logging
  detection/
    detector.py           # YOLOv8 player and sports ball detector
    tracker.py            # ByteTrack wrapper and persistent trajectory engine
  teams/
    classifier.py         # Torso extraction and KMeans jersey color clustering
  pitch/
    line_detection.py     # Grass mask, Canny edge, and Hough line detection
    homography.py         # Homography computation, validation, and temporal smoothing
  events/
    pass_detector.py      # Ball-player proximity and velocity vector pass trigger
    offside.py            # FIFA Law 11 offside evaluation engine
  render/
    overlay.py            # Broadcast TV graphic overlays, perspective lines, minimap
  ui/
    review.py             # Interactive Streamlit and OpenCV review interfaces
  utils/
    sample_video.py       # High-fidelity synthetic match generator for automated testing
  tests/                  # Complete pytest suite for geometry, detection, and offside
```

---

## Installation & Setup

### Prerequisites
- Python 3.11+
- Virtual environment (recommended)

### Installation
```bash
# Clone or navigate to the repository
cd pyvar

# Create and activate a virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

---

## Usage

### 1. Run on Sample Video (Milestone 1 Quickstart)
Generate a synthetic match video and run detection, tracking, and broadcast overlay:
```bash
python main.py --generate-sample --out output/pyvar_sample_review.mp4
```

### 2. Run on Your Own Match Video
```bash
python main.py --video path/to/match.mp4 --out output/reviewed.mp4 --stride 1 --attack-direction right
```

### 3. CLI Arguments
- `--video <path>`: Path to match video file.
- `--out <path>`: Destination path for annotated MP4 video.
- `--stride <int>`: Frame stride (e.g. `2` processes every 2nd frame).
- `--model <name>`: YOLO model weight (e.g. `yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt`).
- `--attack-direction <left|right>`: Attacking team's direction towards goal.
- `--manual-calib`: Enable manual pitch point calibration mode.
- `--generate-sample`: Generate sample match video with ground-truth geometry.
- `--max-frames <int>`: Cap total frames to process for quick debugging.

---

## Running Unit Tests

Execute the comprehensive test suite with `pytest`:
```bash
pytest -v tests/
```

---

## Assistive Technology Disclaimer
`pyvar` is an assistive computer vision system. Single-camera broadcast views are subject to optical perspective distortion, frame-rate motion blur, and lens aberrations. Measurements include an assistive tolerance margin (±5 cm) and should be used to support, not replace, official refereeing.
