"""Unit tests for jersey extraction and KMeans team classification."""

import numpy as np
import pytest

from config import TeamConfig
from detection.detector import DetectedObject
from teams.classifier import TeamClassifier


def test_jersey_color_extraction_excludes_grass():
    """Verify that green grass background is masked out when sampling torso color."""
    classifier = TeamClassifier(TeamConfig())

    # Create dummy frame with green pitch grass (HSV green is roughly BGR (35, 120, 45))
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    frame[:] = (35, 120, 45)  # All grass

    # Player bounding box
    obj = DetectedObject(
        bbox=(50.0, 50.0, 100.0, 150.0),
        confidence=0.9,
        class_id=0,
        label="player"
    )

    # In the player's torso, put a bright red jersey (BGR: (30, 30, 220))
    # torso crop ratios are (0.15, 0.55, 0.15, 0.85)
    # y: 50 + 100*0.15 to 50 + 100*0.55 = 65 to 105
    # x: 50 + 50*0.15 to 50 + 50*0.85 = 57 to 92
    frame[65:105, 57:92] = (30, 30, 220)

    color_rgb = classifier.extract_jersey_color(obj, frame)
    assert color_rgb is not None
    # Extracted color should be high in Red channel (RGB index 0)
    assert color_rgb[0] > 180  # High red
    assert color_rgb[1] < 60   # Low green
    assert color_rgb[2] < 60   # Low blue


def test_kmeans_team_clustering():
    """Verify KMeans successfully separates players into Team A and Team B."""
    classifier = TeamClassifier(TeamConfig(n_teams=2))
    frame = np.zeros((400, 800, 3), dtype=np.uint8)
    frame[:] = (35, 120, 45)  # Grass

    players = []
    # 4 Red players (Team A)
    for i in range(4):
        x = 50 + i * 80
        frame[50:150, x:x+40] = (30, 30, 220)  # BGR Red
        p = DetectedObject(bbox=(x, 50, x+40, 150), confidence=0.9, class_id=0, label="player", track_id=i+1)
        players.append(p)

    # 4 Blue players (Team B)
    for i in range(4):
        x = 400 + i * 80
        frame[50:150, x:x+40] = (220, 120, 30)  # BGR Blue
        p = DetectedObject(bbox=(x, 50, x+40, 150), confidence=0.9, class_id=0, label="player", track_id=i+5)
        players.append(p)

    success = classifier.fit_from_players(players, frame)
    assert success is True
    assert classifier.fitted is True

    # Classify all
    classified = classifier.classify_frame(players, frame)
    team_a_count = sum(1 for p in classified if p.team_id == "team_a")
    team_b_count = sum(1 for p in classified if p.team_id == "team_b")

    assert team_a_count == 4
    assert team_b_count == 4
