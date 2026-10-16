"""Jersey color classification module using KMeans clustering in HSV/RGB space.

Categorizes detected players into Team A, Team B, Goalkeepers, and Referees.
"""

from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from sklearn.cluster import KMeans

from config import TeamConfig, PitchConfig
from detection.detector import DetectedObject

import logging

logger = logging.getLogger("pyvar.teams")


class TeamClassifier:
    """Classifies players into teams based on jersey color distribution."""

    def __init__(
        self,
        config: Optional[TeamConfig] = None,
        pitch_config: Optional[PitchConfig] = None,
    ):
        self.config = config or TeamConfig()
        self.pitch_config = pitch_config or PitchConfig()
        self.kmeans: Optional[KMeans] = None
        self.fitted: bool = False
        self.color_cache: Dict[int, str] = {}  # track_id -> team_id
        self.dominant_colors: Dict[str, Tuple[int, int, int]] = {}

    def extract_jersey_color(
        self,
        player: DetectedObject,
        frame: np.ndarray
    ) -> Optional[np.ndarray]:
        """Extract dominant jersey color feature vector (RGB) excluding pitch grass.

        Args:
            player: Detected player object.
            frame: Full video frame (BGR).

        Returns:
            RGB array of shape (3,) representing dominant jersey color, or None.
        """
        torso = player.crop_torso(frame, self.config.torso_crop)
        if torso.size == 0 or torso.shape[0] < 4 or torso.shape[1] < 4:
            return None

        # Convert to HSV to filter out background grass
        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        
        # Mask grass: hue roughly 30-85, moderate saturation
        lower_grass = np.array([28, 35, 30], dtype=np.uint8)
        upper_grass = np.array([85, 255, 255], dtype=np.uint8)
        grass_mask = cv2.inRange(hsv, lower_grass, upper_grass)

        # Mask dark shadows / borders
        lower_dark = np.array([0, 0, 0], dtype=np.uint8)
        upper_dark = np.array([180, 255, 35], dtype=np.uint8)
        dark_mask = cv2.inRange(hsv, lower_dark, upper_dark)

        # Mask white shorts and pitch line borders
        lower_white = np.array([0, 0, 180], dtype=np.uint8)
        upper_white = np.array([180, 45, 255], dtype=np.uint8)
        white_mask = cv2.inRange(hsv, lower_white, upper_white)

        ignore_mask = cv2.bitwise_or(grass_mask, dark_mask)
        ignore_mask = cv2.bitwise_or(ignore_mask, white_mask)
        valid_mask = cv2.bitwise_not(ignore_mask)

        valid_pixels = torso[valid_mask > 0]
        if len(valid_pixels) < 8:
            # Fallback to entire torso if masking was too aggressive
            valid_pixels = torso.reshape(-1, 3)

        if len(valid_pixels) == 0:
            return None

        # Calculate median RGB color (robust to outliers)
        # Convert BGR -> RGB for standardized feature representation
        valid_pixels_rgb = valid_pixels[:, ::-1]
        median_color = np.median(valid_pixels_rgb, axis=0)
        return median_color.astype(np.float32)

    def fit_from_players(
        self,
        players: List[DetectedObject],
        frame: np.ndarray
    ) -> bool:
        """Fit KMeans model on players extracted from one or more frames.

        Args:
            players: List of detected players.
            frame: Full video frame.

        Returns:
            True if fitting succeeded, False otherwise.
        """
        features = []
        for p in players:
            color = self.extract_jersey_color(p, frame)
            if color is not None:
                features.append(color)

        if len(features) < self.config.n_teams * 2:
            logger.warning("Insufficient player color samples (%d) to fit KMeans", len(features))
            return False

        X = np.array(features, dtype=np.float32)
        self.kmeans = KMeans(
            n_clusters=self.config.n_teams,
            random_state=self.config.kmeans_random_state,
            n_init=10
        )
        self.kmeans.fit(X)
        self.fitted = True

        # Store centroid colors in BGR for display
        for idx, center in enumerate(self.kmeans.cluster_centers_):
            # center is RGB
            r, g, b = int(center[0]), int(center[1]), int(center[2])
            team_key = f"team_{chr(ord('a') + idx)}"
            self.dominant_colors[team_key] = (b, g, r)  # Store as BGR

        logger.info("TeamClassifier fitted successfully. Centers: %s", self.dominant_colors)
        return True

    def classify_player(
        self,
        player: DetectedObject,
        frame: np.ndarray
    ) -> str:
        """Classify a single player into team_a, team_b, referee, or goalkeeper.

        Args:
            player: DetectedObject to classify.
            frame: Video frame.

        Returns:
            team_id string.
        """
        if player.track_id is not None and player.track_id in self.color_cache:
            player.team_id = self.color_cache[player.track_id]
            return player.team_id

        color = self.extract_jersey_color(player, frame)
        if color is None or not self.fitted:
            team_id = "unclassified"
            player.team_id = team_id
            return team_id

        # Check distance to cluster centroids
        color_2d = color.reshape(1, -1)
        distances = np.linalg.norm(self.kmeans.cluster_centers_ - color_2d, axis=1)
        min_dist_idx = int(np.argmin(distances))
        min_dist = distances[min_dist_idx]

        # If color is vastly different from both teams, identify referee/GK
        # In RGB space, Euclidean distance > 140 usually indicates distinct kit (e.g. neon yellow ref)
        if min_dist > 150.0 and len(distances) >= 2:
            # Check if high green/yellow (referee)
            r, g, b = color
            if g > 150 and r > 120 and b < 80:
                team_id = "referee"
            else:
                team_id = "goalkeeper"
        else:
            team_id = f"team_{chr(ord('a') + min_dist_idx)}"

        player.team_id = team_id
        if player.track_id is not None:
            self.color_cache[player.track_id] = team_id

        return team_id

    def classify_frame(
        self,
        players: List[DetectedObject],
        frame: np.ndarray
    ) -> List[DetectedObject]:
        """Classify all players in a frame, fitting model if not already fitted."""
        if not self.fitted:
            self.fit_from_players(players, frame)

        for p in players:
            self.classify_player(p, frame)

        return players
