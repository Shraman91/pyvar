"""FIFA Law 11 Offside geometric reasoning engine.

Evaluates player pitch coordinates against ball and second-last defender,
computing metric margins and broadcast decision verdicts.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from config import OffsideConfig, PitchConfig

import logging

logger = logging.getLogger("pyvar.offside")


@dataclass
class PlayerPitchPosition:
    """Represents a player projected onto the 2D pitch plane in meters."""
    track_id: int
    team_id: str  # 'team_a', 'team_b', 'referee', 'goalkeeper'
    pitch_x: float  # Along length of pitch [0, 105] meters
    pitch_y: float  # Across width of pitch [0, 68] meters
    image_coords: Tuple[float, float] = (0.0, 0.0)


@dataclass
class OffsideResult:
    """Structured decision output for an offside evaluation."""
    is_offside: bool
    verdict: str  # 'OFFSIDE', 'ONSIDE', 'IN_OWN_HALF', 'BEHIND_BALL'
    margin_m: float
    margin_cm: float
    confidence: float
    attacker_id: Optional[int]
    defender_id: Optional[int]
    offside_line_x_pitch: float
    attacker_x_pitch: float
    ball_x_pitch: float
    goal_line_x_pitch: float
    attack_direction: str
    margin_of_error_cm: float
    explanation: str

    def to_dict(self) -> dict:
        return {
            "is_offside": self.is_offside,
            "verdict": self.verdict,
            "margin_m": round(self.margin_m, 3),
            "margin_cm": round(self.margin_cm, 1),
            "confidence": round(self.confidence, 2),
            "attacker_id": self.attacker_id,
            "defender_id": self.defender_id,
            "offside_line_x_pitch": round(self.offside_line_x_pitch, 2),
            "attacker_x_pitch": round(self.attacker_x_pitch, 2),
            "ball_x_pitch": round(self.ball_x_pitch, 2),
            "attack_direction": self.attack_direction,
            "margin_of_error_note": f"+/-{self.margin_of_error_cm:.1f} cm (Single-camera optical perspective limit)",
            "explanation": self.explanation,
        }


class OffsideDetector:
    """Evaluates offside status according to FIFA Law 11."""

    def __init__(
        self,
        offside_config: Optional[OffsideConfig] = None,
        pitch_config: Optional[PitchConfig] = None,
    ):
        self.config = offside_config or OffsideConfig()
        self.pitch_config = pitch_config or PitchConfig()

    def evaluate_offside(
        self,
        attackers: List[PlayerPitchPosition],
        defenders: List[PlayerPitchPosition],
        ball_pos: Tuple[float, float],
        target_attacker_id: Optional[int] = None,
        direction: Optional[str] = None
    ) -> OffsideResult:
        """Evaluate offside condition at the pass moment.

        Args:
            attackers: List of attacking players in pitch coordinates.
            defenders: List of defending players (including goalkeeper).
            ball_pos: (ball_x, ball_y) on pitch in meters.
            target_attacker_id: Specific receiving attacker to test (if None, tests leading attacker).
            direction: Attacking direction 'right' (towards X=105) or 'left' (towards X=0).

        Returns:
            Structured OffsideResult.
        """
        dir_mode = (direction or self.config.attack_direction).lower()
        halfway_x = self.pitch_config.pitch_length / 2.0  # 52.5m
        ball_x = float(ball_pos[0])

        if len(defenders) < 2:
            return OffsideResult(
                is_offside=False,
                verdict="ONSIDE",
                margin_m=0.0,
                margin_cm=0.0,
                confidence=0.5,
                attacker_id=target_attacker_id,
                defender_id=None,
                offside_line_x_pitch=0.0 if dir_mode == "left" else self.pitch_config.pitch_length,
                attacker_x_pitch=0.0,
                ball_x_pitch=ball_x,
                goal_line_x_pitch=self.pitch_config.pitch_length if dir_mode == "right" else 0.0,
                attack_direction=dir_mode,
                margin_of_error_cm=self.config.margin_of_error_cm,
                explanation="Fewer than 2 defending players available on pitch."
            )

        # Sort defenders by proximity to the goal line
        if dir_mode == "right":
            # Attacking towards X = 105.0. Largest X is closest to goal.
            goal_line_x = self.pitch_config.pitch_length
            sorted_defenders = sorted(defenders, key=lambda p: p.pitch_x, reverse=True)
            second_last_defender = sorted_defenders[1]
            offside_line_x = second_last_defender.pitch_x

            # Select target attacker: either specified ID or leading attacker
            if target_attacker_id is not None:
                attacker = next((p for p in attackers if p.track_id == target_attacker_id), None)
                if attacker is None and len(attackers) > 0:
                    attacker = max(attackers, key=lambda p: p.pitch_x)
            else:
                attacker = max(attackers, key=lambda p: p.pitch_x) if attackers else None

            if attacker is None:
                raise ValueError("No attacking players provided to evaluate offside.")

            att_x = attacker.pitch_x
            # Reference line is max(offside_line_x, ball_x, halfway_x)
            reference_x = max(offside_line_x, ball_x)

            if att_x <= halfway_x:
                verdict = "IN_OWN_HALF"
                is_offside = False
                margin_m = att_x - reference_x
            elif att_x <= ball_x:
                verdict = "BEHIND_BALL"
                is_offside = False
                margin_m = att_x - reference_x
            elif att_x > offside_line_x:
                verdict = "OFFSIDE"
                is_offside = True
                margin_m = att_x - reference_x
            else:
                verdict = "ONSIDE"
                is_offside = False
                margin_m = att_x - reference_x

        else:
            # Attacking towards X = 0.0. Smallest X is closest to goal.
            goal_line_x = 0.0
            sorted_defenders = sorted(defenders, key=lambda p: p.pitch_x, reverse=False)
            second_last_defender = sorted_defenders[1]
            offside_line_x = second_last_defender.pitch_x

            if target_attacker_id is not None:
                attacker = next((p for p in attackers if p.track_id == target_attacker_id), None)
                if attacker is None and len(attackers) > 0:
                    attacker = min(attackers, key=lambda p: p.pitch_x)
            else:
                attacker = min(attackers, key=lambda p: p.pitch_x) if attackers else None

            if attacker is None:
                raise ValueError("No attacking players provided to evaluate offside.")

            att_x = attacker.pitch_x
            reference_x = min(offside_line_x, ball_x)

            if att_x >= halfway_x:
                verdict = "IN_OWN_HALF"
                is_offside = False
                margin_m = reference_x - att_x
            elif att_x >= ball_x:
                verdict = "BEHIND_BALL"
                is_offside = False
                margin_m = reference_x - att_x
            elif att_x < offside_line_x:
                verdict = "OFFSIDE"
                is_offside = True
                margin_m = reference_x - att_x
            else:
                verdict = "ONSIDE"
                is_offside = False
                margin_m = reference_x - att_x

        margin_cm = margin_m * 100.0
        confidence = 0.95  # Optical perspective estimation confidence

        explanation = (
            f"Attacker #{attacker.track_id} (X={att_x:.2f}m) vs 2nd-last defender "
            f"#{second_last_defender.track_id} (X={offside_line_x:.2f}m), Ball (X={ball_x:.2f}m). "
            f"Verdict: {verdict} ({margin_m:+.2f}m)."
        )

        return OffsideResult(
            is_offside=is_offside,
            verdict=verdict,
            margin_m=margin_m,
            margin_cm=margin_cm,
            confidence=confidence,
            attacker_id=attacker.track_id,
            defender_id=second_last_defender.track_id,
            offside_line_x_pitch=offside_line_x,
            attacker_x_pitch=att_x,
            ball_x_pitch=ball_x,
            goal_line_x_pitch=goal_line_x,
            attack_direction=dir_mode,
            margin_of_error_cm=self.config.margin_of_error_cm,
            explanation=explanation
        )
