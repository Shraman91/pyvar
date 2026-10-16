"""Unit tests for FIFA Law 11 geometric offside logic."""

import pytest
from config import OffsideConfig, PitchConfig
from events.offside import OffsideDetector, PlayerPitchPosition


@pytest.fixture
def detector() -> OffsideDetector:
    pitch_cfg = PitchConfig(pitch_length=105.0, pitch_width=68.0)
    offside_cfg = OffsideConfig(margin_of_error_cm=5.0)
    return OffsideDetector(offside_cfg, pitch_cfg)


def test_clear_offside_attacking_right(detector):
    """Attacking right (towards X=105): Attacker ahead of 2nd last defender and ball."""
    # Defending goal is at X=105
    # GK at X=103 (last defender)
    # B1 at X=85 (second-last defender -> offside line at X=85)
    # B2 at X=70
    defenders = [
        PlayerPitchPosition(track_id=1, team_id="team_b", pitch_x=103.0, pitch_y=34.0),
        PlayerPitchPosition(track_id=2, team_id="team_b", pitch_x=85.0, pitch_y=30.0),
        PlayerPitchPosition(track_id=3, team_id="team_b", pitch_x=70.0, pitch_y=25.0),
    ]

    # Ball played from X=75
    ball_pos = (75.0, 34.0)

    # Attacker A1 is at X=86.5 (1.5 meters ahead of defender at X=85)
    attackers = [
        PlayerPitchPosition(track_id=10, team_id="team_a", pitch_x=86.5, pitch_y=32.0),
        PlayerPitchPosition(track_id=11, team_id="team_a", pitch_x=75.0, pitch_y=34.0),
    ]

    result = detector.evaluate_offside(
        attackers=attackers,
        defenders=defenders,
        ball_pos=ball_pos,
        target_attacker_id=10,
        direction="right"
    )

    assert result.is_offside is True
    assert result.verdict == "OFFSIDE"
    assert pytest.approx(result.margin_m, 0.01) == 1.5
    assert pytest.approx(result.margin_cm, 0.1) == 150.0
    assert result.attacker_id == 10
    assert result.defender_id == 2  # B1
    assert result.offside_line_x_pitch == 85.0


def test_clear_onside_attacking_right(detector):
    """Attacking right: Attacker behind 2nd last defender."""
    defenders = [
        PlayerPitchPosition(track_id=1, team_id="team_b", pitch_x=103.0, pitch_y=34.0),
        PlayerPitchPosition(track_id=2, team_id="team_b", pitch_x=85.0, pitch_y=30.0),
    ]
    ball_pos = (70.0, 34.0)

    # Attacker A1 is at X=84.2 (0.8m behind offside line at 85)
    attackers = [
        PlayerPitchPosition(track_id=10, team_id="team_a", pitch_x=84.2, pitch_y=32.0),
    ]

    result = detector.evaluate_offside(
        attackers=attackers,
        defenders=defenders,
        ball_pos=ball_pos,
        target_attacker_id=10,
        direction="right"
    )

    assert result.is_offside is False
    assert result.verdict == "ONSIDE"
    assert pytest.approx(result.margin_m, 0.01) == -0.8
    assert pytest.approx(result.margin_cm, 0.1) == -80.0


def test_attacker_behind_ball_is_onside(detector):
    """Attacker ahead of 2nd last defender but behind the ball is ONSIDE."""
    defenders = [
        PlayerPitchPosition(track_id=1, team_id="team_b", pitch_x=103.0, pitch_y=34.0),
        PlayerPitchPosition(track_id=2, team_id="team_b", pitch_x=80.0, pitch_y=30.0),
    ]
    # Ball is dribbled deep into penalty box to X=92
    ball_pos = (92.0, 34.0)

    # Attacker A1 is at X=90.0 (ahead of defender at 80, but cut-back behind ball at 92)
    attackers = [
        PlayerPitchPosition(track_id=10, team_id="team_a", pitch_x=90.0, pitch_y=28.0),
    ]

    result = detector.evaluate_offside(
        attackers=attackers,
        defenders=defenders,
        ball_pos=ball_pos,
        target_attacker_id=10,
        direction="right"
    )

    assert result.is_offside is False
    assert result.verdict == "BEHIND_BALL"


def test_attacker_in_own_half_is_onside(detector):
    """Attacker in their own half (X <= 52.5) cannot be offside."""
    # High defensive line: defenders pushed past halfway line to X=48
    defenders = [
        PlayerPitchPosition(track_id=1, team_id="team_b", pitch_x=103.0, pitch_y=34.0),
        PlayerPitchPosition(track_id=2, team_id="team_b", pitch_x=50.0, pitch_y=30.0),
    ]
    ball_pos = (35.0, 34.0)

    # Attacker A1 is at X=51.0 (ahead of defender at 50, but still in own half <= 52.5)
    attackers = [
        PlayerPitchPosition(track_id=10, team_id="team_a", pitch_x=51.0, pitch_y=32.0),
    ]

    result = detector.evaluate_offside(
        attackers=attackers,
        defenders=defenders,
        ball_pos=ball_pos,
        target_attacker_id=10,
        direction="right"
    )

    assert result.is_offside is False
    assert result.verdict == "IN_OWN_HALF"


def test_attacking_left_symmetry(detector):
    """Verify symmetry when attacking towards left goal (X = 0)."""
    # Defending goal at X=0
    # GK at X=2.0 (last defender)
    # B1 at X=20.0 (second-last defender -> offside line at X=20.0)
    defenders = [
        PlayerPitchPosition(track_id=1, team_id="team_b", pitch_x=2.0, pitch_y=34.0),
        PlayerPitchPosition(track_id=2, team_id="team_b", pitch_x=20.0, pitch_y=30.0),
    ]
    ball_pos = (30.0, 34.0)

    # Attacker A1 at X=18.0 (2 meters closer to goal at 0 than defender at 20.0) -> OFFSIDE
    attackers = [
        PlayerPitchPosition(track_id=10, team_id="team_a", pitch_x=18.0, pitch_y=32.0),
    ]

    result = detector.evaluate_offside(
        attackers=attackers,
        defenders=defenders,
        ball_pos=ball_pos,
        target_attacker_id=10,
        direction="left"
    )

    assert result.is_offside is True
    assert result.verdict == "OFFSIDE"
    assert pytest.approx(result.margin_m, 0.01) == 2.0
    assert result.offside_line_x_pitch == 20.0
