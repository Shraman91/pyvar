"""Event detection and offside geometry subpackage for pyvar."""

from .offside import OffsideDetector, OffsideResult, PlayerPitchPosition
from .pass_detector import PassDetector, PassEvent

__all__ = [
    "OffsideDetector",
    "OffsideResult",
    "PlayerPitchPosition",
    "PassDetector",
    "PassEvent",
]
