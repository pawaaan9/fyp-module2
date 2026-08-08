"""PitchPulse shared package (schema v1.0)."""

from .contract import SCHEMA_VERSION, from_record, validate
from .scorer import HeuristicScorer, get_scorer
from .simulate import apply_edit, joint_optimise, optimise_player, what_if

__all__ = [
    "SCHEMA_VERSION",
    "from_record",
    "validate",
    "HeuristicScorer",
    "get_scorer",
    "apply_edit",
    "what_if",
    "optimise_player",
    "joint_optimise",
]
