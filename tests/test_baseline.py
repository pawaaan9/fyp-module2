"""Tests for the trained logistic baseline. Run:  python -m pytest tests -q"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pitchpulse.baseline import FEATURE_NAMES, LearnedScorer, scenario_features
from pitchpulse.contract import make_player
from pitchpulse.simulate import move_to

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "models" / "module2_baseline_corner.json"


def demo_scenario():
    players = [
        make_player(0, 119.5, 0.5, True, actor=True),
        make_player(1, 112.0, 38.0, True),
        make_player(2, 110.0, 44.0, True),
        make_player(3, 106.0, 40.0, True),
        make_player(4, 100.0, 30.0, True),
        make_player(5, 113.0, 39.0, False),
        make_player(6, 111.0, 45.0, False),
        make_player(7, 107.0, 41.0, False),
        make_player(8, 118.0, 40.0, False, keeper=True),
        make_player(9, 95.0, 50.0, False),
    ]
    return {
        "scenario_id": "demo-1",
        "set_piece_type": "corner",
        "delivery_end": [110.0, 40.0],
        "players": players,
    }


def test_feature_vector_length_matches_names():
    assert len(scenario_features(demo_scenario())) == len(FEATURE_NAMES)


def test_every_feature_is_movement_sensitive():
    """If a feature never changes when players move, the counterfactual layer
    cannot act on it. Guard against dead features creeping in."""
    base = scenario_features(demo_scenario())
    changed = [False] * len(base)
    s = demo_scenario()
    for pid in [p["id"] for p in s["players"]]:
        for dx, dy in ((5, 0), (-5, 0), (0, 5), (0, -5), (3.5, 3.5), (-3.5, -3.5)):
            p = next(q for q in s["players"] if q["id"] == pid)
            f = scenario_features(move_to(s, pid, p["x"] + dx, p["y"] + dy))
            for i, (a, b) in enumerate(zip(base, f)):
                if abs(a - b) > 1e-9:
                    changed[i] = True
    dead = [FEATURE_NAMES[i] for i, c in enumerate(changed) if not c]
    assert not dead, f"features never respond to movement: {dead}"


def test_scorer_returns_probability():
    if not WEIGHTS.exists():
        return  # weights not trained yet on this machine
    s = LearnedScorer(WEIGHTS)
    v = s(demo_scenario())
    assert 0.0 <= v <= 1.0


def test_scorer_is_deterministic():
    if not WEIGHTS.exists():
        return
    s = LearnedScorer(WEIGHTS)
    assert s(demo_scenario()) == s(demo_scenario())


def test_scorer_is_declared_trained():
    if not WEIGHTS.exists():
        return
    assert LearnedScorer(WEIGHTS).is_trained_model is True


def test_scorer_responds_to_an_edit():
    """A scorer that ignores edits would make the whole module pointless."""
    if not WEIGHTS.exists():
        return
    s = LearnedScorer(WEIGHTS)
    base = demo_scenario()
    moved = move_to(base, 3, 111.0, 40.0)
    assert s(moved) != s(base)