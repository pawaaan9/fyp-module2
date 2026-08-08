"""Module 2 unit tests. Run with:  python -m pytest tests -q"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pitchpulse.contract import find_player, make_player, validate
from pitchpulse.scorer import HeuristicScorer
from pitchpulse.simulate import (
    apply_edit,
    candidate_grid,
    clamp_displacement,
    joint_optimise,
    move_to,
    optimise_player,
    what_if,
)

SCORER = HeuristicScorer()


def demo_scenario():
    players = [
        make_player(0, 119.5, 0.5, True, actor=True),          # taker
        make_player(1, 112.0, 38.0, True),                     # near post
        make_player(2, 110.0, 44.0, True),
        make_player(3, 106.0, 40.0, True),
        make_player(4, 100.0, 30.0, True),
        make_player(5, 113.0, 39.0, False),                    # markers
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


def test_scenario_is_valid():
    assert validate(demo_scenario())


def test_zero_move_does_not_change_score():
    s = demo_scenario()
    assert what_if(s, 3, 106.0, 40.0, SCORER)["delta"] == 0.0


def test_input_scenario_is_never_mutated():
    s = demo_scenario()
    before = find_player(s, 3)["x"]
    apply_edit(s, 3, 4.0, 0.0)
    assert find_player(s, 3)["x"] == before


def test_displacement_is_clamped_to_five_yards():
    dx, dy = clamp_displacement(30.0, 40.0)
    assert math.isclose(math.hypot(dx, dy), 5.0, abs_tol=1e-9)


def test_player_cannot_leave_the_pitch():
    edited = apply_edit(demo_scenario(), 0, 5.0, -5.0)
    p = find_player(edited, 0)
    assert 0.0 <= p["x"] <= 120.0 and 0.0 <= p["y"] <= 80.0


def test_grid_is_121_candidates_before_circle_mask():
    grid = candidate_grid(find_player(demo_scenario(), 3), steps=11)
    assert 60 <= len(grid) <= 121


def test_tighter_marking_lowers_the_score():
    """Directional check: drag a defender onto an attacker -> danger falls."""
    s = demo_scenario()
    attacker = find_player(s, 2)
    tightened = move_to(s, 6, attacker["x"] + 0.5, attacker["y"] + 0.5)
    assert SCORER(tightened) < SCORER(s)


def test_optimiser_never_returns_a_worse_score():
    r = optimise_player(demo_scenario(), 3, SCORER, maximise=True)
    assert r["best_score"] >= r["baseline_score"]


def test_defensive_optimiser_lowers_the_score():
    r = optimise_player(demo_scenario(), 7, SCORER, maximise=False)
    assert r["best_score"] <= r["baseline_score"]


def test_joint_optimisation_improves_on_baseline():
    r = joint_optimise(demo_scenario(), [1, 2, 3], SCORER, maximise=True)
    assert r["final_score"] >= r["baseline_score"]
