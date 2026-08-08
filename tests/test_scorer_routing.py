"""The scorer must pick weights by set-piece type.

These exist because of a real bug: api.py held one global scorer built from the
corner weights and used it for free kicks too, which returns a plausible-looking
wrong number rather than failing. Both halves are guarded here -- the scorer
refuses a mismatched scenario, and get_scorer() routes by type.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pitchpulse.baseline import WEIGHTS_PATHS, LearnedScorer
from pitchpulse.contract import make_player
from pitchpulse.scorer import HeuristicScorer, RoutingScorer, get_scorer

ROOT = Path(__file__).resolve().parents[1]


def scenario(set_piece_type):
    return {
        "scenario_id": f"demo-{set_piece_type}",
        "set_piece_type": set_piece_type,
        "delivery_end": [110.0, 40.0],
        "players": [
            make_player(0, 119.5, 0.5, True, actor=True),
            make_player(1, 112.0, 38.0, True),
            make_player(2, 106.0, 40.0, True),
            make_player(3, 113.0, 39.0, False),
            make_player(4, 118.0, 40.0, False, keeper=True),
        ],
    }


def weights(sp_type):
    return ROOT / WEIGHTS_PATHS[sp_type]


def test_scorer_rejects_a_scenario_of_the_wrong_type():
    if not weights("corner").exists():
        pytest.skip("corner weights not trained on this machine")
    s = LearnedScorer(weights("corner"))
    with pytest.raises(ValueError, match="trained on 'corner'"):
        s(scenario("freekick"))


def test_scorer_accepts_its_own_type():
    if not weights("corner").exists():
        pytest.skip("corner weights not trained on this machine")
    assert 0.0 <= LearnedScorer(weights("corner"))(scenario("corner")) <= 1.0


def test_routing_scorer_dispatches_on_set_piece_type():
    corner, freekick = HeuristicScorer(), HeuristicScorer()
    freekick.bias = corner.bias + 2.0  # make the two distinguishable
    r = RoutingScorer({"corner": corner, "freekick": freekick})

    assert r(scenario("corner")) == corner(scenario("corner"))
    assert r(scenario("freekick")) == freekick(scenario("freekick"))
    assert r(scenario("corner")) != r(scenario("freekick"))


def test_routing_scorer_is_not_trained_if_any_branch_is_placeholder():
    """A half-trained system must not advertise itself as a trained model."""
    if not weights("corner").exists():
        pytest.skip("corner weights not trained on this machine")
    r = RoutingScorer(
        {"corner": LearnedScorer(weights("corner"))}, fallback=HeuristicScorer()
    )
    assert r.is_trained_model is False


def test_get_scorer_routes_each_type_to_its_own_weights():
    if not all(weights(t).exists() for t in WEIGHTS_PATHS):
        pytest.skip("per-type weights not trained on this machine")
    s = get_scorer(verbose=False)
    assert hasattr(s, "scorer_for"), "expected a routing scorer"
    for sp_type in WEIGHTS_PATHS:
        assert s.scorer_for(sp_type).set_piece_type == sp_type
    # And the routed call must agree with the model it routes to.
    assert s(scenario("corner")) == s.scorer_for("corner")(scenario("corner"))


def test_explicit_baseline_path_still_forces_one_model():
    """The single-model escape hatch must keep working for ablations."""
    if not weights("corner").exists():
        pytest.skip("corner weights not trained on this machine")
    s = get_scorer(baseline_path=str(weights("corner")), verbose=False)
    assert s.set_piece_type == "corner"
