"""
Step 2 -- directional sanity of the danger score.

Step 1 proved the GAT loads and runs. It says nothing about whether the scores
mean anything. This script asks the only question that matters for a
counterfactual tool:

    when a coach makes a change whose tactical direction is not in doubt,
    does the score move the right way?

Two probes, both deliberately unambiguous:

    TIGHTEN   move a defender 2 yards toward the attacker they are marking.
              Tighter marking should LOWER danger.

    ADVANCE   move an unmarked-ish attacker 2 yards toward goal.
              Closer to goal should RAISE danger.

A model that fails these is not "slightly inaccurate" -- it is unusable for
what-if analysis, because every recommendation it makes is a coin flip. This is
a property of the model, not of the simulation layer, so both scorers are run
side by side and reported separately.

Run from the repo root:

    python step2_perturbation.py
"""

import json
import math

from pitchpulse.baseline import WEIGHTS_PATHS, LearnedScorer
from pitchpulse.contract import from_record
from pitchpulse.graph import CHECKPOINT_PATHS
from pitchpulse.scorer import GATScorer
from pitchpulse.simulate import what_if

GOAL = (120.0, 40.0)
STEP = 2.0
N_SCENARIOS = 150

FILES = {
    "corner": "output/corners_dataset_full.jsonl",
    "freekick": "output/freekicks_dataset_full.jsonl",
}


def load_scenarios(path, sp_type, limit):
    out = []
    with open(path) as f:
        for i, line in enumerate(f):
            rec = json.loads(line)
            if not rec.get("has_freeze_frame"):
                continue
            if sp_type == "freekick" and rec.get("set_piece_type") != "freekick_delivery":
                continue
            scen = from_record(rec, scenario_id=f"{sp_type}-{i}")
            if scen is None:
                continue
            n_att = sum(1 for p in scen["players"] if p["teammate"])
            if n_att >= 3 and (len(scen["players"]) - n_att) >= 1:
                out.append(scen)
            if len(out) >= limit:
                break
    return out


def _dist(a, b):
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


def probe_tighten(scen, scorer):
    """Move a defender toward its nearest attacker. Expect delta < 0."""
    attackers = [p for p in scen["players"] if p["teammate"]]
    defenders = [p for p in scen["players"] if not p["teammate"] and not p.get("keeper")]
    if not defenders:
        return None

    # the defender/attacker pair that is already closest -- an unambiguous marker
    best = min(
        ((d, a) for d in defenders for a in attackers), key=lambda pair: _dist(*pair)
    )
    d, a = best
    gap = _dist(d, a)
    if gap < 0.5:  # already on top of each other, nothing to tighten
        return None

    frac = min(STEP, gap - 0.2) / gap
    return what_if(
        scen, d["id"], d["x"] + (a["x"] - d["x"]) * frac,
        d["y"] + (a["y"] - d["y"]) * frac, scorer
    )["delta"]


def probe_advance(scen, scorer):
    """Move the freest attacker toward goal. Expect delta > 0."""
    attackers = [p for p in scen["players"] if p["teammate"] and not p.get("actor")]
    defenders = [p for p in scen["players"] if not p["teammate"]]
    if not attackers or not defenders:
        return None

    a = max(attackers, key=lambda p: min(_dist(p, d) for d in defenders))
    gx, gy = GOAL
    gap = math.hypot(gx - a["x"], gy - a["y"])
    if gap < 1.0:
        return None

    frac = min(STEP, gap - 0.5) / gap
    return what_if(
        scen, a["id"], a["x"] + (gx - a["x"]) * frac, a["y"] + (gy - a["y"]) * frac,
        scorer
    )["delta"]


PROBES = [
    ("TIGHTEN defender", probe_tighten, -1),  # expected sign
    ("ADVANCE attacker", probe_advance, +1),
]


def run(scorer, scenarios, label):
    print(f"\n  {label}")
    for name, probe, expected_sign in PROBES:
        deltas = []
        for scen in scenarios:
            try:
                d = probe(scen, scorer)
            except Exception:  # noqa: BLE001  -- skip scenarios the model rejects
                d = None
            if d is not None:
                deltas.append(d)

        if not deltas:
            print(f"    {name:18s}  no usable scenarios")
            continue

        correct = sum(1 for d in deltas if d * expected_sign > 0)
        flat = sum(1 for d in deltas if abs(d) < 1e-6)
        mean_abs = sum(abs(d) for d in deltas) / len(deltas)
        pct = 100.0 * correct / len(deltas)

        verdict = "OK" if pct >= 60 else ("COIN FLIP" if pct >= 40 else "INVERTED")
        print(
            f"    {name:18s}  correct direction {pct:5.1f}%  "
            f"(n={len(deltas)}, mean |delta|={mean_abs:.4f}, flat={flat})  {verdict}"
        )


for sp_type, path in FILES.items():
    print(f"\n=== {sp_type} ===")
    scenarios = load_scenarios(path, sp_type, N_SCENARIOS)
    print(f"  {len(scenarios)} scenarios")

    try:
        run(GATScorer(CHECKPOINT_PATHS[sp_type]), scenarios, "Module 1 GAT")
    except Exception as exc:  # noqa: BLE001
        print(f"  GAT unavailable: {exc}")

    try:
        run(LearnedScorer(WEIGHTS_PATHS[sp_type]), scenarios, "Module 2 logistic baseline")
    except Exception as exc:  # noqa: BLE001
        print(f"  baseline unavailable: {exc}")

print(
    "\n50% correct = the model is guessing. A counterfactual tool built on a "
    "guessing model gives coaches advice that is right half the time."
)