"""
Step 1 smoke test -- run this from the repo root after making the edits:

    python step1_check.py

It answers one question only: does the GAT load and score a real scenario?
It does NOT tell you whether the scores are any good. That is Step 2.
"""

import json

from pitchpulse.contract import from_record
from pitchpulse.graph import CHECKPOINT_PATHS
from pitchpulse.scorer import GATScorer
from pitchpulse.simulate import what_if

FILES = {
    "corner": "output/corners_dataset_full.jsonl",
    "freekick": "output/freekicks_dataset_full.jsonl",
}


def first_usable(path, sp_type, n=200):
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            rec = json.loads(line)
            if not rec.get("has_freeze_frame"):
                continue
            if sp_type == "freekick" and rec.get("set_piece_type") != "freekick_delivery":
                continue
            scen = from_record(rec, scenario_id=str(i))
            if scen is None:
                continue
            n_att = sum(1 for p in scen["players"] if p["teammate"])
            n_def = len(scen["players"]) - n_att
            if n_att >= 3 and n_def >= 1:
                return scen
    return None


for sp_type, path in FILES.items():
    print(f"\n=== {sp_type} ===")

    scen = first_usable(path, sp_type)
    if scen is None:
        print("  no usable scenario found -- check the dataset path")
        continue

    print(f"  scenario {scen['scenario_id']}: {len(scen['players'])} players")
    if sp_type == "freekick":
        print(f"  kick_location: {scen.get('kick_location')}")

    try:
        scorer = GATScorer(CHECKPOINT_PATHS[sp_type])
    except Exception as exc:
        print(f"  FAILED to load: {exc}")
        continue

    print(f"  loaded {scorer.name}")

    base = scorer(scen)
    print(f"  base danger score: {base:.4f}")

    # move the first attacker 3 yards toward goal -- does anything change at all?
    attacker = next(p for p in scen["players"] if p["teammate"] and not p.get("actor"))
    result = what_if(scen, attacker["id"], attacker["x"] + 3.0, attacker["y"], scorer)
    print(f"  after moving player {attacker['id']} +3 yards toward goal: "
          f"{result['score_after']:.4f}  (delta {result['delta']:+.4f})")

    if abs(result["delta"]) < 1e-6:
        print("  WARNING: score did not move. The model may be ignoring position.")