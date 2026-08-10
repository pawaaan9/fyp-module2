"""
Step 3 -- the control experiment.

    python step3_control.py

Step 2 showed Module 1's real GAT does not move the expected way when a coach
makes an unambiguous edit. That finding has two possible causes and step 2
cannot separate them:

    (a) the GAT learned correlation rather than causation, or
    (b) Module 2's simulation layer is broken.

This script separates them. It runs the SAME two probes against models whose
correct answer is known by construction:

    synthetic-pos   trained on labels that rise when an attacker advances and
                    fall when a defender tightens
    synthetic-neg   trained on the exact same geometry with the labels inverted

Read the results like this:

    pos recovers the directions  ->  the pipeline transmits direction faithfully,
                                     so (b) is ruled out and step 2's result is
                                     a fact about the real GAT
    neg comes out INVERTED       ->  Module 2 reports what the scorer actually
                                     says. This is the load-bearing half: a
                                     pipeline that always reported "improvement"
                                     would score pos and neg identically, and
                                     its agreement with pos would mean nothing
    pos fails                    ->  the pipeline is broken; fix it before
                                     drawing any conclusion about the real GAT

Percentages are ALWAYS measured against the sign=+1 expectation (advance raises
danger, tighten lowers it), for every scorer in the table. That is what makes
the pos and neg columns directly comparable: neg is working correctly when its
number is LOW.

    The synthetic models are NOT football models. They predict
    pitchpulse/synthetic.py's geometry rule and say nothing about real football.
"""

import json
import os

from pitchpulse.contract import from_record
from pitchpulse.graph import CHECKPOINT_PATHS
from pitchpulse.scorer import GATScorer
from pitchpulse.synthetic import PROBES, SyntheticScorer

N_SCENARIOS = 150
STEP = 2.0

FILES = {
    "corner": "output/corners_dataset_full.jsonl",
    "freekick": "output/freekicks_dataset_full.jsonl",
}

OK_THRESHOLD = 60.0        # >= this = direction recovered
INVERTED_THRESHOLD = 40.0  # <= this = direction inverted


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
            if sp_type == "freekick" and not scen.get("kick_location"):
                continue
            n_att = sum(1 for p in scen["players"] if p["teammate"])
            if n_att >= 3 and (len(scen["players"]) - n_att) >= 1:
                out.append(scen)
            if len(out) >= limit:
                break
    return out


def run_probes(scorer, scenarios):
    """{probe_name: {pct, n, flat, mean_abs}} against the sign=+1 expectation."""
    results = {}
    for name, probe, expected in PROBES:
        deltas = []
        for scen in scenarios:
            try:
                d = probe(scen, scorer, STEP)
            except Exception:  # noqa: BLE001 -- scenarios a model rejects are not results
                d = None
            if d is not None:
                deltas.append(d)

        correct = sum(1 for d in deltas if d * expected > 0)
        results[name] = {
            "pct": 100.0 * correct / len(deltas) if deltas else float("nan"),
            "n": len(deltas),
            "flat": sum(1 for d in deltas if abs(d) < 1e-9),
            "mean_abs": (sum(abs(d) for d in deltas) / len(deltas)) if deltas else 0.0,
        }
    return results


def synthetic_checkpoint(sp_type, sign):
    return f"models/module2_synthetic_gat_{sp_type}_{sign}.pt"


def calibration_for(sp_type, sign):
    path = f"output/synthetic_{sp_type}_{sign}.meta.json"
    with open(path) as f:
        return json.load(f)["calibration"]


PROBE_NAMES = [name for name, _, _ in PROBES]
VERDICTS = {}


def verdict_for(pct):
    if pct != pct:  # NaN
        return "n/a"
    if pct >= OK_THRESHOLD:
        return "RECOVERED"
    if pct <= INVERTED_THRESHOLD:
        return "INVERTED"
    return "COIN FLIP"


for sp_type, path in FILES.items():
    print(f"\n{'=' * 78}")
    print(f"=== {sp_type} ===")
    print(f"{'=' * 78}")

    scenarios = load_scenarios(path, sp_type, N_SCENARIOS)
    print(f"  {len(scenarios)} real scenarios (unedited StatsBomb geometry)\n")

    scorers = []

    # The rule itself: the ceiling no model trained on it can beat.
    try:
        cal = calibration_for(sp_type, "pos")
        scorers.append(("synthetic RULE pos  [ceiling]", SyntheticScorer(cal, +1.0)))
        scorers.append(("synthetic RULE neg  [ceiling]", SyntheticScorer(cal, -1.0)))
    except Exception as exc:  # noqa: BLE001
        print(f"  synthetic rule unavailable: {exc}")

    for sign in ("pos", "neg"):
        ckpt = synthetic_checkpoint(sp_type, sign)
        if not os.path.exists(ckpt):
            print(f"  MISSING checkpoint {ckpt}")
            continue
        try:
            scorers.append((f"synthetic GAT {sign}", GATScorer(ckpt)))
        except Exception as exc:  # noqa: BLE001
            print(f"  failed to load {ckpt}: {exc}")

    # Module 1's real GAT, for reference -- this is the thing under suspicion.
    try:
        scorers.append(("module 1 GAT (real)", GATScorer(CHECKPOINT_PATHS[sp_type])))
    except Exception as exc:  # noqa: BLE001
        print(f"  module 1 GAT unavailable: {exc}")

    header = f"  {'scorer':<30}"
    for name in PROBE_NAMES:
        header += f"{name.upper():>24}"
    print(header)
    sub = f"  {'':<30}"
    for _ in PROBE_NAMES:
        sub += f"{'%corr':>8}{'n':>5}{'mean|d|':>11}"
    print(sub)
    print(f"  {'-' * 74}")

    for label, scorer in scorers:
        res = run_probes(scorer, scenarios)
        line = f"  {label:<30}"
        for name in PROBE_NAMES:
            r = res[name]
            line += f"{r['pct']:8.1f}{r['n']:5d}{r['mean_abs']:11.5f}"
        print(line)
        VERDICTS[(sp_type, label)] = res


# --- the read-out -----------------------------------------------------------

print(f"\n\n{'=' * 78}")
print("=== CONTROL VERDICT ===")
print(f"{'=' * 78}")
print(
    "\nPercentages are measured against the sign=+1 expectation for every row,\n"
    "so a correctly working NEG model scores LOW. That asymmetry is the test.\n"
)

for sp_type in FILES:
    print(f"\n  --- {sp_type} ---")
    for sign, want in (("pos", "RECOVERED"), ("neg", "INVERTED")):
        key = (sp_type, f"synthetic GAT {sign}")
        res = VERDICTS.get(key)
        if res is None:
            print(f"    synthetic GAT {sign}: NOT RUN (checkpoint missing)")
            continue

        ceiling = VERDICTS.get((sp_type, f"synthetic RULE {sign}  [ceiling]"), {})
        # For pos the rule is the ceiling the model cannot beat; for neg the same
        # row is a floor, since low is correct there. Call it a reference either way.
        for name in PROBE_NAMES:
            r = res[name]
            got = verdict_for(r["pct"])
            ceil_pct = ceiling.get(name, {}).get("pct")
            ceil_txt = f"  (rule reference {ceil_pct:.1f}%)" if ceil_pct is not None else ""
            flag = "OK" if got == want else "*** UNEXPECTED ***"
            print(
                f"    {sign} / {name:<8} {r['pct']:6.1f}%  {got:<10}"
                f"{ceil_txt:<28} expected {want:<10} {flag}"
            )

print(
    "\n  A trained model cannot beat the rule it is imitating, so compare each\n"
    "  model against its rule ceiling rather than against 100%.\n"
)
print(
    "  If pos RECOVERED and neg INVERTED on the same probes, Module 2 transmits\n"
    "  the scorer's direction faithfully in both directions -- it reports what the\n"
    "  scorer says rather than always reporting improvement. Step 2's result for\n"
    "  the real GAT is then a fact about that model, not a pipeline artefact.\n"
)
print(
    "  REMINDER: the synthetic models are geometry controls, not football models.\n"
)
