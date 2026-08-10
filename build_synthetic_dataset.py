"""
Build the synthetic-label control datasets.

    python build_synthetic_dataset.py                 # all four
    python build_synthetic_dataset.py --type corner   # one type, both signs
    python build_synthetic_dataset.py --variants 20   # denser supervision

Real StatsBomb geometry, synthetic labels from pitchpulse/synthetic.py. Writes,
for each set-piece type and each sign:

    output/synthetic_{type}_{sign}.jsonl        one row per graph
    output/synthetic_{type}_{sign}.meta.json    calibration + provenance

The pos and neg files hold IDENTICAL geometry and differ only in the label,
because that is what makes the pair a control: any difference in how Module 2
reports on the two models has to come from the labels.

WHY VARIANTS
------------
A corpus of untouched real frames teaches a model where danger lives, but says
nothing about what happens when a player MOVES -- and moving players is the
only thing Module 2 does. So each real frame is expanded into `--variants`
perturbed copies (attackers advanced or dropped off, defenders tightened or
loosened, plus random jitter), each relabelled by the same rule. The model
therefore sees the local response surface around every frame, which is the
supervision a counterfactual tool actually needs.
"""

import argparse
import json
import math
import os
import random

from pitchpulse.contract import GOAL, from_record
from pitchpulse.simulate import MAX_DISPLACEMENT, apply_edit
from pitchpulse.synthetic import (
    DEFAULT_CALIBRATION,
    RULE_VERSION,
    fit_calibration,
    synthetic_label,
)

SOURCES = {
    "corner": "output/corners_dataset_full.jsonl",
    "freekick": "output/freekicks_dataset_full.jsonl",
}

MIN_ATTACKERS = 3   # graph.py's training guard -- rows outside it can never be scored
MIN_DEFENDERS = 1


def load_scenarios(path, sp_type, limit=None):
    """Real frames that the shared graph builder is willing to score."""
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
                continue  # the free-kick GAT needs kick_x/kick_y as graph features

            n_att = sum(1 for p in scen["players"] if p["teammate"])
            if n_att < MIN_ATTACKERS or (len(scen["players"]) - n_att) < MIN_DEFENDERS:
                continue

            out.append(scen)
            if limit and len(out) >= limit:
                break
    return out


def _dist(a, b):
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


def make_variant(scenario, rng):
    """One perturbed copy of a real frame. Returns (scenario, variant_name)."""
    attackers = [p for p in scenario["players"] if p["teammate"]]
    defenders = [p for p in scenario["players"] if not p["teammate"]]
    kind = rng.choice(["advance", "retreat", "tighten", "loosen", "jitter"])
    step = rng.uniform(0.5, MAX_DISPLACEMENT)
    gx, gy = GOAL

    if kind in ("advance", "retreat") and attackers:
        a = rng.choice(attackers)
        gap = math.hypot(gx - a["x"], gy - a["y"])
        if gap < 1e-6:
            kind = "jitter"
        else:
            s = step if kind == "advance" else -step
            return (
                apply_edit(
                    scenario, a["id"], (gx - a["x"]) / gap * s, (gy - a["y"]) / gap * s
                ),
                kind,
            )

    if kind in ("tighten", "loosen") and defenders and attackers:
        d = rng.choice(defenders)
        a = min(attackers, key=lambda p: _dist(d, p))
        gap = _dist(d, a)
        if gap < 1e-6:
            kind = "jitter"
        else:
            s = step if kind == "tighten" else -step
            return (
                apply_edit(
                    scenario, d["id"], (a["x"] - d["x"]) / gap * s, (a["y"] - d["y"]) / gap * s
                ),
                kind,
            )

    p = rng.choice(scenario["players"])
    theta = rng.uniform(0, 2 * math.pi)
    return (
        apply_edit(scenario, p["id"], math.cos(theta) * step, math.sin(theta) * step),
        "jitter",
    )


def row_for(scenario, variant, calibration, sign):
    return {
        "scenario_id": scenario["scenario_id"],
        "set_piece_type": scenario["set_piece_type"],
        "kick_location": scenario.get("kick_location"),
        "variant": variant,
        "players": scenario["players"],
        "label": synthetic_label(scenario, calibration, sign),
    }


def build(sp_type, n_variants, limit, seed, outdir):
    path = SOURCES[sp_type]
    if not os.path.exists(path):
        raise SystemExit(f"missing source dataset: {path}")

    scenarios = load_scenarios(path, sp_type, limit)
    if not scenarios:
        raise SystemExit(f"no usable {sp_type} scenarios in {path}")
    print(f"[{sp_type}] {len(scenarios)} real frames pass the graph guard")

    # Calibrate on the real frames only. Fitting on variants too would let the
    # augmentation scheme decide the label distribution.
    calibration = fit_calibration(scenarios)
    print(
        f"[{sp_type}] calibration: mu={calibration['mu']:.4f} "
        f"sigma={calibration['sigma']:.4f} gain={calibration['gain']}"
    )

    handles = {
        sign_name: open(os.path.join(outdir, f"synthetic_{sp_type}_{sign_name}.jsonl"), "w")
        for sign_name in ("pos", "neg")
    }
    signs = {"pos": 1.0, "neg": -1.0}
    counts = {"original": 0}

    try:
        for idx, scen in enumerate(scenarios):
            rng = random.Random(f"{seed}:{sp_type}:{idx}")  # per-frame, so --limit is stable
            pairs = [(scen, "original")]
            for _ in range(max(n_variants - 1, 0)):
                try:
                    pairs.append(make_variant(scen, rng))
                except Exception:  # noqa: BLE001 -- an unplaceable edit is not fatal
                    continue

            for variant_scen, variant in pairs:
                counts[variant] = counts.get(variant, 0) + 1
                for sign_name, sign in signs.items():
                    handles[sign_name].write(
                        json.dumps(row_for(variant_scen, variant, calibration, sign))
                        + "\n"
                    )
    finally:
        for h in handles.values():
            h.close()

    total = sum(counts.values())
    for sign_name, sign in signs.items():
        meta = {
            "set_piece_type": sp_type,
            "sign": sign_name,
            "sign_value": sign,
            "rule_version": RULE_VERSION,
            "calibration": calibration,
            "n_real_frames": len(scenarios),
            "n_rows": total,
            "variants_per_frame": n_variants,
            "variant_counts": counts,
            "seed": seed,
            "source": SOURCES[sp_type],
            "labels_are_synthetic": True,
            "warning": (
                "Synthetic geometric labels, NOT football outcomes. Models trained "
                "on this file are directional controls and must never be reported "
                "as football results."
            ),
        }
        mpath = os.path.join(outdir, f"synthetic_{sp_type}_{sign_name}.meta.json")
        with open(mpath, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[{sp_type}] wrote {total} rows -> synthetic_{sp_type}_{sign_name}.jsonl")

    return calibration, scenarios


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--type", choices=sorted(SOURCES), default=None)
    ap.add_argument("--variants", type=int, default=12,
                    help="rows per real frame, including the untouched original")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap on real frames, for a quick smoke build")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="output")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    for sp_type in ([args.type] if args.type else sorted(SOURCES)):
        build(sp_type, args.variants, args.limit, args.seed, args.outdir)


if __name__ == "__main__":
    main()
