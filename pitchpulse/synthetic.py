"""
Synthetic labels for the Module 2 CONTROL experiment.

WHAT THIS IS FOR
----------------
Module 1's real GAT does not respond to coaching edits in the expected
direction. That observation is ambiguous: either the GAT learned correlation
rather than causation, or Module 2's simulation pipeline is broken. One
observation, two explanations, no way to tell them apart.

This file removes the ambiguity by supplying a label whose causal structure is
known *by construction*. Models trained on it use REAL StatsBomb geometry but
SYNTHETIC labels, so the correct directional response is not an empirical
question -- it is a property of the arithmetic below. If a model trained here
recovers the direction and Module 2 reports it correctly, the pipeline works,
and the real GAT's failure is the GAT's.

    THESE ARE NOT FOOTBALL MODELS. A model trained on these labels predicts
    this file's geometry rule, not the probability of a goal. Nothing produced
    from them may be reported as a football result.

THE RULE
--------
Two terms, both computed from the same geometry the probes manipulate:

    presence  mean over attackers of exp(-dist_to_goal / tau)
              -- rises monotonically as any attacker nears the goal

    freedom   mean over attackers of exp(-dist_to_goal / tau) * m
              where m = min(dist_to_nearest_defender, cap) / cap
              -- falls as a defender closes on the attacker they mark

    z = w_goal * presence + w_free * freedom
    label = sigmoid(sign * gain * (z - mu) / sigma)

`sign` is the whole point of the control. sign=+1 gives a scorer that behaves
like a sane danger model. sign=-1 gives sigmoid(-x) = 1 - sigmoid(x), which
inverts every directional response exactly. A Module 2 that reports "improved"
for both is reporting its own optimism rather than what the scorer said.

WHY THE DIRECTIONS ARE GUARANTEED (for sign=+1)
-----------------------------------------------
ADVANCE -- move an attacker toward goal. Its dist_to_goal strictly falls, so
    its exp(-d/tau) weight strictly rises, so `presence` rises. The same
    weight also multiplies its freedom term, so unless the move pushes it into
    a defender hard enough for the `m` drop to outweigh both, z rises.
    w_goal > w_free makes the presence channel dominant.

TIGHTEN -- move the defender nearest an attacker 2 yards toward them. No
    attacker moves, so `presence` is untouched; the marked attacker's nearest-
    defender distance strictly falls, so `m` falls, so `freedom` falls and z
    falls. The one escape is saturation: if that attacker's marker was already
    further than `cap` and still is, `m` stays pinned at 1.0 and the label does
    not move at all. That produces an exactly-zero delta, which scores as
    incorrect -- and is the main reason the ceiling below sits under 100%.

The ceiling is a property of the rule, not of any model. Measure it with
`verify_label_directions` and treat it as the maximum a trained model could
score on the same probes. See [[synthetic-control-ceiling]] in the run notes.
"""

import math
import random

from .contract import GOAL, validate

RULE_VERSION = "1.0"

# The rule's shape constants. Changing any of these changes the labels, so they
# are written into every dataset's .meta.json and re-read at training time
# rather than being imported as globals by the trainer.
DEFAULT_CALIBRATION = {
    "rule_version": RULE_VERSION,
    "tau": 12.0,      # yards; decay length of the goal-proximity weight
    "cap": 10.0,      # yards; marking distance beyond which an attacker counts as free
    "w_goal": 3.0,    # weight on presence -- dominant, so ADVANCE stays unambiguous
    "w_free": 1.0,    # weight on freedom  -- the only channel TIGHTEN moves
    "gain": 1.6,      # logistic steepness after standardising z
    "mu": 0.0,        # filled in by fit_calibration
    "sigma": 1.0,     # filled in by fit_calibration
}


def _sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)  # avoid overflow on large negative z
    return e / (1.0 + e)


def _dist(a, b):
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


def terms(scenario, calibration=None):
    """Return (presence, freedom), the two geometric ingredients of the rule.

    Exposed separately so the directional argument in the docstring can be
    checked term by term in tests rather than only through the final label.
    """
    cal = calibration or DEFAULT_CALIBRATION
    tau = float(cal["tau"])
    cap = float(cal["cap"])

    gx, gy = GOAL
    attackers = [p for p in scenario["players"] if p["teammate"]]
    defenders = [p for p in scenario["players"] if not p["teammate"]]

    if not attackers:
        return 0.0, 0.0

    presence = 0.0
    freedom = 0.0
    for a in attackers:
        weight = math.exp(-math.hypot(a["x"] - gx, a["y"] - gy) / tau)
        nearest = min((_dist(a, d) for d in defenders), default=cap)
        presence += weight
        freedom += weight * min(nearest, cap) / cap

    n = len(attackers)
    return presence / n, freedom / n


def raw_z(scenario, calibration=None):
    """The unstandardised rule score. Unbounded, higher = more dangerous."""
    cal = calibration or DEFAULT_CALIBRATION
    presence, freedom = terms(scenario, cal)
    return float(cal["w_goal"]) * presence + float(cal["w_free"]) * freedom


def synthetic_label(scenario, calibration=None, sign=1.0):
    """The label in [0, 1]. `sign=-1.0` inverts every directional response."""
    cal = calibration or DEFAULT_CALIBRATION
    z = raw_z(scenario, cal)
    sigma = float(cal.get("sigma", 1.0)) or 1.0
    standardised = (z - float(cal.get("mu", 0.0))) / sigma
    return _sigmoid(float(sign) * float(cal.get("gain", 1.0)) * standardised)


def fit_calibration(scenarios, base=None):
    """Centre and scale the rule over a corpus so labels span [0, 1].

    Without this the raw z values sit in a narrow band and every label lands
    near 0.5 -- a model would hit low loss by predicting the mean, which is
    exactly the collapse the trainer's pred_sd check is there to catch.
    """
    cal = dict(base or DEFAULT_CALIBRATION)
    zs = [raw_z(s, cal) for s in scenarios]
    if not zs:
        raise ValueError("cannot fit calibration on zero scenarios")

    mu = sum(zs) / len(zs)
    var = sum((z - mu) ** 2 for z in zs) / max(len(zs) - 1, 1)
    sigma = math.sqrt(var)
    if sigma < 1e-9:
        raise ValueError(
            "rule score has no variance across the corpus; labels would be "
            "constant and the trained control would be meaningless"
        )

    cal["mu"] = mu
    cal["sigma"] = sigma
    cal["rule_version"] = RULE_VERSION
    return cal


# --- the probes -------------------------------------------------------------
#
# These mirror step2_perturbation.py exactly. They are defined here rather than
# imported from it so that the ceiling measurement and the control experiment
# provoke the models in precisely the same way, and so that changing step 2
# cannot silently move the control's goalposts.

def probe_tighten(scenario, scorer, step=2.0):
    """Move the closest marker 2 yards onto their attacker. Expect delta < 0."""
    from .simulate import what_if

    attackers = [p for p in scenario["players"] if p["teammate"]]
    defenders = [
        p for p in scenario["players"] if not p["teammate"] and not p.get("keeper")
    ]
    if not defenders or not attackers:
        return None

    d, a = min(
        ((d, a) for d in defenders for a in attackers), key=lambda pair: _dist(*pair)
    )
    gap = _dist(d, a)
    if gap < 0.5:
        return None

    frac = min(step, gap - 0.2) / gap
    return what_if(
        scenario,
        d["id"],
        d["x"] + (a["x"] - d["x"]) * frac,
        d["y"] + (a["y"] - d["y"]) * frac,
        scorer,
    )["delta"]


def probe_advance(scenario, scorer, step=2.0):
    """Move the freest attacker 2 yards toward goal. Expect delta > 0."""
    from .simulate import what_if

    attackers = [
        p for p in scenario["players"] if p["teammate"] and not p.get("actor")
    ]
    defenders = [p for p in scenario["players"] if not p["teammate"]]
    if not attackers or not defenders:
        return None

    a = max(attackers, key=lambda p: min(_dist(p, d) for d in defenders))
    gx, gy = GOAL
    gap = math.hypot(gx - a["x"], gy - a["y"])
    if gap < 1.0:
        return None

    frac = min(step, gap - 0.5) / gap
    return what_if(
        scenario,
        a["id"],
        a["x"] + (gx - a["x"]) * frac,
        a["y"] + (gy - a["y"]) * frac,
        scorer,
    )["delta"]


# name, probe, expected sign of the delta when sign=+1
PROBES = [
    ("advance", probe_advance, +1),
    ("tighten", probe_tighten, -1),
]


def verify_label_directions(scenarios, calibration=None, sign=1.0, step=2.0, seed=0):
    """Measure the label rule's own directional correctness -- the CEILING.

    A trained model cannot beat the function it is imitating, so this number is
    the ceiling for any checkpoint trained on these labels. Report a model
    against this, not against 100%.

    Returns {probe_name: {"pct", "n", "flat", "mean_abs"}}.
    """
    cal = calibration or DEFAULT_CALIBRATION
    rng = random.Random(seed)
    scenarios = list(scenarios)
    rng.shuffle(scenarios)

    def scorer(s):
        return synthetic_label(s, cal, sign)

    out = {}
    for name, probe, expected in PROBES:
        deltas = []
        for scen in scenarios:
            try:
                d = probe(scen, scorer, step)
            except Exception:  # noqa: BLE001 -- a rejected scenario is not a result
                d = None
            if d is not None:
                deltas.append(d)

        # `sign` flips what "correct" means, which is the entire control.
        want = expected * (1 if sign >= 0 else -1)
        correct = sum(1 for d in deltas if d * want > 0)
        out[name] = {
            "pct": 100.0 * correct / len(deltas) if deltas else float("nan"),
            "n": len(deltas),
            "flat": sum(1 for d in deltas if abs(d) < 1e-9),
            "mean_abs": (sum(abs(d) for d in deltas) / len(deltas)) if deltas else 0.0,
        }
    return out


class SyntheticScorer:
    """Scorer interface over the rule itself, for comparison against the models.

    Same `scorer(scenario) -> float` contract as everything in scorer.py, so it
    drops into simulate.py unchanged.
    """

    is_trained_model = False

    def __init__(self, calibration=None, sign=1.0, name=None):
        self.calibration = calibration or DEFAULT_CALIBRATION
        self.sign = float(sign)
        self.name = name or f"synthetic-rule-{'pos' if self.sign >= 0 else 'neg'}"

    def __call__(self, scenario):
        validate(scenario)
        return synthetic_label(scenario, self.calibration, self.sign)
