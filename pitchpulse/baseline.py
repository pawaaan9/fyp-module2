"""
A TRAINED baseline scorer for Module 2.

Why this exists
---------------
The HeuristicScorer has hand-typed coefficients. It is fine for exercising the
simulation layer, but it cannot support any tactical claim, because its weights
were chosen rather than learned.

This module fits a logistic regression on the SAME node features that
graph.py exposes to Module 1, mean-pooled per team. That makes it a proper
ablation: it is the GAT with the message passing removed. The gap between this
model and Chamara's GAT is exactly the value added by the graph structure.

No torch. No torch_geometric. sklearn only.

Train:   python -m pitchpulse.baseline
Load:    from pitchpulse.baseline import LearnedScorer; s = LearnedScorer()
"""

import json
import math
import os

from .contract import GOAL, PITCH_LENGTH, PITCH_WIDTH, validate

# Corners and free kicks are fitted separately, so there is no single "the"
# weights file. WEIGHTS_PATH names the corner model only because a default has
# to name something; anything scoring both types should route by set-piece type
# (see scorer.RoutingScorer) rather than rely on this.
WEIGHTS_PATHS = {
    "corner": "models/module2_baseline_corner.json",
    "freekick": "models/module2_baseline_freekick.json",
}

WEIGHTS_PATH = os.environ.get("PITCHPULSE_BASELINE", WEIGHTS_PATHS["corner"])

FEATURE_VERSION = "1.0"

FEATURE_NAMES = [
    # attacker block -- mean pooled
    "att_mean_x",
    "att_mean_dist_goal",
    "att_min_dist_goal",
    "att_mean_marking",
    "att_min_marking",
    "att_spread_y",
    # counts in the danger zones (these are what a coach actually manipulates)
    "att_within_6",
    "att_within_12",
    "att_within_18",
    # defender block
    "def_mean_dist_goal",
    "def_min_dist_goal",
    "def_within_10",
    "def_within_18",
]

# Deliberately EXCLUDED: n_attackers, n_defenders, att_def_ratio.
# Two reasons. (a) They cannot change when a coach repositions a player, so
# they contribute predictive signal the counterfactual layer can never act on.
# (b) Player counts here are an artefact of StatsBomb 360 only capturing what
# was inside the broadcast camera frame -- 5 to 22 players per scenario. A
# model leaning on them would partly be learning camera framing, not football.


def _d(a, b):
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


def _dg(p):
    return math.hypot(p["x"] - GOAL[0], p["y"] - GOAL[1])


def scenario_features(scenario):
    """Mean-pool the per-player geometry into one fixed-length vector.

    Every feature here MUST change when a player moves, otherwise the
    counterfactual simulation would have nothing to respond to. That is why
    there are no static features (competition, minute, team id) in this list.
    """
    validate(scenario)
    atts = [p for p in scenario["players"] if p["teammate"]]
    defs = [p for p in scenario["players"] if not p["teammate"]]

    if not atts:
        raise ValueError("no attackers")

    att_dg = [_dg(p) for p in atts]
    def_dg = [_dg(p) for p in defs] if defs else [PITCH_LENGTH]

    if defs:
        marking = [min(_d(a, d) for d in defs) for a in atts]
    else:
        marking = [30.0] * len(atts)

    ys = [p["y"] for p in atts]
    mean_y = sum(ys) / len(ys)
    spread_y = (sum((y - mean_y) ** 2 for y in ys) / len(ys)) ** 0.5

    n_a, n_d = len(atts), len(defs)

    return [
        sum(p["x"] for p in atts) / n_a / PITCH_LENGTH,
        sum(att_dg) / n_a / PITCH_LENGTH,
        min(att_dg) / PITCH_LENGTH,
        sum(marking) / n_a / 30.0,
        min(marking) / 30.0,
        spread_y / PITCH_WIDTH,
        sum(1 for d in att_dg if d < 6.0) / 5.0,
        sum(1 for d in att_dg if d < 12.0) / 5.0,
        sum(1 for d in att_dg if d < 18.0) / 5.0,
        sum(def_dg) / max(n_d, 1) / PITCH_LENGTH,
        min(def_dg) / PITCH_LENGTH,
        sum(1 for d in def_dg if d < 10.0) / 6.0,
        sum(1 for d in def_dg if d < 18.0) / 6.0,
    ]


class LearnedScorer:
    """Trained logistic-regression scorer. Same call signature as the others:
    scorer(scenario) -> float in [0, 1]."""

    name = "module2-logreg"
    is_trained_model = True

    def __init__(self, weights_path=WEIGHTS_PATH):
        with open(weights_path) as f:
            blob = json.load(f)
        if blob.get("feature_version") != FEATURE_VERSION:
            raise ValueError(
                f"feature version mismatch: checkpoint {blob.get('feature_version')} "
                f"vs code {FEATURE_VERSION}"
            )
        self.coef = blob["coef"]
        self.intercept = blob["intercept"]
        self.mean = blob["mean"]
        self.scale = blob["scale"]
        self.val_auc = blob.get("val_auc")
        self.feature_names = blob["feature_names"]
        # Which set-piece type this model was fitted on. Older checkpoints
        # predate the field; those stay unchecked rather than failing to load.
        self.set_piece_type = blob.get("set_piece_type")

    def __call__(self, scenario):
        # Scoring a free kick with corner weights is silent and plausible-looking
        # -- it returns a number in [0, 1] that is simply wrong. Refuse instead.
        if (
            self.set_piece_type is not None
            and scenario.get("set_piece_type") != self.set_piece_type
        ):
            raise ValueError(
                f"this scorer was trained on '{self.set_piece_type}' but was "
                f"given a '{scenario.get('set_piece_type')}' scenario; load the "
                f"matching weights or use scorer.get_scorer() to route by type"
            )
        f = scenario_features(scenario)
        z = self.intercept
        for v, m, s, w in zip(f, self.mean, self.scale, self.coef):
            z += w * (v - m) / (s if s else 1.0)
        z = max(min(z, 30.0), -30.0)
        return 1.0 / (1.0 + math.exp(-z))


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------


def _is_shot(label):
    """Same binary collapse Module 1 uses: anything that ends in a shot = 1."""
    return 0 if label == "no-shot" else 1


def load_labelled(path, freekick=False):
    """Return (scenarios, labels, match_ids)."""
    from .contract import from_record

    rows = [json.loads(l) for l in open(path)]
    if freekick:
        rows = [r for r in rows if r.get("set_piece_type") == "freekick_delivery"]
    rows = [r for r in rows if r.get("has_freeze_frame")]

    scen, y, groups = [], [], []
    for r in rows:
        s = from_record(r)
        if not s:
            continue
        try:
            scenario_features(s)
        except ValueError:
            continue
        scen.append(s)
        y.append(_is_shot(r["label"]))
        groups.append(r["match_id"])
    return scen, y, groups


def train(path, freekick=False, out_path=None, seed=42):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    scen, y, groups = load_labelled(path, freekick)
    X = np.array([scenario_features(s) for s in scen])
    y = np.array(y)
    groups = np.array(groups)

    print(f"  {len(y)} scenarios | positives {y.mean():.1%} | {X.shape[1]} features")

    # Grouped by match, so the same game never straddles the split.
    gkf = GroupKFold(n_splits=5)
    aucs = []
    for tr, te in gkf.split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(
            max_iter=2000, class_weight="balanced", C=1.0, random_state=seed
        ).fit(sc.transform(X[tr]), y[tr])
        p = clf.predict_proba(sc.transform(X[te]))[:, 1]
        aucs.append(roc_auc_score(y[te], p))

    mean_auc = float(np.mean(aucs))
    print(f"  5-fold grouped AUC: {mean_auc:.3f} "
          f"(folds: {', '.join(f'{a:.3f}' for a in aucs)})")

    # Refit on everything for the deployed scorer.
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(
        max_iter=2000, class_weight="balanced", C=1.0, random_state=seed
    ).fit(sc.transform(X), y)

    blob = {
        "feature_version": FEATURE_VERSION,
        # Recorded so LearnedScorer can refuse a scenario of the wrong type.
        "set_piece_type": "freekick" if freekick else "corner",
        "feature_names": FEATURE_NAMES,
        "coef": clf.coef_[0].tolist(),
        "intercept": float(clf.intercept_[0]),
        "mean": sc.mean_.tolist(),
        "scale": sc.scale_.tolist(),
        "val_auc": mean_auc,
        "fold_aucs": [float(a) for a in aucs],
        "n_train": int(len(y)),
        "positive_rate": float(y.mean()),
    }

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(blob, f, indent=2)
        print(f"  saved -> {out_path}")

    return blob


if __name__ == "__main__":
    print("Corners")
    train("output/corners_dataset_full.jsonl",
          out_path="models/module2_baseline_corner.json")
    print("\nFree kicks")
    train("output/freekicks_dataset_full.jsonl", freekick=True,
          out_path="models/module2_baseline_freekick.json")