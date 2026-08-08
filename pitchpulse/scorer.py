"""
The Danger Score interface.

Module 2 only ever calls `scorer(scenario) -> float in [0, 1]`.

Two implementations:
  * HeuristicScorer -- no model, no torch. Use this TODAY so Module 2 can be
    built and demoed before Module 1's GAT is ready.
  * GATScorer       -- loads Chamara's saved checkpoint.

Swapping between them is one line. Nothing else in Module 2 changes.
"""

import math
import os

from .contract import GOAL, validate

CHECKPOINT_PATH = os.environ.get("PITCHPULSE_CHECKPOINT", "models/module1_gat.pt")


def _sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))


class HeuristicScorer:
    """A deliberately simple, transparent stand-in for the GAT.

    It is NOT a trained model and must never be reported as one. It exists so
    that the simulation layer has something sane to consume: it responds in the
    expected direction to the edits a coach would make, which is exactly what
    Module 2's plumbing needs in order to be built and tested.

    Three ingredients, all standard set-piece intuitions:
      1. attackers close to goal raise danger
      2. attackers who are tightly marked lower it
      3. defenders crowding the six-yard box lower it
    """

    name = "heuristic-placeholder"
    is_trained_model = False

    def __init__(self, w_goal=1.5, w_free=1.1, w_crowd=-0.9, bias=-1.6):
        self.w_goal = w_goal
        self.w_free = w_free
        self.w_crowd = w_crowd
        self.bias = bias

    def __call__(self, scenario):
        validate(scenario)
        gx, gy = GOAL
        attackers = [p for p in scenario["players"] if p["teammate"]]
        defenders = [p for p in scenario["players"] if not p["teammate"]]

        def dist(a, b):
            return math.hypot(a["x"] - b["x"], a["y"] - b["y"])

        # 1. attacking presence near the goal (closer = more weight)
        presence = 0.0
        for a in attackers:
            d = math.hypot(a["x"] - gx, a["y"] - gy)
            presence += math.exp(-d / 12.0)
        presence /= max(len(attackers), 1)

        # 2. how free those attackers are from their nearest marker
        freedom = 0.0
        for a in attackers:
            if defenders:
                nearest = min(dist(a, d) for d in defenders)
            else:
                nearest = 20.0
            weight = math.exp(-math.hypot(a["x"] - gx, a["y"] - gy) / 12.0)
            freedom += weight * min(nearest, 10.0) / 10.0
        freedom /= max(len(attackers), 1)

        # 3. defensive crowding of the danger zone
        crowd = sum(
            1 for d in defenders if math.hypot(d["x"] - gx, d["y"] - gy) < 10.0
        ) / 6.0

        z = (
            self.bias
            + self.w_goal * presence
            + self.w_free * freedom
            + self.w_crowd * crowd
        )
        return _sigmoid(z)


class GATScorer:
    """Wraps Module 1's trained model. Torch is imported lazily so that this
    file stays importable on a machine without torch installed."""

    name = "module1-gat"
    is_trained_model = True

    def __init__(self, checkpoint_path=CHECKPOINT_PATH, device=None):
        import torch  # noqa: F401  (lazy on purpose)

        from .graph import build_graph, load_model  # Module 1 owns these

        self.torch = torch
        self.build_graph = build_graph
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model(checkpoint_path, self.device)
        self.model.eval()

    def __call__(self, scenario):
        validate(scenario)
        data = self.build_graph(scenario).to(self.device)
        with self.torch.no_grad():
            batch = self.torch.zeros(
                data.x.size(0), dtype=self.torch.long, device=self.device
            )
            logit, _, _ = self.model(data.x, data.edge_index, data.edge_attr, batch)
            return float(self.torch.sigmoid(logit).item())


BASELINE_PATH = os.environ.get(
    "PITCHPULSE_BASELINE", "models/module2_baseline_corner.json"
)


class RoutingScorer:
    """One scorer object that dispatches on the scenario's set-piece type.

    Corners and free kicks have separately fitted baselines, so "the scorer" is
    really two models. Module 2 hands a single callable to simulate.py, so the
    routing has to hide behind the same `scorer(scenario) -> float` interface
    rather than leak out to every call site.

    A per-type mapping also means a partially trained system degrades honestly:
    types with weights use them, the rest fall back to the placeholder, and
    `is_trained_model` goes False for the whole object so nothing downstream
    reports placeholder output as a model prediction.
    """

    def __init__(self, by_type, fallback=None):
        if not by_type and fallback is None:
            raise ValueError("RoutingScorer needs at least one scorer")
        self.by_type = dict(by_type)
        self.fallback = fallback
        self.is_trained_model = bool(self.by_type) and all(
            s.is_trained_model for s in self.by_type.values()
        ) and (fallback is None or fallback.is_trained_model)
        inner = ", ".join(f"{k}={s.name}" for k, s in sorted(self.by_type.items()))
        if fallback is not None:
            inner += f", *={fallback.name}"
        self.name = f"routing({inner})"

    def scorer_for(self, set_piece_type):
        s = self.by_type.get(set_piece_type, self.fallback)
        if s is None:
            raise ValueError(
                f"no scorer available for set_piece_type '{set_piece_type}'"
            )
        return s

    def __call__(self, scenario):
        return self.scorer_for(scenario.get("set_piece_type"))(scenario)


def get_scorer(
    prefer_trained=True,
    checkpoint_path=CHECKPOINT_PATH,
    baseline_path=None,
    verbose=True,
):
    """Three-step fallback chain, best first:

        1. Module 1's trained GAT          (graph structure + learned weights)
        2. Module 2's logistic baseline    (learned weights, no graph)
        3. HeuristicScorer                 (neither -- placeholder only)

    Whichever loads first wins, and the choice is printed so no result is ever
    produced without knowing which scorer made it.

    At step 2 the per-set-piece-type baselines are loaded and wrapped in a
    RoutingScorer, because a corner model scoring a free kick returns a
    plausible-looking wrong number. Pass an explicit `baseline_path` (or set
    PITCHPULSE_BASELINE) to force one specific model for everything.
    """
    if prefer_trained and os.path.exists(checkpoint_path):
        try:
            scorer = GATScorer(checkpoint_path)
            if verbose:
                print(f"[scorer] using trained GAT from {checkpoint_path}")
            return scorer
        except Exception as exc:  # noqa: BLE001
            if verbose:
                print(f"[scorer] failed to load GAT ({exc}); falling back")

    if prefer_trained:
        try:
            from .baseline import WEIGHTS_PATHS, LearnedScorer

            forced = baseline_path or os.environ.get("PITCHPULSE_BASELINE")
            if forced:
                scorer = LearnedScorer(forced)
                if verbose:
                    print(
                        f"[scorer] using trained logistic baseline from {forced} "
                        f"(AUC {scorer.val_auc:.3f}) for ALL set-piece types"
                    )
                return scorer

            loaded, missing = {}, []
            for sp_type, path in WEIGHTS_PATHS.items():
                if os.path.exists(path):
                    loaded[sp_type] = LearnedScorer(path)
                else:
                    missing.append(sp_type)

            if loaded:
                scorer = RoutingScorer(
                    loaded, fallback=HeuristicScorer() if missing else None
                )
                if verbose:
                    for sp_type, s in sorted(loaded.items()):
                        print(
                            f"[scorer] {sp_type}: trained logistic baseline "
                            f"from {WEIGHTS_PATHS[sp_type]} (AUC {s.val_auc:.3f})"
                        )
                    for sp_type in missing:
                        print(
                            f"[scorer] {sp_type}: NO trained weights at "
                            f"{WEIGHTS_PATHS[sp_type]} -- using PLACEHOLDER heuristic"
                        )
                return scorer
        except Exception as exc:  # noqa: BLE001
            if verbose:
                print(f"[scorer] failed to load baseline ({exc}); falling back")

    if verbose:
        print(
            "[scorer] using HeuristicScorer -- PLACEHOLDER, not a trained model. "
            "Do not report accuracy figures produced with this."
        )
    return HeuristicScorer()