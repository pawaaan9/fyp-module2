"""
The Danger Score interface.

Module 2 only ever calls `scorer(scenario) -> float in [0, 1]`.

Three implementations, best first:
  * GATScorer      -- Module 1's trained GAT. One instance per set-piece type,
                      because corner and free-kick models have different
                      graph-feature dimensions (6 vs 4) and are not
                      interchangeable.
  * LearnedScorer  -- Module 2's logistic baseline (see baseline.py).
  * HeuristicScorer-- no model, no torch. A transparent placeholder.

Swapping between them is one line. Nothing else in Module 2 changes.
"""

import math
import os

from .contract import GOAL, validate
from .graph import CHECKPOINT_PATHS


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
    """Wraps ONE of Module 1's trained models.

    Deliberately single-type. The corner and free-kick checkpoints have
    different `graph_feat_dim`, so a shared instance is not merely unwise --
    the tensors would not even concatenate. Binding one scorer to one type and
    refusing everything else turns that into a clear error at the boundary
    rather than a shape error deep inside the classifier.

    Torch is imported lazily so this file stays importable on a machine without
    torch installed.
    """

    is_trained_model = True

    def __init__(self, checkpoint_path, device=None):
        import torch

        from .graph import build_graph, load_model

        self.torch = torch
        self.build_graph = build_graph
        self.checkpoint_path = str(checkpoint_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model(self.checkpoint_path, self.device)
        self.set_piece_type = getattr(self.model, "set_piece_type", None)
        self.val_auc = None  # Module 1's checkpoints do not record one
        self.name = f"module1-gat-{self.set_piece_type or 'unknown'}"

    def __call__(self, scenario):
        validate(scenario)

        got = scenario.get("set_piece_type")
        if self.set_piece_type and got != self.set_piece_type:
            raise ValueError(
                f"{self.name} was trained on '{self.set_piece_type}' but was "
                f"given a '{got}' scenario."
            )

        data = self.build_graph(scenario).to(self.device)
        with self.torch.no_grad():
            batch = self.torch.zeros(
                data.x.size(0), dtype=self.torch.long, device=self.device
            )
            logit, _, _ = self.model(
                data.x,
                data.edge_index,
                data.edge_attr,
                batch,
                data.graph_feat.to(self.device),
            )
            return float(self.torch.sigmoid(logit).item())


BASELINE_PATH = os.environ.get(
    "PITCHPULSE_BASELINE", "models/module2_baseline_corner.json"
)


class RoutingScorer:
    """One scorer object that dispatches on the scenario's set-piece type.

    Corners and free kicks have separately fitted models, so "the scorer" is
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
    checkpoint_paths=None,
    baseline_path=None,
    verbose=True,
):
    """Three-step fallback chain, best first:

        1. Module 1's trained GAT          (graph structure + learned weights)
        2. Module 2's logistic baseline    (learned weights, no graph)
        3. HeuristicScorer                 (neither -- placeholder only)

    Whichever loads first wins, and the choice is printed so no result is ever
    produced without knowing which scorer made it.

    Both step 1 and step 2 load per-set-piece-type and wrap the result in a
    RoutingScorer, because a corner model scoring a free kick returns a
    plausible-looking wrong number. Pass an explicit `baseline_path` (or set
    PITCHPULSE_BASELINE) to force one specific baseline for everything.
    """

    # An explicit baseline_path (or PITCHPULSE_BASELINE) is an ablation
    # instruction: "score everything with this one model". Honour it before
    # reaching for the GAT, otherwise step 1 always wins and the escape hatch
    # can never actually be used.
    forced_baseline = baseline_path or os.environ.get("PITCHPULSE_BASELINE")

    # ---- step 1: Module 1's trained GAT, one checkpoint per type ----
    if prefer_trained and not forced_baseline:
        paths = checkpoint_paths or CHECKPOINT_PATHS
        loaded_gat, failed = {}, {}

        for sp_type, path in paths.items():
            if not os.path.exists(path):
                failed[sp_type] = "no checkpoint file"
                continue
            try:
                loaded_gat[sp_type] = GATScorer(path)
            except Exception as exc:  # noqa: BLE001
                failed[sp_type] = str(exc)

        if loaded_gat:
            if verbose:
                for sp_type, s in sorted(loaded_gat.items()):
                    print(f"[scorer] {sp_type}: trained GAT from {paths[sp_type]}")
                for sp_type, why in sorted(failed.items()):
                    print(f"[scorer] {sp_type}: GAT unavailable ({why})")

            if not failed:
                return RoutingScorer(loaded_gat)

            # Partial load. Fall through to the baseline for the missing types
            # rather than to the placeholder, since a trained-but-weaker model
            # beats no model at all.
            try:
                from .baseline import WEIGHTS_PATHS, LearnedScorer

                for sp_type in list(failed):
                    wpath = WEIGHTS_PATHS.get(sp_type)
                    if wpath and os.path.exists(wpath):
                        loaded_gat[sp_type] = LearnedScorer(wpath)
                        if verbose:
                            print(
                                f"[scorer] {sp_type}: falling back to logistic "
                                f"baseline from {wpath}"
                            )
                        failed.pop(sp_type)
            except Exception as exc:  # noqa: BLE001
                if verbose:
                    print(f"[scorer] baseline fallback failed ({exc})")

            return RoutingScorer(
                loaded_gat, fallback=HeuristicScorer() if failed else None
            )

        if verbose:
            for sp_type, why in sorted(failed.items()):
                print(f"[scorer] {sp_type}: GAT unavailable ({why})")

    # ---- step 2: Module 2's logistic baseline ----
    if prefer_trained:
        try:
            from .baseline import WEIGHTS_PATHS, LearnedScorer

            if forced_baseline:
                scorer = LearnedScorer(forced_baseline)
                if verbose:
                    print(
                        f"[scorer] using trained logistic baseline from "
                        f"{forced_baseline} (AUC {scorer.val_auc:.3f}) for ALL "
                        f"set-piece types"
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

    # ---- step 3: placeholder ----
    if verbose:
        print(
            "[scorer] using HeuristicScorer -- PLACEHOLDER, not a trained model. "
            "Do not report accuracy figures produced with this."
        )
    return HeuristicScorer()