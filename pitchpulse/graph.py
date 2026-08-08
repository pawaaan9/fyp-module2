"""
THE SINGLE SHARED GRAPH BUILDER.  Owned by Module 1 (Chamara).

Module 1's notebook currently has four near-identical graph builders
(cells 28, 44, 54, 58). They must collapse into the one function below.
Module 2 and Module 3 import it from here and never reimplement it --
if the two modules featurise differently, the model silently sees garbage.

Chamara only needs to fill in two functions:

    build_graph(scenario) -> torch_geometric.data.Data
    load_model(path, device) -> nn.Module

and make sure the training notebook calls the SAME build_graph, fed from
contract.from_record(). That single change is what guarantees the score a
coach sees in the dashboard is the score the model was trained to produce.
"""

from .contract import GOAL, PITCH_LENGTH, PITCH_WIDTH, validate

# Bump this whenever the feature layout changes, and save it into the
# checkpoint. If they ever disagree at load time, fail loudly.
FEATURE_VERSION = "1.0"
NODE_FEATURE_NAMES = [
    "x_norm",
    "y_norm",
    "teammate",
    "actor",
    "dist_to_goal_norm",
    "dist_to_nearest_opponent_norm",
]


def node_features(scenario):
    """Plain-python feature extraction, no torch. Useful for the logistic
    regression baseline and for unit tests."""
    validate(scenario)
    gx, gy = GOAL
    players = scenario["players"]
    feats = []
    for p in players:
        others = [q for q in players if q["teammate"] != p["teammate"]]
        if others:
            nearest = min(
                ((q["x"] - p["x"]) ** 2 + (q["y"] - p["y"]) ** 2) ** 0.5 for q in others
            )
        else:
            nearest = 30.0
        dist_goal = ((p["x"] - gx) ** 2 + (p["y"] - gy) ** 2) ** 0.5
        feats.append(
            [
                p["x"] / PITCH_LENGTH,
                p["y"] / PITCH_WIDTH,
                float(p["teammate"]),
                float(p["actor"]),
                dist_goal / PITCH_LENGTH,
                min(nearest, 30.0) / 30.0,
            ]
        )
    return feats


def build_graph(scenario):
    """TODO (Module 1): return a torch_geometric Data object.

    Must be deterministic and must NOT depend on anything outside `scenario` --
    no global dataframes, no notebook state. Module 2 calls this thousands of
    times per optimisation run.
    """
    raise NotImplementedError(
        "Module 1 must implement build_graph(scenario). Until then, Module 2 "
        "runs against scorer.HeuristicScorer, which does not need this."
    )


def load_model(path, device="cpu"):
    """TODO (Module 1): rebuild the architecture and load the state dict.

    Suggested save format on the Module 1 side:

        torch.save({
            "state_dict":       model.state_dict(),
            "arch":             {"node_dim": 6, "hidden_dim": 32, "heads": 4},
            "feature_version":  FEATURE_VERSION,
            "val_auc":          0.63,
        }, "models/module1_gat.pt")
    """
    raise NotImplementedError(
        "Module 1 must implement load_model(path, device) and save a checkpoint."
    )
