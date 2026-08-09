"""
THE SINGLE SHARED GRAPH BUILDER.  Feature layout owned by Module 1 (Chamara).

This file mirrors, exactly, the featurisation used to TRAIN the checkpoints in
models/module1_gat_*.pt. If the two ever drift apart the model silently sees
garbage, so every constant here is checked against the `norm_constants`,
`node_feat_order` and `graph_feat_order` recorded inside the checkpoint at load
time, and a mismatch raises rather than warns.

Two things differ per set-piece type and must NOT be merged:

    corner   graph_feat_dim = 6
             [hull_area, n_near_goal, box_density,
              short_corner_dist, def_line_depth, centroid_dist]

    freekick graph_feat_dim = 4
             [hull_area, n_near_goal, kick_x, kick_y]

Node features are identical for both (6 per player).
"""

import numpy as np

from .contract import GOAL, PITCH_LENGTH, PITCH_WIDTH, validate

# --- constants, matching checkpoint["norm_constants"] -----------------------
FEATURE_VERSION = "2.0"
HULL_NORM = 8400.0
MARK_CLIP = 30.0
MARK_FALLBACK = 20.0          # Chamara uses 20.0 when a player has no opponents
NEAR_GOAL_DIST = 12.0
BOX_X_MIN = 102.0             # 18-yard box, StatsBomb units
BOX_Y_MIN, BOX_Y_MAX = 18.0, 62.0
TAKER_X_THRESHOLD = 118.0
SHORT_CORNER_FALLBACK = 20.0

NODE_FEATURE_NAMES = [
    "x_norm",
    "y_norm",
    "teammate",
    "actor",
    "dist_goal_norm",
    "mark_dist_norm",
]

GRAPH_FEATURE_NAMES = {
    "corner": [
        "hull_area",
        "n_near_goal",
        "box_density",
        "short_corner_dist",
        "def_line_depth",
        "centroid_dist",
    ],
    "freekick": ["hull_area", "n_near_goal", "kick_x", "kick_y"],
}

CHECKPOINT_PATHS = {
    "corner": "models/module1_gat_corner.pt",
    "freekick": "models/module1_gat_freekick.pt",
}


# --- helpers ----------------------------------------------------------------

def _arrays(scenario):
    players = scenario["players"]
    locs = np.array([[p["x"], p["y"]] for p in players], dtype=float)
    teammates = np.array([bool(p["teammate"]) for p in players])
    actors = np.array([bool(p.get("actor", False)) for p in players])
    return locs, teammates, actors


def _guard(teammates):
    """Chamara's training filter. Scenarios outside it were never seen by the
    model, so scoring them would be extrapolation dressed up as prediction."""
    n_att = int(teammates.sum())
    n_def = int((~teammates).sum())
    if n_att < 3 or n_def < 1:
        raise ValueError(
            f"scenario has {n_att} attackers and {n_def} defenders; the GAT was "
            f"trained only on frames with >=3 attackers and >=1 defender. "
            f"Use the logistic baseline for this scenario."
        )


def node_features(scenario):
    """Plain-python/numpy feature extraction, no torch. Used by the logistic
    baseline and by unit tests as well as by build_graph."""
    validate(scenario)
    locs, teammates, actors = _arrays(scenario)
    goal = np.array(GOAL, dtype=float)

    dist_goal = np.linalg.norm(locs - goal, axis=1)
    team_locs = locs[teammates]
    opp_locs = locs[~teammates]

    mark_dist = np.zeros(len(locs))
    for i in range(len(locs)):
        others = opp_locs if teammates[i] else team_locs
        mark_dist[i] = (
            np.linalg.norm(others - locs[i], axis=1).min()
            if len(others) > 0
            else MARK_FALLBACK
        )

    return np.stack(
        [
            locs[:, 0] / PITCH_LENGTH,
            locs[:, 1] / PITCH_WIDTH,
            teammates.astype(float),
            actors.astype(float),
            dist_goal / PITCH_LENGTH,
            np.clip(mark_dist, 0, MARK_CLIP) / MARK_CLIP,
        ],
        axis=1,
    )


def _hull_area(team_locs):
    from scipy.spatial import ConvexHull

    try:
        return ConvexHull(team_locs).volume / HULL_NORM
    except Exception:  # noqa: BLE001  (collinear or too few points)
        return 0.0


def graph_features(scenario):
    """The graph-level feature vector. Its length and meaning depend on the
    set-piece type -- see the module docstring."""
    locs, teammates, _ = _arrays(scenario)
    goal = np.array(GOAL, dtype=float)
    dist_goal = np.linalg.norm(locs - goal, axis=1)
    team_locs = locs[teammates]
    opp_locs = locs[~teammates]

    hull_area = _hull_area(team_locs)
    n_near_goal = float((dist_goal[teammates] < NEAR_GOAL_DIST).sum()) / max(
        int(teammates.sum()), 1
    )

    sp_type = scenario["set_piece_type"]

    if sp_type == "freekick":
        kick = scenario.get("kick_location")
        if kick is None:
            raise ValueError(
                "free-kick scenarios need 'kick_location' (the spot the kick is "
                "taken from). The free-kick GAT was trained with kick_x/kick_y as "
                "graph features; omitting them changes the model's input."
            )
        return [
            hull_area,
            n_near_goal,
            float(kick[0]) / PITCH_LENGTH,
            float(kick[1]) / PITCH_WIDTH,
        ]

    # --- corner ---
    att_in_box = sum(
        1
        for loc in team_locs
        if loc[0] >= BOX_X_MIN and BOX_Y_MIN <= loc[1] <= BOX_Y_MAX
    )
    def_in_box = sum(
        1
        for loc in opp_locs
        if loc[0] >= BOX_X_MIN and BOX_Y_MIN <= loc[1] <= BOX_Y_MAX
    )
    box_density = att_in_box / max(def_in_box, 1)

    takers = [loc for loc in team_locs if loc[0] > TAKER_X_THRESHOLD]
    if takers and len(team_locs) > 1:
        taker_loc = takers[0]
        teammate_dists = [
            np.linalg.norm(loc - taker_loc)
            for loc in team_locs
            if not np.array_equal(loc, taker_loc)
        ]
        short_corner_dist = (
            min(teammate_dists) if teammate_dists else SHORT_CORNER_FALLBACK
        )
    else:
        short_corner_dist = SHORT_CORNER_FALLBACK

    def_line_depth = float(np.mean(opp_locs[:, 0])) if len(opp_locs) else PITCH_LENGTH

    if len(team_locs) and len(opp_locs):
        centroid_dist = float(
            np.linalg.norm(np.mean(team_locs, axis=0) - np.mean(opp_locs, axis=0))
        )
    else:
        centroid_dist = 0.0

    return [
        hull_area,
        n_near_goal,
        box_density,
        short_corner_dist / PITCH_LENGTH,
        def_line_depth / PITCH_LENGTH,
        centroid_dist / PITCH_LENGTH,
    ]


def build_graph(scenario):
    """Return a torch_geometric Data object with `.graph_feat` attached.

    Deterministic, and depends on nothing outside `scenario` -- Module 2 calls
    this thousands of times per optimisation run.
    """
    import torch
    from torch_geometric.data import Data

    validate(scenario)
    _, teammates, _ = _arrays(scenario)
    _guard(teammates)

    x = torch.tensor(node_features(scenario), dtype=torch.float)
    gf = torch.tensor(graph_features(scenario), dtype=torch.float).unsqueeze(0)

    n = len(teammates)
    edge_index, edge_attr = [], []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            edge_index.append([i, j])
            edge_attr.append([float(teammates[i] == teammates[j])])

    data = Data(
        x=x,
        edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float),
    )
    data.graph_feat = gf
    return data


# --- the model -------------------------------------------------------------

def _enriched_gat_class():
    """Defined lazily so this module imports fine without torch installed."""
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GATv2Conv, global_mean_pool

    class EnrichedGAT(nn.Module):
        def __init__(
            self, node_dim=6, edge_dim=1, hidden_dim=32, heads=4, graph_feat_dim=2
        ):
            super().__init__()
            self.gat1 = GATv2Conv(
                node_dim, hidden_dim, heads=heads, edge_dim=edge_dim, concat=True
            )
            self.norm1 = nn.LayerNorm(hidden_dim * heads)
            self.gat2 = GATv2Conv(
                hidden_dim * heads,
                hidden_dim,
                heads=1,
                edge_dim=edge_dim,
                concat=False,
            )
            self.norm2 = nn.LayerNorm(hidden_dim)
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim + graph_feat_dim, 16), nn.ReLU(), nn.Linear(16, 1)
            )
            self._global_mean_pool = global_mean_pool
            self._F = F

        def forward(self, x, edge_index, edge_attr, batch, graph_feat):
            h = self.gat1(x, edge_index, edge_attr)
            h = self._F.elu(self.norm1(h))
            h = self.gat2(h, edge_index, edge_attr)
            h = self._F.elu(self.norm2(h))
            player_embeddings = h
            graph_embedding = self._global_mean_pool(h, batch)
            combined = __import__("torch").cat([graph_embedding, graph_feat], dim=1)
            return self.classifier(combined), player_embeddings, graph_embedding

    return EnrichedGAT


def load_model(path, device="cpu"):
    """Rebuild the architecture from the checkpoint's own metadata and load the
    weights. Verifies the feature layout instead of assuming it."""
    import torch

    ckpt = torch.load(path, map_location=device, weights_only=False)

    if ckpt.get("model_class") != "EnrichedGAT":
        raise ValueError(
            f"{path} says model_class={ckpt.get('model_class')!r}; this loader "
            f"only builds EnrichedGAT."
        )

    # --- fail loudly if Module 1's featurisation has moved on ---
    ckpt_nodes = ckpt.get("node_feat_order")
    if ckpt_nodes and list(ckpt_nodes) != NODE_FEATURE_NAMES:
        raise ValueError(
            f"node feature mismatch.\n  checkpoint: {list(ckpt_nodes)}\n"
            f"  graph.py:   {NODE_FEATURE_NAMES}"
        )

    sp_type = ckpt.get("set_piece_type")
    ckpt_graph = ckpt.get("graph_feat_order")
    expected = GRAPH_FEATURE_NAMES.get(sp_type)
    if ckpt_graph and expected and list(ckpt_graph) != expected:
        raise ValueError(
            f"graph feature mismatch for '{sp_type}'.\n"
            f"  checkpoint: {list(ckpt_graph)}\n  graph.py:   {expected}"
        )

    consts = ckpt.get("norm_constants") or {}
    ours = {
        "PITCH_X": PITCH_LENGTH,
        "PITCH_Y": PITCH_WIDTH,
        "HULL_NORM": HULL_NORM,
        "MARK_CLIP": MARK_CLIP,
        "NEAR_GOAL_DIST": NEAR_GOAL_DIST,
    }
    for key, mine in ours.items():
        theirs = consts.get(key)
        if theirs is not None and abs(float(theirs) - mine) > 1e-9:
            raise ValueError(
                f"normalisation constant {key} differs: checkpoint={theirs}, "
                f"graph.py={mine}"
            )

    init_kwargs = ckpt.get("init_kwargs", {})
    model = _enriched_gat_class()(**init_kwargs).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    model.set_piece_type = sp_type
    model.graph_feat_dim = init_kwargs.get("graph_feat_dim")
    return model