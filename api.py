"""
PitchPulse Module 2 — simulation service.

Run from the Code/ folder:
    uvicorn api:app --reload --port 8000

Then open:
    http://127.0.0.1:8000/       coach dashboard
    http://127.0.0.1:8000/docs   interactive API browser
"""

import json
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from pitchpulse.contract import find_player, from_record
from pitchpulse.scorer import get_scorer
from pitchpulse.simulate import (
    MAX_DISPLACEMENT,
    heatmap_sweep,
    joint_optimise,
    optimise_player,
    what_if,
)

DATA_DIR = Path(__file__).parent / "output"
UI_FILE = Path(__file__).parent / "ui.html"

app = FastAPI(
    title="PitchPulse Module 2",
    description="Counterfactual what-if simulation for football set pieces",
    version="1.0",
)

SCORER = get_scorer()
SCENARIOS = {}


def _load():
    """Load corners and free-kick deliveries into an id-keyed store."""
    store = {}

    def add(path, key, freekick=False):
        if not path.exists():
            return
        rows = [json.loads(l) for l in open(path)]
        if freekick:
            rows = [r for r in rows if r.get("set_piece_type") == "freekick_delivery"]
        rows = [r for r in rows if r.get("has_freeze_frame")]
        for i, r in enumerate(rows):
            s = from_record(r)
            if s:
                s["scenario_id"] = f"{key}-{i}"
                store[s["scenario_id"]] = s

    add(DATA_DIR / "corners_dataset_full.jsonl", "corner")
    add(DATA_DIR / "freekicks_dataset_full.jsonl", "freekick", freekick=True)
    return store


@app.on_event("startup")
def startup():
    SCENARIOS.update(_load())
    print(f"[api] {len(SCENARIOS)} scenarios loaded | scorer: {SCORER.name}")
    if not SCENARIOS:
        # An empty store looks identical to a working board with nothing on it.
        # Say why, rather than letting the dashboard render a blank pitch.
        print(
            f"[api] WARNING: no scenarios found in {DATA_DIR}. Expected "
            "corners_dataset_full.jsonl / freekicks_dataset_full.jsonl there."
        )
    print(f"[api] dashboard: http://127.0.0.1:8000/")


# ----------------------------- schemas -----------------------------


class WhatIfRequest(BaseModel):
    scenario_id: str = Field(..., description="e.g. 'freekick-0'")
    player_id: int = Field(..., description="player to move")
    x: float = Field(..., ge=0, le=120, description="target x in StatsBomb units")
    y: float = Field(..., ge=0, le=80, description="target y in StatsBomb units")
    max_displacement: float = Field(MAX_DISPLACEMENT, gt=0, le=20)


class OptimiseRequest(BaseModel):
    scenario_id: str
    player_ids: List[int] = Field(..., description="1-6 players to optimise")
    maximise: bool = Field(True, description="True = attacking edit, False = defending")
    max_displacement: float = Field(MAX_DISPLACEMENT, gt=0, le=20)
    steps: int = Field(11, ge=3, le=25, description="lattice resolution per axis")
    rounds: int = Field(2, ge=1, le=5, description="coordinate-ascent passes")


class HeatmapRequest(BaseModel):
    scenario_id: str
    player_id: int
    max_displacement: float = Field(MAX_DISPLACEMENT, gt=0, le=20)
    steps: int = Field(11, ge=3, le=25)


def _get(scenario_id: str):
    s = SCENARIOS.get(scenario_id)
    if s is None:
        raise HTTPException(404, f"unknown scenario_id '{scenario_id}'")
    return s


def _check_player(scen, player_id):
    try:
        return find_player(scen, player_id)
    except KeyError:
        raise HTTPException(404, f"player {player_id} not in scenario")


# ----------------------------- dashboard -----------------------------


@app.get("/", include_in_schema=False)
def dashboard():
    """Serve the coach board from the same origin as the API, so the
    browser never treats the fetch calls as cross-origin."""
    if not UI_FILE.exists():
        raise HTTPException(
            404,
            "ui.html not found next to api.py. Put the dashboard file in the "
            "same folder as this script.",
        )
    return FileResponse(UI_FILE, media_type="text/html")


# ----------------------------- endpoints -----------------------------


@app.get("/health")
def health():
    """Service status and — importantly — which scorer is active."""
    return {
        "status": "ok",
        "scenarios_loaded": len(SCENARIOS),
        "scorer": SCORER.name,
        "trained_model": SCORER.is_trained_model,
        "warning": None
        if SCORER.is_trained_model
        else "placeholder scorer active; scores are not model predictions",
        "max_displacement_yards": MAX_DISPLACEMENT,
    }


@app.get("/scenarios")
def list_scenarios(set_piece_type: Optional[str] = None, limit: int = 50):
    """List available scenarios, optionally filtered to 'corner' or 'freekick'."""
    items = list(SCENARIOS.values())
    if set_piece_type:
        items = [s for s in items if s["set_piece_type"] == set_piece_type]
    return {
        "total": len(items),
        "returned": min(limit, len(items)),
        "scenarios": [
            {
                "scenario_id": s["scenario_id"],
                "set_piece_type": s["set_piece_type"],
                "n_players": len(s["players"]),
            }
            for s in items[:limit]
        ],
    }


@app.get("/scenarios/{scenario_id}")
def get_scenario(scenario_id: str):
    """Full player arrangement plus its baseline Danger Score."""
    s = _get(scenario_id)
    return {
        "scenario_id": s["scenario_id"],
        "set_piece_type": s["set_piece_type"],
        "delivery_end": s["delivery_end"],
        "players": s["players"],
        "danger_score": SCORER(s),
    }


@app.post("/whatif")
def post_whatif(req: WhatIfRequest):
    """Move one player to a target position and re-score the arrangement."""
    s = _get(req.scenario_id)
    _check_player(s, req.player_id)

    r = what_if(s, req.player_id, req.x, req.y, SCORER, req.max_displacement)
    r.pop("scenario", None)
    r["scenario_id"] = req.scenario_id
    return r


@app.post("/optimise")
def post_optimise(req: OptimiseRequest):
    """Grid search for the best position. One player = single search,
    several players = joint coordinate ascent."""
    s = _get(req.scenario_id)
    for pid in req.player_ids:
        _check_player(s, pid)

    if len(req.player_ids) == 1:
        r = optimise_player(
            s, req.player_ids[0], SCORER, req.maximise,
            req.max_displacement, req.steps,
        )
    else:
        r = joint_optimise(
            s, req.player_ids, SCORER, req.maximise,
            req.max_displacement, req.steps, req.rounds,
        )

    r["scenario_id"] = req.scenario_id
    r["adjusted_players"] = [
        {"id": p["id"], "x": p["x"], "y": p["y"]}
        for p in r["scenario"]["players"]
        if p["id"] in req.player_ids
    ]
    r.pop("scenario", None)
    return r


@app.post("/heatmap")
def post_heatmap(req: HeatmapRequest):
    """Score every candidate cell around a player — drives the UI heatmap."""
    s = _get(req.scenario_id)
    _check_player(s, req.player_id)

    hm = heatmap_sweep(s, req.player_id, SCORER, req.max_displacement, req.steps)
    scores = [c["score"] for c in hm["cells"]]
    hm["scenario_id"] = req.scenario_id
    hm["baseline"] = SCORER(s)
    hm["min_score"] = min(scores)
    hm["max_score"] = max(scores)
    hm["n_cells"] = len(scores)
    return hm