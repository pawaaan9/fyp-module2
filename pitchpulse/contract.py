"""
PitchPulse shared data contract  (schema v1.0)

This file defines the ONE format that Module 1, 2 and 3 all agree on.
Nobody invents their own dict shape. If this file changes, all three
modules change together.

A "scenario" is a single set-piece arrangement:

{
  "scenario_id":    str,
  "set_piece_type": "corner" | "freekick",
  "delivery_end":   [x, y] or None,      # where the ball is delivered to
  "players": [
      {"id": 0, "x": 105.8, "y": 43.6, "teammate": True,
       "actor": False, "keeper": False},
      ...
  ]
}

Coordinates are RAW StatsBomb pitch units: x in [0, 120], y in [0, 80].
The attacking goal is at x = 120. Normalisation happens inside graph.py,
never here -- so Module 2 can move players in real yards.
"""

from copy import deepcopy

PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0
GOAL = (120.0, 40.0)

SCHEMA_VERSION = "1.0"

# StatsBomb 360 occasionally reports a player a fraction of a yard behind the
# goal line or touchline. Tolerate that on input rather than rejecting the row.
OFF_PITCH_TOLERANCE = 2.0


def make_player(pid, x, y, teammate, actor=False, keeper=False):
    return {
        "id": int(pid),
        "x": float(x),
        "y": float(y),
        "teammate": bool(teammate),
        "actor": bool(actor),
        "keeper": bool(keeper),
    }


def from_record(record, scenario_id=None):
    """Convert one row of corners_dataset_full.jsonl / freekicks_dataset_full.jsonl
    into a scenario dict. Returns None if the row has no freeze frame."""
    ff = record.get("freeze_frame")
    if not ff:
        return None

    players = [
        make_player(
            pid=i,
            x=min(max(p["location"][0], 0.0), PITCH_LENGTH),
            y=min(max(p["location"][1], 0.0), PITCH_WIDTH),
            teammate=p.get("teammate", False),
            actor=p.get("actor", False),
            keeper=p.get("keeper", False),
        )
        for i, p in enumerate(ff)
    ]

    is_corner = "corner_event_id" in record
    return {
        "scenario_id": scenario_id
        or record.get("corner_event_id")
        or record.get("event_id"),
        "set_piece_type": "corner" if is_corner else "freekick",
        "delivery_end": record.get("pass_end_location"),
        "players": players,
    }


def validate(scenario):
    """Raise ValueError if the scenario is malformed. Call this at every
    API boundary -- it turns silent wrong answers into loud errors."""
    if scenario.get("set_piece_type") not in ("corner", "freekick"):
        raise ValueError("set_piece_type must be 'corner' or 'freekick'")

    players = scenario.get("players")
    if not players:
        raise ValueError("scenario has no players")

    seen = set()
    for p in players:
        for key in ("id", "x", "y", "teammate"):
            if key not in p:
                raise ValueError(f"player missing required field '{key}': {p}")
        if p["id"] in seen:
            raise ValueError(f"duplicate player id {p['id']}")
        seen.add(p["id"])
        tol = OFF_PITCH_TOLERANCE
        if not (-tol <= p["x"] <= PITCH_LENGTH + tol and -tol <= p["y"] <= PITCH_WIDTH + tol):
            raise ValueError(f"player {p['id']} is off the pitch: {p['x']}, {p['y']}")

    if not any(p["teammate"] for p in players):
        raise ValueError("scenario has no attacking players")
    return True


def copy_scenario(scenario):
    return deepcopy(scenario)


def find_player(scenario, player_id):
    for p in scenario["players"]:
        if p["id"] == player_id:
            return p
    raise KeyError(f"no player with id {player_id}")
