"""
Module 2 core: counterfactual editing of a set-piece scenario.

Pure functions. No web server, no model, no torch. The scorer is passed
in as an argument, so this file works identically with the heuristic
placeholder and with Chamara's trained GAT.
"""

import math

from .contract import (
    PITCH_LENGTH,
    PITCH_WIDTH,
    copy_scenario,
    find_player,
    validate,
)

MAX_DISPLACEMENT = 5.0  # yards, per the project spec


def clamp_to_pitch(x, y, margin=0.1):
    """Keep a player inside the pitch boundary."""
    x = min(max(x, margin), PITCH_LENGTH - margin)
    y = min(max(y, margin), PITCH_WIDTH - margin)
    return x, y


def clamp_displacement(dx, dy, max_disp=MAX_DISPLACEMENT):
    """Shrink an edit that exceeds the allowed radius, keeping its direction."""
    dist = math.hypot(dx, dy)
    if dist <= max_disp or dist == 0.0:
        return dx, dy
    scale = max_disp / dist
    return dx * scale, dy * scale


def apply_edit(scenario, player_id, dx, dy, max_disp=MAX_DISPLACEMENT):
    """Return a NEW scenario with one player moved by (dx, dy).
    The input scenario is never mutated."""
    edited = copy_scenario(scenario)
    player = find_player(edited, player_id)

    dx, dy = clamp_displacement(dx, dy, max_disp)
    player["x"], player["y"] = clamp_to_pitch(player["x"] + dx, player["y"] + dy)

    validate(edited)
    return edited


def move_to(scenario, player_id, new_x, new_y, max_disp=MAX_DISPLACEMENT):
    """Same as apply_edit but expressed as an absolute target position --
    this is what the drag-and-drop UI sends."""
    player = find_player(scenario, player_id)
    return apply_edit(
        scenario, player_id, new_x - player["x"], new_y - player["y"], max_disp
    )


def what_if(scenario, player_id, new_x, new_y, scorer, max_disp=MAX_DISPLACEMENT):
    """The headline Module 2 operation: score before, score after, delta."""
    before = scorer(scenario)
    edited = move_to(scenario, player_id, new_x, new_y, max_disp)
    after = scorer(edited)
    moved = find_player(edited, player_id)
    return {
        "player_id": player_id,
        "requested_position": [new_x, new_y],
        "applied_position": [moved["x"], moved["y"]],
        "was_clamped": abs(moved["x"] - new_x) > 1e-6 or abs(moved["y"] - new_y) > 1e-6,
        "score_before": before,
        "score_after": after,
        "delta": after - before,
        "scenario": edited,
    }


def candidate_grid(player, max_disp=MAX_DISPLACEMENT, steps=11):
    """The 11 x 11 = 121 lattice of candidate positions around a player."""
    if steps < 2:
        raise ValueError("steps must be >= 2")
    span = [-max_disp + 2 * max_disp * i / (steps - 1) for i in range(steps)]
    out = []
    for dx in span:
        for dy in span:
            if math.hypot(dx, dy) > max_disp + 1e-9:
                continue  # keep the edit inside the circle, not the square
            out.append(clamp_to_pitch(player["x"] + dx, player["y"] + dy))
    return out


def optimise_player(
    scenario, player_id, scorer, maximise=True, max_disp=MAX_DISPLACEMENT, steps=11
):
    """Grid search over candidate positions for one player.

    maximise=True  -> attacking edit, find the highest Danger Score
    maximise=False -> defensive edit, find the lowest
    """
    baseline = scorer(scenario)
    player = find_player(scenario, player_id)
    origin = (player["x"], player["y"])

    best_scenario = scenario
    best_score = baseline
    best_pos = origin
    evaluated = 0

    for cx, cy in candidate_grid(player, max_disp, steps):
        candidate = move_to(scenario, player_id, cx, cy, max_disp)
        score = scorer(candidate)
        evaluated += 1
        better = score > best_score if maximise else score < best_score
        if better:
            best_score, best_scenario, best_pos = score, candidate, (cx, cy)

    return {
        "player_id": player_id,
        "maximise": maximise,
        "original_position": list(origin),
        "best_position": list(best_pos),
        "baseline_score": baseline,
        "best_score": best_score,
        "improvement": (best_score - baseline) if maximise else (baseline - best_score),
        "candidates_evaluated": evaluated,
        "improvement_found": best_pos != origin,
        "scenario": best_scenario,
    }


def joint_optimise(
    scenario,
    player_ids,
    scorer,
    maximise=True,
    max_disp=MAX_DISPLACEMENT,
    steps=11,
    rounds=2,
):
    """Coordinate ascent: optimise each player in turn, repeat for `rounds`
    passes, so later players can react to earlier moves."""
    current = scenario
    baseline = scorer(scenario)
    history = []

    for r in range(rounds):
        for pid in player_ids:
            result = optimise_player(current, pid, scorer, maximise, max_disp, steps)
            current = result["scenario"]
            history.append(
                {
                    "round": r,
                    "player_id": pid,
                    "position": result["best_position"],
                    "score": result["best_score"],
                }
            )

    final = scorer(current)
    return {
        "player_ids": list(player_ids),
        "baseline_score": baseline,
        "final_score": final,
        "improvement": (final - baseline) if maximise else (baseline - final),
        "history": history,
        "scenario": current,
    }


def heatmap_sweep(scenario, player_id, scorer, max_disp=MAX_DISPLACEMENT, steps=11):
    """Score every candidate position so the UI can render a danger heatmap
    around the selected player."""
    player = find_player(scenario, player_id)
    cells = []
    for cx, cy in candidate_grid(player, max_disp, steps):
        cells.append(
            {"x": cx, "y": cy, "score": scorer(move_to(scenario, player_id, cx, cy, max_disp))}
        )
    return {
        "player_id": player_id,
        "origin": [player["x"], player["y"]],
        "cells": cells,
    }
