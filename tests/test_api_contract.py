"""Offline checks for the API layer that don't need a running server."""

import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SRC = Path(__file__).resolve().parents[1] / "api.py"


def test_api_file_parses():
    ast.parse(SRC.read_text())


def test_six_endpoints_declared():
    tree = ast.parse(SRC.read_text())
    routes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for d in node.decorator_list:
                if isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute):
                    if d.func.attr in ("get", "post", "put", "delete"):
                        routes.append((d.func.attr, d.args[0].value))
    paths = {p for _, p in routes}
    for expected in ["/health", "/scenarios", "/scenarios/{scenario_id}",
                     "/whatif", "/optimise", "/heatmap"]:
        assert expected in paths, f"missing endpoint {expected}"
    assert len(routes) >= 6


def test_displacement_bounds_declared():
    src = SRC.read_text()
    assert "ge=0, le=120" in src
    assert "ge=0, le=80" in src
    assert "MAX_DISPLACEMENT" in src