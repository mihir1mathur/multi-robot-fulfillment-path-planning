"""Service layer called directly (no HTTP), plus a regression check that the
underlying robotics algorithms are untouched."""

from __future__ import annotations

import pytest

from robotics.planning import PLANNERS
from robotics.services.errors import (
    ResourceNotFoundError,
    ValidationFailedError,
)
from robotics.services.planning_service import PlanningService
from robotics.services.robot_service import RobotService
from robotics.services.world import build_warehouse
from robotics.warehouse.grid import Position


def test_robot_service_create_and_get(db_session):
    service = RobotService(db_session)
    service.create_robot("R1", 0, 1, battery_level=75.0)
    record = service.get_robot("R1")
    assert record.battery_level == 75.0


def test_robot_service_missing_raises_not_found(db_session):
    with pytest.raises(ResourceNotFoundError):
        RobotService(db_session).get_robot("NOPE")


def test_robot_service_rejects_impossible_state(db_session):
    with pytest.raises(ValidationFailedError):
        RobotService(db_session).create_robot(
            "R1", 0, 1, current_payload=99.0, payload_capacity=1.0
        )


def test_planning_service_matches_direct_call(db_session):
    outcome = PlanningService(db_session).plan_path(0, 0, 4, 1, algorithm="astar")
    direct = PLANNERS["astar"](Position(0, 0), Position(4, 1), build_warehouse())
    assert outcome.result.total_cost == direct.total_cost
    assert outcome.result.nodes_expanded == direct.nodes_expanded
    assert [(p.row, p.col) for p in outcome.result.path] == [
        (p.row, p.col) for p in direct.path
    ]


def test_direct_planner_still_optimal_and_consistent():
    """Regression: A* and Dijkstra still agree on cost, A* expands no more."""
    warehouse = build_warehouse()
    start, goal = Position(0, 0), Position(11, 10)
    a = PLANNERS["astar"](start, goal, warehouse)
    d = PLANNERS["dijkstra"](start, goal, warehouse)
    assert a.success and d.success
    assert a.total_cost == d.total_cost
    assert a.nodes_expanded <= d.nodes_expanded


def test_service_layer_does_not_import_fastapi():
    """The service layer must not depend on the web framework."""
    import ast
    import pathlib

    services_dir = pathlib.Path(__file__).resolve().parents[1] / "robotics" / "services"
    offenders = []
    for path in services_dir.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            if any(n.split(".")[0] in {"fastapi", "starlette"} for n in names):
                offenders.append(path.name)
    assert offenders == [], f"service modules importing fastapi: {offenders}"


def test_persistence_layer_does_not_import_algorithms():
    """The persistence layer must not depend on planning / optimisation code."""
    import ast
    import pathlib

    persistence_dir = (
        pathlib.Path(__file__).resolve().parents[1] / "robotics" / "persistence"
    )
    banned = ("robotics.planning", "robotics.allocation", "robotics.coordination",
              "robotics.recovery", "robotics.services", "robotics.api")
    offenders = []
    for path in persistence_dir.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            if any(n.startswith(b) for n in names for b in banned):
                offenders.append(path.name)
    assert offenders == [], f"persistence modules importing algorithms: {offenders}"
