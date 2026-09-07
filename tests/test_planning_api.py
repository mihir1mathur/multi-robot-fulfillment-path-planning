"""Path-planning endpoint - and that it agrees with the direct planner."""

from __future__ import annotations

from robotics.planning import PLANNERS
from robotics.services.world import build_warehouse
from robotics.warehouse.grid import Position


def _plan(client, start, goal, algorithm="astar", **extra):
    return client.post(
        "/planning/path",
        json={
            "start": {"row": start[0], "col": start[1]},
            "goal": {"row": goal[0], "col": goal[1]},
            "algorithm": algorithm,
            **extra,
        },
    )


def test_astar_success_matches_direct_planner(api_client):
    response = _plan(api_client, (0, 0), (4, 1), "astar")
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True

    direct = PLANNERS["astar"](Position(0, 0), Position(4, 1), build_warehouse())
    assert body["total_cost"] == direct.total_cost
    assert body["nodes_expanded"] == direct.nodes_expanded
    assert len(body["path"]) == len(direct.path)


def test_dijkstra_success(api_client):
    response = _plan(api_client, (0, 0), (0, 5), "dijkstra")
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["algorithm"] == "dijkstra"
    assert body["path"][0] == {"row": 0, "col": 0}
    assert body["path"][-1] == {"row": 0, "col": 5}


def test_start_equals_goal_is_zero_cost(api_client):
    body = _plan(api_client, (3, 0), (3, 0)).json()
    assert body["success"] is True
    assert body["total_cost"] == 0.0
    assert body["path"] == [{"row": 3, "col": 0}]


def test_unreachable_goal_is_a_200_domain_failure(api_client):
    # Wall a 4x4 warehouse goal into its own corner.
    response = api_client.post(
        "/planning/path",
        json={
            "start": {"row": 0, "col": 0},
            "goal": {"row": 0, "col": 3},
            "algorithm": "astar",
            "warehouse": {
                "width": 4,
                "height": 4,
                "static_obstacles": [{"row": 0, "col": 2}, {"row": 1, "col": 3}],
            },
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is False
    assert body["failure_reason"]
    assert body["path"] == []


def test_dynamic_obstacle_forces_a_detour(api_client):
    direct = _plan(api_client, (4, 0), (4, 4), "astar").json()
    detoured = _plan(
        api_client,
        (4, 0),
        (4, 4),
        "astar",
        dynamic_obstacles=[{"row": 4, "col": 2}],
    ).json()
    assert detoured["success"] is True
    assert detoured["total_cost"] > direct["total_cost"]
    assert {"row": 4, "col": 2} not in detoured["path"]


def test_invalid_coordinate_is_422(api_client):
    response = api_client.post(
        "/planning/path",
        json={"start": {"row": -1, "col": 0}, "goal": {"row": 4, "col": 1}},
    )
    assert response.status_code == 422


def test_unsupported_algorithm_is_422(api_client):
    response = api_client.post(
        "/planning/path",
        json={
            "start": {"row": 0, "col": 0},
            "goal": {"row": 1, "col": 0},
            "algorithm": "bfs",
        },
    )
    assert response.status_code == 422


def test_out_of_bounds_goal_is_422(api_client):
    response = _plan(api_client, (0, 0), (99, 99))
    assert response.status_code == 422
    assert response.json()["error"] == "unprocessable"


def test_planning_run_is_persisted(api_client, db_session):
    from robotics.persistence.repositories import RunRepository

    _plan(api_client, (0, 0), (4, 1), "astar")
    runs = RunRepository(db_session).recent_planning_runs()
    assert len(runs) == 1
    assert runs[0].algorithm == "astar"
    assert runs[0].success is True
