"""Multi-robot coordination endpoint."""

from __future__ import annotations

from robotics.coordination.conflicts import find_coordination_problems
from robotics.coordination.timed_path import TimedPath, TimedStep
from robotics.services.world import build_warehouse


def _timed_paths_from_response(body):
    paths = {}
    for rid, route in body["routes"].items():
        if not route["success"]:
            continue
        steps = [
            TimedStep(_pos(s["row"], s["col"]), s["timestep"]) for s in route["steps"]
        ]
        paths[rid] = TimedPath.found(rid, steps)
    return paths


def _pos(row, col):
    from robotics.warehouse.grid import Position

    return Position(row, col)


def test_successful_coordination_is_conflict_free(api_client):
    response = api_client.post(
        "/coordination/plan",
        json={
            "robots": [
                {"robot_id": "R1", "start": {"row": 0, "col": 0}, "goal": {"row": 0, "col": 4}},
                {"robot_id": "R2", "start": {"row": 1, "col": 2}, "goal": {"row": 0, "col": 2}},
            ]
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    problems = find_coordination_problems(
        _timed_paths_from_response(body), build_warehouse(), check_obstacles=False
    )
    assert problems == []


def test_vertex_conflict_is_resolved_with_a_wait(api_client):
    # Plus-shaped junction (corners blocked): the only free cells form a +, so
    # R1 (horizontal) and R2 (vertical) both must pass through the centre and
    # there is no room to detour - the lower-priority robot WAITs.
    body = api_client.post(
        "/coordination/plan",
        json={
            "robots": [
                {"robot_id": "R1", "start": {"row": 1, "col": 0}, "goal": {"row": 1, "col": 2}},
                {"robot_id": "R2", "start": {"row": 0, "col": 1}, "goal": {"row": 2, "col": 1}},
            ],
            "warehouse": {
                "width": 3,
                "height": 3,
                "static_obstacles": [
                    {"row": 0, "col": 0},
                    {"row": 0, "col": 2},
                    {"row": 2, "col": 0},
                    {"row": 2, "col": 2},
                ],
            },
        },
    ).json()
    assert body["success"] is True
    assert body["total_wait_steps"] >= 1
    junction = _build_plus_warehouse()
    assert find_coordination_problems(
        _timed_paths_from_response(body), junction, check_obstacles=False
    ) == []


def _build_plus_warehouse():
    from robotics.warehouse.warehouse import Warehouse

    warehouse = Warehouse(width=3, height=3, name="plus")
    for r, c in ((0, 0), (0, 2), (2, 0), (2, 2)):
        warehouse.add_static_obstacle(f"x{r}{c}", _pos(r, c), "corner")
    return warehouse


def test_edge_swap_is_handled(api_client):
    # Head-on in a one-wide corridor: one robot is honestly reported unroutable.
    body = api_client.post(
        "/coordination/plan",
        json={
            "robots": [
                {"robot_id": "R1", "start": {"row": 0, "col": 0}, "goal": {"row": 0, "col": 2}},
                {"robot_id": "R2", "start": {"row": 0, "col": 2}, "goal": {"row": 0, "col": 0}},
            ],
            "warehouse": {"width": 3, "height": 1},
        },
    ).json()
    # exactly one robot planned, the other reported failed - no swap in the plan
    assert body["planned_robot_count"] == 1
    assert len(body["failed_robot_ids"]) == 1
    assert body["success"] is False


def test_prioritized_planning_failure_is_reported_honestly(api_client):
    body = api_client.post(
        "/coordination/plan",
        json={
            "robots": [
                {"robot_id": "R1", "start": {"row": 0, "col": 0}, "goal": {"row": 0, "col": 3}},
                {"robot_id": "R2", "start": {"row": 0, "col": 3}, "goal": {"row": 0, "col": 0}},
            ],
            "warehouse": {"width": 4, "height": 1},
        },
    ).json()
    assert body["success"] is False
    assert body["failure_reason"]


def test_bad_priority_order_is_422(api_client):
    response = api_client.post(
        "/coordination/plan",
        json={
            "robots": [
                {"robot_id": "R1", "start": {"row": 0, "col": 0}, "goal": {"row": 0, "col": 1}}
            ],
            "priority_order": ["R1", "R2"],
        },
    )
    assert response.status_code == 422


def test_empty_fleet_is_422(api_client):
    assert api_client.post("/coordination/plan", json={"robots": []}).status_code == 422


def test_coordination_run_is_persisted(api_client, db_session):
    from robotics.persistence.repositories import RunRepository

    api_client.post(
        "/coordination/plan",
        json={
            "robots": [
                {"robot_id": "R1", "start": {"row": 0, "col": 0}, "goal": {"row": 0, "col": 2}}
            ]
        },
    )
    runs = RunRepository(db_session).recent_coordination_runs()
    assert len(runs) == 1
    assert runs[0].robot_count == 1
