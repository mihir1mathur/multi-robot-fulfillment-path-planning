"""Task-allocation endpoint."""

from __future__ import annotations

_ROBOTS = [
    {"robot_id": "R1", "position": {"row": 0, "col": 0}, "payload_capacity": 10.0},
    {"robot_id": "R2", "position": {"row": 0, "col": 1}, "payload_capacity": 10.0},
]
_TASKS = [
    {
        "task_id": "T1",
        "pickup": {"row": 2, "col": 4},
        "dropoff": {"row": 11, "col": 1},
        "payload_weight": 3.0,
    },
    {
        "task_id": "T2",
        "pickup": {"row": 8, "col": 7},
        "dropoff": {"row": 11, "col": 10},
        "payload_weight": 4.0,
    },
]


def test_greedy_allocation(api_client):
    response = api_client.post(
        "/allocation/run",
        json={"algorithm": "greedy", "robots": _ROBOTS, "tasks": _TASKS},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["algorithm"] == "greedy"
    assert body["solver_status"] == "GREEDY"
    assert len(body["assignments"]) == 2
    assert set(a["task_id"] for a in body["assignments"]) == {"T1", "T2"}


def test_cp_sat_allocation(api_client):
    response = api_client.post(
        "/allocation/run",
        json={"algorithm": "cp_sat", "robots": _ROBOTS, "tasks": _TASKS},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["algorithm"] == "cp_sat"
    assert body["solver_status"] in {"OPTIMAL", "FEASIBLE"}
    assert len(body["assignments"]) == 2


def test_infeasible_pairing_is_reported_not_errored(api_client):
    # A 20kg task that no 10kg robot can carry: valid result, zero assignments.
    response = api_client.post(
        "/allocation/run",
        json={
            "algorithm": "greedy",
            "robots": _ROBOTS,
            "tasks": [
                {
                    "task_id": "TX",
                    "pickup": {"row": 2, "col": 4},
                    "dropoff": {"row": 11, "col": 1},
                    "payload_weight": 20.0,
                }
            ],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["assignments"] == []
    assert body["unassigned_task_ids"] == ["TX"]
    assert body["infeasible_pair_count"] >= 1


def test_no_route_pairing_yields_no_assignment(api_client):
    response = api_client.post(
        "/allocation/run",
        json={
            "algorithm": "cp_sat",
            "robots": [{"robot_id": "R1", "position": {"row": 0, "col": 0}}],
            "tasks": [
                {
                    "task_id": "T1",
                    "pickup": {"row": 0, "col": 3},
                    "dropoff": {"row": 3, "col": 3},
                }
            ],
            "warehouse": {
                "width": 4,
                "height": 4,
                "static_obstacles": [{"row": 0, "col": 2}, {"row": 1, "col": 3}],
            },
        },
    )
    assert response.status_code == 200
    assert response.json()["assignments"] == []


def test_missing_inputs_is_422(api_client):
    response = api_client.post("/allocation/run", json={"algorithm": "greedy"})
    assert response.status_code == 422


def test_commit_without_persisted_is_422(api_client):
    response = api_client.post(
        "/allocation/run",
        json={
            "algorithm": "greedy",
            "robots": _ROBOTS,
            "tasks": _TASKS,
            "commit": True,
        },
    )
    assert response.status_code == 422


def test_from_persisted_and_commit_updates_state(api_client):
    api_client.post("/robots", json={"robot_id": "R1", "position": {"row": 0, "col": 0}})
    api_client.post(
        "/tasks",
        json={
            "task_id": "T1",
            "pickup": {"row": 2, "col": 4},
            "dropoff": {"row": 11, "col": 1},
            "payload_weight": 3.0,
        },
    )
    response = api_client.post(
        "/allocation/run",
        json={"algorithm": "greedy", "from_persisted": True, "commit": True},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["committed"] is True
    assert body["applied"] == ["R1->T1"]

    assert api_client.get("/tasks/T1").json()["status"] == "assigned"
    assert api_client.get("/robots/R1").json()["assigned_task_id"] == "T1"


def test_unsupported_allocator_is_422(api_client):
    response = api_client.post(
        "/allocation/run",
        json={"algorithm": "hungarian", "robots": _ROBOTS, "tasks": _TASKS},
    )
    assert response.status_code == 422
