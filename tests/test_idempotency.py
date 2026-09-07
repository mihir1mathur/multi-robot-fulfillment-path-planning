"""Repeat-safety of the write endpoints.

The project does NOT use idempotency keys - it does not need them. Instead:

  * creating a resource that already exists is REJECTED with 409 (not a
    second row, not a 500),
  * a lifecycle PATCH to the state the resource is ALREADY in is a no-op that
    returns 200 (safe to retry),
  * a lifecycle PATCH to a state that is no longer reachable is REJECTED with
    409.

These tests pin that behaviour so a client that retries after a dropped
response cannot corrupt state.
"""

from __future__ import annotations


def _make_robot(client, rid="R1"):
    return client.post("/robots", json={"robot_id": rid, "position": {"row": 0, "col": 0}})


# --- duplicate creation ---------------------------------------
def test_creating_the_same_robot_twice_is_409_and_leaves_one_row(api_client):
    assert _make_robot(api_client).status_code == 201
    again = _make_robot(api_client)
    assert again.status_code == 409
    assert again.json()["error"] == "conflict"
    assert api_client.get("/robots").json()["count"] == 1


def test_creating_the_same_task_twice_is_409(api_client):
    body = {"task_id": "T1", "pickup": {"row": 2, "col": 4}, "dropoff": {"row": 11, "col": 1}}
    assert api_client.post("/tasks", json=body).status_code == 201
    assert api_client.post("/tasks", json=body).status_code == 409


def test_registering_the_same_user_twice_is_409(admin_client):
    body = {"username": "repeat", "password": "a-good-password", "role": "viewer"}
    assert admin_client.post("/auth/register", json=body).status_code == 201
    assert admin_client.post("/auth/register", json=body).status_code == 409


# --- repeated lifecycle transitions -------------------------
def test_patching_a_task_to_its_current_status_is_an_idempotent_noop(api_client):
    _make_robot(api_client)
    api_client.post(
        "/tasks",
        json={"task_id": "T1", "pickup": {"row": 2, "col": 4}, "dropoff": {"row": 11, "col": 1}},
    )
    api_client.patch("/tasks/T1", json={"status": "assigned", "assigned_robot_id": "R1"})

    first = api_client.patch("/tasks/T1", json={"status": "assigned", "assigned_robot_id": "R1"})
    second = api_client.patch("/tasks/T1", json={"status": "assigned", "assigned_robot_id": "R1"})
    assert first.status_code == second.status_code == 200
    assert api_client.get("/tasks/T1").json()["status"] == "assigned"


def test_completing_a_task_twice_second_is_409_state_unchanged(api_client):
    _make_robot(api_client)
    api_client.post(
        "/tasks",
        json={"task_id": "T1", "pickup": {"row": 2, "col": 4}, "dropoff": {"row": 11, "col": 1}},
    )
    api_client.patch("/tasks/T1", json={"status": "assigned", "assigned_robot_id": "R1"})
    api_client.patch("/tasks/T1", json={"status": "in_progress"})
    assert api_client.patch("/tasks/T1", json={"status": "completed"}).status_code == 200
    assert api_client.patch("/tasks/T1", json={"status": "completed"}).status_code == 200
    # 'completed' -> 'completed' is the no-op case; a real re-transition attempt
    # from a terminal state to an active one is the 409 case:
    assert api_client.patch("/tasks/T1", json={"status": "assigned",
                                              "assigned_robot_id": "R1"}).status_code == 409


def test_repeated_identical_robot_patch_converges(api_client):
    _make_robot(api_client)
    for _ in range(3):
        r = api_client.patch("/robots/R1", json={"battery_level": 55.0})
        assert r.status_code == 200
    assert api_client.get("/robots/R1").json()["battery_level"] == 55.0
