"""Task CRUD endpoints and lifecycle enforcement."""

from __future__ import annotations


def _make_task(client, task_id="T1", pickup=(2, 4), dropoff=(11, 1), **extra):
    body = {
        "task_id": task_id,
        "pickup": {"row": pickup[0], "col": pickup[1]},
        "dropoff": {"row": dropoff[0], "col": dropoff[1]},
        **extra,
    }
    return client.post("/tasks", json=body)


def test_create_and_get_task(api_client):
    created = _make_task(api_client, "T1", payload_weight=3.0, priority="high")
    assert created.status_code == 201
    body = created.json()
    assert body["status"] == "pending"
    assert body["priority"] == "high"
    assert body["payload_weight"] == 3.0

    assert api_client.get("/tasks/T1").json() == body


def test_list_tasks_filter_by_status(api_client):
    _make_task(api_client, "T1")
    _make_task(api_client, "T2")
    pending = api_client.get("/tasks", params={"status": "pending"}).json()
    assert pending["count"] == 2


def test_get_missing_task_is_404(api_client):
    assert api_client.get("/tasks/NOPE").status_code == 404


def test_duplicate_task_is_409(api_client):
    _make_task(api_client, "T1")
    assert _make_task(api_client, "T1").status_code == 409


def test_pickup_equals_dropoff_is_422(api_client):
    response = _make_task(api_client, "T1", pickup=(1, 1), dropoff=(1, 1))
    assert response.status_code == 422


def test_endpoint_on_a_wall_is_422(api_client):
    # (2, 2) is a rack in the default warehouse.
    response = _make_task(api_client, "T1", pickup=(2, 2))
    assert response.status_code == 422


def test_valid_lifecycle_transition(api_client):
    api_client.post("/robots", json={"robot_id": "R1", "position": {"row": 0, "col": 1}})
    _make_task(api_client, "T1")

    assigned = api_client.patch(
        "/tasks/T1", json={"status": "assigned", "assigned_robot_id": "R1"}
    )
    assert assigned.status_code == 200
    assert assigned.json()["status"] == "assigned"
    assert assigned.json()["assigned_robot_id"] == "R1"

    started = api_client.patch("/tasks/T1", json={"status": "in_progress"})
    assert started.status_code == 200

    done = api_client.patch("/tasks/T1", json={"status": "completed"})
    assert done.status_code == 200
    assert done.json()["status"] == "completed"


def test_illegal_transition_is_409(api_client):
    _make_task(api_client, "T1")
    # PENDING -> COMPLETED is not allowed by the task state machine.
    response = api_client.patch("/tasks/T1", json={"status": "completed"})
    assert response.status_code == 409
    assert response.json()["error"] == "conflict"


def test_assign_without_robot_id_is_422(api_client):
    _make_task(api_client, "T1")
    response = api_client.patch("/tasks/T1", json={"status": "assigned"})
    assert response.status_code == 422


def test_editing_fields_after_pending_is_409(api_client):
    api_client.post("/robots", json={"robot_id": "R1", "position": {"row": 0, "col": 1}})
    _make_task(api_client, "T1")
    api_client.patch("/tasks/T1", json={"status": "assigned", "assigned_robot_id": "R1"})
    response = api_client.patch("/tasks/T1", json={"payload_weight": 9.0})
    assert response.status_code == 409


def test_patch_missing_task_is_404(api_client):
    assert api_client.patch("/tasks/NOPE", json={"status": "assigned"}).status_code == 404


def test_delete_task_removes_the_row(api_client):
    _make_task(api_client, "T1")
    deleted = api_client.request("DELETE", "/tasks/T1")
    assert deleted.status_code == 204
    assert not deleted.content
    assert api_client.get("/tasks/T1").status_code == 404


def test_delete_missing_task_is_404(api_client):
    assert api_client.request("DELETE", "/tasks/NOPE").status_code == 404


def test_delete_assigned_task_releases_the_robot_pointer(api_client):
    api_client.post("/robots", json={"robot_id": "R1", "position": {"row": 0, "col": 1}})
    _make_task(api_client, "T1")
    api_client.patch("/tasks/T1", json={"status": "assigned", "assigned_robot_id": "R1"})

    assert api_client.request("DELETE", "/tasks/T1").status_code == 204
    # the robot row survives; only the task is gone
    assert api_client.get("/robots/R1").status_code == 200
    assert api_client.get("/tasks/T1").status_code == 404
