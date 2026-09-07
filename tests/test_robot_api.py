"""Robot CRUD endpoints."""

from __future__ import annotations


def _make(client, robot_id="R1", row=0, col=1, **extra):
    body = {"robot_id": robot_id, "position": {"row": row, "col": col}, **extra}
    return client.post("/robots", json=body)


def test_create_and_get_robot(api_client):
    created = _make(api_client, "R1", 0, 1, battery_level=80.0, payload_capacity=12.0)
    assert created.status_code == 201
    body = created.json()
    assert body["robot_id"] == "R1"
    assert body["position"] == {"row": 0, "col": 1}
    assert body["battery_level"] == 80.0
    assert body["status"] == "idle"

    fetched = api_client.get("/robots/R1")
    assert fetched.status_code == 200
    assert fetched.json() == body


def test_list_robots_and_filter_by_status(api_client):
    _make(api_client, "R1", 0, 1)
    _make(api_client, "R2", 0, 2)
    api_client.patch("/robots/R2", json={"status": "offline"})

    all_robots = api_client.get("/robots").json()
    assert all_robots["count"] == 2

    idle = api_client.get("/robots", params={"status": "idle"}).json()
    assert [r["robot_id"] for r in idle["robots"]] == ["R1"]


def test_get_missing_robot_is_404(api_client):
    response = api_client.get("/robots/NOPE")
    assert response.status_code == 404
    assert response.json()["error"] == "not_found"


def test_duplicate_robot_is_409(api_client):
    _make(api_client, "R1", 0, 1)
    dup = _make(api_client, "R1", 0, 2)
    assert dup.status_code == 409
    assert dup.json()["error"] == "conflict"


def test_create_on_blocked_cell_is_422(api_client):
    # (2, 2) is a rack in the default warehouse profile.
    response = _make(api_client, "RX", 2, 2)
    assert response.status_code == 422


def test_create_on_occupied_cell_is_409(api_client):
    _make(api_client, "R1", 0, 1)
    response = _make(api_client, "R2", 0, 1)
    assert response.status_code == 409


def test_invalid_field_values_are_422(api_client):
    # battery over 100 - rejected by the schema
    bad = api_client.post(
        "/robots",
        json={"robot_id": "R1", "position": {"row": 0, "col": 1}, "battery_level": 150},
    )
    assert bad.status_code == 422

    # negative coordinate - rejected by the schema
    bad = api_client.post(
        "/robots", json={"robot_id": "R2", "position": {"row": -1, "col": 0}}
    )
    assert bad.status_code == 422

    # unknown status - rejected by the schema enum
    bad = api_client.post(
        "/robots",
        json={"robot_id": "R3", "position": {"row": 0, "col": 1}, "status": "flying"},
    )
    assert bad.status_code == 422


def test_patch_updates_only_supplied_fields(api_client):
    _make(api_client, "R1", 0, 1, battery_level=90.0)
    patched = api_client.patch(
        "/robots/R1", json={"battery_level": 42.0, "position": {"row": 0, "col": 0}}
    )
    assert patched.status_code == 200
    body = patched.json()
    assert body["battery_level"] == 42.0
    assert body["position"] == {"row": 0, "col": 0}
    assert body["payload_capacity"] == 10.0  # untouched


def test_patch_missing_robot_is_404(api_client):
    assert api_client.patch("/robots/NOPE", json={"battery_level": 1.0}).status_code == 404


def test_patch_onto_occupied_cell_is_409(api_client):
    _make(api_client, "R1", 0, 1)
    _make(api_client, "R2", 0, 2)
    response = api_client.patch("/robots/R2", json={"position": {"row": 0, "col": 1}})
    assert response.status_code == 409


def test_delete_robot_removes_the_row(api_client):
    _make(api_client, "R1", 0, 1)
    deleted = api_client.request("DELETE", "/robots/R1")
    assert deleted.status_code == 204
    assert not deleted.content
    assert api_client.get("/robots/R1").status_code == 404


def test_delete_missing_robot_is_404(api_client):
    assert api_client.request("DELETE", "/robots/NOPE").status_code == 404


def test_delete_robot_releases_its_task(api_client):
    _make(api_client, "R1", 0, 1)
    api_client.post(
        "/tasks",
        json={"task_id": "T1", "pickup": {"row": 2, "col": 4},
              "dropoff": {"row": 11, "col": 1}},
    )
    api_client.patch("/tasks/T1", json={"status": "assigned", "assigned_robot_id": "R1"})

    assert api_client.request("DELETE", "/robots/R1").status_code == 204
    # ON DELETE SET NULL: the task survives with no robot rather than dangling
    task = api_client.get("/tasks/T1")
    assert task.status_code == 200
    assert task.json()["assigned_robot_id"] is None
