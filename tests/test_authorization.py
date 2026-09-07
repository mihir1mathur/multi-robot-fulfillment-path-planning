"""Authorization: who may call which endpoint.

    public      GET /health, GET /ready, POST /auth/token, POST /auth/login
    viewer+     GET /robots, GET /tasks
    operator+   POST/PATCH /robots & /tasks, POST /planning|allocation|
                coordination|recovery
    admin       POST /auth/register
"""

from __future__ import annotations

import pytest

_ROBOT = {"robot_id": "R1", "position": {"row": 0, "col": 1}}
_TASK = {
    "task_id": "T1",
    "pickup": {"row": 2, "col": 4},
    "dropoff": {"row": 11, "col": 1},
}
_PLAN = {"start": {"row": 0, "col": 0}, "goal": {"row": 4, "col": 1}, "algorithm": "astar"}


# --- public endpoints need no token --------------------------------
@pytest.mark.parametrize("path", ["/health", "/ready"])
def test_health_and_readiness_are_public(anon_client, path):
    assert anon_client.get(path).status_code in (200, 503)


# --- unauthenticated -> 401 ---------------------------------------
@pytest.mark.parametrize(
    "method,path",
    [
        ("get", "/robots"),
        ("get", "/tasks"),
        ("post", "/robots"),
        ("post", "/tasks"),
        ("post", "/planning/path"),
        ("post", "/allocation/run"),
        ("post", "/coordination/plan"),
        ("post", "/recovery/obstacle"),
    ],
)
def test_protected_endpoints_reject_anonymous_with_401(anon_client, method, path):
    # httpx's .get() takes no body; only send one for the POST endpoints.
    kwargs = {"json": {}} if method != "get" else {}
    response = getattr(anon_client, method)(path, **kwargs)
    assert response.status_code == 401
    assert response.json()["error"] == "unauthenticated"


# --- viewer: may read, may NOT write (403) -----------------------
def test_viewer_can_read_robots(viewer_client):
    assert viewer_client.get("/robots").status_code == 200


def test_viewer_cannot_create_a_robot(viewer_client):
    r = viewer_client.post("/robots", json=_ROBOT)
    assert r.status_code == 403
    assert r.json()["error"] == "forbidden"


def test_viewer_cannot_run_planning(viewer_client):
    assert viewer_client.post("/planning/path", json=_PLAN).status_code == 403


def test_viewer_cannot_register_users(viewer_client):
    r = viewer_client.post(
        "/auth/register", json={"username": "x", "password": "abcdefgh", "role": "viewer"}
    )
    assert r.status_code == 403


# --- operator: may read AND write ------------------------------
def test_operator_can_create_a_robot_and_run_planning(api_client):
    assert api_client.post("/robots", json=_ROBOT).status_code == 201
    assert api_client.post("/tasks", json=_TASK).status_code == 201
    assert api_client.post("/planning/path", json=_PLAN).json()["success"] is True


def test_operator_cannot_register_users(api_client):
    r = api_client.post(
        "/auth/register", json={"username": "x", "password": "abcdefgh", "role": "viewer"}
    )
    assert r.status_code == 403


# --- admin: everything ----------------------------------------
def test_admin_can_register_a_user(admin_client):
    r = admin_client.post(
        "/auth/register",
        json={"username": "fresh_viewer", "password": "a-good-password", "role": "viewer"},
    )
    assert r.status_code == 201
    assert r.json()["role"] == "viewer"


def test_admin_can_also_do_operator_things(admin_client):
    assert admin_client.post("/robots", json=_ROBOT).status_code == 201
    assert admin_client.get("/robots").status_code == 200


def test_registering_a_duplicate_user_is_409(admin_client):
    body = {"username": "dupe", "password": "a-good-password", "role": "viewer"}
    assert admin_client.post("/auth/register", json=body).status_code == 201
    assert admin_client.post("/auth/register", json=body).status_code == 409


def test_registering_with_a_short_password_is_422(admin_client):
    r = admin_client.post(
        "/auth/register", json={"username": "shorty", "password": "abc", "role": "viewer"}
    )
    assert r.status_code == 422


def test_registering_with_an_unknown_role_is_422(admin_client):
    r = admin_client.post(
        "/auth/register",
        json={"username": "weird", "password": "a-good-password", "role": "superuser"},
    )
    assert r.status_code == 422


# --- a freshly-registered viewer really is limited ---------------
def test_newly_registered_viewer_gets_a_working_but_limited_token(admin_client, anon_client):
    admin_client.post(
        "/auth/register",
        json={"username": "limited", "password": "a-good-password", "role": "viewer"},
    )
    token = anon_client.post(
        "/auth/login", json={"username": "limited", "password": "a-good-password"}
    ).json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    assert anon_client.get("/robots", headers=headers).status_code == 200
    assert anon_client.post("/robots", json=_ROBOT, headers=headers).status_code == 403
