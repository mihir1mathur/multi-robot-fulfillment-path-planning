"""Every error path returns the ONE documented body shape and leaks nothing.

Body shape (from robotics/api/errors.py):

    {"error": "<machine code>", "message": "<human text>"[, "details": ...]}

Status codes:
    400/401/403/404/409/422  -> deliberate, described in the body
    503                      -> transient (database unavailable), Retry-After set
    500                      -> a bug; body carries only an incident id
"""

from __future__ import annotations

from sqlalchemy.exc import OperationalError


def _assert_error_body(response, code: str):
    body = response.json()
    assert body["error"] == code
    assert isinstance(body["message"], str) and body["message"]
    assert set(body) <= {"error", "message", "details"}


# --- the ordinary 4xx family ------------------------------------
def test_unknown_route_is_404(api_client):
    assert api_client.get("/does-not-exist").status_code == 404


def test_missing_resource_is_404_with_shape(api_client):
    r = api_client.get("/robots/GHOST")
    assert r.status_code == 404
    _assert_error_body(r, "not_found")


def test_malformed_body_is_422_with_details(api_client):
    r = api_client.post("/robots", json={"robot_id": "R1"})  # missing position
    assert r.status_code == 422
    _assert_error_body(r, "validation_error")
    assert r.json()["details"]  # pydantic error list


def test_domain_rejection_is_422(api_client):
    # a task pickup outside the warehouse: well-formed JSON, unacceptable content
    r = api_client.post(
        "/tasks",
        json={"task_id": "T1", "pickup": {"row": 99, "col": 99}, "dropoff": {"row": 11, "col": 1}},
    )
    assert r.status_code == 422
    assert r.json()["error"] in {"unprocessable", "domain_rejected"}


def test_duplicate_is_409(api_client):
    api_client.post("/robots", json={"robot_id": "R1", "position": {"row": 0, "col": 0}})
    r = api_client.post("/robots", json={"robot_id": "R1", "position": {"row": 0, "col": 1}})
    assert r.status_code == 409
    _assert_error_body(r, "conflict")


def test_unauthenticated_is_401_with_www_authenticate(anon_client):
    r = anon_client.get("/robots")
    assert r.status_code == 401
    assert r.headers.get("www-authenticate") == "Bearer"
    _assert_error_body(r, "unauthenticated")


def test_forbidden_is_403(viewer_client):
    r = viewer_client.post("/robots", json={"robot_id": "R1", "position": {"row": 0, "col": 0}})
    assert r.status_code == 403
    _assert_error_body(r, "forbidden")


# --- database-unavailable -> 503 (transient, not a bug) --------
def test_database_operational_error_mid_request_is_503_not_500(api_app, api_database, monkeypatch):
    from fastapi.testclient import TestClient

    from tests.conftest import _token_for

    token = _token_for(api_database, "operator")  # mint before breaking the DB

    def _dropped_connection(*_a, **_k):
        # what SQLAlchemy raises when the socket to the server is gone
        raise OperationalError("SELECT 1", {}, Exception("server closed the connection"))

    monkeypatch.setattr(api_database, "session", _dropped_connection)

    with TestClient(api_app, raise_server_exceptions=False) as client:
        client.headers["Authorization"] = f"Bearer {token}"
        r = client.get("/robots")

    assert r.status_code == 503
    body = r.json()
    assert body["error"] == "database_unavailable"
    assert "incident" in body["message"]
    assert r.headers.get("retry-after") is not None
    assert "server closed the connection" not in r.text  # driver detail not leaked


# --- no error body ever leaks internals -----------------------
def test_no_error_response_leaks_secrets_or_sql(api_client, anon_client):
    responses = [
        api_client.get("/robots/GHOST"),
        api_client.post("/robots", json={"bad": "body"}),
        anon_client.get("/robots"),
        api_client.get("/nope"),
    ]
    for r in responses:
        text = r.text.lower()
        for forbidden in ("traceback", "password", "sqlite3", "psycopg", "5432",
                          "jwt_secret", "eyj"):  # 'eyj' = a leaked JWT prefix
            assert forbidden not in text
