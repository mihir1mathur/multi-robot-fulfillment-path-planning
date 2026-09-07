"""Health and readiness endpoints."""

from __future__ import annotations


def test_health_is_ok(api_client):
    response = api_client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["app"]


def test_readiness_ok_when_database_is_up(api_client):
    response = api_client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ready", "checks": {"database": "ok"}}


def test_readiness_reports_not_ready_when_database_is_down(api_app, monkeypatch):
    """If the DB ping fails, /ready must be 503 and say so - never a false 'ready'."""
    from fastapi.testclient import TestClient

    monkeypatch.setattr(api_app.state.database, "ping", lambda: False)
    with TestClient(api_app) as client:
        response = client.get("/ready")
    assert response.status_code == 503
    assert response.json() == {
        "status": "not_ready",
        "checks": {"database": "unavailable"},
    }
    # health does NOT depend on the database
    with TestClient(api_app) as client:
        assert client.get("/health").status_code == 200


def test_correlation_id_header_is_returned(api_client):
    response = api_client.get("/health")
    assert response.headers.get("x-correlation-id")

    supplied = "abc123def456"
    response = api_client.get("/health", headers={"x-correlation-id": supplied})
    assert response.headers["x-correlation-id"] == supplied
