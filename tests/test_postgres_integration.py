"""Integration tests that run against a REAL PostgreSQL database.

They SKIP unless ``TEST_DATABASE_URL`` (in the environment / .env) points at a
reachable PostgreSQL. In CI it points at the ``postgres`` service container; on
a dev machine it points at a dedicated test database.

The ``pg_database`` / ``pg_client`` fixtures (in conftest.py) bring the schema
to head with the real Alembic migrations, seed the test users, and TRUNCATE the
app tables around each test.
"""

from __future__ import annotations

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError

from robotics.persistence.models import RobotRecord, TaskRecord, UserRecord
from robotics.persistence.repositories import RobotRepository, TaskRepository

pytestmark = pytest.mark.integration


# --- connection + schema ------------------------------------------
def test_it_is_actually_postgresql(pg_database):
    with pg_database.engine.connect() as conn:
        version = conn.execute(text("SELECT version()")).scalar()
    assert version.lower().startswith("postgresql")


def test_migrations_created_every_table_with_the_right_indexes(pg_database):
    insp = inspect(pg_database.engine)
    names = set(insp.get_table_names())
    assert {
        "users", "robots", "tasks",
        "planning_runs", "allocation_runs", "coordination_runs", "recovery_events",
        "alembic_version",
    } <= names
    task_indexes = {i["name"] for i in insp.get_indexes("tasks")}
    assert "ix_tasks_status" in task_indexes
    assert "ix_tasks_assigned_robot_id" in task_indexes


def test_alembic_version_is_at_head(pg_database):
    with pg_database.engine.connect() as conn:
        rev = conn.execute(text("SELECT version_num FROM alembic_version")).scalar()
    assert rev == "0001"


# --- CRUD round trips -------------------------------------------
def test_create_and_read_a_robot(pg_database):
    with pg_database.session_scope() as session:
        RobotRepository(session).add(RobotRecord(robot_id="PG1", row=0, col=1))
    with pg_database.session() as session:
        got = RobotRepository(session).get("PG1")
        assert got is not None and (got.row, got.col) == (0, 1)
        assert got.created_at is not None  # server default populated by PostgreSQL


def test_foreign_key_on_delete_set_null_is_enforced_by_postgres(pg_database):
    with pg_database.session_scope() as session:
        session.add(RobotRecord(robot_id="PGR", row=0, col=1))
        session.flush()
        session.add(TaskRecord(
            task_id="PGT", pickup_row=2, pickup_col=4, dropoff_row=11, dropoff_col=1,
            status="assigned", assigned_robot_id="PGR",
        ))
    with pg_database.session_scope() as session:
        session.delete(RobotRepository(session).get("PGR"))
    with pg_database.session() as session:
        task = TaskRepository(session).get("PGT")
        assert task is not None and task.assigned_robot_id is None


def test_unique_constraint_violation_raises(pg_database):
    with pg_database.session_scope() as session:
        session.add(UserRecord(username="dup", password_hash="x", role="viewer"))
    with pytest.raises(IntegrityError):
        with pg_database.session_scope() as session:
            session.add(UserRecord(username="dup", password_hash="y", role="viewer"))
            session.flush()


def test_rollback_after_failure_leaves_nothing(pg_database):
    class Boom(RuntimeError):
        pass

    with pytest.raises(Boom):
        with pg_database.session_scope() as session:
            session.add(RobotRecord(robot_id="ROLLBACK", row=0, col=1))
            raise Boom()
    with pg_database.session() as session:
        assert RobotRepository(session).get("ROLLBACK") is None


def test_atomic_multi_write_on_postgres(pg_database):
    """A second write violates a constraint -> the first write is undone too."""
    with pytest.raises(IntegrityError):
        with pg_database.session_scope() as session:
            session.add(RobotRecord(robot_id="A1", row=0, col=1))
            session.add(RobotRecord(robot_id="A1", row=2, col=2))  # dup PK
            session.flush()
    with pg_database.session() as session:
        assert RobotRepository(session).list() == []


# --- API -> service -> ORM -> PostgreSQL round trips ----------------
def test_api_create_and_read_robot_via_postgres(pg_client):
    created = pg_client.post("/robots", json={"robot_id": "API1", "position": {"row": 0, "col": 1}})
    assert created.status_code == 201
    assert pg_client.get("/robots/API1").json()["position"] == {"row": 0, "col": 1}


def test_api_planning_run_is_persisted_in_postgres(pg_client, pg_database):
    r = pg_client.post(
        "/planning/path",
        json={"start": {"row": 0, "col": 0}, "goal": {"row": 11, "col": 10}, "algorithm": "astar"},
    )
    assert r.status_code == 200 and r.json()["success"] is True
    with pg_database.engine.connect() as conn:
        count = conn.execute(text("SELECT count(*) FROM planning_runs")).scalar()
    assert count == 1


def test_api_allocation_commit_round_trip_via_postgres(pg_client):
    pg_client.post("/robots", json={"robot_id": "R1", "position": {"row": 0, "col": 0}})
    pg_client.post("/tasks", json={
        "task_id": "T1", "pickup": {"row": 2, "col": 4},
        "dropoff": {"row": 11, "col": 1}, "payload_weight": 3.0,
    })
    out = pg_client.post(
        "/allocation/run", json={"algorithm": "greedy", "from_persisted": True, "commit": True}
    )
    assert out.status_code == 200 and out.json()["committed"] is True
    assert pg_client.get("/tasks/T1").json()["status"] == "assigned"


def test_recovery_event_persisted_in_postgres(pg_client, pg_database):
    pg_client.post("/recovery/obstacle", json={
        "robots": [{"robot_id": "R1", "start": {"row": 4, "col": 0}, "goal": {"row": 4, "col": 6}}],
        "obstacle": {"row": 4, "col": 3}, "timestep": 2,
    })
    with pg_database.engine.connect() as conn:
        count = conn.execute(text("SELECT count(*) FROM recovery_events")).scalar()
    assert count == 1


# --- auth + readiness against PostgreSQL ------------------------
def test_protected_endpoint_rejects_a_bad_token_even_on_postgres(pg_client):
    pg_client.post("/robots", json={"robot_id": "R9", "position": {"row": 0, "col": 1}})
    bad = pg_client.get("/robots", headers={"Authorization": "Bearer not.a.valid.token"})
    assert bad.status_code == 401


def test_readiness_is_ok_when_postgres_is_up(pg_client):
    r = pg_client.get("/ready")
    assert r.status_code == 200
    assert r.json() == {"status": "ready", "checks": {"database": "ok"}}
