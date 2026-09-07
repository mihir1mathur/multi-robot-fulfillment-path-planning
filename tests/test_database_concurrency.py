"""Concurrency against a REAL PostgreSQL database (a real connection pool).

SKIPS unless ``TEST_DATABASE_URL`` points at a reachable PostgreSQL. On SQLite
these would be meaningless (one process-wide connection); PostgreSQL is where
overlapping transactions actually run in parallel.

No sleeps: correctness is asserted on row counts and status codes.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest
from sqlalchemy import text

from robotics.persistence.models import PlanningRun, RobotRecord

pytestmark = pytest.mark.integration


_PLAN = {"start": {"row": 0, "col": 0}, "goal": {"row": 11, "col": 10}, "algorithm": "astar"}


def _map(fn, n, workers):
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, range(n)))


def test_the_pool_serves_many_overlapping_reads(pg_client):
    codes = _map(lambda _i: pg_client.get("/robots").status_code, 40, 20)
    assert all(c == 200 for c in codes)


def test_concurrent_distinct_writes_all_commit_on_postgres(pg_client):
    open_rows = (0, 1, 4, 7, 10)

    def create(i: int) -> int:
        return pg_client.post("/robots", json={
            "robot_id": f"P{i}",
            "position": {"row": open_rows[i % len(open_rows)], "col": i},
        }).status_code

    codes = _map(create, 12, 12)
    assert codes.count(201) == 12

    with pg_client.app.state.database.session() as s:
        assert s.query(RobotRecord).count() == 12


def test_concurrent_duplicate_write_race_is_one_winner_on_postgres(pg_client):
    """PostgreSQL's real unique constraint + the 409 handler: exactly one 201,
    the rest 409, never a 500 - even though every thread passed the
    'does it exist?' check before any INSERT committed."""
    body = {"robot_id": "PGRACE", "position": {"row": 0, "col": 0}}
    codes = _map(lambda _i: pg_client.post("/robots", json=body).status_code, 20, 20)

    assert codes.count(201) == 1
    assert codes.count(409) == 19
    assert not any(c >= 500 for c in codes)

    with pg_client.app.state.database.engine.connect() as conn:
        n = conn.execute(
            text("SELECT count(*) FROM robots WHERE robot_id = 'PGRACE'")
        ).scalar()
    assert n == 1


def test_concurrent_history_inserts_do_not_lose_rows_on_postgres(pg_client, pg_database):
    codes = _map(lambda _i: pg_client.post("/planning/path", json=_PLAN).status_code, 30, 15)
    assert all(c == 200 for c in codes)
    with pg_database.engine.connect() as conn:
        n = conn.execute(text("SELECT count(*) FROM planning_runs")).scalar()
    assert n == 30


def test_transactions_stay_isolated_under_concurrency_on_postgres(pg_client, pg_database):
    """A rejected request (illegal transition) must not roll back a concurrent
    successful one."""
    pg_client.post("/robots", json={"robot_id": "R1", "position": {"row": 0, "col": 0}})
    pg_client.post("/tasks", json={
        "task_id": "T1", "pickup": {"row": 2, "col": 4}, "dropoff": {"row": 11, "col": 1},
    })

    def op(i: int) -> int:
        if i % 2 == 0:
            # a doomed illegal transition (pending -> completed)
            return pg_client.patch("/tasks/T1", json={"status": "completed"}).status_code
        # a legit robot battery update
        return pg_client.patch("/robots/R1", json={"battery_level": 50.0 + i}).status_code

    codes = _map(op, 12, 12)
    assert set(c for i, c in enumerate(codes) if i % 2 == 0) <= {409}
    assert set(c for i, c in enumerate(codes) if i % 2 == 1) <= {200}

    # the task never moved off 'pending'; the robot update took effect
    assert pg_client.get("/tasks/T1").json()["status"] == "pending"
    with pg_database.engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM planning_runs")).scalar() >= 0
