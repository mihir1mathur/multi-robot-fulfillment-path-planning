"""Concurrent requests through the API must not corrupt state.

These tests drive the in-process ASGI app from a real thread pool. FastAPI
runs each sync endpoint in its own worker thread, so the requests genuinely
overlap. The database here is the per-test in-memory SQLite (file-less,
`StaticPool`), which serialises at the connection level - that is fine: the
point is *correctness under interleaving*, not throughput. Throughput and a
real connection pool are measured separately in
``benchmarks/benchmark_concurrency.py --postgres``.

No test here sleeps. Determinism comes from fixed request bodies and from
asserting on counts / status codes, never on timing.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from robotics.persistence.models import PlanningRun, RobotRecord

_PLAN = {"start": {"row": 0, "col": 0}, "goal": {"row": 11, "col": 10}, "algorithm": "astar"}


def _run(fn, count, workers):
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, range(count)))


# --- a create race resolves to exactly one winner -----------------
def test_many_concurrent_creates_of_the_same_id_yield_one_201_and_rest_409(concurrent_client):
    body = {"robot_id": "DUP", "position": {"row": 0, "col": 0}}
    codes = _run(lambda _i: concurrent_client.post("/robots", json=body).status_code, 20, 20)

    assert codes.count(201) == 1
    assert codes.count(409) == 19
    assert not any(c >= 500 for c in codes)  # never a crash

    # exactly one row actually exists
    with concurrent_client.app.state.database.session_scope() as s:
        assert s.query(RobotRecord).filter_by(robot_id="DUP").count() == 1


# --- distinct concurrent creates all succeed --------------------
_OPEN_ROWS = (0, 1, 4, 7, 10)  # rows with no racks in the default 12x12 profile


def test_concurrent_creates_of_distinct_ids_all_succeed(concurrent_client):
    def create(i: int) -> int:
        return concurrent_client.post(
            "/robots",
            json={
                "robot_id": f"C{i}",
                "position": {"row": _OPEN_ROWS[i % len(_OPEN_ROWS)], "col": i},
            },
        ).status_code

    codes = _run(create, 12, 10)
    assert codes.count(201) == 12

    with concurrent_client.app.state.database.session_scope() as s:
        assert s.query(RobotRecord).count() == 12


# --- concurrent writes to the run-history table ----------------
def test_concurrent_planning_runs_all_persist_exactly_once(concurrent_client):
    codes = _run(lambda _i: concurrent_client.post("/planning/path", json=_PLAN).status_code, 25, 12)
    assert all(c == 200 for c in codes)

    with concurrent_client.app.state.database.session_scope() as s:
        assert s.query(PlanningRun).count() == 25  # no lost or duplicated rows


# --- concurrent lifecycle transitions on one task -------------
def test_concurrent_completion_attempts_complete_the_task_once(concurrent_client):
    # one robot, one task, assigned then in-progress
    concurrent_client.post("/robots", json={"robot_id": "R1", "position": {"row": 0, "col": 0}})
    concurrent_client.post(
        "/tasks",
        json={"task_id": "T1", "pickup": {"row": 2, "col": 4}, "dropoff": {"row": 11, "col": 1}},
    )
    concurrent_client.patch("/tasks/T1", json={"status": "assigned", "assigned_robot_id": "R1"})
    concurrent_client.patch("/tasks/T1", json={"status": "in_progress"})

    codes = _run(
        lambda _i: concurrent_client.patch("/tasks/T1", json={"status": "completed"}).status_code,
        12, 12,
    )
    # at least one 200; any loser sees 409 (already completed) - never a 500
    assert codes.count(200) >= 1
    assert set(codes) <= {200, 409}
    assert concurrent_client.get("/tasks/T1").json()["status"] == "completed"
