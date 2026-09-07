"""Reproducible benchmark: the service / API layer and its database.

WHAT IT MEASURES
----------------
For each API operation - robot create/read, task create/read, path planning,
task allocation, coordination, recovery - it fires a fixed number of
deterministic authenticated requests through the FastAPI app (in-process, via
TestClient - NO network), warms up first, and records:

    operation, request count, success rate, HTTP status mix,
    P50 / P95 / min / max total latency (ms)

For the pure-compute operations (planning, allocation, coordination) it also
calls the SAME algorithm DIRECTLY (no HTTP, no ORM) on identical inputs and
reports:

    direct P50, service P50, absolute overhead, relative overhead

BACKENDS
--------
    default            SQLite in-memory (fast, zero setup)
    --postgres         real PostgreSQL from DATABASE_URL / TEST_DATABASE_URL

    python benchmarks/benchmark_service.py
    python benchmarks/benchmark_service.py --postgres

IMPORTANT FRAMING
-----------------
These are LOCAL, in-process numbers on the development machine. They are NOT
"production latency": there is no real network hop and no concurrent load. The
point is engineering measurement - how much the HTTP + validation + auth + ORM
wrapper costs on top of the algorithm, and how much a real PostgreSQL round
trip adds over SQLite - not a performance claim.

Writes benchmark_service_results.json / .csv next to this file.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import create_engine, text  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from robotics.api.app import create_app  # noqa: E402
from robotics.persistence.config import Settings, get_settings  # noqa: E402
from robotics.persistence.database import Base, Database  # noqa: E402
from robotics.persistence.models import UserRecord  # noqa: E402
from robotics.planning import PLANNERS  # noqa: E402
from robotics.allocation.cost_estimator import CostEstimator  # noqa: E402
from robotics.allocation.greedy_allocator import GreedyAllocator  # noqa: E402
from robotics.api.security import create_access_token, hash_password  # noqa: E402
from robotics.coordination.coordinator import MultiRobotCoordinator  # noqa: E402
from robotics.robots.robot import Robot  # noqa: E402
from robotics.services.world import build_warehouse  # noqa: E402
from robotics.tasks.task import Task, TaskPriority  # noqa: E402
from robotics.warehouse.grid import Position  # noqa: E402

REQUESTS_PER_OP = 40
WARMUP = 5
REPEATS_FOR_DIRECT = 40
_JWT_KEY = "benchmark-only-signing-key-at-least-32-bytes-long"
_OPERATOR_HASH = hash_password("benchmark-operator-pw")
OUT_JSON = Path(__file__).with_name("benchmark_service_results.json")
OUT_CSV = Path(__file__).with_name("benchmark_service_results.csv")


# ----------------------------------------------------------------------
def _postgres_url() -> Optional[str]:
    s = get_settings()
    if s.test_database_url and s.test_database_url.startswith("postgresql"):
        return s.test_database_url
    if s.is_postgres:
        return s.database_url
    return None


def _make_client(backend: str):
    """Return an authenticated TestClient + the backend label actually used."""
    if backend == "postgres":
        url = _postgres_url()
        if not url:
            raise SystemExit(
                "--postgres requested but no PostgreSQL DATABASE_URL / "
                "TEST_DATABASE_URL is configured"
            )
        settings = Settings(
            database_url=url, log_level="ERROR", jwt_secret_key=_JWT_KEY,
            auto_create_tables=False,
        )
        engine = create_engine(url, future=True, pool_pre_ping=True)
        database = Database(settings=settings, engine=engine)
        # a clean slate every run
        Base.metadata.drop_all(engine)
        database.create_all()
        label = f"postgresql ({engine.dialect.server_version_info})"
    else:
        settings = Settings(
            database_url="sqlite+pysqlite://", log_level="ERROR",
            jwt_secret_key=_JWT_KEY, auto_create_tables=False,
        )
        engine = create_engine(
            "sqlite+pysqlite://", future=True,
            connect_args={"check_same_thread": False}, poolclass=StaticPool,
        )
        database = Database(settings=settings, engine=engine)
        database.create_all()
        label = "sqlite in-memory"

    with database.session_scope() as session:
        session.add(
            UserRecord(username="bench_op", password_hash=_OPERATOR_HASH, role="operator")
        )

    token = create_access_token("bench_op", "operator", _JWT_KEY, expires_minutes=120)
    app = create_app(settings=settings, database=database, create_tables=False)
    client = TestClient(app)
    client.headers["Authorization"] = f"Bearer {token}"
    return client, label


def _db_roundtrip_ms(client) -> float:
    """Median wall-clock of a bare 'SELECT 1' through the app's engine."""
    engine = client.app.state.database.engine
    samples = []
    for _ in range(WARMUP):
        with engine.connect() as c:
            c.execute(text("SELECT 1"))
    for _ in range(30):
        started = time.perf_counter()
        with engine.connect() as c:
            c.execute(text("SELECT 1"))
        samples.append((time.perf_counter() - started) * 1000)
    return round(statistics.median(samples), 4)


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = pct * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def _time_calls(count: int, call: Callable[[int], object], warmup: int = WARMUP):
    for i in range(warmup):
        call(-1 - i)
    latencies: List[float] = []
    outcomes: List = []
    for i in range(count):
        started = time.perf_counter()
        outcome = call(i)
        latencies.append((time.perf_counter() - started) * 1000)
        outcomes.append(outcome)
    return latencies, outcomes


# ----------------------------------------------------------------------
_OPEN_ROWS = (0, 1, 4, 7, 10)


def _robot_cell(i: int) -> tuple[int, int]:
    return (_OPEN_ROWS[abs(i) // 12 % len(_OPEN_ROWS)], abs(i) % 12)


def _bench_robot_crud(client, records: list) -> None:
    # 40 unique valid cells: rows 0/1/4/7 x cols 0..11
    def create(i: int):
        if i < 0:  # warmup - separate ids and a dedicated open row
            return client.post(
                "/robots",
                json={"robot_id": f"WARM{-i}", "position": {"row": 10, "col": -i - 1}},
            )
        row, col = _OPEN_ROWS[i // 12], i % 12
        return client.post(
            "/robots", json={"robot_id": f"R{i}", "position": {"row": row, "col": col}}
        )

    lat, res = _time_calls(REQUESTS_PER_OP, create)
    _record(records, "robot_create", lat, res)

    def read(i: int):
        return client.get(f"/robots/R{max(i, 0)}")

    lat, res = _time_calls(REQUESTS_PER_OP, read)
    _record(records, "robot_read", lat, res)


def _bench_task_crud(client, records: list) -> None:
    pickups = [(2, 4), (8, 7), (2, 1), (5, 1)]
    drops = [(11, 1), (11, 10), (11, 10), (11, 1)]

    def create(i: int):
        tag = f"W{-i}" if i < 0 else str(i)
        p, d = pickups[abs(i) % 4], drops[abs(i) % 4]
        return client.post(
            "/tasks",
            json={
                "task_id": f"T{tag}",
                "pickup": {"row": p[0], "col": p[1]},
                "dropoff": {"row": d[0], "col": d[1]},
                "payload_weight": 1.0 + (abs(i) % 5),
            },
        )

    lat, res = _time_calls(REQUESTS_PER_OP, create)
    _record(records, "task_create", lat, res)

    def read(i: int):
        return client.get(f"/tasks/T{max(i, 0)}")

    lat, res = _time_calls(REQUESTS_PER_OP, read)
    _record(records, "task_read", lat, res)


def _plan_bodies() -> list[dict]:
    goals = [(11, 10), (4, 6), (9, 4), (0, 11), (11, 1)]
    return [
        {
            "start": {"row": 0, "col": 0},
            "goal": {"row": g[0], "col": g[1]},
            "algorithm": "astar" if i % 2 == 0 else "dijkstra",
        }
        for i, g in enumerate(goals * 8)
    ][:REQUESTS_PER_OP]


def _bench_planning(client, records: list) -> dict:
    bodies = _plan_bodies()

    def call(i: int):
        return client.post("/planning/path", json=bodies[abs(i) % len(bodies)])

    lat, res = _time_calls(len(bodies), call)
    _record(records, "planning_path", lat, res)

    warehouse = build_warehouse()
    direct_lat: List[float] = []
    for body in bodies:
        planner = PLANNERS[body["algorithm"]]
        s, g = Position(**body["start"]), Position(**body["goal"])
        planner(s, g, warehouse)  # warmup
        started = time.perf_counter()
        planner(s, g, warehouse)
        direct_lat.append((time.perf_counter() - started) * 1000)
    return _overhead("planning_path", direct_lat, lat)


def _bench_allocation(client, records: list) -> dict:
    robots = [
        {"robot_id": "AR1", "position": {"row": 0, "col": 0}, "payload_capacity": 10.0},
        {"robot_id": "AR2", "position": {"row": 0, "col": 1}, "payload_capacity": 10.0},
        {"robot_id": "AR3", "position": {"row": 0, "col": 5}, "payload_capacity": 8.0},
    ]
    tasks = [
        {"task_id": "AT1", "pickup": {"row": 2, "col": 4}, "dropoff": {"row": 11, "col": 1}, "payload_weight": 3.0},
        {"task_id": "AT2", "pickup": {"row": 8, "col": 7}, "dropoff": {"row": 11, "col": 10}, "payload_weight": 4.0},
        {"task_id": "AT3", "pickup": {"row": 2, "col": 1}, "dropoff": {"row": 11, "col": 10}, "payload_weight": 2.0},
    ]
    body = {"algorithm": "greedy", "robots": robots, "tasks": tasks}

    lat, res = _time_calls(REQUESTS_PER_OP, lambda i: client.post("/allocation/run", json=body))
    _record(records, "allocation_run", lat, res)

    warehouse = build_warehouse()
    domain_robots = [
        Robot(robot_id=r["robot_id"], position=Position(**r["position"]),
              payload_capacity=r["payload_capacity"])
        for r in robots
    ]
    domain_tasks = [
        Task(task_id=t["task_id"], pickup_location=Position(**t["pickup"]),
             dropoff_location=Position(**t["dropoff"]), payload_weight=t["payload_weight"],
             priority=TaskPriority.NORMAL)
        for t in tasks
    ]
    direct_lat: List[float] = []
    for _ in range(REPEATS_FOR_DIRECT):
        estimator = CostEstimator(warehouse)
        started = time.perf_counter()
        GreedyAllocator(estimator).allocate(domain_robots, domain_tasks)
        direct_lat.append((time.perf_counter() - started) * 1000)
    return _overhead("allocation_run", direct_lat, lat)


def _bench_coordination(client, records: list) -> dict:
    body = {
        "robots": [
            {"robot_id": "CR1", "start": {"row": 4, "col": 0}, "goal": {"row": 4, "col": 6}},
            {"robot_id": "CR2", "start": {"row": 4, "col": 6}, "goal": {"row": 4, "col": 0}},
            {"robot_id": "CR3", "start": {"row": 0, "col": 0}, "goal": {"row": 7, "col": 0}},
        ]
    }
    lat, res = _time_calls(REQUESTS_PER_OP, lambda i: client.post("/coordination/plan", json=body))
    _record(records, "coordination_plan", lat, res)

    warehouse = build_warehouse()
    starts = {r["robot_id"]: Position(**r["start"]) for r in body["robots"]}
    goals = {r["robot_id"]: Position(**r["goal"]) for r in body["robots"]}
    direct_lat: List[float] = []
    for _ in range(REPEATS_FOR_DIRECT):
        started = time.perf_counter()
        MultiRobotCoordinator(warehouse).plan(starts, goals)
        direct_lat.append((time.perf_counter() - started) * 1000)
    return _overhead("coordination_plan", direct_lat, lat)


def _bench_recovery(client, records: list) -> None:
    obstacle = {
        "robots": [{"robot_id": "XR1", "start": {"row": 4, "col": 0}, "goal": {"row": 4, "col": 6}}],
        "obstacle": {"row": 4, "col": 3}, "timestep": 2,
    }
    lat, res = _time_calls(REQUESTS_PER_OP, lambda i: client.post("/recovery/obstacle", json=obstacle))
    _record(records, "recovery_obstacle", lat, res)

    failure = {
        "robots": [
            {"robot_id": "XR1", "start": {"row": 4, "col": 0}, "goal": {"row": 4, "col": 4},
             "task": {"task_id": "XT1", "pickup": {"row": 4, "col": 4},
                      "dropoff": {"row": 11, "col": 1}, "payload_weight": 2.0}},
            {"robot_id": "XR2", "start": {"row": 7, "col": 0}, "goal": {"row": 7, "col": 0}},
        ],
        "failed_robot_id": "XR1", "timestep": 1,
    }
    lat, res = _time_calls(REQUESTS_PER_OP, lambda i: client.post("/recovery/robot-failure", json=failure))
    _record(records, "recovery_robot_failure", lat, res)


# ----------------------------------------------------------------------
def _record(records: list, operation: str, latencies: List[float], responses: List) -> None:
    statuses: Dict[int, int] = {}
    ok = 0
    for r in responses:
        code = getattr(r, "status_code", 0)
        statuses[code] = statuses.get(code, 0) + 1
        if 200 <= code < 300:
            ok += 1
    records.append({
        "operation": operation,
        "requests": len(responses),
        "success_rate": round(ok / len(responses), 4) if responses else 0.0,
        "status_counts": {str(k): v for k, v in sorted(statuses.items())},
        "p50_latency_ms": round(_percentile(latencies, 0.50), 4),
        "p95_latency_ms": round(_percentile(latencies, 0.95), 4),
        "min_latency_ms": round(min(latencies), 4) if latencies else 0.0,
        "max_latency_ms": round(max(latencies), 4) if latencies else 0.0,
        "mean_latency_ms": round(statistics.fmean(latencies), 4) if latencies else 0.0,
    })


def _overhead(operation: str, direct: List[float], service: List[float]) -> dict:
    d50 = _percentile(direct, 0.50)
    s50 = _percentile(service, 0.50)
    return {
        "operation": operation,
        "direct_p50_ms": round(d50, 4),
        "service_p50_ms": round(s50, 4),
        "absolute_overhead_ms": round(s50 - d50, 4),
        "relative_overhead_x": round(s50 / d50, 2) if d50 > 0 else None,
    }


# ----------------------------------------------------------------------
def run(backend: str) -> dict:
    client, label = _make_client(backend)
    db_ms = _db_roundtrip_ms(client)

    records: List[dict] = []
    overheads: List[dict] = []
    _bench_robot_crud(client, records)
    _bench_task_crud(client, records)
    overheads.append(_bench_planning(client, records))
    overheads.append(_bench_allocation(client, records))
    overheads.append(_bench_coordination(client, records))
    _bench_recovery(client, records)

    total = sum(r["requests"] for r in records)
    overall_success = sum(r["requests"] * r["success_rate"] for r in records) / total
    return {
        "config": {
            "backend": label,
            "requests_per_operation": REQUESTS_PER_OP,
            "warmup_per_operation": WARMUP,
            "transport": "fastapi.testclient (in-process, authenticated as operator)",
            "timer": "time.perf_counter",
            "python": platform.python_version(),
            "platform": platform.platform(),
            "note": "Local in-process latency on the development machine. NOT "
            "production latency - no network hop, no concurrent load. Numbers "
            "swing with machine load; only success rate is deterministic.",
        },
        "database_select1_median_ms": db_ms,
        "total_requests": total,
        "overall_success_rate": round(overall_success, 4),
        "operations": records,
        "direct_vs_service": overheads,
    }


def main(argv: Optional[list] = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    backend = "postgres" if "--postgres" in argv else "sqlite"

    print("=" * 78)
    print("SERVICE / API / DATABASE BENCHMARK")
    print("=" * 78)

    report = run(backend)
    report["config"]["requested_backend"] = backend

    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["operation", "requests", "success_rate", "p50_latency_ms",
                         "p95_latency_ms", "min_latency_ms", "max_latency_ms", "mean_latency_ms"])
        for r in report["operations"]:
            writer.writerow([r["operation"], r["requests"], r["success_rate"],
                             r["p50_latency_ms"], r["p95_latency_ms"],
                             r["min_latency_ms"], r["max_latency_ms"], r["mean_latency_ms"]])

    _print_summary(report)
    print(f"\n  wrote {OUT_JSON.name} and {OUT_CSV.name} to {OUT_JSON.parent}")
    return 0


def _print_summary(report: dict) -> None:
    print(f"  backend                : {report['config']['backend']}")
    print(f"  requests per operation : {report['config']['requests_per_operation']} "
          f"(+ {report['config']['warmup_per_operation']} warmup)")
    print(f"  bare SELECT 1 (median) : {report['database_select1_median_ms']:.4f} ms")
    print(f"  total requests         : {report['total_requests']}")
    print(f"  overall success rate   : {report['overall_success_rate'] * 100:.1f}%")
    print()
    print(f"  {'operation':<24}{'reqs':>5}{'ok%':>6}{'p50':>9}{'p95':>9}{'min':>9}{'max':>9}")
    for r in report["operations"]:
        print(f"  {r['operation']:<24}{r['requests']:>5}{r['success_rate']*100:>5.0f}%"
              f"{r['p50_latency_ms']:>9.3f}{r['p95_latency_ms']:>9.3f}"
              f"{r['min_latency_ms']:>9.3f}{r['max_latency_ms']:>9.3f}")
    print()
    print("  DIRECT vs SERVICE (compute operations, P50 ms):")
    print(f"  {'operation':<22}{'direct':>10}{'service':>10}{'+abs':>10}{'x':>7}")
    for o in report["direct_vs_service"]:
        print(f"  {o['operation']:<22}{o['direct_p50_ms']:>10.3f}{o['service_p50_ms']:>10.3f}"
              f"{o['absolute_overhead_ms']:>10.3f}{(o['relative_overhead_x'] or 0):>7.1f}")
    print()
    print("  " + report["config"]["note"])


if __name__ == "__main__":
    sys.exit(main())
