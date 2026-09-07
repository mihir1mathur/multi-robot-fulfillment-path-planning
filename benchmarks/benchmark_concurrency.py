"""Reproducible benchmark: behaviour under CONCURRENCY.

WHAT IT MEASURES
----------------
Three things, all in-process on the development machine:

  A. API CONCURRENCY - the same fixed set of authenticated requests driven
     through the FastAPI app by a thread pool at increasing concurrency levels
     (1, 5, 10, 20). Reports throughput, latency percentiles, and how many
     requests succeeded. The point is the SHAPE (does throughput hold up, do
     errors appear), not the millisecond values.

  B. STATE INTEGRITY UNDER A RACE - many threads POST the *same* robot id at
     once. Exactly one must win (201); every other must get a clean 409; none
     may get a 500. This proves the "already exists?" check plus the database
     unique constraint plus the 409 handler hold together when requests
     interleave.

  C. FLEET-WORKLOAD DETERMINISM - a set of fixed multi-robot coordination
     scenarios run once sequentially and once through a thread pool. The
     per-scenario results (paths, makespan, waits, conflicts) must be
     byte-for-byte identical: the deterministic algorithms do not depend on
     wall-clock interleaving. Reports the elapsed time of each.

BACKENDS (A and B)
------------------
    default       SQLite in-memory  (one shared connection - writes serialise)
    --postgres    real PostgreSQL from TEST_DATABASE_URL / DATABASE_URL
                  (a real connection pool - writes run concurrently)

    python benchmarks/benchmark_concurrency.py
    python benchmarks/benchmark_concurrency.py --postgres

IMPORTANT FRAMING
-----------------
This is an IN-PROCESS benchmark. Requests go through the ASGI app via
``TestClient`` on one event loop; there is no network socket, no separate
server process, no OS scheduler contention with other machines. It is NOT a
load test and the numbers are NOT production latency or production throughput.
It is an engineering check that the concurrency handling is correct and does
not fall over.

Writes benchmark_concurrency_results.json / .csv next to this file.
"""

from __future__ import annotations

import csv
import json
import platform
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402

from robotics.api.app import create_app  # noqa: E402
from robotics.api.security import create_access_token, hash_password  # noqa: E402
from robotics.coordination.coordinator import MultiRobotCoordinator  # noqa: E402
from robotics.persistence.config import Settings, get_settings  # noqa: E402
from robotics.persistence.database import Base, Database  # noqa: E402
from robotics.persistence.models import UserRecord  # noqa: E402
from robotics.warehouse.grid import Position  # noqa: E402
from robotics.warehouse.warehouse import Warehouse  # noqa: E402

CONCURRENCY_LEVELS = (1, 5, 10, 20)
REQUESTS_PER_LEVEL = 120
RACE_THREADS = 24
FLEET_SIZES = (2, 5, 10, 20)
_JWT_KEY = "concurrency-benchmark-signing-key-at-least-32-bytes"
_OPERATOR_HASH = hash_password("bench-operator-pw")

OUT_JSON = Path(__file__).with_name("benchmark_concurrency_results.json")
OUT_CSV = Path(__file__).with_name("benchmark_concurrency_results.csv")


# ----------------------------------------------------------------------
# backend wiring (mirrors benchmark_service.py)
# ----------------------------------------------------------------------
def _postgres_url() -> str | None:
    s = get_settings()
    if s.test_database_url and s.test_database_url.startswith("postgresql"):
        return s.test_database_url
    if s.is_postgres:
        return s.database_url
    return None


def _make_app(backend: str):
    """Return (app, token, label, database, cleanup)."""
    cleanup: Callable[[], None] = lambda: None  # noqa: E731

    if backend == "postgres":
        url = _postgres_url()
        if not url:
            raise SystemExit(
                "--postgres requested but no PostgreSQL TEST_DATABASE_URL / "
                "DATABASE_URL is configured"
            )
        settings = Settings(
            database_url=url, log_level="ERROR", jwt_secret_key=_JWT_KEY,
            auto_create_tables=False, db_pool_size=10, db_max_overflow=10,
        )
        engine = create_engine(url, future=True, pool_pre_ping=True,
                               pool_size=10, max_overflow=10)
        database = Database(settings=settings, engine=engine)
        Base.metadata.drop_all(engine)
        database.create_all()
        label = f"postgresql ({engine.dialect.server_version_info})"
    else:
        # A FILE-based SQLite database (not :memory:) so each pooled connection
        # is a real, independent connection. SQLite still serialises writers
        # (one writer at a time, readers wait) - that is the point of comparing
        # it with PostgreSQL below - but with a busy timeout it does not error.
        tmpdir = tempfile.mkdtemp(prefix="bench_conc_")
        db_path = Path(tmpdir) / "bench.db"
        url = f"sqlite+pysqlite:///{db_path}"
        settings = Settings(
            database_url=url, log_level="ERROR", jwt_secret_key=_JWT_KEY,
            auto_create_tables=False,
        )
        engine = create_engine(
            url, future=True,
            connect_args={"check_same_thread": False, "timeout": 30},
        )
        database = Database(settings=settings, engine=engine)
        database.create_all()
        label = "sqlite file (writers serialise)"
        cleanup = lambda: shutil.rmtree(tmpdir, ignore_errors=True)  # noqa: E731

    with database.session_scope() as session:
        session.add(UserRecord(username="bench_op", password_hash=_OPERATOR_HASH,
                               role="operator"))

    token = create_access_token("bench_op", "operator", _JWT_KEY, expires_minutes=120)
    app = create_app(settings=settings, database=database, create_tables=False)
    return app, token, label, database, cleanup


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = pct * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


# ----------------------------------------------------------------------
# A. API concurrency
# ----------------------------------------------------------------------
_PLAN_BODIES = [
    {"start": {"row": 0, "col": 0}, "goal": {"row": 11, "col": 10}, "algorithm": "astar"},
    {"start": {"row": 0, "col": 0}, "goal": {"row": 4, "col": 6}, "algorithm": "dijkstra"},
    {"start": {"row": 11, "col": 0}, "goal": {"row": 0, "col": 11}, "algorithm": "astar"},
]


def _api_concurrency(app, token: str) -> list[dict]:
    headers = {"Authorization": f"Bearer {token}"}
    rows: list[dict] = []

    with TestClient(app) as client:
        # warmup
        for body in _PLAN_BODIES:
            client.post("/planning/path", json=body, headers=headers)

        for level in CONCURRENCY_LEVELS:
            latencies: list[float] = []
            statuses: list[int] = []

            def one(i: int) -> None:
                body = _PLAN_BODIES[i % len(_PLAN_BODIES)]
                started = time.perf_counter()
                r = client.post("/planning/path", json=body, headers=headers)
                latencies.append((time.perf_counter() - started) * 1000)
                statuses.append(r.status_code)

            wall_start = time.perf_counter()
            with ThreadPoolExecutor(max_workers=level) as pool:
                list(pool.map(one, range(REQUESTS_PER_LEVEL)))
            elapsed = time.perf_counter() - wall_start

            ok = sum(1 for s in statuses if s == 200)
            rows.append({
                "concurrency": level,
                "requests": REQUESTS_PER_LEVEL,
                "successful": ok,
                "failed": REQUESTS_PER_LEVEL - ok,
                "status_counts": _count(statuses),
                "elapsed_s": round(elapsed, 4),
                "throughput_req_per_s": round(REQUESTS_PER_LEVEL / elapsed, 1),
                "p50_latency_ms": round(_percentile(latencies, 0.50), 3),
                "p95_latency_ms": round(_percentile(latencies, 0.95), 3),
                "max_latency_ms": round(max(latencies), 3),
            })
    return rows


def _count(items) -> dict:
    out: dict = {}
    for x in items:
        out[str(x)] = out.get(str(x), 0) + 1
    return out


# ----------------------------------------------------------------------
# B. state integrity under a race
# ----------------------------------------------------------------------
def _race_on_duplicate_create(app, token: str) -> dict:
    headers = {"Authorization": f"Bearer {token}"}
    with TestClient(app) as client:
        body = {"robot_id": "RACE", "position": {"row": 0, "col": 0}}

        def attempt(_i: int) -> int:
            return client.post("/robots", json=body, headers=headers).status_code

        with ThreadPoolExecutor(max_workers=RACE_THREADS) as pool:
            codes = list(pool.map(attempt, range(RACE_THREADS)))

        listing = client.get("/robots", headers=headers).json()

    created = sum(1 for c in codes if c == 201)
    conflict = sum(1 for c in codes if c == 409)
    server_error = sum(1 for c in codes if c >= 500)
    return {
        "threads": RACE_THREADS,
        "status_counts": _count(codes),
        "created_201": created,
        "conflict_409": conflict,
        "server_error_5xx": server_error,
        "rows_for_RACE_in_db": sum(1 for r in listing["robots"] if r["robot_id"] == "RACE"),
        "integrity_held": created == 1 and server_error == 0
        and conflict == RACE_THREADS - 1,
    }


# ----------------------------------------------------------------------
# C. fleet-workload determinism (sequential vs thread pool)
# ----------------------------------------------------------------------
def _fleet_scenarios() -> list[dict]:
    """Fixed, reproducible open-warehouse coordination scenarios."""
    scenarios = []
    for fleet in FLEET_SIZES:
        size = max(8, fleet + 4)
        wh = Warehouse(width=size, height=size, name=f"conc-{fleet}")
        starts = {f"R{i:02d}": Position(i % size, 0) for i in range(fleet)}
        goals = {f"R{i:02d}": Position(i % size, size - 1) for i in range(fleet)}
        scenarios.append({"fleet": fleet, "warehouse": wh, "starts": starts, "goals": goals})
    return scenarios


def _solve(scenario: dict) -> tuple[int, tuple]:
    result = MultiRobotCoordinator(scenario["warehouse"]).plan(
        scenario["starts"], scenario["goals"]
    )
    signature = (
        result.success,
        result.planned_robot_count,
        result.makespan,
        result.total_move_steps,
        result.total_wait_steps,
        tuple(sorted(result.failed_robot_ids)),
    )
    return scenario["fleet"], signature


def _fleet_determinism() -> dict:
    scenarios = _fleet_scenarios()

    seq_start = time.perf_counter()
    sequential = dict(_solve(s) for s in scenarios)
    seq_elapsed = time.perf_counter() - seq_start

    par_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(scenarios)) as pool:
        concurrent = dict(pool.map(_solve, scenarios))
    par_elapsed = time.perf_counter() - par_start

    identical = sequential == concurrent
    return {
        "fleet_sizes": list(FLEET_SIZES),
        "scenarios": len(scenarios),
        "sequential_elapsed_s": round(seq_elapsed, 4),
        "concurrent_elapsed_s": round(par_elapsed, 4),
        "results_identical": identical,
        "per_fleet": {
            str(f): {
                "success": sig[0],
                "planned": sig[1],
                "makespan": sig[2],
                "move_steps": sig[3],
                "wait_steps": sig[4],
            }
            for f, sig in sorted(sequential.items())
        },
    }


# ----------------------------------------------------------------------
def run(backend: str) -> dict:
    app, token, label, database, cleanup = _make_app(backend)
    try:
        api_rows = _api_concurrency(app, token)
        race = _race_on_duplicate_create(app, token)
    finally:
        database.dispose()
        cleanup()

    fleet = _fleet_determinism()  # backend-independent (pure algorithm)

    return {
        "config": {
            "backend": label,
            "requested_backend": backend,
            "concurrency_levels": list(CONCURRENCY_LEVELS),
            "requests_per_level": REQUESTS_PER_LEVEL,
            "race_threads": RACE_THREADS,
            "transport": "fastapi.testclient (in-process ASGI, thread pool of clients)",
            "timer": "time.perf_counter",
            "python": platform.python_version(),
            "platform": platform.platform(),
            "note": "IN-PROCESS concurrency check on the dev machine. NOT a load "
            "test, NOT production latency/throughput. Only the success counts and "
            "the integrity/determinism booleans are deterministic.",
        },
        "api_concurrency": api_rows,
        "race_on_duplicate_create": race,
        "fleet_workload_determinism": fleet,
    }


def _write(report: dict) -> None:
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["concurrency", "requests", "successful", "failed",
                    "throughput_req_per_s", "p50_latency_ms", "p95_latency_ms",
                    "max_latency_ms", "elapsed_s"])
        for r in report["api_concurrency"]:
            w.writerow([r["concurrency"], r["requests"], r["successful"], r["failed"],
                        r["throughput_req_per_s"], r["p50_latency_ms"],
                        r["p95_latency_ms"], r["max_latency_ms"], r["elapsed_s"]])


def _print_summary(report: dict) -> None:
    print("=" * 78)
    print("CONCURRENCY BENCHMARK")
    print("=" * 78)
    print(f"  backend : {report['config']['backend']}")
    print()
    print("  A. API CONCURRENCY")
    print(f"  {'conc':>5}{'reqs':>7}{'ok':>6}{'fail':>6}{'req/s':>10}{'p50':>9}{'p95':>9}{'max':>9}")
    for r in report["api_concurrency"]:
        print(f"  {r['concurrency']:>5}{r['requests']:>7}{r['successful']:>6}{r['failed']:>6}"
              f"{r['throughput_req_per_s']:>10}{r['p50_latency_ms']:>9.2f}"
              f"{r['p95_latency_ms']:>9.2f}{r['max_latency_ms']:>9.2f}")
    print()
    race = report["race_on_duplicate_create"]
    print("  B. STATE INTEGRITY UNDER A RACE (same robot id from many threads)")
    print(f"     {race['threads']} threads -> 201:{race['created_201']}  "
          f"409:{race['conflict_409']}  5xx:{race['server_error_5xx']}  "
          f"rows in db:{race['rows_for_RACE_in_db']}")
    print(f"     integrity held: {race['integrity_held']}")
    print()
    fleet = report["fleet_workload_determinism"]
    print("  C. FLEET-WORKLOAD DETERMINISM (sequential vs thread pool)")
    print(f"     fleets {fleet['fleet_sizes']}  "
          f"sequential {fleet['sequential_elapsed_s']}s  "
          f"concurrent {fleet['concurrent_elapsed_s']}s")
    print(f"     results identical: {fleet['results_identical']}")
    print()
    print("  " + report["config"]["note"])


def main(argv: list | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    backend = "postgres" if "--postgres" in argv else "sqlite"
    report = run(backend)
    _write(report)
    _print_summary(report)
    print(f"\n  wrote {OUT_JSON.name} and {OUT_CSV.name} to {OUT_JSON.parent}")

    # a non-zero exit if an integrity / determinism invariant broke
    ok = (
        report["race_on_duplicate_create"]["integrity_held"]
        and report["fleet_workload_determinism"]["results_identical"]
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
