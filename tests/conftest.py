"""Shared pytest fixtures and import setup.

`conftest.py` is a file pytest loads automatically before running any test in
this directory. Two jobs here:

1. Put the project root on `sys.path` so `import robotics` works whether the
   suite is started with `pytest`, `python -m pytest`, or from an IDE.
2. Provide small fixtures so individual tests do not each rebuild the same
   warehouse by hand. A fixture is just a named, reusable piece of setup;
   pytest calls it fresh for every test, so tests cannot leak state into one
   another.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robotics.robots.robot import Robot  # noqa: E402
from robotics.simulation.simulator import WarehouseSimulator  # noqa: E402
from robotics.tasks.task import Task, TaskPriority  # noqa: E402
from robotics.warehouse.grid import Grid, Position  # noqa: E402
from robotics.warehouse.warehouse import LocationType, Warehouse  # noqa: E402


@pytest.fixture
def empty_grid() -> Grid:
    """A 5x5 grid (5 rows, 5 cols) with nothing blocked."""
    return Grid(width=5, height=5)


@pytest.fixture
def small_warehouse() -> Warehouse:
    """A 5x5 warehouse with one rack, one charger and one drop-off.

        col  0  1  2  3  4
    row 0    C  .  .  .  .
    row 1    .  .  #  .  .
    row 2    .  .  .  .  .
    row 3    .  .  .  .  .
    row 4    .  .  .  .  D
    """
    warehouse = Warehouse(width=5, height=5, name="test-warehouse")
    warehouse.add_static_obstacle("rack-1", Position(1, 2), "test rack")
    warehouse.mark_location(Position(0, 0), LocationType.CHARGING)
    warehouse.mark_location(Position(4, 4), LocationType.DROPOFF)
    return warehouse


@pytest.fixture
def simulator(small_warehouse: Warehouse) -> WarehouseSimulator:
    """A simulator over `small_warehouse` with no robots or tasks yet."""
    return WarehouseSimulator(small_warehouse)


@pytest.fixture
def robot() -> Robot:
    """A default robot standing at (2, 2) with a full battery."""
    return Robot(robot_id="R1", position=Position(2, 2))


@pytest.fixture
def task() -> Task:
    """A default 2kg task from (0, 1) to (4, 4)."""
    return Task(
        task_id="T1",
        pickup_location=Position(0, 1),
        dropoff_location=Position(4, 4),
        priority=TaskPriority.NORMAL,
        payload_weight=2.0,
    )


# ----------------------------------------------------------------------
# Fixtures for path planning
# ----------------------------------------------------------------------
@pytest.fixture
def open_warehouse() -> Warehouse:
    """A 6x6 warehouse with nothing in it at all.

    The simplest possible planning case: the shortest route between any two
    cells is exactly their Manhattan distance, which makes expected costs easy
    to state by hand.
    """
    return Warehouse(width=6, height=6, name="open-warehouse")


@pytest.fixture
def wall_warehouse() -> Warehouse:
    """A 7x7 warehouse split by a wall with a single gap at the bottom.

        col  0  1  2  3  4  5  6
    row 0    .  .  .  #  .  .  .
    row 1    .  .  .  #  .  .  .
    row 2    .  .  .  #  .  .  .
    row 3    .  .  .  #  .  .  .
    row 4    .  .  .  #  .  .  .
    row 5    .  .  .  #  .  .  .
    row 6    .  .  .  .  .  .  .     <- the only way through

    Any route from the left half to the right half must detour to row 6, so
    the shortest cost is easy to reason about by hand and a planner that
    ignored obstacles would be caught immediately.
    """
    warehouse = Warehouse(width=7, height=7, name="wall-warehouse")
    for row in range(6):
        warehouse.add_static_obstacle(f"wall-{row}", Position(row, 3), "dividing wall")
    return warehouse


@pytest.fixture
def sealed_warehouse() -> Warehouse:
    """A 5x5 warehouse where (0, 4) is walled into its own corner.

        col  0  1  2  3  4
    row 0    .  .  .  #  X     <- X = (0,4), reachable only through blocked cells
    row 1    .  .  .  #  #
    row 2    .  .  .  .  .
    row 3    .  .  .  .  .
    row 4    .  .  .  .  .

    Used to test that both planners fail cleanly on a genuinely unreachable
    goal, rather than looping forever or returning a nonsense path.
    """
    warehouse = Warehouse(width=5, height=5, name="sealed-warehouse")
    warehouse.add_static_obstacle("seal-a", Position(0, 3), "sealing wall")
    warehouse.add_static_obstacle("seal-b", Position(1, 3), "sealing wall")
    warehouse.add_static_obstacle("seal-c", Position(1, 4), "sealing wall")
    return warehouse


# ----------------------------------------------------------------------
# Fixtures for the service / persistence / API layer
#
# Fast tests get their OWN in-memory SQLite database (a StaticPool so every
# connection shares the one in-memory instance). Nothing touches an external
# server, so these tests stay deterministic and need no setup.
#
# The dedicated PostgreSQL integration tests (tests/test_postgres_integration.py)
# use the `pg_database` fixture, which skips unless TEST_DATABASE_URL points at a
# real PostgreSQL. In CI that variable points at the postgres service container.
# ----------------------------------------------------------------------
_TEST_USERS = (
    ("viewer_user", "viewer-pw-123", "viewer"),
    ("operator_user", "operator-pw-123", "operator"),
    ("admin_user", "admin-pw-123", "admin"),
)

# bcrypt at the production cost factor is deliberately slow (~0.25 s per hash).
# Seeding three users in every API test would add ~100 s to the suite, so the
# test hashes are computed ONCE here and inserted directly. Production hashing
# (robotics.api.security.hash_password) is untouched and still uses the default
# cost. tests/test_authentication.py exercises the real hash/verify path.
_HASH_CACHE: dict = {}


def _test_hash(password: str) -> str:
    if password not in _HASH_CACHE:
        import bcrypt

        _HASH_CACHE[password] = bcrypt.hashpw(
            password.encode(), bcrypt.gensalt(rounds=4)
        ).decode()
    return _HASH_CACHE[password]


def _seed_users(database) -> None:
    from robotics.persistence.models import UserRecord

    with database.session_scope() as session:
        for username, password, role in _TEST_USERS:
            session.add(
                UserRecord(
                    username=username,
                    password_hash=_test_hash(password),
                    role=role,
                )
            )


def _token_for(database, role: str) -> str:
    from robotics.services.auth_service import AuthService

    with database.session_scope() as session:
        service = AuthService(session, database.settings)
        username = next(u for u, _, r in _TEST_USERS if r == role)
        user = service.get_by_username(username)
        token, _ = service.issue_token(user)
        return token


@pytest.fixture
def api_database():
    """A fresh, isolated in-memory SQLite database with the schema + test users."""
    from sqlalchemy import create_engine
    from sqlalchemy.pool import StaticPool

    from robotics.persistence.config import Settings
    from robotics.persistence.database import Database

    settings = Settings(
        database_url="sqlite+pysqlite://",
        log_level="WARNING",
        jwt_secret_key="test-only-signing-key-that-is-at-least-32-bytes-long",
        auto_create_tables=False,
    )
    engine = create_engine(
        "sqlite+pysqlite://",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    database = Database(settings=settings, engine=engine)
    database.create_all()
    _seed_users(database)
    try:
        yield database
    finally:
        database.drop_all()
        engine.dispose()


@pytest.fixture
def db_session(api_database):
    """A single session against the test database, for direct persistence tests."""
    session = api_database.session()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture
def api_app(api_database):
    """A FastAPI app wired to the isolated test database."""
    from robotics.api.app import create_app

    return create_app(
        settings=api_database.settings,
        database=api_database,
        create_tables=False,
    )


@pytest.fixture
def anon_client(api_app):
    """A TestClient with NO Authorization header - for testing 401 paths."""
    from fastapi.testclient import TestClient

    with TestClient(api_app) as client:
        yield client


def _authed_client(api_app, api_database, role: str):
    from fastapi.testclient import TestClient

    client = TestClient(api_app)
    client.headers["Authorization"] = f"Bearer {_token_for(api_database, role)}"
    with client:
        yield client


@pytest.fixture
def api_client(api_app, api_database):
    """The default TestClient - authenticated as an OPERATOR (can read + mutate
    + run planning/allocation/coordination/recovery). Used by the bulk of the
    existing API tests."""
    yield from _authed_client(api_app, api_database, "operator")


@pytest.fixture
def viewer_client(api_app, api_database):
    """Authenticated as a VIEWER - may GET, must be 403 on any mutation."""
    yield from _authed_client(api_app, api_database, "viewer")


@pytest.fixture
def admin_client(api_app, api_database):
    """Authenticated as an ADMIN - everything, plus /auth/register."""
    yield from _authed_client(api_app, api_database, "admin")


# ----------------------------------------------------------------------
# REAL PostgreSQL fixtures (used only by tests/test_postgres_integration.py).
#
# `pg_database` skips the test unless TEST_DATABASE_URL points at a live
# PostgreSQL. It applies the Alembic migrations (so the integration tests also
# prove the migrations produce a working schema), seeds the test users, and
# TRUNCATEs the app tables after each test so tests stay isolated without
# dropping the schema every time.
# ----------------------------------------------------------------------
def _pg_url() -> str | None:
    from robotics.persistence.config import Settings

    url = Settings().test_database_url
    return url if url and url.startswith("postgresql") else None


@pytest.fixture(scope="session")
def _pg_engine():
    url = _pg_url()
    if url is None:
        pytest.skip("TEST_DATABASE_URL is not a PostgreSQL URL - skipping PG integration")
    from sqlalchemy import create_engine

    engine = create_engine(url, future=True, pool_pre_ping=True)
    try:
        with engine.connect() as conn:
            conn.execute(__import__("sqlalchemy").text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"cannot reach TEST_DATABASE_URL PostgreSQL: {exc}")

    # Bring the schema to head via the real migrations.
    from alembic import command
    from alembic.config import Config

    cfg = Config(str(PROJECT_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(PROJECT_ROOT / "alembic"))
    # Pass the URL via config.attributes rather than set_main_option: the latter
    # routes through configparser, whose %-interpolation rejects the %-encoded
    # characters in a real URL-encoded PostgreSQL password. env.py reads this key.
    cfg.attributes["db_url"] = url
    command.upgrade(cfg, "head")

    yield engine
    engine.dispose()


@pytest.fixture
def pg_database(_pg_engine):
    """A `Database` bound to the real PostgreSQL test DB, cleaned per test."""
    from sqlalchemy import text

    from robotics.persistence.config import Settings
    from robotics.persistence.database import Database

    settings = Settings(
        database_url=_pg_url(),
        log_level="WARNING",
        jwt_secret_key="test-only-signing-key-that-is-at-least-32-bytes-long",
        auto_create_tables=False,
    )
    database = Database(settings=settings, engine=_pg_engine)

    _tables = (
        "recovery_events", "coordination_runs", "allocation_runs", "planning_runs",
        "tasks", "robots", "users",
    )

    def _wipe():
        with _pg_engine.begin() as conn:
            conn.execute(text("TRUNCATE " + ", ".join(_tables) + " RESTART IDENTITY CASCADE"))

    _wipe()
    _seed_users(database)
    try:
        yield database
    finally:
        _wipe()


@pytest.fixture
def pg_client(pg_database):
    """An operator-authenticated TestClient backed by real PostgreSQL."""
    from fastapi.testclient import TestClient

    from robotics.api.app import create_app

    app = create_app(settings=pg_database.settings, database=pg_database, create_tables=False)
    client = TestClient(app)
    client.headers["Authorization"] = f"Bearer {_token_for(pg_database, 'operator')}"
    with client:
        yield client


# ----------------------------------------------------------------------
# Concurrency fixtures.
#
# The fast API tests share ONE in-memory SQLite connection (StaticPool), which
# cannot service overlapping requests. Concurrency tests instead get a
# FILE-based SQLite database with a normal pool, so every worker thread has its
# own real connection. SQLite still serialises writers (one at a time, a busy
# timeout makes them wait instead of erroring) - fine for asserting correctness
# under interleaving. Real concurrent throughput is a PostgreSQL concern and is
# measured in benchmarks/benchmark_concurrency.py and
# tests/test_database_concurrency.py.
# ----------------------------------------------------------------------
@pytest.fixture
def concurrent_database(tmp_path):
    from sqlalchemy import create_engine

    from robotics.persistence.config import Settings
    from robotics.persistence.database import Database

    db_file = tmp_path / "concurrent.db"
    url = f"sqlite+pysqlite:///{db_file}"
    settings = Settings(
        database_url=url,
        log_level="WARNING",
        jwt_secret_key="test-only-signing-key-that-is-at-least-32-bytes-long",
        auto_create_tables=False,
    )
    engine = create_engine(
        url,
        future=True,
        connect_args={"check_same_thread": False, "timeout": 30},
    )
    database = Database(settings=settings, engine=engine)
    database.create_all()
    _seed_users(database)
    try:
        yield database
    finally:
        database.drop_all()
        engine.dispose()


@pytest.fixture
def concurrent_client(concurrent_database):
    """Operator-authenticated TestClient over a file-based SQLite DB whose pool
    gives each worker thread its own connection - safe for overlapping calls."""
    from fastapi.testclient import TestClient

    from robotics.api.app import create_app

    app = create_app(
        settings=concurrent_database.settings,
        database=concurrent_database,
        create_tables=False,
    )
    client = TestClient(app)
    client.headers["Authorization"] = f"Bearer {_token_for(concurrent_database, 'operator')}"
    with client:
        yield client
