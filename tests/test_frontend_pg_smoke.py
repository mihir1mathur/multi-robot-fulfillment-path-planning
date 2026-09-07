"""One end-to-end dashboard-flow run against real PostgreSQL.

Skips unless TEST_DATABASE_URL points at PostgreSQL (same rule as
tests/test_postgres_integration.py). Mirrors the SQLite end-to-end test but
proves the persisted allocation/coordination/recovery flow the dashboard drives
also holds on the production database engine.
"""

from __future__ import annotations

from frontend import scenario, view_model
from frontend.api_client import FulfillmentClient


def _client(pg_app, pg_database) -> FulfillmentClient:
    from fastapi.testclient import TestClient

    from tests.conftest import _token_for

    c = FulfillmentClient(base_url="http://testserver", _client=TestClient(pg_app))
    c.token = _token_for(pg_database, "operator")
    return c


def test_dashboard_flow_against_real_postgres(pg_database) -> None:
    from robotics.api.app import create_app

    app = create_app(settings=pg_database.settings, database=pg_database,
                     create_tables=False)
    client = _client(app, pg_database)

    for body in scenario.robot_bodies():
        client.create_robot(body)
    for body in scenario.task_bodies():
        client.create_task(body)

    alloc = client.run_allocation(
        {"algorithm": "cp_sat", "from_persisted": True, "commit": True}
    )
    assert alloc["committed"] and len(alloc["assignments"]) == 3

    robots = {r["robot_id"]: r for r in client.list_robots()}
    tasks = {t["task_id"]: t for t in client.list_tasks()}
    for a in alloc["assignments"]:
        assert robots[a["robot_id"]]["assigned_task_id"] == a["task_id"]
        assert tasks[a["task_id"]]["assigned_robot_id"] == a["robot_id"]
    assert robots[scenario.SPARE_ROBOT_ID]["status"] == "idle"

    body = view_model.coordination_request_robots(alloc["assignments"], robots, tasks)
    coord = client.plan_coordination({"robots": body})
    assert coord["planned_robot_count"] == 3 and coord["failed_robot_ids"] == []

    picked = view_model.pick_dynamic_obstacle(coord)
    obstacle = client.recover_obstacle(
        {"robots": body, "obstacle": picked["cell"], "timestep": picked["timestep"]}
    )
    assert obstacle["affected_robot_ids"] and obstacle["replanning_attempted"]

    fail_body = view_model.recovery_request_robots(
        alloc["assignments"], robots, tasks, scenario.SPARE_ROBOT_ID
    )
    failed = sorted(a["robot_id"] for a in alloc["assignments"])[0]
    failed_task = next(
        a["task_id"] for a in alloc["assignments"] if a["robot_id"] == failed
    )
    failure = client.recover_robot_failure(
        {"robots": fail_body, "failed_robot_id": failed,
         "timestep": scenario.FAILURE_TIMESTEP, "commit": True}
    )
    assert failure["task_reassignment_success"]
    assert failure["reassigned_to"] == scenario.SPARE_ROBOT_ID
    assert failure["committed"] is True

    # the successful reassignment is now reflected in real PostgreSQL
    robots_after = {r["robot_id"]: r for r in client.list_robots()}
    tasks_after = {t["task_id"]: t for t in client.list_tasks()}
    assert robots_after[failed]["status"] == "offline"
    assert robots_after[failed]["assigned_task_id"] is None
    assert robots_after[scenario.SPARE_ROBOT_ID]["assigned_task_id"] == failed_task
    assert tasks_after[failed_task]["assigned_robot_id"] == scenario.SPARE_ROBOT_ID


def test_initialise_scenario_clears_stale_rows_against_real_postgres(pg_database) -> None:
    """The Event-Log count and the persisted tables agree, even when an older
    version of the demo left extra rows behind."""
    from robotics.api.app import create_app
    from robotics.simulation import scenario as old_scenario

    app = create_app(settings=pg_database.settings, database=pg_database,
                     create_tables=False)
    client = _client(app, pg_database)

    # an earlier run persisted the old 4-task demo set
    for r in old_scenario.build_sample_robots():
        client.create_robot({
            "robot_id": r.robot_id,
            "position": {"row": r.position.row, "col": r.position.col},
            "battery_level": r.battery_level,
            "payload_capacity": r.payload_capacity,
            "status": "idle",
        })
    for t in old_scenario.build_sample_tasks():
        client.create_task({
            "task_id": t.task_id,
            "pickup": {"row": t.pickup_location.row, "col": t.pickup_location.col},
            "dropoff": {"row": t.dropoff_location.row, "col": t.dropoff_location.col},
            "payload_weight": t.payload_weight,
            "priority": t.priority.name.lower(),
        })

    summary = scenario.sync_persisted_scenario(client)
    assert summary == {"robots": 4, "tasks": 3, "removed": ["T4"]}
    assert {t["task_id"] for t in client.list_tasks()} == {"T1", "T2", "T3"}
    assert len(client.list_tasks()) == summary["tasks"]
    assert len(client.list_robots()) == summary["robots"]

    alloc = client.run_allocation(
        {"algorithm": "cp_sat", "from_persisted": True, "commit": True}
    )
    assert alloc["committed"] and len(alloc["assignments"]) == 3
    robots = {r["robot_id"]: r for r in client.list_robots()}
    assert robots[scenario.SPARE_ROBOT_ID]["status"] == "idle"


def test_reset_then_initialise_clean_scenario_against_real_postgres(pg_database) -> None:
    """Reset (session-only) + Initialise gives the exact clean demo scenario on
    real PostgreSQL, and the dashboard warns while it is only reset."""
    from robotics.api.app import create_app

    app = create_app(settings=pg_database.settings, database=pg_database,
                     create_tables=False)
    client = _client(app, pg_database)

    scenario.sync_persisted_scenario(client)
    client.run_allocation(
        {"algorithm": "cp_sat", "from_persisted": True, "commit": True}
    )

    # Reset scenario is session-only: persisted assignments survive, and the
    # dashboard surfaces a notice rather than a silent stale table.
    notice = view_model.persisted_scenario_notice(
        False, client.list_robots(), client.list_tasks()
    )
    assert notice is not None and "1 - Initialise scenario" in notice

    summary = scenario.sync_persisted_scenario(client)   # 1 - Initialise scenario
    assert summary == {"robots": 4, "tasks": 3, "removed": []}
    robots = client.list_robots()
    tasks = client.list_tasks()
    assert all(r["status"] == "idle" and r["assigned_task_id"] is None for r in robots)
    assert all(t["status"] == "pending" and t["assigned_robot_id"] is None for t in tasks)
    assert view_model.persisted_scenario_notice(True, robots, tasks) is None
