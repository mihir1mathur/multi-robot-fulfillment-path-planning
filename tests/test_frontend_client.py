"""Tests for the Streamlit dashboard's non-UI logic.

The dashboard itself (frontend/app.py) is not tested pixel-by-pixel. What is
tested here is everything the dashboard's correctness actually depends on:

    * the HTTP client - headers, timeouts turned into ApiUnavailable, 4xx/5xx
      turned into ApiError carrying the service's message, no credential ever
      placed in an exception or URL,
    * the pure view-model transforms - API payload -> table rows / grid data /
      event lines / metrics,
    * the fixed scenario bodies are well-formed for the API schemas.

The backend is a mock httpx transport - no server, no database.
"""

from __future__ import annotations

import httpx
import pytest

from frontend import scenario, view_model
from frontend.api_client import ApiError, ApiUnavailable, FulfillmentClient


# ----------------------------------------------------------------------
# HTTP client
# ----------------------------------------------------------------------
def make_client(handler) -> FulfillmentClient:
    transport = httpx.MockTransport(handler)
    http = httpx.Client(transport=transport, base_url="http://testserver")
    return FulfillmentClient(base_url="http://testserver", _client=http)


def test_login_posts_form_and_stores_token() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = request.content.decode()
        return httpx.Response(200, json={"access_token": "tok-123", "role": "operator",
                                         "expires_in": 1800})

    client = make_client(handler)
    token = client.login("operator_user", "s3cret-pw")
    assert token == "tok-123"
    assert client.token == "tok-123"
    assert "username=operator_user" in seen["body"]
    # the password is in the form body (required) but never in the URL
    assert "s3cret-pw" not in seen["url"]


def test_auth_header_is_attached_after_login() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/auth/token":
            return httpx.Response(200, json={"access_token": "tok-xyz", "role": "operator",
                                             "expires_in": 1800})
        assert request.headers.get("Authorization") == "Bearer tok-xyz"
        return httpx.Response(200, json={"robots": [], "count": 0})

    client = make_client(handler)
    client.login("u", "p")
    assert client.list_robots() == []


def test_ready_maps_503_to_not_ready() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"status": "not_ready",
                                         "checks": {"database": "unavailable"}})

    client = make_client(handler)
    assert client.ready()["status"] == "not_ready"


def test_4xx_becomes_api_error_with_service_message() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json={"error": "validation_failed",
                                         "message": "start cell is blocked"})

    client = make_client(handler)
    with pytest.raises(ApiError) as excinfo:
        client.plan_path({"start": {"row": 0, "col": 0}, "goal": {"row": 1, "col": 1}})
    assert excinfo.value.status_code == 422
    assert excinfo.value.message == "start cell is blocked"
    assert excinfo.value.error_code == "validation_failed"


def test_connection_error_becomes_api_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    client = make_client(handler)
    with pytest.raises(ApiUnavailable):
        client.health()


def test_delete_robot_and_task_issue_delete_and_accept_204() -> None:
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path))
        return httpx.Response(204)

    client = make_client(handler)
    assert client.delete_robot("R1") is None
    assert client.delete_task("T4") is None
    assert seen == [("DELETE", "/robots/R1"), ("DELETE", "/tasks/T4")]


def test_delete_surfaces_404_as_api_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "not_found",
                                         "message": "no task with id 'T9'"})

    client = make_client(handler)
    with pytest.raises(ApiError) as excinfo:
        client.delete_task("T9")
    assert excinfo.value.status_code == 404


def test_list_robots_passes_status_filter() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params.get("status") == "idle"
        return httpx.Response(200, json={"robots": [{"robot_id": "R1"}], "count": 1})

    client = make_client(handler)
    assert client.list_robots(status="idle") == [{"robot_id": "R1"}]


# ----------------------------------------------------------------------
# view_model transforms
# ----------------------------------------------------------------------
def test_persisted_scenario_notice_lifecycle() -> None:
    idle = [{"robot_id": "R1", "assigned_task_id": None}]
    pending = [{"task_id": "T1", "assigned_robot_id": None}]
    assigned_r = [{"robot_id": "R1", "assigned_task_id": "T1"}]
    assigned_t = [{"task_id": "T1", "assigned_robot_id": "R1"}]

    # scenario initialised this session -> never warn
    assert view_model.persisted_scenario_notice(True, assigned_r, assigned_t) is None
    # nothing persisted -> nothing to be confused by
    assert view_model.persisted_scenario_notice(False, [], []) is None
    # not initialised this session, rows exist -> warn, and mention assignments
    msg = view_model.persisted_scenario_notice(False, assigned_r, assigned_t)
    assert msg is not None
    assert "1 - Initialise scenario" in msg
    assert "live assignment" in msg
    # not initialised, idle/pending leftovers -> still warn, without the count
    msg2 = view_model.persisted_scenario_notice(False, idle, pending)
    assert msg2 is not None and "live assignment" not in msg2


def test_robot_rows_shape_and_sort() -> None:
    rows = view_model.robot_rows([
        {"robot_id": "R2", "status": "moving", "position": {"row": 3, "col": 4},
         "battery_level": 88.25, "assigned_task_id": "T1"},
        {"robot_id": "R1", "status": "idle", "position": {"row": 0, "col": 0},
         "battery_level": 100.0, "assigned_task_id": None},
    ])
    assert [r["robot"] for r in rows] == ["R1", "R2"]
    assert rows[1]["position"] == "(3, 4)"
    assert rows[1]["assigned task"] == "T1"
    assert rows[0]["assigned task"] == "-"


def test_coordinated_routes_keeps_wait_steps() -> None:
    resp = {"routes": {"R1": {"steps": [{"row": 0, "col": 0, "timestep": 0},
                                         {"row": 0, "col": 0, "timestep": 1},
                                         {"row": 0, "col": 1, "timestep": 2}]}}}
    routes = view_model.coordinated_routes(resp)
    assert routes["R1"] == [(0, 0), (0, 0), (0, 1)]


def test_scenario_metrics_only_reports_present_data() -> None:
    metrics = view_model.scenario_metrics(None, None, [])
    assert metrics == {}

    alloc = {"algorithm": "cp_sat", "assignments": [{"robot_id": "R1", "task_id": "T1",
             "total_estimated_cost": 10}], "unassigned_task_ids": ["T2"],
             "total_estimated_cost": 10}
    metrics = view_model.scenario_metrics(alloc, None, [])
    assert metrics["tasks assigned"] == 1
    assert metrics["tasks unassigned"] == 1
    assert metrics["assignment rate"] == 0.5
    assert "makespan" not in metrics  # no coordination given


def test_recovery_events_reports_reassignment_and_safe_stop() -> None:
    resp = {"affected_robot_ids": ["R1"], "task_reassignment_attempted": True,
            "reassigned_from": "R1", "reassigned_to": "R3",
            "task_reassignment_success": True, "safe_stop": False}
    events = view_model.recovery_events(resp, "failure")
    assert any("R1 -> R3" in e for e in events)


def test_health_banner_handles_missing_backend() -> None:
    assert view_model.health_banner(None, None) == "backend: unreachable"
    assert "ok" in view_model.health_banner({"status": "ok"}, {"checks": {"database": "ok"}})


# ----------------------------------------------------------------------
# scenario bodies match the API schemas
# ----------------------------------------------------------------------
def test_robot_bodies_are_valid_for_robotcreate() -> None:
    from robotics.api.schemas.robots import RobotCreate

    bodies = scenario.robot_bodies()
    assert len(bodies) == 4
    for body in bodies:
        RobotCreate.model_validate(body)  # raises on a bad body


def test_task_bodies_are_valid_for_taskcreate() -> None:
    from robotics.api.schemas.tasks import TaskCreate

    bodies = scenario.task_bodies()
    assert len(bodies) == 3
    for body in bodies:
        TaskCreate.model_validate(body)


def test_scenario_has_a_real_spare_robot() -> None:
    # 4 robots, 3 tasks -> CP-SAT leaves exactly one robot idle, and the module
    # names it. The failure demo depends on this.
    robot_ids = {b["robot_id"] for b in scenario.robot_bodies()}
    task_ids = {b["task_id"] for b in scenario.task_bodies()}
    assert scenario.SPARE_ROBOT_ID in robot_ids
    assert len(robot_ids) == len(task_ids) + 1


def test_allocation_specs_are_valid_for_allocationrequest() -> None:
    from robotics.api.schemas.allocation import AllocationRequest

    specs = scenario.allocation_specs()
    AllocationRequest.model_validate({"algorithm": "cp_sat", **specs})


# ----------------------------------------------------------------------
# view_model: request builders and the dynamic-obstacle picker
# ----------------------------------------------------------------------
def test_coordination_request_robots_pairs_start_and_dropoff() -> None:
    body = view_model.coordination_request_robots(
        [{"robot_id": "R1", "task_id": "T1"}],
        {"R1": {"robot_id": "R1", "position": {"row": 0, "col": 4}}},
        {"T1": {"task_id": "T1", "pickup": {"row": 1, "col": 4},
                "dropoff": {"row": 11, "col": 4}}},
    )
    assert body == [{"robot_id": "R1", "start": {"row": 0, "col": 4},
                     "goal": {"row": 11, "col": 4}}]


def test_recovery_request_robots_appends_parked_spare() -> None:
    body = view_model.recovery_request_robots(
        [{"robot_id": "R1", "task_id": "T1"}],
        {"R1": {"robot_id": "R1", "position": {"row": 0, "col": 4}},
         "R4": {"robot_id": "R4", "position": {"row": 0, "col": 10}}},
        {"T1": {"task_id": "T1", "pickup": {"row": 1, "col": 4},
                "dropoff": {"row": 11, "col": 4}, "payload_weight": 3.0,
                "priority": "high"}},
        spare_robot_id="R4",
    )
    assert [r["robot_id"] for r in body] == ["R1", "R4"]
    assert body[0]["task"]["task_id"] == "T1"
    assert "task" not in body[1]
    assert body[1]["start"] == body[1]["goal"] == {"row": 0, "col": 10}


def test_pick_dynamic_obstacle_lands_on_a_planned_route() -> None:
    coord = {"routes": {
        "R1": {"success": True, "steps": [
            {"row": 0, "col": 4, "timestep": t} for t in range(8)]},
        "R2": {"success": True, "steps": [
            {"row": 4, "col": c, "timestep": c} for c in range(5)]},
    }}
    picked = view_model.pick_dynamic_obstacle(coord)
    assert picked is not None
    # longest route is R1 (8 steps); the cell must be one of its cells
    assert picked["robot_id"] == "R1"
    r1_cells = {(s["row"], s["col"]) for s in coord["routes"]["R1"]["steps"]}
    assert (picked["cell"]["row"], picked["cell"]["col"]) in r1_cells
    assert picked["timestep"] >= 1
    assert view_model.pick_dynamic_obstacle(None) is None
    assert view_model.pick_dynamic_obstacle({"routes": {}}) is None


def test_grid_layout_is_the_default_profile() -> None:
    layout = scenario.grid_layout()
    assert layout["width"] == 12 and layout["height"] == 12
    assert layout["blocked"]
    assert layout["pickups"] and layout["dropoffs"] and layout["chargers"]


# ----------------------------------------------------------------------
# plotting (headless) and app module hygiene
# ----------------------------------------------------------------------
def test_draw_warehouse_renders_without_a_display() -> None:
    from frontend.plotting import draw_warehouse

    layout = scenario.grid_layout()
    robots = [{"robot_id": "R1", "status": "idle", "position": {"row": 0, "col": 1}},
              {"robot_id": "R2", "status": "offline", "position": {"row": 5, "col": 5}}]
    routes = {"R1": [(0, 1), (0, 2), (1, 2)], "R2": []}
    fig = draw_warehouse(layout, robots, routes, obstacle=(1, 4))
    assert fig is not None
    assert len(fig.axes) == 1


def test_app_module_parses() -> None:
    import ast
    from pathlib import Path

    source = Path(__file__).resolve().parents[1] / "frontend" / "app.py"
    ast.parse(source.read_text(encoding="utf-8"))


# ----------------------------------------------------------------------
# End-to-end: the dashboard's client against the REAL FastAPI app
# (in-process via TestClient - no network, no separate server)
# ----------------------------------------------------------------------
def _bound_client(api_app, api_database) -> FulfillmentClient:
    from fastapi.testclient import TestClient

    http = TestClient(api_app)
    client = FulfillmentClient(base_url="http://testserver", _client=http)
    from tests.conftest import _token_for

    client.token = _token_for(api_database, "operator")
    return client


def _persisted_maps(client):
    robots = {r["robot_id"]: r for r in client.list_robots()}
    tasks = {t["task_id"]: t for t in client.list_tasks()}
    return robots, tasks


def test_dashboard_flow_against_real_api(api_app, api_database) -> None:
    client = _bound_client(api_app, api_database)

    assert client.health()["status"] == "ok"
    assert client.ready()["status"] == "ready"

    # --- 1. initialise -------------------------------------------------
    for body in scenario.robot_bodies():
        client.create_robot(body)
    for body in scenario.task_bodies():
        client.create_task(body)
    assert len(client.list_robots()) == 4
    assert len(client.list_tasks()) == 3
    assert all(r["status"] == "idle" for r in client.list_robots())

    # --- 2. allocate: from persisted state, committed -----------------
    alloc = client.run_allocation(
        {"algorithm": "cp_sat", "from_persisted": True, "commit": True}
    )
    assert alloc["committed"] is True
    assert len(alloc["assignments"]) == 3
    assert alloc["unassigned_task_ids"] == []

    robots_by_id, tasks_by_id = _persisted_maps(client)
    # the spare robot is left idle; every other robot now carries its task
    assert robots_by_id[scenario.SPARE_ROBOT_ID]["status"] == "idle"
    assert robots_by_id[scenario.SPARE_ROBOT_ID]["assigned_task_id"] is None
    for a in alloc["assignments"]:
        assert robots_by_id[a["robot_id"]]["status"] == "assigned"
        assert robots_by_id[a["robot_id"]]["assigned_task_id"] == a["task_id"]
        assert tasks_by_id[a["task_id"]]["status"] == "assigned"
        assert tasks_by_id[a["task_id"]]["assigned_robot_id"] == a["robot_id"]

    # Fleet Status / Task Status tables agree with the allocation metrics
    robot_rows = view_model.robot_rows(list(robots_by_id.values()))
    assigned_in_rows = {r["robot"]: r["assigned task"] for r in robot_rows}
    for a in alloc["assignments"]:
        assert assigned_in_rows[a["robot_id"]] == a["task_id"]

    # --- 3. coordinate the assigned fleet -----------------------------
    robots_body = view_model.coordination_request_robots(
        alloc["assignments"], robots_by_id, tasks_by_id
    )
    coord = client.plan_coordination({"robots": robots_body})
    assert coord["planned_robot_count"] == 3
    assert coord["failed_robot_ids"] == []
    assert coord["total_wait_steps"] >= 1          # real coordination happened
    routes = view_model.coordinated_routes(coord)
    assert all(routes.values())

    # --- 4. inject an obstacle that actually sits on a planned route --
    picked = view_model.pick_dynamic_obstacle(coord)
    assert picked is not None
    hit_route = {(s["row"], s["col"]) for s in coord["routes"][picked["robot_id"]]["steps"]}
    assert (picked["cell"]["row"], picked["cell"]["col"]) in hit_route
    obstacle = client.recover_obstacle({
        "robots": robots_body,
        "obstacle": picked["cell"],
        "timestep": picked["timestep"],
    })
    assert obstacle["affected_robot_ids"]            # not "affected: none"
    assert obstacle["replanning_attempted"] is True

    # --- 5. fail an assigned robot; the spare takes the task ----------
    fail_body = view_model.recovery_request_robots(
        alloc["assignments"], robots_by_id, tasks_by_id, scenario.SPARE_ROBOT_ID
    )
    assert any(r["robot_id"] == scenario.SPARE_ROBOT_ID for r in fail_body)
    failed = sorted(a["robot_id"] for a in alloc["assignments"])[0]
    failed_task = next(
        a["task_id"] for a in alloc["assignments"] if a["robot_id"] == failed
    )
    failure = client.recover_robot_failure({
        "robots": fail_body,
        "failed_robot_id": failed,
        "timestep": scenario.FAILURE_TIMESTEP,
        "commit": True,
    })
    assert failure["task_reassignment_attempted"] is True
    assert failure["task_reassignment_success"] is True
    assert failure["reassigned_from"] == failed
    assert failure["reassigned_to"] == scenario.SPARE_ROBOT_ID
    assert failure["committed"] is True
    assert set(failure["committed_state"]) == {failed, scenario.SPARE_ROBOT_ID, failed_task}

    # the persisted Fleet / Task tables now tell the same story as the metrics
    robots_after = {r["robot_id"]: r for r in client.list_robots()}
    tasks_after = {t["task_id"]: t for t in client.list_tasks()}
    assert robots_after[failed]["status"] == "offline"
    assert robots_after[failed]["assigned_task_id"] is None
    assert robots_after[scenario.SPARE_ROBOT_ID]["status"] == "assigned"
    assert robots_after[scenario.SPARE_ROBOT_ID]["assigned_task_id"] == failed_task
    assert tasks_after[failed_task]["assigned_robot_id"] == scenario.SPARE_ROBOT_ID
    assert tasks_after[failed_task]["status"] == "assigned"

    metrics = view_model.scenario_metrics(
        alloc, coord, [("obstacle", obstacle), ("failure", failure)]
    )
    assert metrics["tasks assigned"] == 3
    assert metrics["robots coordinated"] == 3
    assert metrics["task reassignments"] == 1
    assert "reassigned" in metrics["last action"]


def test_dashboard_client_reports_domain_errors_cleanly(api_app, api_database) -> None:
    client = _bound_client(api_app, api_database)
    # a blocked / out-of-range plan request -> 422 with a message, not a crash
    with pytest.raises(ApiError) as excinfo:
        client.plan_path({"start": {"row": -1, "col": 0}, "goal": {"row": 1, "col": 1}})
    assert excinfo.value.status_code == 422
    assert excinfo.value.message


# ----------------------------------------------------------------------
# Regression: "1 - Initialise scenario" must not inherit stale persisted rows
# from an earlier run. (Live bug: an old 4-task demo left T4 behind; the event
# log said "3 tasks persisted" while the tables and CP-SAT saw 4, consuming the
# R4 spare.)
# ----------------------------------------------------------------------
def test_initialise_scenario_clears_stale_rows_from_an_earlier_run(
    api_app, api_database
) -> None:
    from robotics.simulation import scenario as old_scenario

    client = _bound_client(api_app, api_database)

    # An earlier run persisted the pre-2026-09-07 4-task demo set (T1..T4).
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
    assert len(client.list_tasks()) == 4  # T4 is the stale one

    # "1 - Initialise scenario"
    summary = scenario.sync_persisted_scenario(client)
    assert summary["removed"] == ["T4"]
    assert summary["robots"] == 4 and summary["tasks"] == 3   # what the log reports

    tasks = {t["task_id"] for t in client.list_tasks()}
    assert tasks == {"T1", "T2", "T3"}
    assert all(r["status"] == "idle" for r in client.list_robots())

    # "2 - Allocate tasks (CP-SAT)" now leaves R4 a genuine spare again
    alloc = client.run_allocation(
        {"algorithm": "cp_sat", "from_persisted": True, "commit": True}
    )
    assert len(alloc["assignments"]) == 3
    assert alloc["unassigned_task_ids"] == []
    robots_by_id = {r["robot_id"]: r for r in client.list_robots()}
    assert robots_by_id[scenario.SPARE_ROBOT_ID]["status"] == "idle"
    assert robots_by_id[scenario.SPARE_ROBOT_ID]["assigned_task_id"] is None


def test_initialise_scenario_is_idempotent_and_resets_a_finished_run(
    api_app, api_database
) -> None:
    client = _bound_client(api_app, api_database)

    scenario.sync_persisted_scenario(client)
    client.run_allocation(
        {"algorithm": "cp_sat", "from_persisted": True, "commit": True}
    )
    # fleet is now all assigned - a second init must return it to a fresh state
    summary = scenario.sync_persisted_scenario(client)
    assert summary == {"robots": 4, "tasks": 3, "removed": []}
    assert all(r["status"] == "idle" for r in client.list_robots())
    assert all(t["status"] == "pending" for t in client.list_tasks())


# ----------------------------------------------------------------------
# "Reset scenario" lifecycle: Reset is session-only; "1 - Initialise scenario"
# is the single authoritative persisted-state reconcile. Between them the
# dashboard shows a notice, it does not silently present stale assignments.
# ----------------------------------------------------------------------
def _reset_scenario_session_only(ss: dict) -> None:
    """Exactly what frontend.app.reset_scenario() does - session state, no API."""
    ss.update(allocation=None, coordination=None, recoveries=[], plan_demo=None,
              obstacle=None, scenario_ready=False, log=None)


def test_reset_is_session_only_and_leaves_persisted_state_for_initialise(
    api_app, api_database
) -> None:
    client = _bound_client(api_app, api_database)

    # a prior run: initialise + allocate -> persisted assigned rows
    scenario.sync_persisted_scenario(client)
    client.run_allocation(
        {"algorithm": "cp_sat", "from_persisted": True, "commit": True}
    )
    before_r = {r["robot_id"]: (r["status"], r["assigned_task_id"])
                for r in client.list_robots()}
    before_t = {t["task_id"]: (t["status"], t["assigned_robot_id"])
                for t in client.list_tasks()}

    # 1. Reset scenario  2. (re)authenticate  3. do NOT click Initialise
    ss: dict = {"scenario_ready": True}
    _reset_scenario_session_only(ss)
    assert ss["scenario_ready"] is False

    after_r = {r["robot_id"]: (r["status"], r["assigned_task_id"])
               for r in client.list_robots()}
    after_t = {t["task_id"]: (t["status"], t["assigned_robot_id"])
               for t in client.list_tasks()}
    # Reset touched no persisted row - intended semantics (design C)
    assert after_r == before_r
    assert after_t == before_t
    assert after_r["R1"] == ("assigned", "T1")

    # ...but the dashboard now surfaces a notice instead of a silent stale table
    notice = view_model.persisted_scenario_notice(
        ss["scenario_ready"], client.list_robots(), client.list_tasks()
    )
    assert notice is not None and "1 - Initialise scenario" in notice


def test_reset_then_initialise_produces_the_exact_clean_scenario(
    api_app, api_database
) -> None:
    client = _bound_client(api_app, api_database)

    # a messy prior run: extra task + a full allocation
    client.create_task({"task_id": "T4", "pickup": {"row": 5, "col": 1},
                        "dropoff": {"row": 11, "col": 1}, "payload_weight": 4.5,
                        "priority": "low"})
    scenario.sync_persisted_scenario(client)
    client.run_allocation(
        {"algorithm": "cp_sat", "from_persisted": True, "commit": True}
    )

    ss: dict = {}
    _reset_scenario_session_only(ss)               # Reset scenario
    summary = scenario.sync_persisted_scenario(client)  # 1 - Initialise scenario

    assert summary == {"robots": 4, "tasks": 3, "removed": []}
    robots = sorted(client.list_robots(), key=lambda r: r["robot_id"])
    tasks = sorted(client.list_tasks(), key=lambda t: t["task_id"])
    assert [r["robot_id"] for r in robots] == ["R1", "R2", "R3", "R4"]
    assert [t["task_id"] for t in tasks] == ["T1", "T2", "T3"]      # no T4
    assert all(r["status"] == "idle" and r["assigned_task_id"] is None
               for r in robots)
    assert all(t["status"] == "pending" and t["assigned_robot_id"] is None
               for t in tasks)
    # the notice clears once the scenario is initialised this session
    ss["scenario_ready"] = True
    assert view_model.persisted_scenario_notice(True, robots, tasks) is None


def test_initialise_reset_initialise_is_deterministic_across_cycles(
    api_app, api_database
) -> None:
    client = _bound_client(api_app, api_database)

    snapshots = []
    for _ in range(3):
        scenario.sync_persisted_scenario(client)          # Initialise
        client.run_allocation(                            # Allocate
            {"algorithm": "cp_sat", "from_persisted": True, "commit": True}
        )
        ss: dict = {}
        _reset_scenario_session_only(ss)                  # Reset
        summary = scenario.sync_persisted_scenario(client)  # Initialise again
        snapshots.append((
            summary,
            sorted((r["robot_id"], r["status"], r["assigned_task_id"])
                   for r in client.list_robots()),
            sorted((t["task_id"], t["status"], t["assigned_robot_id"])
                   for t in client.list_tasks()),
        ))

    assert snapshots[0] == snapshots[1] == snapshots[2]
    summary, robots, tasks = snapshots[0]
    assert summary == {"robots": 4, "tasks": 3, "removed": []}
    assert robots == [("R1", "idle", None), ("R2", "idle", None),
                      ("R3", "idle", None), ("R4", "idle", None)]
    assert tasks == [("T1", "pending", None), ("T2", "pending", None),
                     ("T3", "pending", None)]


# ----------------------------------------------------------------------
# Regression: "5 - Fail a robot" - after a successful failed-robot -> spare
# reassignment the persisted Fleet / Task tables must tell the same story as
# the recovery result and the Scenario Metrics. (Live bug: recovery reported
# "R1 -> R4 success=True" but R1 still owned T1 and R4 stayed idle, because
# the recovery endpoint was a pure stateless simulation.)
# ----------------------------------------------------------------------
def test_button5_successful_reassignment_is_reflected_in_persisted_state(
    api_app, api_database
) -> None:
    client = _bound_client(api_app, api_database)

    scenario.sync_persisted_scenario(client)
    alloc = client.run_allocation(
        {"algorithm": "cp_sat", "from_persisted": True, "commit": True}
    )
    robots_by_id = {r["robot_id"]: r for r in client.list_robots()}
    tasks_by_id = {t["task_id"]: t for t in client.list_tasks()}

    fail_body = view_model.recovery_request_robots(
        alloc["assignments"], robots_by_id, tasks_by_id, scenario.SPARE_ROBOT_ID
    )
    failed = sorted(a["robot_id"] for a in alloc["assignments"])[0]
    failed_task = next(
        a["task_id"] for a in alloc["assignments"] if a["robot_id"] == failed
    )

    failure = client.recover_robot_failure({
        "robots": fail_body,
        "failed_robot_id": failed,
        "timestep": scenario.FAILURE_TIMESTEP,
        "commit": True,
    })
    assert failure["task_reassignment_success"] is True
    assert failure["reassigned_to"] == scenario.SPARE_ROBOT_ID
    assert failure["committed"] is True

    robots_after = {r["robot_id"]: r for r in client.list_robots()}
    tasks_after = {t["task_id"]: t for t in client.list_tasks()}

    # R1 no longer owns T1; existing OFFLINE failure semantics
    assert robots_after[failed]["status"] == "offline"
    assert robots_after[failed]["assigned_task_id"] is None
    # R4 owns T1; T1 references R4
    assert robots_after[scenario.SPARE_ROBOT_ID]["status"] == "assigned"
    assert robots_after[scenario.SPARE_ROBOT_ID]["assigned_task_id"] == failed_task
    assert tasks_after[failed_task]["assigned_robot_id"] == scenario.SPARE_ROBOT_ID
    assert tasks_after[failed_task]["status"] == "assigned"

    # the other two assignments are untouched
    for a in alloc["assignments"]:
        if a["robot_id"] == failed:
            continue
        assert robots_after[a["robot_id"]]["assigned_task_id"] == a["task_id"]
        assert tasks_after[a["task_id"]]["assigned_robot_id"] == a["robot_id"]

    # Fleet Status / Task Status view rows now agree with the recovery result
    fleet = {r["robot"]: r for r in view_model.robot_rows(list(robots_after.values()))}
    tasks_rows = {t["task"]: t for t in view_model.task_rows(list(tasks_after.values()))}
    assert fleet[failed]["assigned task"] == "-"
    assert fleet[scenario.SPARE_ROBOT_ID]["assigned task"] == failed_task
    assert tasks_rows[failed_task]["assigned robot"] == scenario.SPARE_ROBOT_ID
