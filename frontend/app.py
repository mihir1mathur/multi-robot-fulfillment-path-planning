"""Streamlit demonstration dashboard.

    streamlit run frontend/app.py

It talks ONLY to the FastAPI service (default ``http://127.0.0.1:8000``, or set
``FULFILLMENT_API_URL``). It computes nothing itself. Configure credentials with
``FULFILLMENT_API_USERNAME`` / ``FULFILLMENT_API_PASSWORD`` or log in from the
sidebar.

Panels: a warehouse view (grid, racks, pickup/drop-off/charge cells, robots,
planned routes), fleet status, task status, an event log, and verified scenario
metrics. Controls: initialise the scenario, allocate tasks, plan routes,
inject a dynamic obstacle, fail a robot, reset.

Every backend call is wrapped so a missing server, an auth failure or a domain
error shows a plain message - never a stack trace, never a credential.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from frontend import scenario, view_model
from frontend.api_client import ApiError, ApiUnavailable, FulfillmentClient
from frontend.plotting import draw_warehouse

API_URL = os.environ.get("FULFILLMENT_API_URL", "http://127.0.0.1:8000")
API_USER = os.environ.get("FULFILLMENT_API_USERNAME", "")
API_PASSWORD = os.environ.get("FULFILLMENT_API_PASSWORD", "")

st.set_page_config(page_title="Fulfillment System Dashboard", layout="wide")


# ======================================================================
# Session state
# ======================================================================
def _init_state() -> None:
    ss = st.session_state
    ss.setdefault("client", FulfillmentClient(base_url=API_URL))
    ss.setdefault("log", view_model.EventLog())
    ss.setdefault("allocation", None)
    ss.setdefault("coordination", None)
    ss.setdefault("recoveries", [])          # list[(kind, response)]
    ss.setdefault("plan_demo", None)         # single A*/Dijkstra demo response
    ss.setdefault("obstacle", None)          # picked {cell, timestep, robot_id}
    ss.setdefault("scenario_ready", False)
    ss.setdefault("authed", False)


_init_state()
client: FulfillmentClient = st.session_state.client
log: view_model.EventLog = st.session_state.log


# ======================================================================
# Helpers
# ======================================================================
def call(fn, *args, spinner: str = "working...", **kwargs):
    """Run a client call, converting every failure into a user-facing message.

    Returns the payload on success, or ``None`` on any handled failure.
    """
    try:
        with st.spinner(spinner):
            return fn(*args, **kwargs)
    except ApiUnavailable as exc:
        st.error(f"Backend unavailable: {exc}")
    except ApiError as exc:
        if exc.status_code in (401, 403):
            st.session_state.authed = False
            st.error("Not authenticated. Log in from the sidebar.")
        else:
            st.error(f"Request rejected ({exc.status_code}): {exc.message}")
    except Exception as exc:  # noqa: BLE001 - last-resort guard, no traceback to the user
        st.error(f"Unexpected client error: {type(exc).__name__}")
    return None


def ensure_auth() -> bool:
    if st.session_state.authed and client.token:
        return True
    if API_USER and API_PASSWORD:
        token = call(client.login, API_USER, API_PASSWORD, spinner="authenticating...")
        if token:
            st.session_state.authed = True
            return True
    return False


def persisted_maps() -> tuple[dict, dict]:
    """``({robot_id: row}, {task_id: row})`` for the current persisted state."""
    robots = {r["robot_id"]: r for r in (call(client.list_robots) or [])}
    tasks = {t["task_id"]: t for t in (call(client.list_tasks) or [])}
    return robots, tasks


def reset_scenario() -> None:
    st.session_state.allocation = None
    st.session_state.coordination = None
    st.session_state.recoveries = []
    st.session_state.plan_demo = None
    st.session_state.obstacle = None
    st.session_state.scenario_ready = False
    st.session_state.log = view_model.EventLog()


# ======================================================================
# Sidebar - connection + controls
# ======================================================================
st.sidebar.title("Fulfillment dashboard")
st.sidebar.caption(f"API: {API_URL}")

health = call(client.health, spinner="checking backend...")
ready = call(client.ready, spinner="checking database...") if health else None
st.sidebar.info(view_model.health_banner(health, ready))

with st.sidebar.expander("Authentication", expanded=not ensure_auth()):
    if st.session_state.authed:
        st.success("authenticated")
    else:
        with st.form("login"):
            username = st.text_input("username", value=API_USER)
            password = st.text_input("password", type="password")
            if st.form_submit_button("Log in"):
                if call(client.login, username, password, spinner="authenticating..."):
                    st.session_state.authed = True
                    st.rerun()

st.sidebar.divider()
st.sidebar.subheader("Controls")

algorithm = st.sidebar.selectbox("Planning algorithm (single-route demo)",
                                 ["astar", "dijkstra"])

if st.sidebar.button("1 - Initialise scenario", use_container_width=True):
    if ensure_auth():
        # Reconcile persisted state to EXACTLY the fixed scenario: any rows a
        # previous run left behind (including an older version of this demo)
        # are removed, then the fixed 4-robot / 3-task set is recreated.
        summary = call(scenario.sync_persisted_scenario, client,
                       spinner="initialising scenario...")
        if summary is not None:
            st.session_state.scenario_ready = True
            if summary["removed"]:
                log.add("cleared stale rows from an earlier run: "
                        + ", ".join(summary["removed"]))
            # Count comes from the service, not a literal, so the log can never
            # disagree with the Fleet / Task tables.
            log.add(
                f"scenario initialised: {summary['robots']} robots, "
                f"{summary['tasks']} tasks persisted"
            )
    else:
        st.sidebar.warning("Log in first.")

if st.sidebar.button("2 - Allocate tasks (CP-SAT)", use_container_width=True):
    if ensure_auth():
        # Run the optimiser over the PERSISTED idle robots / pending tasks and
        # commit the winning assignments through the existing commit path, so
        # the stored robot / task rows (and the tables below) actually change.
        resp = call(client.run_allocation,
                    {"algorithm": "cp_sat", "from_persisted": True, "commit": True},
                    spinner="running CP-SAT allocation...")
        if resp:
            st.session_state.allocation = resp
            log.extend(view_model.allocation_events(resp))
            if resp.get("applied"):
                log.add("assignments committed to persisted state: "
                        + ", ".join(resp["applied"]))

if st.sidebar.button("3 - Plan & coordinate routes", use_container_width=True):
    if ensure_auth():
        alloc = st.session_state.allocation
        if not alloc or not alloc.get("assignments"):
            st.sidebar.warning("Allocate tasks first.")
        else:
            robots_by_id, tasks_by_id = persisted_maps()
            robots_body = view_model.coordination_request_robots(
                alloc["assignments"], robots_by_id, tasks_by_id
            )
            resp = call(client.plan_coordination, {"robots": robots_body},
                        spinner="prioritized space-time planning...")
            if resp:
                st.session_state.coordination = resp
                log.extend(view_model.coordination_events(resp))
            # a single-route A*/Dijkstra demo for the selected algorithm
            if robots_body:
                first = robots_body[0]
                demo = call(client.plan_path,
                            {"start": first["start"], "goal": first["goal"],
                             "algorithm": algorithm},
                            spinner=f"planning one route with {algorithm}...")
                if demo:
                    st.session_state.plan_demo = demo
                    log.add(
                        f"{algorithm}: {first['robot_id']} route cost "
                        f"{demo.get('total_cost')}, {demo.get('nodes_expanded')} nodes expanded"
                    )

if st.sidebar.button("4 - Inject dynamic obstacle", use_container_width=True):
    if ensure_auth():
        alloc = st.session_state.allocation
        coord = st.session_state.coordination
        picked = view_model.pick_dynamic_obstacle(coord)
        if not alloc or not alloc.get("assignments"):
            st.sidebar.warning("Allocate and plan first.")
        elif not coord or not coord.get("routes"):
            st.sidebar.warning("Plan & coordinate routes first.")
        elif picked is None:
            st.sidebar.warning(
                "No coordinated route is long enough to drop an obstacle on."
            )
        else:
            robots_by_id, tasks_by_id = persisted_maps()
            robots_body = view_model.coordination_request_robots(
                alloc["assignments"], robots_by_id, tasks_by_id
            )
            resp = call(client.recover_obstacle,
                        {"robots": robots_body,
                         "obstacle": picked["cell"],
                         "timestep": picked["timestep"]},
                        spinner="replanning around the obstacle...")
            if resp:
                st.session_state.obstacle = picked
                st.session_state.recoveries.append(("obstacle", resp))
                cell = picked["cell"]
                log.add(
                    f"obstacle dropped at ({cell['row']}, {cell['col']}) on "
                    f"{picked['robot_id']}'s planned route at t={picked['timestep']}"
                )
                log.extend(view_model.recovery_events(resp, "obstacle"))

if st.sidebar.button("5 - Fail a robot", use_container_width=True):
    if ensure_auth():
        alloc = st.session_state.allocation
        if not alloc or not alloc.get("assignments"):
            st.sidebar.warning("Allocate and plan first.")
        else:
            robots_by_id, tasks_by_id = persisted_maps()
            robots_body = view_model.recovery_request_robots(
                alloc["assignments"], robots_by_id, tasks_by_id,
                scenario.SPARE_ROBOT_ID,
            )
            failed = sorted(a["robot_id"] for a in alloc["assignments"])[0]
            spare_in = any(
                r["robot_id"] == scenario.SPARE_ROBOT_ID for r in robots_body
            )
            log.add(
                f"failing {failed}"
                + (f"; spare {scenario.SPARE_ROBOT_ID} standing by"
                   if spare_in else "; no spare robot available")
            )
            resp = call(client.recover_robot_failure,
                        {"robots": robots_body, "failed_robot_id": failed,
                         "timestep": scenario.FAILURE_TIMESTEP, "commit": True},
                        spinner="recovering from robot failure...")
            if resp:
                st.session_state.recoveries.append(("failure", resp))
                log.extend(view_model.recovery_events(resp, "failure"))
                if resp.get("committed"):
                    log.add("persisted reassignment: "
                            + ", ".join(resp.get("committed_state", [])))

if st.sidebar.button("Reset scenario", use_container_width=True):
    reset_scenario()
    st.rerun()
st.sidebar.caption(
    "Reset clears this dashboard view only. Use \"1 - Initialise scenario\" to "
    "reconcile the persisted database to the clean demo scenario."
)


# ======================================================================
# Main area
# ======================================================================
st.title("Autonomous Multi-Robot Fulfillment")

if health is None:
    st.warning(
        "The API is not reachable. Start it with "
        "`uvicorn robotics.api.app:create_app --factory` and reload."
    )

layout = scenario.grid_layout()

try:
    robots = client.list_robots() if st.session_state.authed else []
except (ApiError, ApiUnavailable):
    robots = []
try:
    tasks = client.list_tasks() if st.session_state.authed else []
except (ApiError, ApiUnavailable):
    tasks = []

notice = view_model.persisted_scenario_notice(
    st.session_state.scenario_ready, robots, tasks
)
if notice:
    st.warning(notice)

routes: dict = {}
coordination = st.session_state.coordination
if coordination:
    routes = view_model.coordinated_routes(coordination)
elif st.session_state.plan_demo:
    routes = {"route": view_model.path_cells(st.session_state.plan_demo)}

obstacle_cell = None
if st.session_state.obstacle:
    _oc = st.session_state.obstacle["cell"]
    obstacle_cell = (_oc["row"], _oc["col"])

left, right = st.columns([3, 2])

with left:
    st.pyplot(draw_warehouse(layout, robots, routes, obstacle_cell))

with right:
    st.subheader("Scenario metrics")
    metrics = view_model.scenario_metrics(
        st.session_state.allocation,
        st.session_state.coordination,
        st.session_state.recoveries,
    )
    if metrics:
        st.table([{"metric": k, "value": v} for k, v in metrics.items()])
    else:
        st.caption("Run allocation / planning to populate metrics.")

    if st.session_state.plan_demo:
        d = st.session_state.plan_demo
        st.caption(
            f"Single-route demo ({d.get('algorithm')}): cost {d.get('total_cost')}, "
            f"{d.get('nodes_expanded')} nodes expanded, "
            f"{d.get('planning_time_ms', 0):.2f} ms"
        )

st.subheader("Fleet status")
st.table(view_model.robot_rows(robots) or [{"robot": "-", "state": "no robots"}])

st.subheader("Task status")
st.table(view_model.task_rows(tasks) or [{"task": "-", "state": "no tasks"}])

st.subheader("Event log")
entries = log.tail(30)
if entries:
    for line in reversed(entries):
        st.text(f"• {line}")
else:
    st.caption("No events yet.")
