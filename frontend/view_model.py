"""Pure transforms: raw API payloads -> things the UI draws.

No Streamlit import, no HTTP. Every function takes a plain dict/list (exactly
what ``FulfillmentClient`` returns) and returns plain data, so all of this is
unit-tested without a browser or a server.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

Cell = Tuple[int, int]


# ----------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------
def robot_rows(robots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for r in robots:
        pos = r.get("position") or {}
        rows.append(
            {
                "robot": r.get("robot_id", "?"),
                "state": r.get("status", "?"),
                "position": f"({pos.get('row', '?')}, {pos.get('col', '?')})",
                "battery %": round(float(r.get("battery_level", 0)), 1),
                "assigned task": r.get("assigned_task_id") or "-",
            }
        )
    return sorted(rows, key=lambda x: x["robot"])


def task_rows(tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for t in tasks:
        pickup = t.get("pickup") or {}
        dropoff = t.get("dropoff") or {}
        rows.append(
            {
                "task": t.get("task_id", "?"),
                "state": t.get("status", "?"),
                "assigned robot": t.get("assigned_robot_id") or "-",
                "pickup": f"({pickup.get('row', '?')}, {pickup.get('col', '?')})",
                "destination": f"({dropoff.get('row', '?')}, {dropoff.get('col', '?')})",
                "priority": t.get("priority", "-"),
            }
        )
    return sorted(rows, key=lambda x: x["task"])


# ----------------------------------------------------------------------
# Geometry for the warehouse view
# ----------------------------------------------------------------------
def path_cells(plan_response: Dict[str, Any]) -> List[Cell]:
    """The (row, col) cells of a ``POST /planning/path`` response."""
    return [(c["row"], c["col"]) for c in plan_response.get("path", [])]


def coordinated_routes(coord_response: Dict[str, Any]) -> Dict[str, List[Cell]]:
    """{robot_id: [(row, col), ...]} from a ``POST /coordination/plan`` response.

    WAIT steps repeat a cell; they are kept so the drawn path length matches
    the makespan.
    """
    routes: Dict[str, List[Cell]] = {}
    for robot_id, route in (coord_response.get("routes") or {}).items():
        steps = route.get("steps", [])
        routes[robot_id] = [(s["row"], s["col"]) for s in steps]
    return routes


def robot_positions_from_specs(specs: List[Dict[str, Any]]) -> Dict[str, Cell]:
    return {
        s["robot_id"]: (s["position"]["row"], s["position"]["col"]) for s in specs
    }


# ----------------------------------------------------------------------
# Turning an allocation result into coordination / recovery request bodies
# ----------------------------------------------------------------------
def coordination_request_robots(
    assignments: List[Dict[str, Any]],
    robots_by_id: Dict[str, Dict[str, Any]],
    tasks_by_id: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """``[{robot_id, start, goal}]`` for ``POST /coordination/plan``.

    ``start`` is the robot's persisted position; ``goal`` is its task's
    drop-off. Only assignments whose robot and task are both known are kept.
    """
    body: List[Dict[str, Any]] = []
    for a in assignments:
        robot = robots_by_id.get(a["robot_id"])
        task = tasks_by_id.get(a["task_id"])
        if robot is None or task is None:
            continue
        body.append(
            {
                "robot_id": a["robot_id"],
                "start": _coord(robot["position"]),
                "goal": _coord(task["dropoff"]),
            }
        )
    return body


def recovery_request_robots(
    assignments: List[Dict[str, Any]],
    robots_by_id: Dict[str, Dict[str, Any]],
    tasks_by_id: Dict[str, Dict[str, Any]],
    spare_robot_id: str = "",
) -> List[Dict[str, Any]]:
    """``[{robot_id, start, goal, task?}]`` for the recovery endpoints.

    Each assigned robot carries its inline task. If ``spare_robot_id`` names a
    persisted robot that holds no assignment, it is appended *parked*
    (``goal == start``, no task) so the failure demo has a real robot to hand
    the orphaned task to.
    """
    assigned_ids = {a["robot_id"] for a in assignments}
    body: List[Dict[str, Any]] = []
    for a in assignments:
        robot = robots_by_id.get(a["robot_id"])
        task = tasks_by_id.get(a["task_id"])
        if robot is None or task is None:
            continue
        body.append(
            {
                "robot_id": a["robot_id"],
                "start": _coord(robot["position"]),
                "goal": _coord(task["dropoff"]),
                "task": {
                    "task_id": task["task_id"],
                    "pickup": _coord(task["pickup"]),
                    "dropoff": _coord(task["dropoff"]),
                    "payload_weight": task["payload_weight"],
                    "priority": task["priority"],
                },
            }
        )
    spare = robots_by_id.get(spare_robot_id)
    if spare is not None and spare_robot_id not in assigned_ids:
        body.append(
            {
                "robot_id": spare_robot_id,
                "start": _coord(spare["position"]),
                "goal": _coord(spare["position"]),
            }
        )
    return body


def pick_dynamic_obstacle(
    coord_response: Optional[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Choose an obstacle cell that actually lies on a planned route.

    Returns ``{"cell": {"row", "col"}, "timestep": int, "robot_id": str}`` for
    the mid-point cell of the longest successful coordinated route (ties broken
    by robot id), or ``None`` if there is no route long enough to obstruct.
    The timestep is early but strictly before the robot reaches that cell, so
    the cell is still part of the robot's *remaining* route when the obstacle
    fires.
    """
    if not coord_response:
        return None
    candidates = []
    for robot_id, route in (coord_response.get("routes") or {}).items():
        steps = route.get("steps", [])
        if route.get("success") and len(steps) >= 5:
            candidates.append((robot_id, steps))
    if not candidates:
        return None
    robot_id, steps = max(candidates, key=lambda rs: (len(rs[1]), rs[0]))
    mid = steps[len(steps) // 2]
    fire_at = max(1, min(2, int(mid.get("timestep", 2)) - 1))
    return {
        "cell": {"row": mid["row"], "col": mid["col"]},
        "timestep": fire_at,
        "robot_id": robot_id,
    }


def _coord(value: Any) -> Dict[str, int]:
    """Normalise a ``{row, col}`` mapping (dropping any extra keys)."""
    return {"row": int(value["row"]), "col": int(value["col"])}


# ----------------------------------------------------------------------
# Event log
# ----------------------------------------------------------------------
class EventLog:
    """An append-only list of short human-readable strings, newest last."""

    def __init__(self, entries: Optional[List[str]] = None) -> None:
        self.entries: List[str] = list(entries or [])

    def add(self, message: str) -> None:
        self.entries.append(message)

    def extend(self, messages: List[str]) -> None:
        self.entries.extend(messages)

    def tail(self, n: int = 25) -> List[str]:
        return self.entries[-n:]


def allocation_events(alloc_response: Dict[str, Any]) -> List[str]:
    events = []
    for a in alloc_response.get("assignments", []):
        events.append(
            f"task assigned: {a['task_id']} -> {a['robot_id']} "
            f"(est. cost {a['total_estimated_cost']})"
        )
    for task_id in alloc_response.get("unassigned_task_ids", []):
        events.append(f"task unassigned: {task_id} (no feasible robot)")
    return events


def coordination_events(coord_response: Dict[str, Any]) -> List[str]:
    events = [
        f"routes planned: {coord_response.get('planned_robot_count', 0)} robot(s), "
        f"makespan {coord_response.get('makespan', 0)}, "
        f"{coord_response.get('total_wait_steps', 0)} wait step(s)"
    ]
    for robot_id, route in (coord_response.get("routes") or {}).items():
        if route.get("wait_count"):
            events.append(
                f"wait inserted: {robot_id} waits {route['wait_count']} step(s)"
            )
    for robot_id in coord_response.get("failed_robot_ids", []):
        events.append(f"planning failed: {robot_id} could not be placed")
    return events


def recovery_events(recovery_response: Dict[str, Any], kind: str) -> List[str]:
    events: List[str] = []
    if kind == "obstacle":
        events.append(
            f"obstacle introduced; affected: "
            f"{recovery_response.get('affected_robot_ids') or 'none'}"
        )
        if recovery_response.get("replanning_attempted"):
            events.append(
                f"replanning triggered: success = "
                f"{recovery_response.get('replanning_success')}"
            )
    else:
        events.append(
            f"robot failure; affected: "
            f"{recovery_response.get('affected_robot_ids') or 'none'}"
        )
        if recovery_response.get("task_reassignment_attempted"):
            events.append(
                f"recovery attempted: {recovery_response.get('reassigned_from')} -> "
                f"{recovery_response.get('reassigned_to')}, success = "
                f"{recovery_response.get('task_reassignment_success')}"
            )
    if recovery_response.get("safe_stop"):
        events.append("safe stop: an affected robot held position")
    return events


# ----------------------------------------------------------------------
# Metrics panel
# ----------------------------------------------------------------------
def scenario_metrics(
    allocation: Optional[Dict[str, Any]],
    coordination: Optional[Dict[str, Any]],
    recoveries: List[Tuple[str, Dict[str, Any]]],
) -> Dict[str, Any]:
    """A flat dict of verified scenario counters for the metrics panel.

    Only values actually present in the responses are reported - nothing is
    invented or defaulted to a flattering number.
    """
    metrics: Dict[str, Any] = {}

    if allocation is not None:
        assigned = len(allocation.get("assignments", []))
        unassigned = len(allocation.get("unassigned_task_ids", []))
        metrics["tasks assigned"] = assigned
        metrics["tasks unassigned"] = unassigned
        total = assigned + unassigned
        if total:
            metrics["assignment rate"] = round(assigned / total, 3)
        metrics["allocation solver"] = allocation.get("algorithm", "-")
        metrics["allocation total est. cost"] = allocation.get("total_estimated_cost")

    if coordination is not None:
        metrics["robots coordinated"] = coordination.get("planned_robot_count", 0)
        metrics["coordination failures"] = len(
            coordination.get("failed_robot_ids", [])
        )
        metrics["wait steps"] = coordination.get("total_wait_steps", 0)
        metrics["move steps"] = coordination.get("total_move_steps", 0)
        metrics["makespan"] = coordination.get("makespan", 0)

    replans = 0
    recoveries_ok = 0
    safe_stops = 0
    reassignments = 0
    unresolved = 0
    for _kind, response in recoveries:
        if response.get("replanning_attempted"):
            replans += 1
        if response.get("recovery_success"):
            recoveries_ok += 1
        if response.get("safe_stop"):
            safe_stops += 1
        if response.get("task_reassignment_success"):
            reassignments += 1
        unresolved += response.get("unresolved_vertex_conflicts", 0)
        unresolved += response.get("unresolved_edge_conflicts", 0)
    if recoveries:
        metrics["replans triggered"] = replans
        metrics["recoveries succeeded"] = recoveries_ok
        metrics["safe stops"] = safe_stops
        metrics["task reassignments"] = reassignments
        metrics["unresolved conflicts"] = unresolved
        # The most recent action's own outcome, so the panel is not read as a
        # snapshot of an earlier step.
        last_kind, last_response = recoveries[-1]
        metrics["last action"] = _recovery_outcome(last_kind, last_response)

    return metrics


def _recovery_outcome(kind: str, response: Dict[str, Any]) -> str:
    affected = response.get("affected_robot_ids") or []
    if kind == "obstacle":
        if not response.get("replanning_attempted"):
            return f"obstacle: no route affected ({', '.join(affected) or 'none'})"
        verb = "replanned" if response.get("replanning_success") else "safe-stopped"
        return f"obstacle: {', '.join(affected) or 'none'} {verb}"
    if response.get("task_reassignment_success"):
        return (
            f"robot failure: {response.get('reassigned_from')} -> "
            f"{response.get('reassigned_to')} reassigned"
        )
    if response.get("safe_stop"):
        return f"robot failure: safe stop ({', '.join(affected) or 'none'})"
    return f"robot failure: {', '.join(affected) or 'none'} affected"


def health_banner(health: Optional[Dict[str, Any]], ready: Optional[Dict[str, Any]]) -> str:
    if not health:
        return "backend: unreachable"
    status = health.get("status", "?")
    db = (ready or {}).get("checks", {}).get("database", "?")
    return f"backend: {status} · database: {db}"


def persisted_scenario_notice(
    scenario_ready: bool,
    robots: List[Dict[str, Any]],
    tasks: List[Dict[str, Any]],
) -> Optional[str]:
    """Warn when the dashboard is showing persisted rows it did not create.

    "Reset scenario" clears only the dashboard session; "1 - Initialise
    scenario" is the single authoritative step that reconciles the persisted
    ``robots`` / ``tasks`` state. Between the two - a fresh session, or right
    after a reset - the tables below still show whatever an earlier run left in
    the database. That is correct data but ambiguous for a screenshot, so this
    returns a message telling the operator to initialise before demoing.

    Returns ``None`` when the scenario was initialised this session, or when
    there is no persisted state to be confused by.
    """
    if scenario_ready or (not robots and not tasks):
        return None
    live_assignments = sum(1 for r in robots if r.get("assigned_task_id")) + sum(
        1 for t in tasks if t.get("assigned_robot_id")
    )
    extra = (
        f", including {live_assignments} live assignment(s) from that run"
        if live_assignments
        else ""
    )
    return (
        f"Persisted state from an earlier run is still in the database: "
        f"{len(robots)} robot(s), {len(tasks)} task(s){extra}. "
        f'"Reset scenario" clears only this dashboard view - click '
        f'"1 - Initialise scenario" to reconcile the database to the clean '
        f"4-robot / 3-task demo scenario before taking screenshots."
    )
