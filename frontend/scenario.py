"""The fixed, deterministic demo scenario the dashboard drives.

It runs on the API's default 12x12 warehouse profile (``build_sample_warehouse``:
racks, storage, pickups, chargers), so every compute request sends no
``warehouse`` block and the service builds the world server-side - and
allocation, coordination and recovery all reason about the *same* map.

The fleet and task set are chosen so the real algorithms tell a complete story
without any of them being weakened to make the picture look good:

* 4 robots, 3 tasks. CP-SAT assigns three robots and leaves **R4 idle as a
  genuine spare** - the robot the failure demo hands the orphaned task to.
* The three delivery routes cross the two open cross-aisles (row 4 and row 7)
  head-on with the vertical run down column 4, so the prioritized coordinator
  has to insert real WAIT steps to keep the plan conflict-free. No two assigned
  robots share a goal cell, so none of them is squeezed out at planning time.
* The dynamic obstacle is not a constant - the dashboard picks a cell that lies
  on an actual planned route (see ``view_model.pick_dynamic_obstacle``), so it
  genuinely forces a replan.

Nothing here computes a route or an assignment - it turns fixed specs into API
request bodies, reads the layout back for drawing, and (``sync_persisted_scenario``)
reconciles the persisted fleet / task queue to exactly this fixed set.
"""

from __future__ import annotations

from typing import Any, Dict, List

from robotics.simulation.scenario import build_sample_warehouse
from robotics.warehouse.warehouse import LocationType

# ----------------------------------------------------------------------
# The fixed fleet and task set (see the module docstring for the geometry).
# ----------------------------------------------------------------------
# robot_id -> (row, col), all with a full battery and a 10 kg deck.
_ROBOTS: Dict[str, Dict] = {
    "R1": {"row": 0, "col": 4},
    "R2": {"row": 4, "col": 0},
    "R3": {"row": 7, "col": 11},
    "R4": {"row": 0, "col": 10},   # the spare - CP-SAT leaves it idle
}
_BATTERY = 100.0
_CAPACITY = 10.0

# The robot CP-SAT is expected to leave unassigned (4 robots, 3 tasks). The
# failure demo needs a real spare; a dashboard test pins this expectation.
SPARE_ROBOT_ID = "R4"

# Timestep the "fail a robot" button injects the failure at. Small, but late
# enough that the failed robot has cleared its own task's pickup cell so the
# spare can actually reach it.
FAILURE_TIMESTEP = 2

# task_id -> (pickup, dropoff, priority, payload_weight_kg)
_TASKS: List[Dict] = [
    {"task_id": "T1", "pickup": {"row": 1, "col": 4}, "dropoff": {"row": 11, "col": 4},
     "priority": "high", "payload_weight": 3.0},
    {"task_id": "T2", "pickup": {"row": 4, "col": 1}, "dropoff": {"row": 4, "col": 11},
     "priority": "urgent", "payload_weight": 7.5},
    {"task_id": "T3", "pickup": {"row": 7, "col": 10}, "dropoff": {"row": 7, "col": 0},
     "priority": "normal", "payload_weight": 2.0},
]


def robot_bodies() -> List[Dict]:
    """API ``POST /robots`` bodies for the demo fleet."""
    return [
        {
            "robot_id": robot_id,
            "position": {"row": pos["row"], "col": pos["col"]},
            "battery_level": _BATTERY,
            "payload_capacity": _CAPACITY,
            "status": "idle",
        }
        for robot_id, pos in _ROBOTS.items()
    ]


def task_bodies() -> List[Dict]:
    """API ``POST /tasks`` bodies for the demo tasks."""
    return [
        {
            "task_id": t["task_id"],
            "pickup": dict(t["pickup"]),
            "dropoff": dict(t["dropoff"]),
            "payload_weight": t["payload_weight"],
            "priority": t["priority"],
        }
        for t in _TASKS
    ]


def allocation_specs() -> Dict[str, List[Dict]]:
    """Inline robot/task specs for ``POST /allocation/run`` request validation.

    The dashboard runs allocation ``from_persisted``/``commit`` against the
    stored fleet; these inline specs are kept only so a test can validate them
    against ``AllocationRequest`` and so callers can look positions up by id.
    """
    robots = [
        {
            "robot_id": b["robot_id"],
            "position": b["position"],
            "battery_level": b["battery_level"],
            "payload_capacity": b["payload_capacity"],
        }
        for b in robot_bodies()
    ]
    tasks = [
        {
            "task_id": b["task_id"],
            "pickup": b["pickup"],
            "dropoff": b["dropoff"],
            "payload_weight": b["payload_weight"],
            "priority": b["priority"],
        }
        for b in task_bodies()
    ]
    return {"robots": robots, "tasks": tasks}


def sync_persisted_scenario(client: Any) -> Dict[str, Any]:
    """Make the persisted fleet + task queue be EXACTLY this fixed scenario.

    "Initialise scenario" means *this* scenario, not "this scenario plus
    whatever an earlier run left behind". The demo dashboard owns the
    ``robots`` / ``tasks`` state, so every persisted row is removed first and
    then the fixed 4-robot / 3-task set is recreated. The append-only run
    history (allocation / coordination / recovery) is never touched.

    ``client`` is a :class:`~frontend.api_client.FulfillmentClient` (or anything
    with the same ``list_*`` / ``create_*`` / ``delete_*`` methods).

    Returns ``{"robots": int, "tasks": int, "removed": [id, ...]}`` describing
    the state after the sync - counts read back from the service, not assumed.
    """
    wanted_robot_ids = {b["robot_id"] for b in robot_bodies()}
    wanted_task_ids = {b["task_id"] for b in task_bodies()}

    removed: List[str] = []
    # Tasks first: with the task queue empty, no task references a robot when
    # the robots are removed.
    for task in client.list_tasks() or []:
        client.delete_task(task["task_id"])
        if task["task_id"] not in wanted_task_ids:
            removed.append(task["task_id"])
    for robot in client.list_robots() or []:
        client.delete_robot(robot["robot_id"])
        if robot["robot_id"] not in wanted_robot_ids:
            removed.append(robot["robot_id"])

    for body in robot_bodies():
        client.create_robot(body)
    for body in task_bodies():
        client.create_task(body)

    robots = client.list_robots() or []
    tasks = client.list_tasks() or []
    return {"robots": len(robots), "tasks": len(tasks), "removed": sorted(removed)}


def grid_layout() -> Dict:
    """Static layout for drawing: size, blocked cells, functional locations.

    Read straight from the same builder the service uses, so the picture
    matches the server's world exactly.
    """
    warehouse = build_sample_warehouse()
    return {
        "width": warehouse.width,
        "height": warehouse.height,
        "blocked": sorted((p.row, p.col) for p in warehouse.blocked_cells()),
        "pickups": sorted((p.row, p.col) for p in warehouse.pickup_locations),
        "dropoffs": sorted((p.row, p.col) for p in warehouse.dropoff_locations),
        "chargers": sorted(
            (p.row, p.col)
            for p in warehouse.positions_of_type(LocationType.CHARGING)
        ),
        "storage": sorted(
            (p.row, p.col) for p in warehouse.positions_of_type(LocationType.STORAGE)
        ),
    }
