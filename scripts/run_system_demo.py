"""One coherent, deterministic end-to-end run of the whole system.

    python scripts/run_system_demo.py
    python scripts/run_system_demo.py --json      # also print the metrics dict

Unlike ``scripts/run_demo.py`` (which is a wide catalogue of every capability,
including the ones that are supposed to fail), this script tells ONE story with
ONE warehouse and ONE fleet, in the order a fulfillment request actually flows
through the system:

    fulfillment tasks
        -> CP-SAT task allocation           (robotics/allocation)
        -> A* / Dijkstra route planning      (robotics/planning)
        -> prioritized space-time coordination (robotics/coordination)
        -> synchronized execution
        -> a dynamic obstacle appears -> local replanning   (robotics/recovery)
        -> a robot fails -> task reassignment + re-coordination
        -> final persisted-style state + measured scenario metrics

DETERMINISM
-----------
The warehouse layout, the fleet, the tasks, the obstacle cell and the failing
robot are all written out literally - nothing is random. Running this twice
produces byte-for-byte identical LOGICAL results (assignments, paths, waits,
conflicts, recovery outcomes). Only wall-clock timings vary and they are not
asserted anywhere.

This script imports the real algorithm and recovery code. It does not
re-implement anything and it does not fake an event to look impressive: every
line of output is the genuine result of the operation above it.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robotics.allocation.commit import commit_allocation  # noqa: E402
from robotics.allocation.cost_estimator import CostEstimator  # noqa: E402
from robotics.allocation.cp_sat_allocator import CpSatAllocator  # noqa: E402
from robotics.allocation.greedy_allocator import GreedyAllocator  # noqa: E402
from robotics.allocation.validation import find_allocation_problems  # noqa: E402
from robotics.coordination.conflicts import find_coordination_problems  # noqa: E402
from robotics.coordination.coordinator import MultiRobotCoordinator  # noqa: E402
from robotics.planning.astar import plan as plan_astar  # noqa: E402
from robotics.planning.dijkstra import plan as plan_dijkstra  # noqa: E402
from robotics.recovery.disruption import (  # noqa: E402
    DisruptionSchedule,
    DynamicObstacleEvent,
    RobotFailureEvent,
)
from robotics.recovery.recovery_manager import RecoveryManager  # noqa: E402
from robotics.robots.robot import Robot  # noqa: E402
from robotics.simulation.simulator import WarehouseSimulator  # noqa: E402
from robotics.tasks.task import Task, TaskPriority  # noqa: E402
from robotics.warehouse.grid import Position  # noqa: E402
from robotics.warehouse.warehouse import LocationType, Warehouse  # noqa: E402

LINE = "=" * 60


# ======================================================================
# The fixed scenario
# ======================================================================
WAREHOUSE_ROWS = 12
WAREHOUSE_COLS = 12

# A rack wall down column 6 (rows 2..9) leaves a two-cell gap at the top
# (rows 0-1) and at the bottom (rows 10-11). Any robot crossing left<->right
# must funnel through one of those gaps - which is what makes the coordinator
# insert a WAIT / detour when two delivery routes cross there.
RACK_CELLS = [Position(r, 6) for r in range(2, 10)]

FLEET = [
    ("R1", Position(0, 0), 12.0),
    ("R2", Position(0, 11), 12.0),
    ("R3", Position(11, 5), 12.0),
    ("R4", Position(11, 11), 12.0),
]

# The delivery routes are deliberately set up so that two of them cross at the
# top gap in opposite directions.
TASKS = [
    # id, pickup, dropoff, priority, weight
    ("T1", Position(0, 10), Position(0, 1), TaskPriority.HIGH, 4.0),    # right -> left
    ("T2", Position(0, 1), Position(0, 10), TaskPriority.URGENT, 6.0),  # left -> right
    ("T3", Position(11, 4), Position(9, 10), TaskPriority.NORMAL, 3.0),
]

# The dynamic obstacle: dropped at t=3 onto the top gap, on a delivery route
# that is using it.
OBSTACLE_TIMESTEP = 3
OBSTACLE_CELL = Position(1, 6)

# The robot that goes offline mid-run is chosen AFTER allocation (see
# run_scenario): it is an assigned, task-carrying robot, and a spare robot is
# guaranteed to exist to take the task over.
FAILURE_TIMESTEP = 2


@dataclass
class DemoResult:
    events: List[str] = field(default_factory=list)
    metrics: Dict[str, object] = field(default_factory=dict)

    def log(self, message: str) -> None:
        self.events.append(message)


# ======================================================================
# Scenario construction
# ======================================================================
def build_warehouse() -> Warehouse:
    warehouse = Warehouse(
        width=WAREHOUSE_COLS, height=WAREHOUSE_ROWS, name="fulfillment-demo"
    )
    for index, cell in enumerate(RACK_CELLS):
        warehouse.add_static_obstacle(f"rack-{index}", cell, "storage rack")
    for _tid, pickup, dropoff, _prio, _w in TASKS:
        warehouse.mark_location(pickup, LocationType.PICKUP)
        warehouse.mark_location(dropoff, LocationType.DROPOFF)
    return warehouse


def build_simulator() -> WarehouseSimulator:
    simulator = WarehouseSimulator(build_warehouse())
    for robot_id, cell, capacity in FLEET:
        simulator.add_robot(
            Robot(robot_id=robot_id, position=cell, payload_capacity=capacity)
        )
    for task_id, pickup, dropoff, priority, weight in TASKS:
        simulator.add_task(
            Task(
                task_id=task_id,
                pickup_location=pickup,
                dropoff_location=dropoff,
                priority=priority,
                payload_weight=weight,
            )
        )
    return simulator


# ======================================================================
# The run
# ======================================================================
def run_scenario() -> DemoResult:
    """Execute the whole scenario and return its events + metrics.

    Pure of console output so it can be asserted against in tests.
    """
    result = DemoResult()
    simulator = build_simulator()
    warehouse = simulator.warehouse
    estimator = CostEstimator(warehouse)

    # --- 1. TASK ALLOCATION (CP-SAT vs greedy baseline) ----------------
    greedy = GreedyAllocator(estimator).allocate(simulator.robots, simulator.tasks)
    cp_sat = CpSatAllocator(estimator).allocate(simulator.robots, simulator.tasks)
    allocation_problems = find_allocation_problems(
        cp_sat, simulator.robots, simulator.tasks, estimator
    )
    commit_report = commit_allocation(simulator, cp_sat)
    assignments = {a.robot_id: a.task_id for a in cp_sat.assignments}
    spare_robots = sorted(
        r.robot_id for r in simulator.robots if r.robot_id not in assignments
    )
    for robot_id, task_id in sorted(assignments.items()):
        result.log(f"task_assigned: {task_id} -> {robot_id}")

    result.metrics["allocation"] = {
        "tasks_offered": len(simulator.tasks),
        "cp_sat_assigned": len(cp_sat.assignments),
        "greedy_assigned": len(greedy.assignments),
        "cp_sat_total_estimated_cost": cp_sat.total_estimated_cost,
        "greedy_total_estimated_cost": greedy.total_estimated_cost,
        "cp_sat_valid": not allocation_problems,
        "committed": commit_report.committed,
        "spare_robots": spare_robots,
    }

    # --- 2. PATH PLANNING (A* vs Dijkstra) ---------------------------
    # Show the comparison on the longest assigned delivery route (robot start
    # -> task drop-off), where the node-expansion gap is actually visible.
    def _delivery(robot_id: str):
        task = simulator.get_task(assignments[robot_id])
        return simulator.get_robot(robot_id).position, task.dropoff_location

    sample_robot = max(
        assignments,
        key=lambda rid: plan_astar(*_delivery(rid), warehouse).total_cost or 0,
    )
    start, goal = _delivery(sample_robot)
    astar_path = plan_astar(start, goal, warehouse)
    dijkstra_path = plan_dijkstra(start, goal, warehouse)
    result.log(
        f"path_planned: {sample_robot} {start}->{goal} "
        f"cost={astar_path.total_cost} (A* expanded {astar_path.nodes_expanded}, "
        f"Dijkstra expanded {dijkstra_path.nodes_expanded})"
    )
    result.metrics["planning_sample"] = {
        "robot": sample_robot,
        "start": [start.row, start.col],
        "goal": [goal.row, goal.col],
        "astar_cost": astar_path.total_cost,
        "dijkstra_cost": dijkstra_path.total_cost,
        "astar_nodes_expanded": astar_path.nodes_expanded,
        "dijkstra_nodes_expanded": dijkstra_path.nodes_expanded,
        "same_cost": astar_path.total_cost == dijkstra_path.total_cost,
    }

    # --- 3. MULTI-ROBOT COORDINATION --------------------------------
    # Coordinate the full delivery routes: each assigned robot's goal is its
    # task's drop-off cell. Two of these routes cross at the top gap.
    goals = {
        robot_id: simulator.get_task(task_id).dropoff_location
        for robot_id, task_id in assignments.items()
    }
    starts = {rid: simulator.get_robot(rid).position for rid in goals}

    # what WOULD conflict if every robot flew its own independent A* route
    independent_paths = {
        rid: plan_astar(starts[rid], goals[rid], warehouse) for rid in goals
    }
    naive_conflicts = _naive_conflicts(independent_paths, warehouse)

    coordination = MultiRobotCoordinator(warehouse).plan_for(simulator, goals)
    coordinated_conflicts = find_coordination_problems(
        coordination.successful_paths, warehouse, starts
    )
    for robot_id in coordination.robot_order:
        timed = coordination.timed_paths[robot_id]
        if timed.wait_count:
            result.log(
                f"wait_inserted: {robot_id} waits {timed.wait_count} step(s) "
                f"to keep the plan conflict-free"
            )
    result.log(
        f"routes_coordinated: {coordination.planned_robot_count}/"
        f"{len(goals)} robots, makespan {coordination.makespan}, "
        f"{coordination.total_wait_steps} total waits"
    )
    result.metrics["coordination"] = {
        "robots": len(goals),
        "planned": coordination.planned_robot_count,
        "failed_robot_ids": list(coordination.failed_robot_ids),
        "makespan": coordination.makespan,
        "total_move_steps": coordination.total_move_steps,
        "total_wait_steps": coordination.total_wait_steps,
        "naive_vertex_conflicts": naive_conflicts["vertex"],
        "naive_edge_conflicts": naive_conflicts["edge"],
        "unresolved_conflicts_after_coordination": len(coordinated_conflicts),
    }

    # --- 4 + 5. DYNAMIC OBSTACLE -> REPLAN, ROBOT FAILURE -> RECOVERY --
    # The failing robot is an assigned, task-carrying robot; a spare exists.
    failed_robot = sorted(assignments)[-1] if assignments else None
    if not spare_robots:
        raise RuntimeError("demo scenario invariant broken: no spare robot")
    result.metrics["disruption"] = {
        "obstacle_cell": [OBSTACLE_CELL.row, OBSTACLE_CELL.col],
        "obstacle_timestep": OBSTACLE_TIMESTEP,
        "failed_robot": failed_robot,
        "failed_robot_task": assignments.get(failed_robot),
        "failure_timestep": FAILURE_TIMESTEP,
        "spare_robots": spare_robots,
    }
    schedule = DisruptionSchedule.of(
        DynamicObstacleEvent(OBSTACLE_TIMESTEP, OBSTACLE_CELL),
        RobotFailureEvent(FAILURE_TIMESTEP, failed_robot),
    )
    result.log(
        f"obstacle_introduced: {OBSTACLE_CELL} at t={OBSTACLE_TIMESTEP}"
    )
    result.log(
        f"robot_failed: {failed_robot} (task {assignments.get(failed_robot)}) "
        f"goes offline at t={FAILURE_TIMESTEP}"
    )
    recovery = RecoveryManager(simulator).run(coordination, schedule)

    obstacle_event = _event_of(recovery, "obstacle")
    failure_event = _event_of(recovery, "failure")

    if obstacle_event is not None:
        result.log(
            f"replanning_triggered: obstacle affected "
            f"{obstacle_event.affected_robot_ids or 'no robot'}; "
            f"replan success = {obstacle_event.replanning_success}"
        )
    if failure_event is not None:
        result.log(
            f"recovery_attempted: {failure_event.reassigned_from} -> "
            f"{failure_event.reassigned_to} for task "
            f"{failure_event.reassigned_task_id}; "
            f"reassignment success = {failure_event.task_reassignment_success}"
        )
    for robot_id, info in sorted(recovery.per_robot.items()):
        if info["safe_stopped"]:
            result.log(f"safe_stop: {robot_id} held position at {info['final_position']}")
        elif info["reached_goal"]:
            result.log(f"task_completed: {robot_id} reached {info['final_position']}")

    duplicate_completions = _duplicate_task_completions(simulator)

    result.metrics["recovery"] = {
        "obstacle_replan_attempted": bool(obstacle_event and obstacle_event.replanning_attempted),
        "obstacle_replan_success": bool(obstacle_event and obstacle_event.replanning_success),
        "obstacle_affected_robots": list(obstacle_event.affected_robot_ids) if obstacle_event else [],
        "failure_reassignment_attempted": bool(
            failure_event and failure_event.task_reassignment_attempted
        ),
        "failure_reassignment_success": bool(
            failure_event and failure_event.task_reassignment_success
        ),
        "reassigned_from": failure_event.reassigned_from if failure_event else None,
        "reassigned_to": failure_event.reassigned_to if failure_event else None,
        "unresolved_vertex_conflicts": recovery.vertex_conflicts,
        "unresolved_edge_conflicts": recovery.edge_conflicts,
        "trace_validates_conflict_free": not recovery.validation_problems,
        "duplicate_task_completions": duplicate_completions,
    }

    # --- 6. FINAL STATE + SCENARIO METRICS ---------------------------
    robots_final = {
        robot_id: {
            "position": [info["final_position"][0], info["final_position"][1]],
            "status": info["status"],
            "reached_goal": info["reached_goal"],
            "offline": info["offline"],
            "moves": info["moves"],
            "waits": info["waits"],
        }
        for robot_id, info in recovery.per_robot.items()
    }
    tasks_final = {
        task.task_id: {
            "status": task.status.value,
            "assigned_robot": task.assigned_robot_id,
        }
        for task in simulator.tasks
    }
    result.metrics["final_state"] = {
        "robots": robots_final,
        "tasks": tasks_final,
        "robots_reached_goal": recovery.robots_reached_goal,
        "robots_offline": sum(1 for i in recovery.per_robot.values() if i["offline"]),
        "safe_stops": sum(1 for i in recovery.per_robot.values() if i["safe_stopped"]),
        "total_move_steps": recovery.total_move_steps,
        "total_wait_steps": recovery.total_wait_steps,
        "episode_conflicts": recovery.vertex_conflicts + recovery.edge_conflicts,
    }
    return result


# ======================================================================
# Helpers
# ======================================================================
def _naive_conflicts(paths: Dict[str, object], warehouse) -> Dict[str, int]:
    """Vertex + edge conflicts if every robot followed its own route at t=step."""
    routes = {
        rid: list(path.path) for rid, path in paths.items() if path.success
    }
    horizon = max((len(r) for r in routes.values()), default=0)
    vertex = 0
    edge = 0
    ids = sorted(routes)
    for step in range(horizon):
        seen: Dict[object, str] = {}
        for rid in ids:
            route = routes[rid]
            cell = route[min(step, len(route) - 1)]
            if cell in seen:
                vertex += 1
            else:
                seen[cell] = rid
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                ra, rb = routes[a], routes[b]
                if step + 1 >= max(len(ra), len(rb)):
                    continue
                a_now = ra[min(step, len(ra) - 1)]
                a_next = ra[min(step + 1, len(ra) - 1)]
                b_now = rb[min(step, len(rb) - 1)]
                b_next = rb[min(step + 1, len(rb) - 1)]
                if a_now == b_next and a_next == b_now and a_now != a_next:
                    edge += 1
    return {"vertex": vertex, "edge": edge}


def _event_of(recovery, kind: str):
    for event in recovery.recovery_events:
        trigger = event.trigger.value if hasattr(event.trigger, "value") else str(event.trigger)
        if kind == "obstacle" and "obstacle" in trigger.lower():
            return event
        if kind == "failure" and ("offline" in trigger.lower() or "failure" in trigger.lower()):
            return event
    return None


def _duplicate_task_completions(simulator) -> int:
    """Tasks must never be marked completed more than once. There is no counter
    in the model for this, so we check the invariant a different way: a task is
    either terminal once or not - a violation would show up as a task both
    COMPLETED and still assigned to a live robot heading for it. Always 0 here;
    kept as an explicit guard."""
    violations = 0
    for task in simulator.tasks:
        if task.status.value == "completed" and task.assigned_robot_id:
            robot = simulator.get_robot(task.assigned_robot_id)
            if robot.assigned_task_id == task.task_id and robot.status.value != "offline":
                violations += 1
    return violations


# ======================================================================
# Console output
# ======================================================================
def _print_report(result: DemoResult) -> None:
    m = result.metrics
    print(LINE)
    print("AUTONOMOUS MULTI-ROBOT FULFILLMENT SYSTEM")
    print(LINE)
    print()
    print("SYSTEM INITIALIZED")
    print(f"  warehouse         : {WAREHOUSE_ROWS}x{WAREHOUSE_COLS}, "
          f"{len(RACK_CELLS)} rack cells")
    print(f"  fleet             : {', '.join(r[0] for r in FLEET)}")
    print(f"  fulfillment tasks : {', '.join(t[0] for t in TASKS)}")
    print()

    a = m["allocation"]
    print("TASK ALLOCATION  (CP-SAT vs greedy baseline)")
    print(f"  tasks offered              : {a['tasks_offered']}")
    print(f"  CP-SAT assigned            : {a['cp_sat_assigned']}")
    print(f"  greedy assigned            : {a['greedy_assigned']}")
    print(f"  CP-SAT est. travel cost    : {a['cp_sat_total_estimated_cost']}")
    print(f"  greedy est. travel cost    : {a['greedy_total_estimated_cost']}")
    print(f"  CP-SAT passed validation   : {a['cp_sat_valid']}")
    print(f"  spare robot(s)             : {a['spare_robots']}")
    print()

    p = m["planning_sample"]
    print("PATH PLANNING  (A* vs Dijkstra, one robot)")
    print(f"  robot {p['robot']}: {tuple(p['start'])} -> {tuple(p['goal'])}")
    print(f"  path cost                  : {p['astar_cost']}  "
          f"(A* == Dijkstra: {p['same_cost']})")
    print(f"  nodes expanded             : A* {p['astar_nodes_expanded']}  "
          f"vs Dijkstra {p['dijkstra_nodes_expanded']}")
    print()

    c = m["coordination"]
    print("MULTI-ROBOT COORDINATION  (prioritized space-time planning)")
    print(f"  robots coordinated         : {c['planned']}/{c['robots']}")
    print(f"  conflicts if independent   : {c['naive_vertex_conflicts']} vertex + "
          f"{c['naive_edge_conflicts']} edge")
    print(f"  waits inserted             : {c['total_wait_steps']}")
    print(f"  unresolved after coord.    : {c['unresolved_conflicts_after_coordination']}")
    print(f"  makespan                   : {c['makespan']}")
    print()

    d = m["disruption"]
    r = m["recovery"]
    print("DYNAMIC DISRUPTION")
    print(f"  obstacle                   : {tuple(d['obstacle_cell'])} at "
          f"t={d['obstacle_timestep']}")
    print(f"  robot failure              : {d['failed_robot']} "
          f"(task {d['failed_robot_task']}) at t={d['failure_timestep']}")
    print()
    print("REPLANNING")
    print(f"  obstacle affected robots   : {r['obstacle_affected_robots']}")
    print(f"  local replan succeeded     : {r['obstacle_replan_success']}")
    print()
    print("ROBOT FAILURE / RECOVERY")
    print(f"  task reassignment          : {r['reassigned_from']} -> {r['reassigned_to']}")
    print(f"  reassignment succeeded     : {r['failure_reassignment_success']}")
    print(f"  unresolved conflicts       : {r['unresolved_vertex_conflicts']} vertex + "
          f"{r['unresolved_edge_conflicts']} edge")
    print(f"  executed trace validates   : {r['trace_validates_conflict_free']}")
    print(f"  duplicate task completions : {r['duplicate_task_completions']}")
    print()

    f = m["final_state"]
    print("FINAL RESULTS")
    for robot_id, info in sorted(f["robots"].items()):
        print(f"  {robot_id}: pos {tuple(info['position'])}, {info['status']}, "
              f"reached_goal={info['reached_goal']}, "
              f"moves={info['moves']}, waits={info['waits']}")
    for task_id, info in sorted(f["tasks"].items()):
        print(f"  {task_id}: {info['status']}, assigned={info['assigned_robot']}")
    print(f"  robots reached goal        : {f['robots_reached_goal']}")
    print(f"  robots offline             : {f['robots_offline']}")
    print(f"  safe stops                 : {f['safe_stops']}")
    print(f"  total moves / waits        : {f['total_move_steps']} / {f['total_wait_steps']}")
    print(f"  episode conflicts          : {f['episode_conflicts']}")
    print()
    print("EVENT LOG")
    for line in result.events:
        print(f"  - {line}")
    print()
    print(LINE)
    print("DEMO COMPLETE")
    print(LINE)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true",
                        help="also print the machine-readable metrics dict")
    args = parser.parse_args(argv)

    result = run_scenario()
    _print_report(result)
    if args.json:
        print()
        print(json.dumps(result.metrics, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
