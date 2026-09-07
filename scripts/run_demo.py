"""End-to-end demonstration script.

Run it with:

    python scripts/run_demo.py

It walks through everything the simulation currently implements, in order,
printing the result of each operation - including the operations that are
SUPPOSED to fail. Showing the rejections is the point: the value of this
foundation is that illegal states cannot be entered, and the only way to show
that is to try.

Sections 1-11 exercise the world model: the warehouse, the fleet, the tasks
and the single-cell movement rules. Every move there is a single, explicitly
chosen step - the world walkthrough deliberately does not plan.

Section 12 then demonstrates the path planning layer: A* and Dijkstra
computing whole obstacle-aware routes between two cells.

Section 13 drives a robot along a planned route, one cell at a time, using the
route executor. Section 14 chooses robot-task assignments across a small fleet
with a greedy baseline and with the CP-SAT optimiser. Section 15 runs the whole
chain: allocate a task, plan the route, execute it.

Sections 16-18 add multi-robot coordination: why independent routes conflict,
space-time planning with a reservation table, WAIT actions, edge/swap
prevention, and synchronised timestep-by-timestep execution.

Sections 19-22 add recovery: replanning from the robot's current cell when an
obstacle appears mid-drive, stopping safely when there is no route, and
reassigning a task when its robot goes offline.

Sections 23-28 add the REST service layer: a FastAPI app (talked to in-process
here, no server needed) backed by SQLAlchemy persistence, exposing the same
planning / allocation / coordination / recovery code through HTTP endpoints
with Pydantic validation, health/readiness checks and transaction-safe writes.

Still not built: recovering from an obstacle or robot failure that is
structurally impossible to route around; globally optimal multi-agent path
finding.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

# Allow `python scripts/run_demo.py` from the project root without
# having to install the package first: put the project root on the import path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robotics.exceptions import RoboticsError  # noqa: E402
from robotics.simulation.renderer import (  # noqa: E402
    LEGEND,
    render_map,
    render_robot_legend,
    render_robot_table,
    render_task_table,
)
from robotics.simulation.scenario import (  # noqa: E402
    SEED,
    SPILL_OBSTACLE_ID,
    build_sample_simulator,
)
from robotics.simulation.simulator import WarehouseSimulator  # noqa: E402
from robotics.allocation.commit import commit_allocation  # noqa: E402
from robotics.allocation.cost_estimator import CostEstimator  # noqa: E402
from robotics.allocation.cp_sat_allocator import CpSatAllocator  # noqa: E402
from robotics.allocation.greedy_allocator import GreedyAllocator  # noqa: E402
from robotics.allocation.validation import find_allocation_problems  # noqa: E402
from robotics.execution.route_executor import RouteExecutor  # noqa: E402
from robotics.coordination.conflicts import find_coordination_problems  # noqa: E402
from robotics.coordination.coordinated_executor import CoordinatedExecutor  # noqa: E402
from robotics.coordination.coordinator import MultiRobotCoordinator  # noqa: E402
from robotics.recovery.disruption import (  # noqa: E402
    DisruptionSchedule,
    DynamicObstacleEvent,
    RobotFailureEvent,
)
from robotics.recovery.recovery_manager import RecoveryManager  # noqa: E402
from robotics.planning.astar import plan as plan_astar  # noqa: E402
from robotics.planning.dijkstra import plan as plan_dijkstra  # noqa: E402
from robotics.planning.path_result import PathResult  # noqa: E402
from robotics.planning.validation import find_path_problems  # noqa: E402
from robotics.robots.robot import Robot  # noqa: E402
from robotics.tasks.task import Task, TaskPriority  # noqa: E402
from robotics.warehouse.grid import Position  # noqa: E402
from robotics.warehouse.warehouse import Warehouse  # noqa: E402

LINE_WIDTH = 78


def section(title: str) -> None:
    """Print a numbered section banner."""
    print()
    print("=" * LINE_WIDTH)
    print(title)
    print("=" * LINE_WIDTH)


def attempt_move(
    simulator: WarehouseSimulator, robot_id: str, destination: Position, note: str = ""
) -> None:
    """Try one move and report the outcome, whether it succeeds or fails.

    Both outcomes are printed in the same format so the demo reads as a single
    consistent log rather than a mix of results and stack traces.
    """
    robot = simulator.get_robot(robot_id)
    origin = robot.position
    battery_before = robot.battery_level
    suffix = f"   ({note})" if note else ""

    try:
        simulator.move_robot(robot_id, destination)
    except RoboticsError as error:
        print(f"  REJECTED  {robot_id}: {origin} -> {destination}{suffix}")
        print(f"            reason: {error}")
        return

    print(f"  OK        {robot_id}: {origin} -> {robot.position}{suffix}")
    print(
        f"            battery {battery_before:.1f}% -> {robot.battery_level:.1f}%, "
        f"steps={robot.steps_taken}, distance={robot.distance_travelled:.1f}"
    )


def attempt_assignment(
    simulator: WarehouseSimulator, task_id: str, robot_id: str
) -> None:
    """Try a manual task assignment and report the outcome."""
    try:
        simulator.assign_task_manually(task_id, robot_id)
    except RoboticsError as error:
        print(f"  REJECTED  task {task_id} -> robot {robot_id}")
        print(f"            reason: {error}")
        return
    print(f"  OK        task {task_id} -> robot {robot_id}")


def show_plan(result: PathResult, warehouse: Warehouse) -> None:
    """Print a planner result with its real measured numbers."""
    if not result.success:
        print(f"  {result.algorithm:<9} FAILED  {result.start} -> {result.goal}")
        print(f"            reason: {result.failure_reason}")
        print(f"            nodes expanded: {result.nodes_expanded}, "
              f"time: {result.planning_time_ms:.3f} ms")
        return

    problems = find_path_problems(result, warehouse)
    validity = "valid" if not problems else f"INVALID: {problems}"
    print(f"  {result.algorithm:<9} OK      {result.start} -> {result.goal}")
    print(f"            path        : {[str(cell) for cell in result.path]}")
    print(f"            total cost  : {result.total_cost:g}  "
          f"({len(result.path)} cells, {len(result.path) - 1} moves)")
    print(f"            nodes exp.  : {result.nodes_expanded}")
    print(f"            planning    : {result.planning_time_ms:.3f} ms")
    print(f"            validation  : {validity}")


def path_planning_demo() -> None:
    """Section 12: A* and Dijkstra planning whole routes around obstacles.

    Each scenario builds its own small warehouse so the expected answer can be
    checked by eye. Every number printed is measured from the actual call.
    """
    section("12. PATH PLANNING (A* and Dijkstra)")
    print("  The planners take a start cell and a goal cell and return the whole")
    print("  route between them, going around every obstacle. They PLAN a route;")
    print("  executing it step by step is not implemented.")

    # A. Simple route across an empty floor.
    print()
    print("  A. Simple route across an open 6x6 warehouse, (0,0) -> (5,5):")
    open_wh = Warehouse(width=6, height=6, name="demo-open")
    show_plan(plan_astar(Position(0, 0), Position(5, 5), open_wh), open_wh)

    # B. Detour forced by a static obstacle.
    print()
    print("  B. Static obstacle at (2,2) forces a detour, (2,1) -> (2,3):")
    detour_wh = Warehouse(width=6, height=6, name="demo-detour")
    detour_wh.add_static_obstacle("rack-1", Position(2, 2), "storage rack")
    show_plan(plan_astar(Position(2, 1), Position(2, 3), detour_wh), detour_wh)
    print("     (direct cost would be 2; routing around the rack costs 4)")

    # C. Dynamic obstacle already present before planning begins.
    print()
    print("  C. Dynamic obstacle (a spill) present BEFORE planning, (2,1) -> (2,3):")
    spill_wh = Warehouse(width=6, height=6, name="demo-spill")
    spill_wh.add_dynamic_obstacle("spill-1", Position(2, 2), "liquid spill")
    show_plan(plan_astar(Position(2, 1), Position(2, 3), spill_wh), spill_wh)
    print("     (a spill that appears AFTER planning would not update the route -")
    print("      replanning during execution is not implemented)")

    # D. Unreachable goal.
    print()
    print("  D. Goal walled into a corner, (4,0) -> (0,4) - no route exists:")
    sealed_wh = Warehouse(width=5, height=5, name="demo-sealed")
    for cell in (Position(0, 3), Position(1, 3), Position(1, 4)):
        sealed_wh.add_static_obstacle(f"seal-{cell.row}-{cell.col}", cell, "wall")
    show_plan(plan_astar(Position(4, 0), Position(0, 4), sealed_wh), sealed_wh)

    # E. start == goal.
    print()
    print("  E. Start equals goal at (2,2) - a zero-cost single-cell path:")
    show_plan(plan_astar(Position(2, 2), Position(2, 2), open_wh), open_wh)

    # F. A* vs Dijkstra on the same problem.
    print()
    print("  F. A* vs Dijkstra on one identical scenario (7x7, wall with a gap):")
    wall_wh = Warehouse(width=7, height=7, name="demo-wall")
    for row in range(6):
        wall_wh.add_static_obstacle(f"wall-{row}", Position(row, 3), "dividing wall")
    start, goal = Position(0, 0), Position(0, 6)
    astar_result = plan_astar(start, goal, wall_wh)
    dijkstra_result = plan_dijkstra(start, goal, wall_wh)
    show_plan(dijkstra_result, wall_wh)
    show_plan(astar_result, wall_wh)
    print()
    same_cost = astar_result.total_cost == dijkstra_result.total_cost
    print(f"     same optimal cost? {same_cost} "
          f"(A*={astar_result.total_cost:g}, Dijkstra={dijkstra_result.total_cost:g})")
    print(f"     nodes expanded   : A*={astar_result.nodes_expanded}, "
          f"Dijkstra={dijkstra_result.nodes_expanded}")
    print("     Both find an optimal-cost route. They need not return the identical")
    print("     list of cells - several shortest paths can exist - but the cost")
    print("     matches. A* is guided towards the goal by the Manhattan heuristic,")
    print("     so here it finalises fewer cells; that is typical, not guaranteed.")


def route_execution_demo() -> None:
    """Section 13: the route executor drives a robot along a planned path."""
    section("13. AUTONOMOUS ROUTE EXECUTION")
    print("  The planner returns a list of cells. The route executor actually")
    print("  moves the robot along it, one cell at a time, using the same move")
    print("  validation as every other move. It never teleports.")

    wh = Warehouse(width=8, height=8, name="demo-exec")
    for cell in (Position(3, 2), Position(3, 3), Position(3, 4)):
        wh.add_static_obstacle(f"rack-{cell.col}", cell, "rack")
    sim = WarehouseSimulator(wh)
    sim.add_robot(Robot("R1", Position(0, 0)))
    executor = RouteExecutor(sim)

    print()
    print("  A. Plan (0,0) -> (5,5) around the racks, then execute it:")
    route = plan_astar(Position(0, 0), Position(5, 5), wh)
    execution = executor.execute("R1", route)
    print(f"     planned steps : {execution.planned_steps}")
    print(f"     executed steps: {execution.executed_steps}")
    print(f"     final position: {execution.final_position}  (goal reached: "
          f"{execution.reached_goal})")
    print(f"     distance      : {execution.distance_travelled:g}")
    print(f"     battery       : {execution.battery_before:.1f}% -> "
          f"{execution.battery_after:.1f}%  (used {execution.battery_consumed:.1f}%)")
    print(f"     execution time: {execution.execution_time_ms:.3f} ms")

    print()
    print("  B. Plan a clear route, then drop a spill on it AFTER planning:")
    sim2 = WarehouseSimulator(Warehouse(width=8, height=8, name="demo-exec-2"))
    sim2.add_robot(Robot("R1", Position(0, 0)))
    blocked_route = plan_astar(Position(0, 0), Position(0, 6), sim2.warehouse)
    sim2.warehouse.add_dynamic_obstacle("spill-1", Position(0, 3), "spill")
    stopped = RouteExecutor(sim2).execute("R1", blocked_route)
    print(f"     success       : {stopped.success}")
    print(f"     executed steps: {stopped.executed_steps}/{stopped.planned_steps}")
    print(f"     stopped at    : {stopped.final_position}")
    print(f"     reason        : {stopped.failure_reason}")
    print("     (the executor stops safely; it does NOT replan - that is later work)")


def _build_allocation_scenario():
    """A fixed small fleet + task set used by sections 14 and 15.

    Chosen so the greedy baseline makes a locally-cheap first choice that
    strands a later task with an expensive robot, and CP-SAT does better.
    """
    wh = Warehouse(width=12, height=12, name="demo-alloc")
    for cell in (Position(5, 5), Position(5, 6), Position(6, 5), Position(6, 6)):
        wh.add_static_obstacle(f"rack-{cell.row}-{cell.col}", cell, "rack")

    robots = [
        Robot("R1", Position(0, 0), battery_level=100.0, payload_capacity=15.0),
        Robot("R2", Position(11, 11), battery_level=100.0, payload_capacity=15.0),
        Robot("R3", Position(0, 11), battery_level=100.0, payload_capacity=3.0),
        Robot("R4", Position(11, 0), battery_level=8.0, payload_capacity=15.0),
    ]
    tasks = [
        # URGENT, so greedy handles it first: only R4 is near and R4 is almost
        # flat, so the route must be short - it is.
        Task("T4", Position(10, 1), Position(8, 1), TaskPriority.URGENT, 2.0),
        # HIGH: greedy gives it R1 (cheapest for THIS task)...
        Task("T1", Position(0, 5), Position(0, 7), TaskPriority.HIGH, 2.0),
        # ...but T2 is even cheaper for R1, and now R1 is taken. CP-SAT swaps.
        Task("T2", Position(0, 3), Position(0, 1), TaskPriority.NORMAL, 2.0),
        # heavy: R3 (3kg cap) cannot carry it - payload infeasible.
        Task("T3", Position(10, 10), Position(8, 10), TaskPriority.NORMAL, 8.0),
        # 5th task, one more than robots: it stays pending this cycle.
        Task("T5", Position(4, 8), Position(7, 9), TaskPriority.LOW, 2.0),
    ]
    return wh, robots, tasks


def _print_allocation(result, warehouse, robots, tasks) -> None:
    problems = find_allocation_problems(result, robots, tasks, CostEstimator(warehouse))
    validity = "valid" if not problems else f"INVALID: {problems}"
    print(f"     solver status : {result.solver_status}")
    print(f"     feasible pairs: {result.eligible_pair_count}   "
          f"infeasible pairs: {result.infeasible_pair_count}")
    for a in result.assignments:
        print(f"       {a.robot_id} -> {a.task_id}   "
              f"R->P {a.robot_to_pickup_cost} + P->D {a.pickup_to_dropoff_cost} "
              f"= {a.total_estimated_cost}")
    print(f"     assigned      : {result.assigned_task_count}   "
          f"unassigned: {result.unassigned_task_ids}")
    print(f"     total est cost: {result.total_estimated_cost}")
    print(f"     solve time    : {result.solve_time_ms:.3f} ms")
    print(f"     validation    : {validity}")


def task_allocation_demo() -> None:
    """Section 14: greedy vs CP-SAT allocation over the same fleet and tasks."""
    section("14. TASK ALLOCATION (greedy baseline vs CP-SAT optimiser)")
    print("  4 robots, 5 pending tasks. Each robot may take at most one task this")
    print("  cycle, so at least one task stays pending. R3 can only carry 3kg;")
    print("  R4 has 8% battery. Both allocators use the SAME A*-based costs.")

    wh, robots, tasks = _build_allocation_scenario()
    estimator = CostEstimator(wh)
    r3, r4 = robots[2], robots[3]
    t3 = next(t for t in tasks if t.task_id == "T3")

    # E. An infeasible robot-task pairing, explained.
    print()
    print("  E. Why some robot-task pairings are infeasible:")
    print(f"       R3 carrying T3 ({t3.payload_weight:.0f}kg): R3 capacity is "
          f"{r3.payload_capacity:.0f}kg  -> payload infeasible")
    cost_r4_t3 = estimator.estimate(r4, t3)
    if cost_r4_t3 is not None:
        need = cost_r4_t3.total * r4.battery_drain_per_move
        print(f"       R4 doing T3: route is ~{cost_r4_t3.total} moves, needs "
              f"~{need:.0f}% battery, R4 has only {r4.battery_level:.0f}%  "
              f"-> battery infeasible")

    print()
    print("  B. GREEDY: each task, most urgent first, to its cheapest free robot:")
    greedy = GreedyAllocator(estimator).allocate(robots, tasks)
    _print_allocation(greedy, wh, robots, tasks)

    print()
    print("  C. CP-SAT: solve the assignment as a binary optimisation problem:")
    cp_sat = CpSatAllocator(estimator).allocate(robots, tasks)
    _print_allocation(cp_sat, wh, robots, tasks)

    print()
    print("  D. Comparison (same 4 robots, same 5 tasks):")
    print(f"     greedy : {greedy.assigned_task_count} tasks, "
          f"total est cost {greedy.total_estimated_cost}")
    print(f"     CP-SAT : {cp_sat.assigned_task_count} tasks, "
          f"total est cost {cp_sat.total_estimated_cost}")
    if greedy.assigned_task_count == cp_sat.assigned_task_count:
        delta = greedy.total_estimated_cost - cp_sat.total_estimated_cost
        print(f"     same number of tasks assigned; CP-SAT total cost is "
              f"{delta} lower" if delta else
              "     same number of tasks assigned; identical total cost here")
    else:
        print("     different task counts - not a like-for-like cost comparison")


def allocation_to_execution_demo() -> None:
    """Section 15: allocate -> commit -> plan -> execute, end to end."""
    section("15. ALLOCATION -> PLANNING -> AUTONOMOUS EXECUTION")
    print("  Take the CP-SAT plan, commit it through the normal task API, then")
    print("  for one assignment: plan the route to the pickup and drive it.")

    wh, robots, tasks = _build_allocation_scenario()
    sim = WarehouseSimulator(wh)
    for robot in robots:
        sim.add_robot(robot)
    for task in tasks:
        sim.add_task(task)

    estimator = CostEstimator(wh)
    plan = CpSatAllocator(estimator).allocate(sim.robots, sim.tasks)
    report = commit_allocation(sim, plan)
    print()
    print(f"  committed: {report.committed}   applied: {report.applied}")

    if not plan.assignments:
        print("  (no assignments to execute)")
        return

    first = plan.assignments[0]
    robot = sim.get_robot(first.robot_id)
    task = sim.get_task(first.task_id)
    print()
    print(f"  Driving {robot.robot_id} to the pickup for {task.task_id} "
          f"{task.pickup_location}:")
    route = plan_astar(robot.position, task.pickup_location, wh)
    execution = RouteExecutor(sim).execute(robot.robot_id, route)
    print(f"     route planned : {[str(c) for c in route.path]}")
    print(f"     executed steps: {execution.executed_steps}/{execution.planned_steps}")
    print(f"     final position: {execution.final_position}  "
          f"(at pickup: {execution.final_position == task.pickup_location})")
    print(f"     battery used  : {execution.battery_consumed:.1f}%")
    print(f"     task status   : {task.status}   robot status: {robot.status}")
    print()
    print("  Note: only ONE robot is driven here. Moving every assigned robot")
    print("  through shared aisles at once needs collision avoidance, which is")
    print("  not implemented - so the demo does not pretend to do it.")


def _place(warehouse: Warehouse, starts: dict) -> WarehouseSimulator:
    sim = WarehouseSimulator(warehouse)
    for robot_id, cell in sorted(starts.items()):
        sim.add_robot(Robot(robot_id, cell))
    return sim


def _independent_execution(warehouse: Warehouse, starts: dict, goals: dict) -> dict:
    """Plan each robot's route with plain A* and drive them one after another,
    exactly the pre-coordination behaviour. Returns {robot_id: reached_goal}."""
    sim = _place(warehouse, starts)
    executor = RouteExecutor(sim)
    outcomes = {}
    for robot_id in sorted(starts):
        route = plan_astar(sim.get_robot(robot_id).position, goals[robot_id], warehouse)
        execution = executor.execute(robot_id, route)
        outcomes[robot_id] = execution.success
    return outcomes


def coordination_demo() -> None:
    """Section 16: why independent routes conflict, and how coordination fixes it."""
    section("16. MULTI-ROBOT COORDINATION (independent vs coordinated)")
    print("  R1 finishes its route ON the central cell. R2's shortest route runs")
    print("  straight through that cell. Independent A* routes ignore each other.")

    wh = Warehouse(width=7, height=7, name="demo-intersection")
    starts = {"R1": Position(3, 0), "R2": Position(0, 3)}
    goals = {"R1": Position(3, 3), "R2": Position(6, 3)}

    print()
    print("  A. INDEPENDENT: plan each route with A*, execute one robot then the")
    print("     other. R1 parks on (3,3); R2 then cannot get past it:")
    independent = _independent_execution(wh, starts, goals)
    for robot_id, ok in independent.items():
        print(f"       {robot_id}: reached goal = {ok}")

    print()
    print("  B. COORDINATED: prioritized space-time planning (default order by ID):")
    coordination = MultiRobotCoordinator(wh).plan(starts, goals)
    for robot_id in coordination.robot_order:
        tp = coordination.timed_paths[robot_id]
        print(f"       {tp}")
    conflict_free = not find_coordination_problems(
        coordination.successful_paths, wh, starts
    )
    print(f"     makespan {coordination.makespan}, "
          f"{coordination.total_move_steps} moves + "
          f"{coordination.total_wait_steps} waits, "
          f"conflict-free validation: {conflict_free}")

    sim = _place(wh, starts)
    execution = CoordinatedExecutor(sim).execute(coordination)
    print(f"     synchronized execution: {execution.robots_reached_goal}/"
          f"{len(starts)} reached goal, "
          f"{execution.vertex_conflicts} vertex + {execution.edge_conflicts} "
          f"edge conflicts during execution")


def coordinated_execution_walkthrough() -> None:
    """Section 17: show the synchronized timestep table for one small scenario."""
    section("17. SYNCHRONIZED TIMESTEP EXECUTION")
    print("  At each timestep the whole joint move is validated, THEN committed.")
    print("  R2 must WAIT one step to let R1 clear the crossing.")

    wh = Warehouse(width=6, height=6, name="demo-sync")
    starts = {"R1": Position(2, 0), "R2": Position(0, 2)}
    goals = {"R1": Position(2, 4), "R2": Position(4, 2)}
    coordination = MultiRobotCoordinator(wh).plan(starts, goals)

    print()
    print("   t | " + " | ".join(f"{rid} pos   action" for rid in coordination.robot_order))
    print("  ---+-" + "-+-".join(["-" * 16 for _ in coordination.robot_order]))
    for t in range(0, coordination.makespan + 1):
        cells = []
        for rid in coordination.robot_order:
            tp = coordination.timed_paths[rid]
            here = tp.position_at(t)
            if t == 0:
                action = "start"
            elif t > tp.arrival_time:
                action = "(done)"
            elif tp.position_at(t) == tp.position_at(t - 1):
                action = "WAIT"
            else:
                action = "MOVE"
            cells.append(f"{str(here):<7} {action:<7}")
        print(f"  {t:2d} | " + " | ".join(cells))

    sim = _place(wh, starts)
    execution = CoordinatedExecutor(sim).execute(coordination)
    print()
    print(f"  executed: {execution.total_move_steps} moves, "
          f"{execution.total_wait_steps} waits, "
          f"makespan {execution.makespan_executed}; "
          f"trace validates conflict-free: {execution.validated_conflict_free}")


def coordination_edge_and_scale_demo() -> None:
    """Section 18: edge/swap prevention, and a larger coordinated fleet."""
    section("18. EDGE/SWAP PREVENTION AND A LARGER FLEET")

    print("  A. HEAD-ON in a one-cell-wide corridor - a swap is impossible, so")
    print("     prioritized planning places R1 and reports R2 cannot be routed:")
    corridor = Warehouse(width=6, height=1, name="demo-corridor")
    res = MultiRobotCoordinator(corridor).plan(
        {"R1": Position(0, 0), "R2": Position(0, 5)},
        {"R1": Position(0, 5), "R2": Position(0, 0)},
    )
    for rid in res.robot_order:
        print(f"       {res.timed_paths[rid]}")
    print(f"     result: {res.planned_robot_count}/2 planned, "
          f"failed = {res.failed_robot_ids}  (no unsafe swap was allowed)")

    print()
    print("  B. Four robots swapping corners of a 9x9 room through the middle:")
    wh = Warehouse(width=9, height=9, name="demo-swap4")
    starts = {"R1": Position(4, 0), "R2": Position(0, 4),
              "R3": Position(8, 4), "R4": Position(4, 8)}
    goals = {"R1": Position(4, 8), "R2": Position(8, 4),
             "R3": Position(0, 4), "R4": Position(4, 0)}
    coordination = MultiRobotCoordinator(wh).plan(starts, goals)
    sim = _place(wh, starts)
    execution = CoordinatedExecutor(sim).execute(coordination)
    print(f"     planned {coordination.planned_robot_count}/4, "
          f"makespan {coordination.makespan}, "
          f"{coordination.total_wait_steps} total waits")
    print(f"     executed: {execution.robots_reached_goal}/4 reached goal, "
          f"{execution.vertex_conflicts} vertex + {execution.edge_conflicts} "
          f"edge conflicts, trace valid: {execution.validated_conflict_free}")

    print()
    print("  C. Bottleneck: three robots must file through a single doorway:")
    room = Warehouse(width=7, height=7, name="demo-bottleneck")
    for row in range(7):
        if row != 3:
            room.add_static_obstacle(f"wall-{row}", Position(row, 3))
    b_starts = {"R1": Position(1, 1), "R2": Position(2, 1), "R3": Position(4, 1)}
    b_goals = {"R1": Position(1, 5), "R2": Position(2, 5), "R3": Position(4, 5)}
    b_coord = MultiRobotCoordinator(room).plan(b_starts, b_goals)
    b_sim = _place(room, b_starts)
    b_exec = CoordinatedExecutor(b_sim).execute(b_coord)
    straight = max(
        b_starts[r].manhattan_distance_to(b_goals[r]) for r in b_starts
    )
    print(f"     straight-line best makespan would be {straight}; "
          f"coordinated makespan is {b_coord.makespan} "
          f"(+{b_coord.makespan - straight} from queueing)")
    print(f"     {b_coord.total_wait_steps} wait steps, "
          f"executed {b_exec.robots_reached_goal}/3 to goal, "
          f"{b_exec.vertex_conflicts + b_exec.edge_conflicts} conflicts")


def dynamic_replan_demo() -> None:
    """Section 19: an obstacle appears mid-drive; the robot replans and resumes."""
    section("19. DYNAMIC OBSTACLE - REPLAN FROM THE CURRENT CELL")
    print("  R1 is driving a planned route. Partway through, an obstacle lands on")
    print("  a cell it was about to use. Recovery replans FROM WHERE R1 IS NOW -")
    print("  not from the original start - and R1 continues.")

    wh = Warehouse(width=7, height=7, name="demo-replan")
    coordination = MultiRobotCoordinator(wh).plan(
        {"R1": Position(3, 0)}, {"R1": Position(3, 6)}
    )
    original = coordination.timed_paths["R1"]
    print()
    print(f"  original route : {[str(s.position) for s in original.steps]}")

    sim = _place(wh, {"R1": Position(3, 0)})
    # R1 reaches (3,3) at t=3; the obstacle blocks (3,4) just ahead of it
    schedule = DisruptionSchedule.of(DynamicObstacleEvent(3, Position(3, 4)))
    result = RecoveryManager(sim).run(coordination, schedule)

    event = result.recovery_events[0]
    replan = event.robot_replans[0]
    print(f"  event          : obstacle at (3,4) at t={event.trigger_timestep}")
    print(f"  R1 was at      : {replan.current_position}  "
          f"(NOT the start {original.start})")
    print(f"  abandoned tail : {[str(c) for c in replan.old_remaining_path]}")
    print(f"  new route tail : {[str(c) for c in replan.new_path]}")
    print(f"  extra distance : {replan.additional_distance:+d} moves, "
          f"{replan.wait_actions_introduced} waits added")
    print(f"  replan latency : {replan.replanning_time_ms:.3f} ms")
    print(f"  reservations   : {event.reservations_released} released, "
          f"{event.reservations_created} recreated")
    print(f"  outcome        : reached goal = {result.per_robot['R1']['reached_goal']}, "
          f"final = {sim.get_robot('R1').position}, "
          f"conflicts = {result.vertex_conflicts}v + {result.edge_conflicts}e")


def no_route_recovery_demo() -> None:
    """Section 20: the obstacle boxes the robot in; recovery reports a safe stop."""
    section("20. DYNAMIC OBSTACLE - NO ROUTE, SAFE STOP")
    print("  A one-cell-wide corridor. An obstacle drops onto the only path.")
    print("  There is nowhere to go, so recovery stops R1 safely and says so.")

    wh = Warehouse(width=6, height=1, name="demo-noroute")
    coordination = MultiRobotCoordinator(wh).plan(
        {"R1": Position(0, 0)}, {"R1": Position(0, 5)}
    )
    sim = _place(wh, {"R1": Position(0, 0)})
    result = RecoveryManager(sim).run(
        coordination, DisruptionSchedule.of(DynamicObstacleEvent(2, Position(0, 3)))
    )
    event = result.recovery_events[0]
    print()
    print(f"  event          : obstacle at (0,3) at t={event.trigger_timestep}")
    print(f"  replanning     : attempted = {event.replanning_attempted}, "
          f"succeeded = {event.replanning_success}")
    print(f"  R1 stopped at  : {sim.get_robot('R1').position}  "
          f"(never entered the blocked (0,3))")
    print(f"  result         : success = {result.success}, "
          f"safe_stop = {result.safe_stop}")
    print(f"  reason         : {event.failure_reason}")
    print(f"  conflicts      : {result.vertex_conflicts}v + {result.edge_conflicts}e "
          f"(a safe stop is never an unsafe move)")


def _recovery_fleet():
    """A fixed 10x10 warehouse, 3 robots (one spare), 2 tasks."""
    wh = Warehouse(width=10, height=10, name="demo-recovery")
    robots = {"R1": Position(0, 0), "R2": Position(9, 9), "R3": Position(9, 0)}
    tasks = [
        Task("T1", Position(0, 6), Position(3, 6), TaskPriority.NORMAL, 2.0),
        Task("T2", Position(8, 4), Position(6, 4), TaskPriority.NORMAL, 2.0),
    ]
    return wh, robots, tasks


def robot_failure_demo() -> None:
    """Section 21: a robot goes offline; its task is released and reassigned."""
    section("21. ROBOT FAILURE - TASK REASSIGNMENT")
    print("  R1 is assigned T1 and heading for the pickup. It goes OFFLINE.")
    print("  Recovery releases T1 (ASSIGNED -> PENDING), the allocator offers it")
    print("  to the remaining robots, and the winner drives to the pickup.")

    wh, robot_cells, tasks = _recovery_fleet()
    sim = _place(wh, robot_cells)
    for task in tasks:
        sim.add_task(task)
    allocation = CpSatAllocator(CostEstimator(wh)).allocate(sim.robots, sim.tasks)
    commit_allocation(sim, allocation)
    assigned = {a.robot_id: a.task_id for a in allocation.assignments}
    print()
    print(f"  initial assignment : {assigned}")

    goals = {a.robot_id: sim.get_task(a.task_id).pickup_location
             for a in allocation.assignments}
    coordination = MultiRobotCoordinator(wh).plan_for(sim, goals)
    victim = allocation.assignments[0].robot_id
    result = RecoveryManager(sim).run(
        coordination, DisruptionSchedule.of(RobotFailureEvent(3, victim))
    )

    event = result.recovery_events[0]
    task = sim.get_task(event.reassigned_task_id or assigned[victim])
    print(f"  {victim} fails at t={event.trigger_timestep}, "
          f"stays at {sim.get_robot(victim).position}, status {sim.get_robot(victim).status}")
    print(f"  task {task.task_id}: {task.status}  (never falsely COMPLETED)")
    print(f"  reassignment   : attempted = {event.task_reassignment_attempted}, "
          f"{event.reassigned_from} -> {event.reassigned_to}, "
          f"success = {event.task_reassignment_success}")
    if event.task_reassignment_success:
        rep = next(r for r in event.robot_replans if r.robot_id == event.reassigned_to)
        print(f"  replacement {event.reassigned_to}: planned {rep.new_length}-move route, "
              f"final = {sim.get_robot(event.reassigned_to).position}, "
              f"at pickup = {sim.get_robot(event.reassigned_to).position == task.pickup_location}")
    print(f"  conflicts      : {result.vertex_conflicts}v + {result.edge_conflicts}e, "
          f"trace valid = {result.validated_conflict_free}")


def coordinated_recovery_demo() -> None:
    """Section 22: an obstacle invalidates two robots of a coordinated fleet."""
    section("22. COORDINATED RECOVERY - RE-COORDINATE THE AFFECTED SUBSET")
    print("  R1 and R2 cross the same central cell at different timesteps. An")
    print("  obstacle lands there early. BOTH routes are invalid; recovery")
    print("  replans and re-coordinates the pair; R3 is untouched.")

    wh = Warehouse(width=9, height=9, name="demo-corecovery")
    starts = {"R1": Position(4, 0), "R2": Position(0, 4), "R3": Position(8, 0)}
    goals = {"R1": Position(4, 8), "R2": Position(8, 4), "R3": Position(0, 0)}
    coordination = MultiRobotCoordinator(wh).plan(starts, goals)

    sim = _place(wh, starts)
    result = RecoveryManager(sim).run(
        coordination, DisruptionSchedule.of(DynamicObstacleEvent(2, Position(4, 4)))
    )
    event = result.recovery_events[0]
    print()
    print(f"  affected robots : {event.affected_robot_ids}  (R3 not disturbed)")
    for replan in event.robot_replans:
        print(f"    {replan.robot_id}: replan {'ok' if replan.success else 'FAILED'}, "
              f"{replan.additional_distance:+d} moves, "
              f"{replan.wait_actions_introduced} waits added")
    print(f"  re-coordination : {event.unresolved_vertex_conflicts} vertex + "
          f"{event.unresolved_edge_conflicts} edge conflicts remain (target: 0)")
    print(f"  reservations    : {event.reservations_released} released, "
          f"{event.reservations_created} recreated")
    reached = {r: result.per_robot[r]["reached_goal"] for r in sorted(starts)}
    print(f"  outcome         : reached goal = {reached}, "
          f"trace valid = {result.validated_conflict_free}")


def print_map(simulator: WarehouseSimulator, caption: Optional[str] = None) -> None:
    if caption:
        print(caption)
    print(render_map(simulator))
    print(LEGEND)
    print(render_robot_legend(simulator))


# ======================================================================
# Sections 23-28: the REST service / persistence layer
#
# These sections talk to the FastAPI app IN-PROCESS through the test client
# (no server to start, no port to bind) and against a throwaway SQLite
# database, so the demo stays deterministic and self-contained. The exact
# same app and services run under uvicorn + PostgreSQL in production.
# ======================================================================
def _demo_api_client():
    from fastapi.testclient import TestClient
    from sqlalchemy import create_engine
    from sqlalchemy.pool import StaticPool

    from robotics.api.app import create_app
    from robotics.api.security import create_access_token, hash_password
    from robotics.persistence.config import Settings
    from robotics.persistence.database import Database
    from robotics.persistence.models import UserRecord

    _jwt_key = "demo-only-signing-key-at-least-32-bytes-long-xxxx"
    settings = Settings(
        database_url="sqlite+pysqlite://",
        log_level="ERROR",
        jwt_secret_key=_jwt_key,
    )
    engine = create_engine(
        "sqlite+pysqlite://",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    database = Database(settings=settings, engine=engine)
    database.create_all()

    # The API enforces JWT auth + role-based authorization. Seed one operator
    # (may read, write, and run planning/allocation/coordination/recovery) and
    # attach its bearer token so the demo can exercise every endpoint.
    with database.session_scope() as session:
        session.add(
            UserRecord(
                username="demo_operator",
                password_hash=hash_password("demo-operator-pw"),
                role="operator",
            )
        )
    token = create_access_token("demo_operator", "operator", _jwt_key, expires_minutes=60)

    app = create_app(settings=settings, database=database, create_tables=False)
    client = TestClient(app)
    client.headers["Authorization"] = f"Bearer {token}"
    return client, database


def _demo_concurrent_client():
    """An operator TestClient over a FILE-based SQLite database with a normal
    pool, so overlapping requests each get their own connection. Returns
    (client, database, cleanup)."""
    import shutil
    import tempfile
    from pathlib import Path

    from fastapi.testclient import TestClient
    from sqlalchemy import create_engine

    from robotics.api.app import create_app
    from robotics.api.security import create_access_token, hash_password
    from robotics.persistence.config import Settings
    from robotics.persistence.database import Database
    from robotics.persistence.models import UserRecord

    _jwt_key = "demo-only-signing-key-at-least-32-bytes-long-xxxx"
    tmpdir = tempfile.mkdtemp(prefix="demo_conc_")
    url = f"sqlite+pysqlite:///{Path(tmpdir) / 'demo.db'}"
    settings = Settings(database_url=url, log_level="ERROR", jwt_secret_key=_jwt_key,
                        auto_create_tables=False)
    engine = create_engine(url, future=True,
                           connect_args={"check_same_thread": False, "timeout": 30})
    database = Database(settings=settings, engine=engine)
    database.create_all()
    with database.session_scope() as session:
        session.add(UserRecord(username="demo_operator",
                               password_hash=hash_password("demo-operator-pw"),
                               role="operator"))
    token = create_access_token("demo_operator", "operator", _jwt_key, expires_minutes=60)
    app = create_app(settings=settings, database=database, create_tables=False)
    client = TestClient(app)
    client.headers["Authorization"] = f"Bearer {token}"

    def cleanup() -> None:
        engine.dispose()
        shutil.rmtree(tmpdir, ignore_errors=True)

    return client, database, cleanup


def service_architecture_demo() -> None:
    section("23. SERVICE + API ARCHITECTURE")
    print("  The REST layer wraps the SAME robotics code the sections above use.")
    print("  Layers: FastAPI router -> Pydantic validation -> service -> "
          "planner/allocator/")
    print("          coordinator/recovery + SQLAlchemy -> database.")
    print()
    client, _ = _demo_api_client()
    health = client.get("/health").json()
    ready = client.get("/ready").json()
    print(f"  GET /health -> {health}")
    print(f"  GET /ready  -> {ready}")
    print("  health = 'is the process alive?' (no DB call)")
    print("  ready  = 'can it serve DB-backed requests?' (runs SELECT 1)")


def database_persistence_demo() -> None:
    section("24. DATABASE PERSISTENCE (SQLAlchemy)")
    client, database = _demo_api_client()

    created = client.post(
        "/robots", json={"robot_id": "R1", "position": {"row": 0, "col": 1},
                         "battery_level": 80.0}
    ).json()
    print(f"  POST /robots -> stored {created['robot_id']} at "
          f"{tuple(created['position'].values())}, battery {created['battery_level']}%")

    # a brand-new session proves the row is really on disk, not just in memory
    from robotics.persistence.repositories import RobotRepository

    with database.session() as session:
        again = RobotRepository(session).get("R1")
        print(f"  new session       -> R1 still there: battery {again.battery_level}%")

    # a deliberate failure rolls back cleanly
    from robotics.persistence.models import RobotRecord

    try:
        with database.session_scope() as session:
            session.add(RobotRecord(robot_id="R2", row=1, col=1))
            session.flush()
            raise RuntimeError("boom - simulated mid-transaction failure")
    except RuntimeError as error:
        print(f"  transaction failed -> {error}")
    with database.session() as session:
        rolled_back = RobotRepository(session).get("R2") is None
        print(f"  after ROLLBACK     -> R2 persisted? {not rolled_back}  "
              f"(nothing half-written)")


def planning_service_demo() -> None:
    section("25. PATH PLANNING THROUGH THE SERVICE LAYER")
    from robotics.services.world import build_warehouse

    client, _ = _demo_api_client()
    body = {"start": {"row": 0, "col": 0}, "goal": {"row": 11, "col": 10},
            "algorithm": "astar"}
    result = client.post("/planning/path", json=body).json()
    print(f"  POST /planning/path  A* (0,0) -> (11,10)")
    print(f"    success={result['success']}  cost={result['total_cost']}  "
          f"nodes_expanded={result['nodes_expanded']}  run_id={result['run_id']}")

    direct = plan_astar(Position(0, 0), Position(11, 10), build_warehouse())
    print(f"  direct planner call  cost={direct.total_cost}  "
          f"nodes_expanded={direct.nodes_expanded}")
    print(f"  API and direct call agree: "
          f"{result['total_cost'] == direct.total_cost and result['nodes_expanded'] == direct.nodes_expanded}")

    blocked = client.post("/planning/path", json={**body, "goal": {"row": 0, "col": 3},
        "warehouse": {"width": 4, "height": 4,
                      "static_obstacles": [{"row": 0, "col": 2}, {"row": 1, "col": 3}]}}).json()
    print(f"  walled-in goal -> success={blocked['success']}  "
          f"reason='{blocked['failure_reason']}'  (a 200 honest 'no route', not a 500)")


def allocation_service_demo() -> None:
    section("26. TASK ALLOCATION THROUGH THE SERVICE LAYER")
    client, _ = _demo_api_client()
    robots = [
        {"robot_id": "R1", "position": {"row": 0, "col": 0}, "payload_capacity": 10.0},
        {"robot_id": "R2", "position": {"row": 0, "col": 1}, "payload_capacity": 10.0},
    ]
    tasks = [
        {"task_id": "T1", "pickup": {"row": 2, "col": 4}, "dropoff": {"row": 11, "col": 1}, "payload_weight": 3.0},
        {"task_id": "T2", "pickup": {"row": 8, "col": 7}, "dropoff": {"row": 11, "col": 10}, "payload_weight": 4.0},
    ]
    for algo in ("greedy", "cp_sat"):
        result = client.post("/allocation/run",
                             json={"algorithm": algo, "robots": robots, "tasks": tasks}).json()
        pairs = ", ".join(f"{a['robot_id']}->{a['task_id']}" for a in result["assignments"])
        print(f"  POST /allocation/run [{algo:>6}] -> {pairs}  "
              f"cost={result['total_estimated_cost']}  status={result['solver_status']}")

    infeasible = client.post("/allocation/run", json={"algorithm": "greedy", "robots": robots,
        "tasks": [{"task_id": "TX", "pickup": {"row": 2, "col": 4},
                   "dropoff": {"row": 11, "col": 1}, "payload_weight": 20.0}]}).json()
    print(f"  20kg task, 10kg robots -> assignments={infeasible['assignments']}  "
          f"unassigned={infeasible['unassigned_task_ids']}  (reported, not errored)")


def coordination_service_demo() -> None:
    section("27. MULTI-ROBOT COORDINATION THROUGH THE SERVICE LAYER")
    client, _ = _demo_api_client()
    body = {"robots": [
        {"robot_id": "R1", "start": {"row": 4, "col": 0}, "goal": {"row": 4, "col": 6}},
        {"robot_id": "R2", "start": {"row": 4, "col": 6}, "goal": {"row": 4, "col": 0}},
        {"robot_id": "R3", "start": {"row": 0, "col": 0}, "goal": {"row": 7, "col": 0}},
    ]}
    result = client.post("/coordination/plan", json=body).json()
    print(f"  POST /coordination/plan  3 robots (R1/R2 head-on on row 4)")
    print(f"    success={result['success']}  makespan={result['makespan']}  "
          f"moves={result['total_move_steps']}  waits={result['total_wait_steps']}")
    print(f"    reservation lookups while planning: {result['reservation_lookups']}")
    print("    (prioritized planning - deterministic, not globally optimal MAPF)")


def recovery_service_demo() -> None:
    section("28. FAULT RECOVERY THROUGH THE SERVICE LAYER")
    client, _ = _demo_api_client()

    obstacle = client.post("/recovery/obstacle", json={
        "robots": [{"robot_id": "R1", "start": {"row": 4, "col": 0}, "goal": {"row": 4, "col": 6}}],
        "obstacle": {"row": 4, "col": 3}, "timestep": 2}).json()
    print(f"  POST /recovery/obstacle  obstacle drops on R1's route mid-drive")
    print(f"    replanning_success={obstacle['replanning_success']}  "
          f"recovery_success={obstacle['recovery_success']}  "
          f"safe_stop={obstacle['safe_stop']}  "
          f"conflicts={obstacle['unresolved_vertex_conflicts']}v+"
          f"{obstacle['unresolved_edge_conflicts']}e")

    failure = client.post("/recovery/robot-failure", json={
        "robots": [
            {"robot_id": "R1", "start": {"row": 4, "col": 0}, "goal": {"row": 4, "col": 4},
             "task": {"task_id": "T1", "pickup": {"row": 4, "col": 4},
                      "dropoff": {"row": 11, "col": 1}, "payload_weight": 2.0}},
            {"robot_id": "R2", "start": {"row": 7, "col": 0}, "goal": {"row": 7, "col": 0}},
        ],
        "failed_robot_id": "R1", "timestep": 1}).json()
    print(f"  POST /recovery/robot-failure  R1 (carrying T1) goes offline")
    print(f"    task reassigned {failure['reassigned_from']} -> {failure['reassigned_to']}  "
          f"success={failure['task_reassignment_success']}  safe_stop={failure['safe_stop']}")

    no_spare = client.post("/recovery/robot-failure", json={
        "robots": [
            {"robot_id": "R1", "start": {"row": 4, "col": 0}, "goal": {"row": 4, "col": 4},
             "task": {"task_id": "T1", "pickup": {"row": 4, "col": 4},
                      "dropoff": {"row": 11, "col": 1}, "payload_weight": 2.0}},
        ],
        "failed_robot_id": "R1", "timestep": 1}).json()
    print(f"  same, but no spare robot -> reassignment success="
          f"{no_spare['task_reassignment_success']}, safe_stop={no_spare['safe_stop']}  "
          f"(honest 'could not recover')")


def auth_service_demo() -> None:
    section("29. AUTHENTICATION & AUTHORIZATION THROUGH THE SERVICE LAYER")
    from fastapi.testclient import TestClient

    client, database = _demo_api_client()
    app = client.app

    # seed a viewer and an admin alongside the operator _demo_api_client made
    from robotics.api.security import hash_password
    from robotics.persistence.models import UserRecord

    with database.session_scope() as session:
        session.add(UserRecord(username="demo_viewer",
                               password_hash=hash_password("demo-viewer-pw"),
                               role="viewer"))

    anon = TestClient(app)  # no Authorization header
    print("  missing token   -> GET /robots  ->",
          f"{anon.get('/robots').status_code} (expect 401)")
    print("  bad token       -> GET /robots  ->",
          anon.get("/robots", headers={"Authorization": "Bearer not.a.token"}).status_code,
          "(expect 401)")

    login = anon.post("/auth/login",
                      json={"username": "demo_viewer", "password": "demo-viewer-pw"})
    viewer_token = login.json()["access_token"]
    viewer = TestClient(app)
    viewer.headers["Authorization"] = f"Bearer {viewer_token}"
    print("  valid login     -> POST /auth/login ->", login.status_code,
          f"role={login.json()['role']}")
    print("  viewer reads    -> GET /robots  ->", viewer.get("/robots").status_code,
          "(expect 200)")
    print("  viewer writes   -> POST /robots ->",
          viewer.post("/robots", json={"robot_id": "RX", "position": {"row": 0, "col": 1}}).status_code,
          "(expect 403 - authenticated but not authorized)")
    print("  operator writes -> POST /robots ->",
          client.post("/robots", json={"robot_id": "RX", "position": {"row": 0, "col": 1}}).status_code,
          "(expect 201)")
    print("  malformed body  -> POST /robots ->",
          client.post("/robots", json={"nope": True}).status_code, "(expect 422)")
    print("  roles: viewer < operator < admin; JWT (HS256) carries sub + role;")
    print("  passwords stored as bcrypt hashes; 401 = who are you, 403 = not allowed.")


def reliability_demo() -> None:
    section("30. RELIABILITY: RETRY, CORRELATION IDS, CONCURRENT-SAFE WRITES")
    from concurrent.futures import ThreadPoolExecutor

    from robotics.common.retry import RetryError, RetryPolicy, retry_call

    # --- bounded retry with exponential backoff (injected clock: no real wait) ---
    waited: list[float] = []
    attempts = {"n": 0}

    def flaky() -> str:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ConnectionError("transient")
        return "ok"

    result = retry_call(
        flaky,
        policy=RetryPolicy(attempts=4, base_delay=0.1, jitter=0.0),
        retry_on=(ConnectionError,),
        sleep=waited.append,
        operation="demo.probe",
    )
    print(f"  retry (transient x2 then ok) -> {result!r} after {attempts['n']} "
          f"attempts; backoff waits = {waited} (exponential, injected clock)")

    attempts["n"] = 0
    try:
        retry_call(
            lambda: (_ for _ in ()).throw(ConnectionError("always down")),
            policy=RetryPolicy(attempts=3, base_delay=0.0, jitter=0.0),
            retry_on=(ConnectionError,),
            sleep=lambda _s: None,
            operation="demo.always_fails",
        )
    except RetryError as exc:
        print(f"  retry exhausted -> RetryError after {exc.attempts} attempts "
              f"(bounded; a deterministic failure is NOT retried forever)")

    # --- per-request correlation id round trip ---
    client, database = _demo_api_client()
    r = client.get("/health", headers={"x-correlation-id": "demo-trace-001"})
    print(f"  GET /health x-correlation-id supplied 'demo-trace-001' -> "
          f"echoed back {r.headers.get('x-correlation-id')!r} "
          f"(also stamped on every log line for that request)")
    r2 = client.get("/health")
    print(f"  GET /health with no header -> generated id "
          f"{r2.headers.get('x-correlation-id')!r}")

    # --- concurrent duplicate create: the DB UNIQUE constraint + 409 handler ---
    # A file-based SQLite database with a normal pool, so each worker thread gets
    # its own real connection (the shared in-memory client above cannot service
    # overlapping writes). PostgreSQL behaves the same, with real parallelism.
    conc_client, conc_db, _cleanup = _demo_concurrent_client()
    try:
        body = {"robot_id": "RACE", "position": {"row": 0, "col": 0}}
        with ThreadPoolExecutor(max_workers=16) as pool:
            codes = list(pool.map(
                lambda _i: conc_client.post("/robots", json=body).status_code,
                range(16)))
        with conc_db.session_scope() as session:
            from robotics.persistence.models import RobotRecord
            rows = session.query(RobotRecord).filter_by(robot_id="RACE").count()
        print(f"  16 threads POST the same robot id -> "
              f"201x{codes.count(201)}  409x{codes.count(409)}  "
              f"5xx x{sum(1 for c in codes if c >= 500)}  rows_in_db={rows}")
        print("  (no application lock: the UNIQUE constraint decides the winner, "
              "the IntegrityError handler returns a clean 409)")
    finally:
        _cleanup()

    # --- a transient DB failure mid-request is a 503, not a 500 ---
    from fastapi.testclient import TestClient
    from sqlalchemy.exc import OperationalError

    from robotics.api.security import create_access_token

    app = client.app
    token = create_access_token(
        "demo_operator", "operator",
        "demo-only-signing-key-at-least-32-bytes-long-xxxx", expires_minutes=60)
    original_session = database.session
    database.session = lambda: (_ for _ in ()).throw(
        OperationalError("SELECT 1", {}, Exception("server closed the connection")))
    try:
        with TestClient(app, raise_server_exceptions=False) as broken:
            broken.headers["Authorization"] = f"Bearer {token}"
            resp = broken.get("/robots")
        leaked = "server closed the connection" in resp.text
        print(f"  DB drops mid-request -> HTTP {resp.status_code} "
              f"({resp.json()['error']}), Retry-After={resp.headers.get('retry-after')}, "
              f"driver text leaked in body: {leaked}")
    finally:
        database.session = original_session


def main() -> None:
    print("=" * LINE_WIDTH)
    print("AUTONOMOUS MULTI-ROBOT FULFILLMENT & DYNAMIC PATH PLANNING SYSTEM")
    print("Simulation foundation - demonstration walkthrough")
    print("=" * LINE_WIDTH)
    print(f"Random seed: {SEED} (the scenario is identical on every run)")

    # ------------------------------------------------------------------
    section("1. WAREHOUSE CREATION")
    # ------------------------------------------------------------------
    simulator = build_sample_simulator()
    warehouse = simulator.warehouse

    print(f"  warehouse         : {warehouse.name}")
    print(f"  size              : {warehouse.height} rows x {warehouse.width} cols")
    print(f"  total cells       : {warehouse.height * warehouse.width}")
    print(f"  static obstacles  : {len(warehouse.static_obstacles)} (storage racks)")
    print(f"  storage cells     : {[str(p) for p in warehouse.storage_cells]}")
    print(f"  pickup locations  : {[str(p) for p in warehouse.pickup_locations]}")
    print(f"  dropoff locations : {[str(p) for p in warehouse.dropoff_locations]}")
    print(f"  charging stations : {[str(p) for p in warehouse.charging_stations]}")
    print(f"  dynamic obstacles : {[str(o) for o in warehouse.dynamic_obstacles]}")
    print()
    print_map(simulator, "Initial warehouse map:")

    # ------------------------------------------------------------------
    section("2. ROBOT CREATION")
    # ------------------------------------------------------------------
    print(f"  {len(simulator.robots)} robots registered with the simulator.")
    print()
    print(render_robot_table(simulator))

    # ------------------------------------------------------------------
    section("3. TASK CREATION")
    # ------------------------------------------------------------------
    print(f"  {len(simulator.tasks)} fulfillment tasks created, all PENDING.")
    print()
    print(render_task_table(simulator))

    # ------------------------------------------------------------------
    section("4. MANUAL TASK ASSIGNMENT")
    # ------------------------------------------------------------------
    print("  Tasks are assigned by hand. Automatic optimisation is future work.")
    print()
    attempt_assignment(simulator, "T3", "R1")
    print()
    print("  T2 weighs 7.5kg. Robot R3 can only carry 5kg, so it is refused;")
    print("  robot R4 can carry 15kg, so it is accepted:")
    attempt_assignment(simulator, "T2", "R3")
    attempt_assignment(simulator, "T2", "R4")
    print()
    print("  R3 takes the lighter task T1 (3kg), and is then no longer free:")
    attempt_assignment(simulator, "T1", "R3")
    attempt_assignment(simulator, "T4", "R3")
    print()
    print(render_task_table(simulator))

    # ------------------------------------------------------------------
    section("5. VALID ROBOT MOVEMENT (one cell at a time)")
    # ------------------------------------------------------------------
    print("  R1 walks from the top aisle down to the storage cell at (2, 1).")
    print("  Each step is chosen by this script, NOT by a path planner.")
    print()
    attempt_move(simulator, "R1", Position(1, 1), "DOWN")
    attempt_move(simulator, "R1", Position(2, 1), "DOWN, onto a storage cell")

    # ------------------------------------------------------------------
    section("6. INVALID MOVEMENT - INTO A STATIC OBSTACLE")
    # ------------------------------------------------------------------
    print("  R1 is at (2, 1). The cell to its right, (2, 2), is a storage rack.")
    print()
    attempt_move(simulator, "R1", Position(2, 2), "RIGHT into a rack")

    # ------------------------------------------------------------------
    section("7. INVALID MOVEMENT - OUTSIDE THE WAREHOUSE")
    # ------------------------------------------------------------------
    print("  R2 is at (0, 4), on the top row. There is no row -1.")
    print()
    attempt_move(simulator, "R2", Position(-1, 4), "UP off the map")

    # ------------------------------------------------------------------
    section("8. OTHER INVALID MOVEMENTS")
    # ------------------------------------------------------------------
    print("  a) Not adjacent - a robot may not teleport:")
    attempt_move(simulator, "R1", Position(5, 5), "three cells away")
    print()
    print("  b) Blocked by the dynamic obstacle (the spill at (1, 4)):")
    attempt_move(simulator, "R2", Position(1, 4), "DOWN into the spill")
    print()
    print("  c) Occupied by another robot - R3 is standing at (0, 5):")
    attempt_move(simulator, "R2", Position(0, 5), "RIGHT onto R3")

    # ------------------------------------------------------------------
    section("9. BATTERY: DRAIN THROUGH MOVEMENT, THEN RECHARGE")
    # ------------------------------------------------------------------
    robot_r1 = simulator.get_robot("R1")
    print(f"  R1 battery now: {robot_r1.battery_level:.1f}% after "
          f"{robot_r1.steps_taken} steps "
          f"(each move costs {robot_r1.battery_drain_per_move:.1f}%).")
    print()
    print("  R1 drives back to the charging station at (0, 0):")
    attempt_move(simulator, "R1", Position(1, 1), "UP")
    attempt_move(simulator, "R1", Position(0, 1), "UP")
    attempt_move(simulator, "R1", Position(0, 0), "LEFT, onto the charger")
    print()

    battery_before_charge = robot_r1.battery_level
    added = simulator.charge_robot("R1")
    print(f"  charge_robot('R1'): battery {battery_before_charge:.1f}% -> "
          f"{robot_r1.battery_level:.1f}%  (+{added:.1f} points), "
          f"status now '{robot_r1.status}'")
    print()
    print("  And charging somewhere that is not a charging station is refused:")
    try:
        simulator.charge_robot("R4")
    except RoboticsError as error:
        print(f"  REJECTED  charge R4 -> reason: {error}")

    # ------------------------------------------------------------------
    section("10. DYNAMIC OBSTACLES: REMOVE AND ADD")
    # ------------------------------------------------------------------
    print("  Cleaning crew clears the spill at cell (1, 4):")
    removed = simulator.remove_dynamic_obstacle(SPILL_OBSTACLE_ID)
    print(f"  removed: {removed}")
    print()
    print("  The move that was refused in step 8b now succeeds:")
    attempt_move(simulator, "R2", Position(1, 4), "DOWN, spill cleared")
    print()
    print("  A maintenance cart is parked at (4, 7):")
    added_obstacle = simulator.add_dynamic_obstacle(
        obstacle_id="cart-1",
        position=Position(4, 7),
        description="maintenance cart",
    )
    print(f"  added: {added_obstacle}")
    print()
    print("  Dropping an obstacle onto a cell a robot occupies is refused:")
    try:
        simulator.add_dynamic_obstacle("cart-2", simulator.get_robot("R4").position)
    except RoboticsError as error:
        print(f"  REJECTED  reason: {error}")

    # ------------------------------------------------------------------
    section("11. FINAL SIMULATION STATE")
    # ------------------------------------------------------------------
    print_map(simulator, "Map after the demo:")
    print()
    print(render_robot_table(simulator))
    print()
    print(render_task_table(simulator))

    state = simulator.get_current_state()
    print()
    print("  get_current_state() summary:")
    print(f"    robots            : {len(state['robots'])}")
    print(f"    tasks             : {len(state['tasks'])}")
    print(f"    dynamic obstacles : {state['warehouse']['dynamic_obstacles']}")

    # ------------------------------------------------------------------
    path_planning_demo()
    route_execution_demo()
    task_allocation_demo()
    allocation_to_execution_demo()
    coordination_demo()
    coordinated_execution_walkthrough()
    coordination_edge_and_scale_demo()
    dynamic_replan_demo()
    no_route_recovery_demo()
    robot_failure_demo()
    coordinated_recovery_demo()
    service_architecture_demo()
    database_persistence_demo()
    planning_service_demo()
    allocation_service_demo()
    coordination_service_demo()
    recovery_service_demo()
    auth_service_demo()
    reliability_demo()

    # ------------------------------------------------------------------
    section("DEMO COMPLETE")
    print("  Implemented:")
    print("    - 2D warehouse grid, named locations, static and dynamic obstacles")
    print("    - robots with battery and payload, tasks with a lifecycle")
    print("    - manual task assignment and validated single-cell movement")
    print("    - single-robot path planning: Dijkstra and A* from first principles,")
    print("      Manhattan heuristic, obstacle-aware routing, path validation")
    print("    - autonomous route execution: driving a robot along a planned path")
    print("      cell by cell, stopping safely if a step is refused")
    print("    - task allocation across the fleet: a greedy baseline and a CP-SAT")
    print("      optimiser (OR-Tools), with feasibility (payload, battery, reach),")
    print("      an A*-based cost estimate, and plan-first / commit-second")
    print("    - multi-robot coordination: space-time A* with a reservation table,")
    print("      WAIT actions, vertex + edge/swap conflict prevention, prioritized")
    print("      planning, an independent conflict validator, and synchronized")
    print("      validate-first / commit-second execution")
    print("    - dynamic replanning: an obstacle appearing mid-drive triggers a")
    print("      replan from the robot's current cell and a re-coordination of")
    print("      the affected subset - or a safe stop when no route exists")
    print("    - robot fault recovery: an OFFLINE robot's task is released and")
    print("      reassigned through the existing allocator; its cell blocks others")
    print("    - a REST service layer: FastAPI + Pydantic endpoints for robots,")
    print("      tasks, planning, allocation, coordination and recovery, backed by")
    print("      SQLAlchemy persistence (PostgreSQL in production, SQLite in tests),")
    print("      with health/readiness checks and transaction-safe writes")
    print("    - JWT authentication + role-based authorization (viewer/operator/")
    print("      admin), bcrypt password hashing, 401/403 handling")
    print("    - Alembic schema migrations (upgrade/downgrade), a container image")
    print("      and Compose stack (app + PostgreSQL), and a GitHub Actions CI")
    print("      pipeline that runs the suite against a real PostgreSQL service")
    print("    - reproducible benchmarks for planning, allocation, coordination,")
    print("      recovery, the service/database layer, and behaviour under concurrency")
    print("    - reliability hardening: one transaction per request, a bounded")
    print("      retry/backoff utility, database-constraint race safety (409, not")
    print("      500), 503 + Retry-After on a transient DB outage, per-request")
    print("      correlation ids, and a graceful startup/shutdown lifecycle")
    print()
    print("  NOT implemented yet:")
    print("    - recovery when the disruption is structurally impossible to route")
    print("      around (a fully blocked corridor, a failure with no spare robot)")
    print("    - globally optimal multi-agent path finding (this is prioritized,")
    print("      not optimal - priority order can affect whether every robot fits)")
    print("    - one robot doing several tasks in a cycle (vehicle routing)")
    print("    - async orchestration of independent robot execution, and")
    print("      deployment to a hosting target")
    print()


if __name__ == "__main__":
    main()
