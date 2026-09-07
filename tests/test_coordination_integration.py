"""End-to-end and adversarial coordination scenarios.

These exercise the whole chain - allocation -> coordination -> synchronised
execution - and the hard conflict patterns (intersection, head-on corridor,
shared goal region, bottleneck).
"""

from __future__ import annotations

from robotics.allocation import CostEstimator, CpSatAllocator, commit_allocation
from robotics.coordination.conflicts import find_coordination_problems
from robotics.coordination.coordinated_executor import CoordinatedExecutor
from robotics.coordination.coordinator import MultiRobotCoordinator
from robotics.robots.robot import Robot
from robotics.simulation.simulator import WarehouseSimulator
from robotics.tasks.task import Task, TaskPriority
from robotics.warehouse.grid import Position
from robotics.warehouse.warehouse import Warehouse


def P(r, c):
    return Position(r, c)


# ----------------------------------------------------------------------
# Full pipeline: allocation -> coordination -> execution
# ----------------------------------------------------------------------
def test_allocation_then_coordination_then_execution() -> None:
    wh = Warehouse(width=10, height=10, name="pipeline")
    sim = WarehouseSimulator(wh)
    corners = [("R1", P(0, 0)), ("R2", P(9, 9)), ("R3", P(0, 9)), ("R4", P(9, 0))]
    for rid, cell in corners:
        sim.add_robot(Robot(rid, cell))
    tasks = [
        Task("T1", P(4, 4), P(1, 1), TaskPriority.NORMAL, 2.0),
        Task("T2", P(5, 5), P(8, 8), TaskPriority.NORMAL, 2.0),
        Task("T3", P(4, 5), P(1, 8), TaskPriority.NORMAL, 2.0),
        Task("T4", P(5, 4), P(8, 1), TaskPriority.NORMAL, 2.0),
    ]
    for t in tasks:
        sim.add_task(t)

    estimator = CostEstimator(wh)
    plan = CpSatAllocator(estimator).allocate(sim.robots, sim.tasks)
    commit_allocation(sim, plan)

    goals = {a.robot_id: sim.get_task(a.task_id).pickup_location for a in plan.assignments}
    coordination = MultiRobotCoordinator(wh).plan_for(sim, goals)
    assert coordination.success

    execution = CoordinatedExecutor(sim).execute(coordination)
    assert execution.success
    assert execution.robots_reached_goal == len(plan.assignments)
    assert execution.vertex_conflicts == 0
    assert execution.edge_conflicts == 0
    for assignment in plan.assignments:
        robot = sim.get_robot(assignment.robot_id)
        assert robot.position == sim.get_task(assignment.task_id).pickup_location


# ----------------------------------------------------------------------
# Adversarial: bottleneck
# ----------------------------------------------------------------------
def _bottleneck_warehouse() -> Warehouse:
    """A 7x7 room split by a wall with a single one-cell doorway at (3,3)."""
    wh = Warehouse(width=7, height=7, name="bottleneck")
    for row in range(7):
        if row != 3:
            wh.add_static_obstacle(f"wall-{row}", P(row, 3))
    return wh


def test_bottleneck_serialises_robots_through_one_doorway() -> None:
    wh = _bottleneck_warehouse()
    starts = {"R1": P(1, 1), "R2": P(2, 1), "R3": P(4, 1)}
    goals = {"R1": P(1, 5), "R2": P(2, 5), "R3": P(4, 5)}   # all must use (3,3)

    result = MultiRobotCoordinator(wh).plan(starts, goals)

    assert result.success
    # they cannot all cross the doorway at once, so there must be waiting
    assert result.total_wait_steps >= 1
    problems = find_coordination_problems(result.successful_paths, wh, starts)
    assert problems == []

    # and it executes collision-free
    sim = WarehouseSimulator(wh)
    for rid, cell in sorted(starts.items()):
        sim.add_robot(Robot(rid, cell))
    execution = CoordinatedExecutor(sim).execute(result)
    assert execution.success
    assert execution.robots_reached_goal == 3
    assert execution.vertex_conflicts == 0 and execution.edge_conflicts == 0


def test_bottleneck_waiting_overhead_is_reported() -> None:
    wh = _bottleneck_warehouse()
    starts = {"R1": P(1, 1), "R2": P(2, 1), "R3": P(4, 1), "R4": P(5, 1)}
    goals = {"R1": P(1, 5), "R2": P(2, 5), "R3": P(4, 5), "R4": P(5, 5)}

    result = MultiRobotCoordinator(wh).plan(starts, goals)
    if result.success:
        # makespan is inflated by the queueing at the doorway
        best_case = max(
            starts[r].manhattan_distance_to(goals[r]) for r in starts
        )
        assert result.makespan >= best_case
        assert result.total_wait_steps == result.total_space_time_cost - result.total_move_steps


# ----------------------------------------------------------------------
# Adversarial: shared goal region
# ----------------------------------------------------------------------
def test_shared_goal_region_one_robot_finishes_where_another_would_pass() -> None:
    wh = Warehouse(width=7, height=7, name="shared")
    # R1 finishes at (3,3). R2 would naturally drive straight through (3,3).
    starts = {"R1": P(3, 1), "R2": P(3, 6)}
    goals = {"R1": P(3, 3), "R2": P(3, 0)}
    result = MultiRobotCoordinator(wh).plan(starts, goals)

    assert result.success
    r1, r2 = result.timed_paths["R1"], result.timed_paths["R2"]
    # once R1 parks on (3,3), R2 is never there
    for t in range(r1.arrival_time, result.makespan + 1):
        assert r2.position_at(t) != P(3, 3)
    assert find_coordination_problems(result.successful_paths, wh, starts) == []
