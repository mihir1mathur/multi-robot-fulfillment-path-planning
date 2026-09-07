"""Independently checking an allocator's output.

Same idea as the path validator: "the solver returned assignments" and "the
assignments are actually legal" are different claims. A bug in the model - a
missing constraint, a wrong cost, a stale variable - would still produce a
confident-looking list of pairings. This checker knows nothing about how the
allocation was produced; it just walks the assignments and asks simple
questions.

Every allocation test asserts its result passes `find_allocation_problems`,
and the benchmark records the pass rate.
"""

from __future__ import annotations

from typing import List, Sequence

from robotics.allocation.allocation_result import AllocationResult
from robotics.allocation.cost_estimator import CostEstimator
from robotics.allocation.feasibility import (
    eligible_robots,
    is_battery_feasible,
    pending_tasks,
)
from robotics.robots.robot import Robot
from robotics.tasks.task import Task


def find_allocation_problems(
    result: AllocationResult,
    robots: Sequence[Robot],
    tasks: Sequence[Task],
    estimator: CostEstimator,
) -> List[str]:
    """Return a list of problems with an allocation result. Empty means valid.

    Checks:
        1. no task assigned twice
        2. no robot assigned twice in this cycle
        3. every assigned robot is a real, eligible robot
        4. every assigned task is a real, pending task
        5. every assignment is feasible (payload, battery, reachable) and its
           reported leg costs match a fresh estimate
        6. reported total_estimated_cost equals the sum of assignment costs
        7. unassigned_task_ids is exactly the candidate tasks minus the
           assigned ones
        8. counts are self-consistent
    """
    problems: List[str] = []

    if not result.success:
        # A failed allocation should carry no assignments.
        if result.assignments:
            problems.append(
                f"unsuccessful allocation should have no assignments, "
                f"got {len(result.assignments)}"
            )
        return problems

    eligible_ids = {r.robot_id for r in eligible_robots(robots)}
    robot_by_id = {r.robot_id: r for r in robots}
    candidate_tasks = pending_tasks(tasks)
    candidate_ids = [t.task_id for t in candidate_tasks]
    task_by_id = {t.task_id: t for t in tasks}

    seen_tasks: set = set()
    seen_robots: set = set()

    for a in result.assignments:
        if a.task_id in seen_tasks:
            problems.append(f"task {a.task_id} is assigned more than once")
        seen_tasks.add(a.task_id)

        if a.robot_id in seen_robots:
            problems.append(f"robot {a.robot_id} is assigned more than once")
        seen_robots.add(a.robot_id)

        if a.robot_id not in robot_by_id:
            problems.append(f"assignment names unknown robot {a.robot_id}")
            continue
        if a.robot_id not in eligible_ids:
            problems.append(f"robot {a.robot_id} was not eligible for a task")

        if a.task_id not in task_by_id:
            problems.append(f"assignment names unknown task {a.task_id}")
            continue
        if a.task_id not in candidate_ids:
            problems.append(f"task {a.task_id} was not a pending candidate")

        robot = robot_by_id[a.robot_id]
        task = task_by_id[a.task_id]

        if not robot.can_carry(task.payload_weight):
            problems.append(
                f"{a.robot_id}->{a.task_id}: payload {task.payload_weight}kg "
                f"exceeds capacity"
            )

        cost = estimator.estimate(robot, task)
        if cost is None:
            problems.append(f"{a.robot_id}->{a.task_id}: no walkable route exists")
        else:
            if not is_battery_feasible(robot, cost.total):
                problems.append(
                    f"{a.robot_id}->{a.task_id}: battery too low for "
                    f"{cost.total} moves"
                )
            if a.robot_to_pickup_cost != cost.robot_to_pickup:
                problems.append(
                    f"{a.robot_id}->{a.task_id}: reported robot_to_pickup_cost "
                    f"{a.robot_to_pickup_cost} != {cost.robot_to_pickup}"
                )
            if a.pickup_to_dropoff_cost != cost.pickup_to_dropoff:
                problems.append(
                    f"{a.robot_id}->{a.task_id}: reported pickup_to_dropoff_cost "
                    f"{a.pickup_to_dropoff_cost} != {cost.pickup_to_dropoff}"
                )

    reported_total = result.total_estimated_cost
    actual_total = sum(a.total_estimated_cost for a in result.assignments)
    if reported_total != actual_total:
        problems.append(
            f"reported total_estimated_cost {reported_total} != sum of "
            f"assignment costs {actual_total}"
        )

    expected_unassigned = [tid for tid in candidate_ids if tid not in seen_tasks]
    if sorted(result.unassigned_task_ids) != sorted(expected_unassigned):
        problems.append(
            f"unassigned_task_ids {sorted(result.unassigned_task_ids)} != "
            f"expected {sorted(expected_unassigned)}"
        )

    if result.assigned_task_count + result.unassigned_task_count != len(candidate_ids):
        problems.append(
            f"assigned ({result.assigned_task_count}) + unassigned "
            f"({result.unassigned_task_count}) != candidate tasks "
            f"({len(candidate_ids)})"
        )

    return problems


def is_valid_allocation(
    result: AllocationResult,
    robots: Sequence[Robot],
    tasks: Sequence[Task],
    estimator: CostEstimator,
) -> bool:
    """True if the allocation passes every check."""
    return not find_allocation_problems(result, robots, tasks, estimator)
