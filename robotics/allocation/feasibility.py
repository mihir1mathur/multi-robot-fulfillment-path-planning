"""Deciding which robot-task pairings are even allowed, before optimising.

Both allocators - the greedy baseline and the CP-SAT optimiser - must agree on
exactly which pairings are legal and what each one costs, or their comparison
is meaningless. That shared logic lives here, in one place.

A pairing (robot r, task t) is a CANDIDATE only if ALL of these hold:

    1. r is ELIGIBLE          - status IDLE and not already holding a task
    2. t is a PENDING task    - not assigned, in progress, completed or failed
    3. PAYLOAD fits           - t.payload_weight <= r's remaining capacity
    4. a ROUTE exists         - both legs R->P and P->D are walkable
    5. BATTERY is enough      - r has charge for the whole estimated route,
                               using r's own battery_drain_per_move

If any check fails the pairing is INFEASIBLE and no allocator may select it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from robotics.allocation.cost_estimator import AssignmentCost, CostEstimator
from robotics.robots.robot import Robot
from robotics.tasks.task import Task, TaskStatus


@dataclass(frozen=True)
class CandidatePair:
    """One feasible (robot, task) pairing and its priced-out cost."""

    robot_id: str
    task_id: str
    cost: AssignmentCost

    @property
    def total_cost(self) -> int:
        return self.cost.total


def eligible_robots(robots: Sequence[Robot]) -> List[Robot]:
    """Robots that can take a new task right now, in a stable order (by ID)."""
    return sorted(
        (r for r in robots if r.is_available), key=lambda r: r.robot_id
    )


def pending_tasks(tasks: Sequence[Task]) -> List[Task]:
    """PENDING tasks, most urgent first, ties broken by task ID.

    This matches `WarehouseSimulator.pending_tasks` so the allocator processes
    tasks in the same order the rest of the system reports them.
    """
    return sorted(
        (t for t in tasks if t.status is TaskStatus.PENDING),
        key=lambda t: (-int(t.priority), t.task_id),
    )


def battery_moves_affordable(robot: Robot) -> float:
    """How many unit moves the robot's current charge can pay for.

    Uses the existing battery model: each move costs
    `robot.battery_drain_per_move` percentage points. A drain of 0 (a robot
    configured never to lose charge) means effectively unlimited moves.
    """
    if robot.battery_drain_per_move <= 0:
        return float("inf")
    return robot.battery_level / robot.battery_drain_per_move


def is_battery_feasible(robot: Robot, route_moves: int) -> bool:
    """True if the robot has enough charge for a route of `route_moves` moves.

    A simple, defensible estimate: the robot must be able to afford every move
    of the planned route from its current charge. It does not model a detour to
    a charger - that is a later, battery-aware-routing capability.
    """
    return route_moves <= battery_moves_affordable(robot)


def build_candidate_pairs(
    robots: Sequence[Robot],
    tasks: Sequence[Task],
    estimator: CostEstimator,
) -> Tuple[List[CandidatePair], int]:
    """Return (feasible pairs, infeasible pair count).

    Only ELIGIBLE robots and PENDING tasks are considered. Every such
    combination is either turned into a `CandidatePair` or counted as
    infeasible. The feasible list is deterministically ordered:
    (robot ID, then task priority desc, then task ID).
    """
    robots_in_order = eligible_robots(robots)
    tasks_in_order = pending_tasks(tasks)

    feasible: List[CandidatePair] = []
    infeasible = 0

    for robot in robots_in_order:
        for task in tasks_in_order:
            if not robot.can_carry(task.payload_weight):
                infeasible += 1
                continue

            cost = estimator.estimate(robot, task)
            if cost is None:
                infeasible += 1
                continue

            if not is_battery_feasible(robot, cost.total):
                infeasible += 1
                continue

            feasible.append(
                CandidatePair(
                    robot_id=robot.robot_id, task_id=task.task_id, cost=cost
                )
            )

    return feasible, infeasible
