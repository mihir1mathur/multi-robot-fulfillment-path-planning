"""Committing an allocation plan to the simulator, through the normal task API.

PLAN FIRST, COMMIT SECOND
------------------------
An allocator returns an `AllocationResult` and changes nothing. This module is
the second step: it takes that plan and actually assigns the tasks, using the
same `WarehouseSimulator.assign_task_manually` path a human would use. No robot
or task field is poked directly.

ALL-OR-NOTHING
--------------
Before touching anything, every assignment is dry-run checked against the live
world (task still PENDING, robot still available, robot can still carry the
load). Only if EVERY assignment passes does the commit proceed. This means a
plan that has gone stale - because the world changed between planning and
committing - is rejected cleanly, leaving the simulator exactly as it was,
rather than half-applied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from robotics.allocation.allocation_result import AllocationResult
from robotics.tasks.task import TaskStatus


@dataclass
class CommitReport:
    """What happened when an allocation plan was committed.

    Attributes:
        committed: True if every assignment was applied.
        applied: "robot->task" strings that were actually assigned.
        rejected_reason: why the commit was refused, or None on success. When
            set, NOTHING was applied.
    """

    committed: bool
    applied: List[str] = field(default_factory=list)
    rejected_reason: Optional[str] = None


def _why_not_committable(simulator, robot_id: str, task_id: str) -> Optional[str]:
    """Return a reason this one assignment cannot be committed now, or None."""
    try:
        task = simulator.get_task(task_id)
        robot = simulator.get_robot(robot_id)
    except Exception as error:  # SimulationError for an unknown ID
        return str(error)

    if task.status is not TaskStatus.PENDING:
        return f"task {task_id} is no longer pending (status {task.status.value})"
    if not robot.is_available:
        return (
            f"robot {robot_id} is no longer available "
            f"(status {robot.status}, task {robot.assigned_task_id})"
        )
    if not robot.can_carry(task.payload_weight):
        return (
            f"robot {robot_id} can no longer carry task {task_id} "
            f"({task.payload_weight}kg)"
        )
    return None


def commit_allocation(simulator, result: AllocationResult) -> CommitReport:
    """Apply every assignment in `result`, or none of them.

    Args:
        simulator: the `WarehouseSimulator` to apply the plan to.
        result: the plan produced by an allocator.

    Returns:
        A `CommitReport`. If `committed` is False, the simulator is unchanged.
    """
    if not result.success:
        return CommitReport(
            committed=False,
            rejected_reason="allocation result was not successful; nothing to commit",
        )

    if not result.assignments:
        return CommitReport(committed=True, applied=[])

    # --- Dry run: check them all before changing anything ---
    for assignment in result.assignments:
        reason = _why_not_committable(
            simulator, assignment.robot_id, assignment.task_id
        )
        if reason is not None:
            return CommitReport(
                committed=False,
                rejected_reason=f"cannot commit {assignment.robot_id}->"
                f"{assignment.task_id}: {reason}",
            )

    # --- Commit: every check passed, so each of these should succeed ---
    applied: List[str] = []
    for assignment in result.assignments:
        simulator.assign_task_manually(assignment.task_id, assignment.robot_id)
        applied.append(f"{assignment.robot_id}->{assignment.task_id}")

    return CommitReport(committed=True, applied=applied)
