"""The result of one coordinated planning pass over a fleet.

The coordinator PLANS - it does not move anything. This object describes the
conflict-free timed routes it found (or reports which robots it could not place
and why). A separate `CoordinatedExecutor` then drives the robots.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from robotics.coordination.timed_path import TimedPath


@dataclass
class CoordinationResult:
    """What `MultiRobotCoordinator.plan` returns.

    Attributes:
        success: True only if EVERY requested robot got a conflict-free route.
        timed_paths: {robot_id: TimedPath} for every requested robot (failed
            ones carry success=False and a reason).
        robot_order: the priority order the robots were planned in.
        planned_robot_count: how many robots got a successful timed path.
        failed_robot_ids: robots that could not be placed, in priority order.
        total_move_steps: sum of MOVE actions across all successful paths.
        total_wait_steps: sum of WAIT actions across all successful paths.
        makespan: the latest arrival timestep across all successful paths.
        total_space_time_cost: sum of every successful path's makespan
            (== total_move_steps + total_wait_steps).
        planning_time_ms: wall-clock time for the whole coordinated pass.
        conflict_checks: reservation-table lookups made during planning.
        horizon: the timestep ceiling used for the search.
        failure_reason: a summary when success is False, else None.
    """

    success: bool
    timed_paths: Dict[str, TimedPath] = field(default_factory=dict)
    robot_order: List[str] = field(default_factory=list)
    planned_robot_count: int = 0
    failed_robot_ids: List[str] = field(default_factory=list)
    total_move_steps: int = 0
    total_wait_steps: int = 0
    makespan: int = 0
    total_space_time_cost: int = 0
    planning_time_ms: float = 0.0
    conflict_checks: int = 0
    horizon: int = 0
    failure_reason: Optional[str] = None

    # ------------------------------------------------------------------
    @property
    def successful_paths(self) -> Dict[str, TimedPath]:
        return {rid: tp for rid, tp in self.timed_paths.items() if tp.success}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "robot_order": list(self.robot_order),
            "planned_robot_count": self.planned_robot_count,
            "failed_robot_ids": list(self.failed_robot_ids),
            "total_move_steps": self.total_move_steps,
            "total_wait_steps": self.total_wait_steps,
            "makespan": self.makespan,
            "total_space_time_cost": self.total_space_time_cost,
            "planning_time_ms": round(self.planning_time_ms, 6),
            "conflict_checks": self.conflict_checks,
            "horizon": self.horizon,
            "failure_reason": self.failure_reason,
            "timed_paths": {rid: tp.to_dict() for rid, tp in self.timed_paths.items()},
        }

    def __str__(self) -> str:
        head = "OK" if self.success else f"PARTIAL ({self.failure_reason})"
        return (
            f"coordination {head}: {self.planned_robot_count}/"
            f"{len(self.timed_paths)} robots | makespan {self.makespan} | "
            f"{self.total_move_steps} moves, {self.total_wait_steps} waits | "
            f"{self.planning_time_ms:.3f} ms"
        )


def summarise(
    timed_paths: Dict[str, TimedPath],
    robot_order: List[str],
    failed_robot_ids: List[str],
    planning_time_ms: float,
    conflict_checks: int,
    horizon: int,
) -> CoordinationResult:
    """Assemble a CoordinationResult, deriving the aggregate numbers.

    Deriving the totals here (rather than trusting the caller) keeps the
    summary numbers and the timed paths from ever disagreeing.
    """
    successful = [tp for tp in timed_paths.values() if tp.success]
    total_moves = sum(tp.move_count for tp in successful)
    total_waits = sum(tp.wait_count for tp in successful)
    makespan = max((tp.arrival_time for tp in successful), default=0)
    space_time_cost = sum(tp.makespan for tp in successful)

    failure_reason: Optional[str] = None
    if failed_robot_ids:
        parts = [
            f"{rid}: {timed_paths[rid].failure_reason}" for rid in failed_robot_ids
        ]
        failure_reason = "; ".join(parts)

    return CoordinationResult(
        success=not failed_robot_ids,
        timed_paths=timed_paths,
        robot_order=robot_order,
        planned_robot_count=len(successful),
        failed_robot_ids=failed_robot_ids,
        total_move_steps=total_moves,
        total_wait_steps=total_waits,
        makespan=makespan,
        total_space_time_cost=space_time_cost,
        planning_time_ms=planning_time_ms,
        conflict_checks=conflict_checks,
        horizon=horizon,
        failure_reason=failure_reason,
    )
