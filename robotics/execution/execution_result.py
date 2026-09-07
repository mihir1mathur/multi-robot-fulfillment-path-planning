"""The result of driving a robot along one planned route.

WHY A SHARED RESULT TYPE (same reasoning as PathResult)
------------------------------------------------------
Execution can succeed, or it can stop part-way for any of several reasons
(the next cell was blocked, the battery ran out, the path did not start where
the robot was standing). Every caller - the demo, the tests, the benchmark -
needs the same shape of answer so it never has to guess.

    completed        did the robot reach the goal?
    executed_steps   how many single-cell moves were actually performed
    planned_steps    how many the route asked for (len(path) - 1)
    completed_path   the cells the robot actually stood on, start included
    failure_reason   why it stopped early, or None on success

All the numeric fields are MEASURED from the robot before and after execution,
never assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from robotics.warehouse.grid import Position


@dataclass
class ExecutionResult:
    """What `RouteExecutor.execute` returns, whether or not the robot arrived.

    Attributes:
        success: True only if the robot reached the goal cell.
        robot_id: the robot that was driven.
        start: the first cell of the route (where the robot began).
        goal: the last cell of the route (the intended destination).
        final_position: where the robot actually ended up.
        planned_steps: moves the route required, i.e. len(path) - 1.
        executed_steps: moves actually performed (<= planned_steps).
        distance_travelled: grid distance covered during THIS execution,
            measured as the change in the robot's odometer.
        battery_before: robot battery percentage before the first step.
        battery_after: robot battery percentage after the last step.
        completed_path: cells the robot actually occupied, start included. On
            success this equals the planned path; on early stop it is the
            prefix that was driven.
        execution_time_ms: wall-clock duration of the execute() call.
        failure_reason: why execution stopped early, or None on success.
    """

    success: bool
    robot_id: str
    start: Position
    goal: Position
    final_position: Position
    planned_steps: int
    executed_steps: int
    distance_travelled: float
    battery_before: float
    battery_after: float
    completed_path: List[Position] = field(default_factory=list)
    execution_time_ms: float = 0.0
    failure_reason: Optional[str] = None

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def completed(
        cls,
        robot_id: str,
        start: Position,
        goal: Position,
        planned_steps: int,
        executed_steps: int,
        distance_travelled: float,
        battery_before: float,
        battery_after: float,
        completed_path: List[Position],
        execution_time_ms: float,
    ) -> "ExecutionResult":
        """Build a successful result: the robot reached the goal."""
        return cls(
            success=True,
            robot_id=robot_id,
            start=start,
            goal=goal,
            final_position=goal,
            planned_steps=planned_steps,
            executed_steps=executed_steps,
            distance_travelled=distance_travelled,
            battery_before=battery_before,
            battery_after=battery_after,
            completed_path=completed_path,
            execution_time_ms=execution_time_ms,
        )

    @classmethod
    def failed(
        cls,
        robot_id: str,
        start: Position,
        goal: Position,
        final_position: Position,
        planned_steps: int,
        executed_steps: int,
        distance_travelled: float,
        battery_before: float,
        battery_after: float,
        completed_path: List[Position],
        execution_time_ms: float,
        reason: str,
    ) -> "ExecutionResult":
        """Build an unsuccessful result, recording how far it got and why it stopped."""
        return cls(
            success=False,
            robot_id=robot_id,
            start=start,
            goal=goal,
            final_position=final_position,
            planned_steps=planned_steps,
            executed_steps=executed_steps,
            distance_travelled=distance_travelled,
            battery_before=battery_before,
            battery_after=battery_after,
            completed_path=completed_path,
            execution_time_ms=execution_time_ms,
            failure_reason=reason,
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    @property
    def battery_consumed(self) -> float:
        """Percentage points of charge used during this execution."""
        return self.battery_before - self.battery_after

    @property
    def reached_goal(self) -> bool:
        """True if the robot's final cell is the goal cell."""
        return self.final_position == self.goal

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """A JSON-safe snapshot, used by the benchmark writer."""
        return {
            "success": self.success,
            "robot_id": self.robot_id,
            "start": [self.start.row, self.start.col],
            "goal": [self.goal.row, self.goal.col],
            "final_position": [self.final_position.row, self.final_position.col],
            "planned_steps": self.planned_steps,
            "executed_steps": self.executed_steps,
            "distance_travelled": self.distance_travelled,
            "battery_before": round(self.battery_before, 4),
            "battery_after": round(self.battery_after, 4),
            "battery_consumed": round(self.battery_consumed, 4),
            "execution_time_ms": round(self.execution_time_ms, 6),
            "failure_reason": self.failure_reason,
        }

    def __str__(self) -> str:
        if self.success:
            return (
                f"{self.robot_id}: EXECUTED {self.start} -> {self.goal} | "
                f"{self.executed_steps}/{self.planned_steps} steps | "
                f"battery {self.battery_before:.1f}% -> {self.battery_after:.1f}% | "
                f"{self.execution_time_ms:.3f} ms"
            )
        return (
            f"{self.robot_id}: STOPPED at {self.final_position} "
            f"({self.executed_steps}/{self.planned_steps} steps) - {self.failure_reason}"
        )
