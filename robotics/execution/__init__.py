"""Autonomous route execution: driving a robot along a planned path.

WHERE THIS SITS
---------------
The planner answers "WHERE should the robot go?" and returns a `PathResult` -
a list of cells from start to goal. It does not move anything.

This package answers "actually MOVE there, one cell at a time." It takes that
list of cells and walks the robot along it, step by step, reusing the
simulator's existing move validation for every single step. No teleporting.

    plan(start, goal, warehouse) -> PathResult        (robotics.planning)
                     |
                     v
    RouteExecutor(simulator).execute(robot_id, result) -> ExecutionResult

SCOPE - WHAT THIS DOES NOT DO
----------------------------
* It moves ONE robot along ONE route. It is not a fleet loop.
* It does NOT replan. If the world changed after the route was planned and the
  next cell is now blocked, execution STOPS SAFELY and reports why. Repairing
  the route mid-drive (dynamic replanning) is a separate, later capability.
* It does not coordinate robots in shared space over time. The simulator still
  refuses any step onto an occupied cell, so a stale route cannot cause an
  overlap - the step is simply rejected and execution stops.
"""

from robotics.execution.execution_result import ExecutionResult
from robotics.execution.route_executor import RouteExecutor

__all__ = ["ExecutionResult", "RouteExecutor"]
