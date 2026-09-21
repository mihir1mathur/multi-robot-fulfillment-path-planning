"""RouteExecutor: walk a robot along a planned route, one cell at a time.

THE DIVISION OF LABOUR
----------------------
    Planner   "Where should I go?"   -> a list of cells (PathResult)
    Executor  "Actually move there." -> performs each single-cell step

The executor deliberately owns NO movement rules of its own. Every step goes
through `WarehouseSimulator.move_robot`, which already checks bounds,
adjacency, static and dynamic obstacles, robot-to-robot occupancy and battery,
and updates the robot's position, odometer, step count and battery. Re-writing
any of that here would risk the executor and the simulator disagreeing.

WHAT THE EXECUTOR ADDS
----------------------
* it iterates the path from path[1] onwards (path[0] is where the robot already
  stands, so it is never executed as a move);
* it checks, before starting, that the route actually begins at the robot's
  current cell;
* it stops SAFELY the moment a step is refused - blocked cell, flat battery,
  anything - and reports how far it got and why it stopped;
* it returns one `ExecutionResult` with measured before/after numbers.

NOT HERE: replanning. If the next cell is blocked, the executor stops. It does
not compute a new route. That is a later capability.
"""

from __future__ import annotations

import time
from typing import List, Sequence, Union

from robotics.exceptions import InvalidMoveError
from robotics.execution.execution_result import ExecutionResult
from robotics.planning.path_result import PathResult
from robotics.robots.robot_state import RobotStatus
from robotics.warehouse.grid import Position

# A route can be handed in as the planner's own result object, or as a bare
# list of cells (useful in tests and for callers that built a path by hand).
Route = Union[PathResult, Sequence[Position]]


class RouteExecutor:
    """Drives one robot along one planned route inside a simulator."""

    def __init__(self, simulator) -> None:
        """
        Args:
            simulator: the `WarehouseSimulator` that owns the robot and the
                warehouse. All movement and validation is delegated to it.
        """
        self.simulator = simulator

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def execute(self, robot_id: str, route: Route) -> ExecutionResult:
        """Move `robot_id` along `route`, stopping safely on the first problem.

        Args:
            robot_id: the robot to drive. Must be registered with the simulator.
            route: a successful `PathResult`, or a non-empty sequence of cells
                starting at the robot's current position.

        Returns:
            An `ExecutionResult`. `success` is True only if the robot reached
            the final cell of the route.

        Raises:
            SimulationError: if no robot has that ID (a programming error, kept
                consistent with the rest of the codebase which raises for
                unknown IDs).
        """
        started_at_ns = time.perf_counter_ns()

        def elapsed_ms() -> float:
            return (time.perf_counter_ns() - started_at_ns) / 1_000_000

        robot = self.simulator.get_robot(robot_id)
        battery_before = robot.battery_level
        odometer_before = robot.distance_travelled

        # --- Normalise the route to a list of cells, rejecting bad input ---
        path, rejection = self._resolve_path(route)
        if rejection is not None:
            return ExecutionResult.failed(
                robot_id=robot_id,
                start=robot.position,
                goal=robot.position,
                final_position=robot.position,
                planned_steps=0,
                executed_steps=0,
                distance_travelled=0.0,
                battery_before=battery_before,
                battery_after=robot.battery_level,
                completed_path=[robot.position],
                execution_time_ms=elapsed_ms(),
                reason=rejection,
            )

        start, goal = path[0], path[-1]
        planned_steps = len(path) - 1

        # --- Pre-flight checks that do not depend on stepping ---
        preflight = self._preflight(robot, path)
        if preflight is not None:
            return ExecutionResult.failed(
                robot_id=robot_id,
                start=start,
                goal=goal,
                final_position=robot.position,
                planned_steps=planned_steps,
                executed_steps=0,
                distance_travelled=0.0,
                battery_before=battery_before,
                battery_after=robot.battery_level,
                completed_path=[robot.position],
                execution_time_ms=elapsed_ms(),
                reason=preflight,
            )

        # --- start == goal: nothing to drive ---
        if planned_steps == 0:
            return ExecutionResult.completed(
                robot_id=robot_id,
                start=start,
                goal=goal,
                planned_steps=0,
                executed_steps=0,
                distance_travelled=0.0,
                battery_before=battery_before,
                battery_after=robot.battery_level,
                completed_path=[start],
                execution_time_ms=elapsed_ms(),
            )

        # --- Walk the route, cell by cell ---
        completed_path: List[Position] = [start]
        executed_steps = 0

        for next_cell in path[1:]:
            try:
                self.simulator.move_robot(robot_id, next_cell)
            except InvalidMoveError as error:
                # A step was refused. The simulator left the robot untouched.
                # Stop here and report exactly how far execution got.
                return ExecutionResult.failed(
                    robot_id=robot_id,
                    start=start,
                    goal=goal,
                    final_position=robot.position,
                    planned_steps=planned_steps,
                    executed_steps=executed_steps,
                    distance_travelled=robot.distance_travelled - odometer_before,
                    battery_before=battery_before,
                    battery_after=robot.battery_level,
                    completed_path=completed_path,
                    execution_time_ms=elapsed_ms(),
                    reason=(
                        f"stopped after {executed_steps} step(s) at "
                        f"{robot.position}: {error}"
                    ),
                )

            executed_steps += 1
            completed_path.append(robot.position)

        # --- Arrived ---
        return ExecutionResult.completed(
            robot_id=robot_id,
            start=start,
            goal=goal,
            planned_steps=planned_steps,
            executed_steps=executed_steps,
            distance_travelled=robot.distance_travelled - odometer_before,
            battery_before=battery_before,
            battery_after=robot.battery_level,
            completed_path=completed_path,
            execution_time_ms=elapsed_ms(),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_path(route: Route):
        """Turn a route argument into a plain list of cells, or a rejection string.

        Returns:
            (path, None) on success, or (None, reason) when the route cannot be
            executed at all.
        """
        if isinstance(route, PathResult):
            if not route.success:
                return None, (
                    f"planner did not find a route "
                    f"({route.failure_reason or 'no path'})"
                )
            path = list(route.path)
        else:
            try:
                path = list(route)
            except TypeError:
                return None, "route is neither a PathResult nor a sequence of cells"

        if not path:
            return None, "route is empty"
        if not all(isinstance(cell, Position) for cell in path):
            return None, "route contains something that is not a Position"

        # Reject an internally broken path before moving anything: consecutive
        # cells must be exactly one orthogonal step apart, and no cell repeats.
        for index in range(len(path) - 1):
            if not path[index].is_adjacent_to(path[index + 1]):
                return None, (
                    f"route is not walkable: cell {index} {path[index]} and cell "
                    f"{index + 1} {path[index + 1]} are not one orthogonal step apart"
                )
        if len(set(path)) != len(path):
            return None, "route visits the same cell more than once"

        return path, None

    def _preflight(self, robot, path: List[Position]):
        """Checks that can be made before the first step. Returns a reason or None."""
        if robot.status is RobotStatus.OFFLINE:
            return f"robot '{robot.robot_id}' is offline and cannot execute a route"

        if path[0] != robot.position:
            return (
                f"route starts at {path[0]} but robot '{robot.robot_id}' is at "
                f"{robot.position}"
            )

        return None
