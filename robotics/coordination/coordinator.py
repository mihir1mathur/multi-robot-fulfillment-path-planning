"""Prioritized multi-robot coordination.

THE APPROACH (deliberately simple)
----------------------------------
1. Put the robots in a fixed, deterministic PRIORITY ORDER (default: by robot
   ID).
2. Plan the highest-priority robot with Space-Time A*. It sees an empty
   reservation table, so it just gets its shortest route.
3. RESERVE that robot's whole timed path - every (cell, timestep), every edge,
   and its goal cell from arrival through the horizon.
4. Plan the next robot. Its Space-Time A* now treats every reserved
   (cell, timestep) as briefly blocked, so it routes around them or WAITs.
5. Reserve it. Repeat.

WHY THIS IS NOT OPTIMAL (and that is fine)
-----------------------------------------
The priority order can matter. A high-priority robot can take a route that
boxes a low-priority robot in, so a later robot may fail to find any
conflict-free route even though a DIFFERENT priority order would have let every
robot through. This is a known limitation of prioritized planning. It buys
simplicity, determinism and a plan you can explain in two sentences.

FAILURE IS ISOLATED
-------------------
If a robot cannot be planned, it is recorded in `failed_robot_ids` and NOTHING
is reserved for it. Every earlier robot's reservations - and timed paths -
stay valid. The coordinator returns a PARTIAL result rather than throwing
everything away.

NO MUTATION
-----------
Planning never touches live robot state. It works purely from the start and
goal cells you pass in. Execution is a separate step.
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional

from robotics.coordination.coordination_result import CoordinationResult, summarise
from robotics.coordination.reservation_table import ReservationTable
from robotics.coordination.space_time_planner import SpaceTimePlanner
from robotics.coordination.timed_path import TimedPath
from robotics.planning.heuristics import manhattan_distance
from robotics.warehouse.grid import Position
from robotics.warehouse.warehouse import Warehouse

Heuristic = Callable[[Position, Position], float]


def default_horizon(warehouse: Warehouse, fleet_size: int) -> int:
    """A deterministic timestep ceiling derived from the grid and fleet size.

    `2 * (width + height)` covers a robot crossing the warehouse twice (a
    detour); `4 * fleet_size` adds slack for later robots that have to wait for
    earlier ones through a bottleneck. Generous on purpose - a too-small
    horizon turns solvable scenarios into failures.
    """
    span = warehouse.width + warehouse.height
    return 2 * span + 4 * max(1, fleet_size)


class MultiRobotCoordinator:
    """Plans conflict-free timed routes for a set of robots, one at a time."""

    def __init__(
        self, warehouse: Warehouse, heuristic: Heuristic = manhattan_distance
    ) -> None:
        self.warehouse = warehouse
        self._planner = SpaceTimePlanner(warehouse, heuristic)

    # ------------------------------------------------------------------
    def plan(
        self,
        starts: Dict[str, Position],
        goals: Dict[str, Position],
        priority_order: Optional[List[str]] = None,
        horizon: Optional[int] = None,
    ) -> CoordinationResult:
        """Plan every robot in `starts` to its goal in `goals`, conflict-free.

        Args:
            starts: {robot_id: current cell}.
            goals: {robot_id: goal cell}. Must have the same keys as `starts`.
            priority_order: the order to plan robots in. Must be a permutation
                of the robot IDs. Defaults to sorted robot IDs.
            horizon: timestep ceiling. Defaults to `default_horizon`.

        Returns:
            A `CoordinationResult`. `success` is True only if every robot got a
            conflict-free route.
        """
        started_ns = time.perf_counter_ns()

        def elapsed_ms() -> float:
            return (time.perf_counter_ns() - started_ns) / 1_000_000

        problem = self._validate_request(starts, goals, priority_order)
        if problem is not None:
            return CoordinationResult(
                success=False,
                timed_paths={},
                robot_order=list(priority_order or sorted(starts)),
                failure_reason=problem,
                planning_time_ms=elapsed_ms(),
                horizon=horizon or 0,
            )

        order = list(priority_order) if priority_order else sorted(starts)
        if horizon is None:
            horizon = default_horizon(self.warehouse, len(starts))

        reservations = ReservationTable()
        timed_paths: Dict[str, TimedPath] = {}
        failed: List[str] = []

        for robot_id in order:
            timed_path = self._planner.plan(
                start=starts[robot_id],
                goal=goals[robot_id],
                reservations=reservations,
                robot_id=robot_id,
                start_time=0,
                horizon=horizon,
            )
            timed_paths[robot_id] = timed_path

            if timed_path.success:
                # Reserving a successful robot's path can only fail if the
                # planner produced a route that clashes with an existing
                # reservation - that would be a planner bug, so let it raise.
                reservations.reserve_timed_path(timed_path, horizon)
            else:
                failed.append(robot_id)

        return summarise(
            timed_paths=timed_paths,
            robot_order=order,
            failed_robot_ids=failed,
            planning_time_ms=elapsed_ms(),
            conflict_checks=reservations.conflict_checks,
            horizon=horizon,
        )

    # ------------------------------------------------------------------
    def plan_for(
        self,
        simulator,
        goals: Dict[str, Position],
        priority_order: Optional[List[str]] = None,
        horizon: Optional[int] = None,
    ) -> CoordinationResult:
        """Convenience: take robot start cells from a simulator's live robots."""
        starts = {
            robot_id: simulator.get_robot(robot_id).position for robot_id in goals
        }
        return self.plan(starts, goals, priority_order, horizon)

    # ------------------------------------------------------------------
    def _validate_request(
        self,
        starts: Dict[str, Position],
        goals: Dict[str, Position],
        priority_order: Optional[List[str]],
    ) -> Optional[str]:
        """Return a reason the request is malformed, or None if it is fine."""
        if not starts:
            return "no robots to coordinate"
        if set(starts) != set(goals):
            missing = set(starts) ^ set(goals)
            return f"starts and goals disagree on robot IDs: {sorted(missing)}"

        # Two robots cannot start on the same cell.
        seen: Dict[Position, str] = {}
        for robot_id in sorted(starts):
            cell = starts[robot_id]
            if cell in seen:
                return (
                    f"invalid start configuration: robots '{seen[cell]}' and "
                    f"'{robot_id}' both start at {cell}"
                )
            seen[cell] = robot_id

        if priority_order is not None:
            if sorted(priority_order) != sorted(starts):
                return (
                    "priority_order must be a permutation of the robot IDs, got "
                    f"{priority_order}"
                )
            if len(priority_order) != len(set(priority_order)):
                return f"priority_order contains a duplicate: {priority_order}"

        return None
