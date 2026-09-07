"""Space-Time A*: shortest route for one robot that also respects RESERVATIONS.

NORMAL A* vs SPACE-TIME A*
-------------------------
Normal A* (robotics.planning.astar) searches over states that are just cells:

    state = (row, col)

Space-Time A* searches over states that are a cell AND a timestep:

    state = (row, col, t)

Everything else is the same idea - a priority queue ordered by f = g + h, a
Manhattan heuristic, a closed set - with two changes:

  1. There is an extra action, WAIT, that stays in the same cell and advances
     time by one:  (r, c, t) -> (r, c, t+1).
  2. A move onto (cell, t+1) is only allowed if the reservation table says that
     (cell, t+1) is free AND the move does not swap head-on with another robot.

WHY THE SAME CELL CAN APPEAR MANY TIMES IN THE SEARCH
----------------------------------------------------
`(2,3)` at t=4 and `(2,3)` at t=7 are DIFFERENT states. A robot might legally
visit a cell, leave, and come back later to let another robot pass. The closed
set is keyed by `(cell, t)`, not by `cell`, so revisiting a cell at a new time
is never blocked.

    g = elapsed timesteps since the robot started  (MOVE costs 1, WAIT costs 1,
        so g is literally "how many timesteps have passed")
    h = Manhattan distance from the cell to the goal (ignores time and
        obstacles - still admissible)
    f = g + h

THE HORIZON
-----------
Time is unbounded, so the search needs a ceiling. `horizon` is the largest
timestep the search will consider. If the goal cannot be reached
conflict-free by then, planning FAILS - and the failure reason distinguishes
"the goal is spatially unreachable" from "no conflict-free route fits inside
the horizon".
"""

from __future__ import annotations

import heapq
import itertools
from typing import Callable, Dict, Optional, Set, Tuple

from robotics.coordination.reservation_table import ReservationTable
from robotics.coordination.timed_path import TimedPath, TimedStep
from robotics.planning.astar import plan as plan_spatial
from robotics.planning.heuristics import manhattan_distance
from robotics.warehouse.grid import Position
from robotics.warehouse.warehouse import Warehouse

Heuristic = Callable[[Position, Position], float]
STState = Tuple[Position, int]  # (cell, timestep)

# Reasons, kept as constants so tests and callers can match on them.
REASON_START_OOB = "start is outside the warehouse"
REASON_GOAL_OOB = "goal is outside the warehouse"
REASON_START_BLOCKED = "start cell is blocked by an obstacle"
REASON_GOAL_BLOCKED = "goal cell is blocked by an obstacle"
REASON_START_RESERVED = "start cell is reserved by another robot at the start time"
REASON_SPATIALLY_UNREACHABLE = "no route exists: the goal is spatially unreachable"
REASON_GOAL_HELD = "the goal cell is reserved by another robot through the horizon"
REASON_HORIZON = "no conflict-free route found within the planning horizon"


class SpaceTimePlanner:
    """Plans one robot's timed route around a reservation table."""

    def __init__(
        self, warehouse: Warehouse, heuristic: Heuristic = manhattan_distance
    ) -> None:
        self.warehouse = warehouse
        self._heuristic = heuristic

    # ------------------------------------------------------------------
    def plan(
        self,
        start: Position,
        goal: Position,
        reservations: ReservationTable,
        robot_id: str,
        start_time: int = 0,
        horizon: Optional[int] = None,
    ) -> TimedPath:
        """Find a conflict-free timed route from `start` to `goal`.

        Args:
            start: the robot's current cell.
            goal: the cell to reach.
            reservations: cells/edges already claimed by other robots. NOT
                mutated - the caller reserves the returned path afterwards.
            robot_id: whose route this is (so its own reservations don't block
                it).
            start_time: the timestep the robot begins at (usually 0).
            horizon: the largest timestep to consider. Defaults to a value
                derived from the warehouse size.

        Returns:
            A `TimedPath`. `success` is False with a specific `failure_reason`
            when no conflict-free route fits the horizon.
        """
        if horizon is None:
            horizon = 4 * (self.warehouse.width + self.warehouse.height)

        # ---- cheap rejections -----------------------------------------
        if not self.warehouse.in_bounds(start):
            return TimedPath.failed(robot_id, REASON_START_OOB)
        if not self.warehouse.in_bounds(goal):
            return TimedPath.failed(robot_id, REASON_GOAL_OOB)
        if not self.warehouse.is_traversable(start):
            return TimedPath.failed(robot_id, REASON_START_BLOCKED)
        if not self.warehouse.is_traversable(goal):
            return TimedPath.failed(robot_id, REASON_GOAL_BLOCKED)
        if not reservations.is_vertex_free(start, start_time, robot_id):
            return TimedPath.failed(robot_id, REASON_START_RESERVED)

        # ---- spatial reachability (reuse the existing single-robot A*) ---
        # This cleanly separates "there is no route at all" from "there is a
        # route but it can't be scheduled inside the horizon".
        spatial = plan_spatial(start, goal, self.warehouse)
        if not spatial.success:
            return TimedPath.failed(robot_id, REASON_SPATIALLY_UNREACHABLE)

        # ---- earliest timestep at which the robot can PARK on the goal ---
        earliest_hold = self._earliest_goal_hold(
            goal, reservations, robot_id, start_time, horizon
        )
        if earliest_hold > horizon:
            return TimedPath.failed(robot_id, REASON_GOAL_HELD)

        # ---- space-time A* -------------------------------------------
        return self._search(
            start, goal, reservations, robot_id, start_time, horizon, earliest_hold
        )

    # ------------------------------------------------------------------
    def _earliest_goal_hold(
        self,
        goal: Position,
        reservations: ReservationTable,
        robot_id: str,
        start_time: int,
        horizon: int,
    ) -> int:
        """Smallest arrival time from which `goal` stays free through `horizon`.

        Scans back from the horizon; the first foreign reservation it hits is
        the latest one, so the robot must arrive at least one step after it.
        """
        for t in range(horizon, start_time - 1, -1):
            reserver = reservations.vertex_reserver(goal, t)
            if reserver is not None and reserver != robot_id:
                return t + 1
        return start_time

    # ------------------------------------------------------------------
    def _search(
        self,
        start: Position,
        goal: Position,
        reservations: ReservationTable,
        robot_id: str,
        start_time: int,
        horizon: int,
        earliest_hold: int,
    ) -> TimedPath:
        def h(cell: Position) -> float:
            return self._heuristic(cell, goal)

        tie = itertools.count()
        start_state: STState = (start, start_time)
        start_h = h(start)
        frontier = [(start_h, start_h, next(tie), start, start_time)]
        came_from: Dict[STState, STState] = {}
        # best elapsed cost (== timestep - start_time) seen for each state
        best_g: Dict[STState, int] = {start_state: 0}
        closed: Set[STState] = set()

        while frontier:
            _, _, _, cell, t = heapq.heappop(frontier)
            state: STState = (cell, t)
            if state in closed:
                continue
            closed.add(state)

            # Goal test: on the goal cell AND late enough to park there for good.
            if cell == goal and t >= earliest_hold:
                return self._reconstruct(
                    came_from, start_state, state, robot_id
                )

            if t >= horizon:
                continue  # cannot advance time any further

            g_next = (t + 1) - start_time

            # --- WAIT: stay put, advance time ---
            if reservations.is_vertex_free(cell, t + 1, robot_id) and \
                    self.warehouse.is_traversable(cell):
                self._relax(
                    (cell, t + 1), state, g_next, h(cell), came_from, best_g,
                    frontier, tie,
                )

            # --- MOVE: to each currently-walkable neighbour ---
            for nb in self.warehouse.neighbors(cell):
                if reservations.is_move_allowed(cell, nb, t, robot_id):
                    self._relax(
                        (nb, t + 1), state, g_next, h(nb), came_from, best_g,
                        frontier, tie,
                    )

        return TimedPath.failed(robot_id, REASON_HORIZON)

    @staticmethod
    def _relax(
        neighbour: STState,
        current: STState,
        tentative_g: int,
        h_value: float,
        came_from: Dict[STState, STState],
        best_g: Dict[STState, int],
        frontier: list,
        tie: "itertools.count",
    ) -> None:
        if tentative_g < best_g.get(neighbour, float("inf")):
            best_g[neighbour] = tentative_g
            came_from[neighbour] = current
            f_value = tentative_g + h_value
            heapq.heappush(
                frontier,
                (f_value, h_value, next(tie), neighbour[0], neighbour[1]),
            )

    @staticmethod
    def _reconstruct(
        came_from: Dict[STState, STState],
        start_state: STState,
        goal_state: STState,
        robot_id: str,
    ) -> TimedPath:
        chain = [goal_state]
        current = goal_state
        while current != start_state:
            current = came_from[current]
            chain.append(current)
        chain.reverse()
        steps = [TimedStep(position=cell, timestep=t) for cell, t in chain]
        return TimedPath.found(robot_id, steps)
