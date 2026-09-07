"""Repairing ONE robot's timed route from where it actually is now.

THE CORE RULE
-------------
Replanning starts from the robot's CURRENT position at the CURRENT timestep -
never from its original start cell. The steps it has already driven are kept
untouched; only the future is rebuilt.

    original : S -> A -> B -> C -> D -> G
    robot is at B (timestep 2), then C becomes blocked
    WRONG    : replan S -> ... (throws away 2 real moves)
    RIGHT    : replan B@2 -> alternate -> G, keep S->A->B as history

WHAT THIS MODULE PROVIDES
------------------------
* `remaining_path_problem`  - is a robot's future route still walkable and
                              conflict-free? (returns a reason, or None)
* `build_future_reservations` - a ReservationTable holding a set of robots'
                              route reservations from a given timestep onward,
                              so a replan can route around them
* `DynamicReplanner.replan` - Space-Time A* from the current cell to the goal,
                              starting at the current timestep
* `splice`                  - executed history + replanned future -> one full
                              TimedPath
"""

from __future__ import annotations

import time
from typing import Dict, Iterable, List, Optional, Sequence

from robotics.coordination.reservation_table import ReservationTable
from robotics.coordination.space_time_planner import SpaceTimePlanner
from robotics.coordination.timed_path import TimedPath, TimedStep
from robotics.planning.heuristics import manhattan_distance
from robotics.warehouse.grid import Position
from robotics.warehouse.warehouse import Warehouse


# ----------------------------------------------------------------------
# Reservation helpers
# ----------------------------------------------------------------------
def build_future_reservations(
    timed_paths: Dict[str, TimedPath],
    from_timestep: int,
    horizon: int,
    exclude: Iterable[str] = (),
) -> ReservationTable:
    """A reservation table of every listed robot's route from `from_timestep` on.

    Each robot's remaining steps (vertices + edges) are reserved, plus its goal
    hold to the horizon. Robots in `exclude` are left out (they are the ones
    being replanned, or have failed).
    """
    excluded = set(exclude)
    table = ReservationTable()

    for robot_id, tp in sorted(timed_paths.items()):
        if robot_id in excluded or not tp.success:
            continue

        remaining = [s for s in tp.steps if s.timestep >= from_timestep]
        # make sure the robot's position AT from_timestep is pinned even if its
        # path ended earlier (it is parked on its goal)
        if not remaining:
            parked = tp.position_at(from_timestep)
            if parked is not None:
                for t in range(from_timestep, horizon + 1):
                    table.reserve_vertex(parked, t, robot_id)
            continue

        for step in remaining:
            table.reserve_vertex(step.position, step.timestep, robot_id)
        for earlier, later in zip(remaining, remaining[1:]):
            if earlier.position != later.position:
                table.reserve_edge(
                    earlier.position, later.position, earlier.timestep, robot_id
                )
        goal = remaining[-1].position
        for t in range(remaining[-1].timestep + 1, horizon + 1):
            table.reserve_vertex(goal, t, robot_id)

    return table


# ----------------------------------------------------------------------
# Validity of a robot's remaining route
# ----------------------------------------------------------------------
def remaining_path_problem(
    timed_path: TimedPath,
    from_timestep: int,
    warehouse: Warehouse,
    reservations: Optional[ReservationTable] = None,
) -> Optional[str]:
    """Is the part of `timed_path` from `from_timestep` onward still safe?

    Checks the CURRENT warehouse (so a newly-appeared obstacle is caught) and,
    if a reservation table is given, that no future step now clashes with
    another robot. Returns a human-readable reason, or None if still fine.
    """
    if not timed_path.success:
        return "path was never successful"

    remaining = [s for s in timed_path.steps if s.timestep >= from_timestep]
    if len(remaining) <= 1:
        return None  # nothing left to drive

    rid = timed_path.robot_id
    for step in remaining:
        if not warehouse.in_bounds(step.position):
            return f"step {step} is out of bounds"
        if not warehouse.is_traversable(step.position):
            return f"cell {step.position} at t={step.timestep} is blocked by an obstacle"

    if reservations is not None:
        for step in remaining:
            reserver = reservations.vertex_reserver(step.position, step.timestep)
            if reserver is not None and reserver != rid:
                return (
                    f"cell {step.position} at t={step.timestep} is now reserved "
                    f"by '{reserver}'"
                )
        for earlier, later in zip(remaining, remaining[1:]):
            if earlier.position != later.position:
                if reservations.is_swap_conflict(
                    earlier.position, later.position, earlier.timestep, rid
                ):
                    return (
                        f"move {earlier.position}->{later.position} at "
                        f"t={earlier.timestep} now swaps with another robot"
                    )
    return None


def route_touches_cell(
    timed_path: TimedPath, cell: Position, from_timestep: int
) -> bool:
    """True if the robot's route occupies `cell` at or after `from_timestep`."""
    return any(
        s.position == cell and s.timestep >= from_timestep
        for s in timed_path.steps
    )


# ----------------------------------------------------------------------
# Splicing history + future
# ----------------------------------------------------------------------
def splice(
    executed_history: Sequence[TimedStep],
    replanned_future: TimedPath,
) -> TimedPath:
    """Join what a robot actually drove with its repaired future route.

    `executed_history` ends at `(current_cell, current_timestep)`.
    `replanned_future` starts at that same `(current_cell, current_timestep)`.
    The joint cell is not duplicated.
    """
    history = list(executed_history)
    future = list(replanned_future.steps)
    if history and future and history[-1] == future[0]:
        future = future[1:]
    return TimedPath.found(replanned_future.robot_id, history + future)


# ----------------------------------------------------------------------
# The replanner
# ----------------------------------------------------------------------
class DynamicReplanner:
    """Space-Time A* from a robot's current cell, around live reservations."""

    def __init__(self, warehouse: Warehouse, heuristic=manhattan_distance) -> None:
        self.warehouse = warehouse
        self._planner = SpaceTimePlanner(warehouse, heuristic)

    def replan(
        self,
        robot_id: str,
        current_position: Position,
        goal: Position,
        current_timestep: int,
        reservations: ReservationTable,
        horizon: int,
    ) -> TimedPath:
        """Plan a conflict-free timed route from `current_position` to `goal`,
        beginning at `current_timestep`.

        Returns a TimedPath whose first step is `(current_position,
        current_timestep)`. `success` is False, with a reason, when no
        conflict-free continuation fits the horizon.
        """
        return self._planner.plan(
            start=current_position,
            goal=goal,
            reservations=reservations,
            robot_id=robot_id,
            start_time=current_timestep,
            horizon=horizon,
        )

    def timed_replan(self, *args, **kwargs):
        """`replan`, but also returns the wall-clock milliseconds it took."""
        started = time.perf_counter_ns()
        result = self.replan(*args, **kwargs)
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
        return result, elapsed_ms
