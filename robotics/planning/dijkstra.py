"""Dijkstra's shortest-path algorithm, implemented from first principles.

THE IDEA IN ONE PARAGRAPH
-------------------------
Start at the start cell with a cost of 0. Repeatedly take the cheapest cell we
have found so far but not yet finalised, mark it finalised, and look at its
neighbours: if going through this cell is cheaper than any route we already
knew to a neighbour, record the better cost and remember that we arrived from
here. Keep going until we finalise the goal.

Picture it as a wave spreading outwards from the start at a constant speed. The
wave reaches nearer cells before further ones, so the first time it touches the
goal, it has arrived by the shortest possible route.

    step 1        step 2        step 3
    . . . . .     . . . . .     . . 3 . .
    . . . . .     . . 1 . .     . 2 1 2 .
    . . S . .     . 1 S 1 .     2 1 S 1 2
    . . . . .     . . 1 . .     . 2 1 2 .
    . . . . .     . . . . .     . . 3 . .

WHY IT IS GUARANTEED TO FIND THE SHORTEST PATH HERE
---------------------------------------------------
Because every move costs the same positive amount (MOVE_COST = 1) and no cost
is ever negative. When a cell is popped as the cheapest remaining option, no
route still waiting in the queue could ever reach it more cheaply - every one
of those routes already costs at least as much, and going further can only add
cost. So the cost it was popped with is final.

WHAT IT DOES NOT KNOW
---------------------
Dijkstra has no idea where the goal is. It searches every direction equally.
That is exactly what A* improves on, and it is why this module exists: it is
the honest baseline that A* is measured against.
"""

from __future__ import annotations

import heapq
import itertools
import time
from typing import Dict, List, Set

from robotics.planning.path_result import PathResult, reconstruct_path
from robotics.planning.validation import check_planning_request
from robotics.warehouse.grid import MOVE_COST, Position
from robotics.warehouse.warehouse import Warehouse

ALGORITHM_NAME = "dijkstra"


def plan(start: Position, goal: Position, warehouse: Warehouse) -> PathResult:
    """Find a shortest path from `start` to `goal` around the obstacles.

    The search uses `warehouse.neighbors()`, which already filters out cells
    that are out of bounds, blocked by a static obstacle, or blocked by a
    currently active dynamic obstacle. Re-implementing that logic here would
    risk the planner and the simulator disagreeing about what is walkable.

    This plans against the warehouse AS IT IS RIGHT NOW. If an obstacle appears
    after the path is returned, the path is not automatically updated -
    replanning during execution is not implemented.

    Robot positions are deliberately NOT considered. See the package docstring.

    Args:
        start: the cell to plan from.
        goal: the cell to plan to.
        warehouse: the warehouse to plan across.

    Returns:
        A `PathResult`. On success, `path` runs from start to goal inclusive
        and `total_cost` is the number of moves.

    Raises:
        InvalidPositionError: if start or goal is outside the warehouse.
    """
    started_at_ns = time.perf_counter_ns()

    def elapsed_ms() -> float:
        return (time.perf_counter_ns() - started_at_ns) / 1_000_000

    rejection = check_planning_request(start, goal, warehouse)
    if rejection is not None:
        return PathResult.not_found(
            algorithm=ALGORITHM_NAME,
            start=start,
            goal=goal,
            reason=rejection,
            planning_time_ms=elapsed_ms(),
        )

    if start == goal:
        # Nothing to search: the robot is already there. Cost 0, and zero nodes
        # expanded because the priority queue is never touched.
        return PathResult.found(
            algorithm=ALGORITHM_NAME,
            start=start,
            goal=goal,
            path=[start],
            nodes_expanded=0,
            planning_time_ms=elapsed_ms(),
        )

    # best_cost[cell] = cheapest cost found SO FAR from start to that cell.
    best_cost: Dict[Position, float] = {start: 0.0}

    # came_from[cell] = the cell we arrived from on that cheapest route.
    came_from: Dict[Position, Position] = {}

    # Cells whose cost is final. Once a cell is here it is never improved
    # again, which is what lets us skip stale queue entries safely.
    finalized: Set[Position] = set()

    # A tie-breaking counter. heapq compares tuples element by element, so when
    # two entries have the same cost it would go on to compare the Positions.
    # An always-increasing counter breaks ties by insertion order instead,
    # which is the textbook approach and keeps the search deterministic without
    # depending on how Position happens to be ordered.
    tie_breaker = itertools.count()

    frontier: List[tuple] = [(0.0, next(tie_breaker), start)]
    nodes_expanded = 0

    while frontier:
        _, _, current = heapq.heappop(frontier)

        # STALE ENTRY. We push a new entry whenever we find a better route to a
        # cell, and heapq cannot remove the old one. So the queue can hold
        # several entries for the same cell, and all but the cheapest are
        # obsolete by the time they surface. Skipping them here is what keeps
        # `nodes_expanded` an honest count of real expansions.
        if current in finalized:
            continue

        finalized.add(current)
        nodes_expanded += 1

        if current == goal:
            return PathResult.found(
                algorithm=ALGORITHM_NAME,
                start=start,
                goal=goal,
                path=reconstruct_path(came_from, start, goal),
                nodes_expanded=nodes_expanded,
                planning_time_ms=elapsed_ms(),
            )

        current_cost = best_cost[current]

        for neighbour in warehouse.neighbors(current):
            if neighbour in finalized:
                continue

            # "Relaxation": ask whether going through `current` beats the best
            # route to `neighbour` we already knew about.
            new_cost = current_cost + MOVE_COST
            if new_cost < best_cost.get(neighbour, float("inf")):
                best_cost[neighbour] = new_cost
                came_from[neighbour] = current
                heapq.heappush(frontier, (new_cost, next(tie_breaker), neighbour))

    # The frontier emptied without reaching the goal, so every cell reachable
    # from the start has been finalised and the goal was not among them.
    return PathResult.not_found(
        algorithm=ALGORITHM_NAME,
        start=start,
        goal=goal,
        reason=f"No route exists from {start} to {goal}; the goal is unreachable",
        nodes_expanded=nodes_expanded,
        planning_time_ms=elapsed_ms(),
    )
