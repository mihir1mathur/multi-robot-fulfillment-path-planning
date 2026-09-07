"""Single-robot shortest-path planning over the warehouse grid.

WHAT THIS PACKAGE DOES
----------------------
Given a start cell and a goal cell, it returns the sequence of cells a robot
should drive through to get there in the fewest moves, going around every
obstacle. Two algorithms are provided:

    dijkstra.plan(start, goal, warehouse)   the baseline: searches outwards in
                                            every direction equally
    astar.plan(start, goal, warehouse)      the same search, guided towards the
                                            goal by a Manhattan-distance guess

Both return a `PathResult` and both are optimal - they find a genuinely
shortest path, or report cleanly that no path exists.

THE PLANNER "INTERFACE"
-----------------------
Both planners share one call signature:

    plan(start, goal, warehouse) -> PathResult

That plain function signature IS the interface. There is deliberately no base
class and no inheritance hierarchy: nothing would be shared by inheriting, and
a class per algorithm would only add ceremony around what is really a single
function. `PLANNERS` below maps a name to each function, which is all the
benchmark and the demo need in order to treat them interchangeably.

SCOPE - WHAT THIS PACKAGE DOES NOT DO
-------------------------------------
* It plans for ONE robot at a time. This is not multi-agent path finding.
* It does NOT consider where other robots are standing. That is a deliberate
  decision, not an oversight: robots move, so a route planned around a snapshot
  of their positions would be stale the moment any of them takes a step, and
  "avoid the cell R2 is in right now" is not the same problem as "make sure R1
  and R2 never want the same cell at the same time". Reserving cells over time
  is a separate piece of work. Until then, the simulator's existing move
  validation still refuses any step onto an occupied cell at execution time, so
  a stale plan cannot cause two robots to overlap - the move is simply refused.
* It plans against the warehouse AS IT IS AT THE MOMENT OF THE CALL. If an
  obstacle appears afterwards, the returned path is not updated. Automatic
  replanning during execution is not implemented.
"""

from typing import Callable, Dict

from robotics.planning import astar, dijkstra
from robotics.planning.astar import plan as plan_astar
from robotics.planning.dijkstra import plan as plan_dijkstra
from robotics.planning.heuristics import manhattan_distance, zero_heuristic
from robotics.planning.path_result import PathResult, reconstruct_path
from robotics.planning.validation import (
    assert_valid_path,
    check_planning_request,
    find_path_problems,
    is_valid_path,
)
from robotics.warehouse.grid import Position
from robotics.warehouse.warehouse import Warehouse

# The shared shape every planner satisfies.
PlannerFunction = Callable[[Position, Position, Warehouse], PathResult]

# Name -> planner, so callers can loop over both without naming them one by one.
# Insertion order puts the baseline first, which is the order the demo and the
# benchmark report them in.
PLANNERS: Dict[str, PlannerFunction] = {
    dijkstra.ALGORITHM_NAME: plan_dijkstra,
    astar.ALGORITHM_NAME: plan_astar,
}

__all__ = [
    "astar",
    "dijkstra",
    "plan_astar",
    "plan_dijkstra",
    "PLANNERS",
    "PlannerFunction",
    "PathResult",
    "reconstruct_path",
    "manhattan_distance",
    "zero_heuristic",
    "check_planning_request",
    "find_path_problems",
    "is_valid_path",
    "assert_valid_path",
]
