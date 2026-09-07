"""The result of one planning request, plus path reconstruction.

WHY A SHARED RESULT TYPE?
-------------------------
Dijkstra and A* answer the same question, so they must answer it in the same
shape. If each planner returned something slightly different - one a list of
cells, the other a tuple with a cost - then every caller (the demo, the tests,
the benchmark) would need to know which planner it was talking to, and a fair
comparison between the two would be impossible.

One `PathResult` for both planners means:
  * benchmarking never has to parse console text,
  * the path validator works on either planner's output unchanged,
  * a cross-algorithm test can compare costs directly.

DEFINITIONS FIXED HERE, USED EVERYWHERE
---------------------------------------
PATH
    Includes BOTH the start cell and the goal cell.
    A path from (0,0) to (0,2) is [(0,0), (0,1), (0,2)] - three cells.

TOTAL COST
    The number of orthogonal moves, i.e. `(len(path) - 1) * MOVE_COST`.
    Three cells means two moves, so the cost is 2.0.
    `None` when no path was found - deliberately not 0.0, because 0.0 is the
    genuine cost of the start == goal case and the two must not be confused.

NODES EXPANDED
    How many cells were removed from the priority queue and finalised. See
    `PathResult.nodes_expanded` for the exact rule, which is identical for
    both planners - that is what makes the comparison meaningful.

PLANNING TIME
    Wall-clock milliseconds for the planner call, measured with
    `time.perf_counter_ns()`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from robotics.warehouse.grid import MOVE_COST, Position


@dataclass
class PathResult:
    """What a planner returns, whether or not it found a path.

    Attributes:
        algorithm: which planner produced this, e.g. "dijkstra" or "astar".
        start: the requested start cell.
        goal: the requested goal cell.
        success: True if a path from start to goal was found.
        path: the cells from start to goal inclusive. Empty when unsuccessful.
        total_cost: number of moves, or None when unsuccessful.
        nodes_expanded: cells removed from the priority queue and finalised.
            A cell is counted exactly once, when it is popped with its
            best-known cost and is not a stale duplicate entry. The goal counts
            as expanded when it is popped, even though the search returns
            immediately instead of examining its neighbours - both planners
            apply that same rule, so the numbers stay comparable.
        planning_time_ms: wall-clock duration of the planner call.
        failure_reason: why no path was found, or None on success.
    """

    algorithm: str
    start: Position
    goal: Position
    success: bool
    path: List[Position] = field(default_factory=list)
    total_cost: Optional[float] = None
    nodes_expanded: int = 0
    planning_time_ms: float = 0.0
    failure_reason: Optional[str] = None

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def found(
        cls,
        algorithm: str,
        start: Position,
        goal: Position,
        path: List[Position],
        nodes_expanded: int,
        planning_time_ms: float,
    ) -> "PathResult":
        """Build a successful result, deriving the cost from the path.

        The cost is DERIVED rather than passed in on purpose: it removes any
        chance of a planner reporting a cost that disagrees with the path it
        returned, which would quietly corrupt every benchmark built on it.
        """
        return cls(
            algorithm=algorithm,
            start=start,
            goal=goal,
            success=True,
            path=path,
            total_cost=(len(path) - 1) * MOVE_COST,
            nodes_expanded=nodes_expanded,
            planning_time_ms=planning_time_ms,
        )

    @classmethod
    def not_found(
        cls,
        algorithm: str,
        start: Position,
        goal: Position,
        reason: str,
        nodes_expanded: int = 0,
        planning_time_ms: float = 0.0,
    ) -> "PathResult":
        """Build an unsuccessful result carrying the reason no path exists."""
        return cls(
            algorithm=algorithm,
            start=start,
            goal=goal,
            success=False,
            path=[],
            total_cost=None,
            nodes_expanded=nodes_expanded,
            planning_time_ms=planning_time_ms,
            failure_reason=reason,
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    @property
    def cells_visited(self) -> int:
        """How many cells the returned path contains (moves + 1)."""
        return len(self.path)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """A JSON-safe snapshot, used by the benchmark writer."""
        return {
            "algorithm": self.algorithm,
            "start": [self.start.row, self.start.col],
            "goal": [self.goal.row, self.goal.col],
            "success": self.success,
            "total_cost": self.total_cost,
            "nodes_expanded": self.nodes_expanded,
            "planning_time_ms": round(self.planning_time_ms, 6),
            "path_cells": len(self.path),
            "failure_reason": self.failure_reason,
        }

    def __str__(self) -> str:
        if not self.success:
            return (
                f"{self.algorithm}: NO PATH {self.start} -> {self.goal} "
                f"({self.failure_reason})"
            )
        return (
            f"{self.algorithm}: {self.start} -> {self.goal} | "
            f"cost {self.total_cost:g} | {self.nodes_expanded} nodes expanded | "
            f"{self.planning_time_ms:.3f} ms"
        )


def reconstruct_path(
    came_from: Dict[Position, Position], start: Position, goal: Position
) -> List[Position]:
    """Turn a "who did I come from?" map into the actual path.

    WHY PLANNERS STORE PREDECESSORS INSTEAD OF WHOLE PATHS
    -----------------------------------------------------
    A search could push a complete path into the priority queue with every
    entry, but that would copy a growing list thousands of times and use far
    more memory. Instead each cell records only the ONE cell it was best
    reached from:

        came_from = {B: A, C: B, D: C}

    To recover the route we start at the goal and walk backwards until we
    reach the start:

        D -> C -> B -> A

    then reverse it:

        A -> B -> C -> D

    Args:
        came_from: maps each discovered cell to the cell it was reached from.
        start: the cell the search began at.
        goal: the cell the search reached.

    Returns:
        The cells from start to goal inclusive.
    """
    path: List[Position] = [goal]
    current = goal

    # The loop terminates because every entry in `came_from` was written when a
    # cell was first reached from a strictly cheaper predecessor, so following
    # the links always moves closer to the start and can never cycle.
    while current != start:
        current = came_from[current]
        path.append(current)

    path.reverse()
    return path
