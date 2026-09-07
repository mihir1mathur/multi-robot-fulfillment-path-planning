"""A* search, implemented from first principles.

A* ("A-star") is Dijkstra's algorithm plus one idea: a guess about which
direction the goal is in.

THE THREE LETTERS
-----------------
    g(n)  the ACTUAL cost of the best route found so far from the start to n.
          This is exactly what Dijkstra tracks.

    h(n)  the ESTIMATED remaining cost from n to the goal, computed instantly
          without searching. Here it is Manhattan distance.

    f(n) = g(n) + h(n)
          the estimated total cost of a route that goes from the start,
          through n, to the goal.

Dijkstra always expands the cell with the smallest g. A* always expands the
cell with the smallest f. That single change is the whole algorithm.

WHY THAT HELPS
--------------
Dijkstra has no idea where the goal is, so it spreads out evenly in every
direction - including directly away from the goal. A* prefers cells whose
total estimated journey is short, which naturally means cells that lie towards
the goal. It therefore tends to expand far fewer cells for the same answer.

    Dijkstra explores            A* explores
    a circle around S            a cone from S towards G

      . x x x . .                  . . . . . .
      x x x x x .                  . x x . . .
      x x S x x G                  . x S x x G
      x x x x x .                  . x x . . .
      . x x x . .                  . . . . . .

WHY IT IS STILL OPTIMAL
-----------------------
Because Manhattan distance never OVER-estimates the real remaining cost (it
ignores obstacles, which can only make the true route longer). A heuristic that
never over-estimates is called ADMISSIBLE, and an admissible heuristic
guarantees A* returns a genuinely shortest path.

If the heuristic lied upwards - claiming a cell was further from the goal than
it really is - A* could dismiss the true best route as expensive and return a
longer one instead.

AN HONEST CAVEAT
----------------
Expanding fewer cells does not automatically mean finishing sooner. A* does
strictly more work per cell than Dijkstra: it calls the heuristic and pushes a
larger tuple. In a small or wide-open warehouse that extra per-cell cost can
outweigh the cells it saves. The benchmark measures both numbers separately
for exactly this reason, and reports whatever actually happened.

THE PRIORITY QUEUE ORDERING
---------------------------
Each frontier entry is the 4-tuple

    (f, h, insertion_order, node)

and heapq compares those fields left to right:

    1. f = g + h   the real A* priority. Smallest estimated total journey wins.
    2. h           a TIE-BREAK, only consulted when two entries have equal f.
                   Among equally-promising cells, prefer the one the heuristic
                   thinks is closer to the goal. This pushes the search along
                   the goal-directed frontier instead of sideways along an
                   equal-f contour, so on an open grid A* expands a narrow
                   corridor of cells rather than a broad diamond.
    3. insertion_order  a strictly increasing counter, so ties in both f and h
                   fall back to "whichever was discovered first". This is what
                   makes the search fully deterministic without ever comparing
                   Position objects.
    4. node        never actually reached as a comparison key (3 already breaks
                   every tie); carried so the tuple holds the cell.

f stays the primary priority, so this is still A* and still optimal for an
admissible heuristic - the h tie-break only reorders cells that A* was going to
expand anyway, it never lets a worse-f cell overtake a better-f one.
"""

from __future__ import annotations

import heapq
import itertools
import time
from typing import Callable, Dict, List, Set

from robotics.planning.heuristics import manhattan_distance
from robotics.planning.path_result import PathResult, reconstruct_path
from robotics.planning.validation import check_planning_request
from robotics.warehouse.grid import MOVE_COST, Position
from robotics.warehouse.warehouse import Warehouse

ALGORITHM_NAME = "astar"

# A heuristic takes two cells and estimates the cost between them.
Heuristic = Callable[[Position, Position], float]


def plan(
    start: Position,
    goal: Position,
    warehouse: Warehouse,
    heuristic: Heuristic = manhattan_distance,
) -> PathResult:
    """Find a shortest path from `start` to `goal` using A*.

    Structurally this is deliberately identical to `dijkstra.plan` - same
    validation, same neighbour source, same stale-entry handling, same
    node-counting rule, same result type. The only difference is how the
    frontier is ordered: A* sorts by `f = g + h` (with `h` as a tie-break;
    see the module docstring), where Dijkstra sorts by `g` alone. Keeping
    everything else the same is what makes the benchmark a fair comparison
    rather than a comparison of two unrelated implementations.

    Args:
        start: the cell to plan from.
        goal: the cell to plan to.
        warehouse: the warehouse to plan across.
        heuristic: the remaining-distance estimate. Defaults to Manhattan
            distance, which is admissible for four-directional unit-cost
            movement. Injectable so tests can show that a zero heuristic makes
            A* behave like Dijkstra.

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
        return PathResult.found(
            algorithm=ALGORITHM_NAME,
            start=start,
            goal=goal,
            path=[start],
            nodes_expanded=0,
            planning_time_ms=elapsed_ms(),
        )

    # g_score[cell] = cheapest ACTUAL cost found so far from start to that cell.
    # This is the same map Dijkstra calls `best_cost`; the name g_score is the
    # conventional one for A* and matches the g/h/f vocabulary above.
    g_score: Dict[Position, float] = {start: 0.0}

    came_from: Dict[Position, Position] = {}
    finalized: Set[Position] = set()

    # A strictly increasing counter used as the third field of every queue
    # entry, so ties in both f and h fall back to insertion order and the
    # search never has to compare Position objects.
    tie_breaker = itertools.count()

    # Each entry is (f, h, insertion_order, node). heapq is a min-heap and
    # compares tuples left to right, so f is the primary priority, h breaks
    # ties in f, and insertion_order breaks ties in h. For the start cell g is
    # 0, so f == h. See the module docstring for why the h tie-break matters.
    start_h = heuristic(start, goal)
    frontier: List[tuple] = [(start_h, start_h, next(tie_breaker), start)]
    nodes_expanded = 0

    while frontier:
        _, _, _, current = heapq.heappop(frontier)

        # Stale entry: a better route to this cell was found after this entry
        # was pushed, so this one is obsolete. Skipping it keeps
        # `nodes_expanded` comparable with Dijkstra's count.
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

        current_cost = g_score[current]

        for neighbour in warehouse.neighbors(current):
            if neighbour in finalized:
                continue

            tentative_g = current_cost + MOVE_COST
            if tentative_g < g_score.get(neighbour, float("inf")):
                g_score[neighbour] = tentative_g
                came_from[neighbour] = current
                h = heuristic(neighbour, goal)
                priority = tentative_g + h
                # Push (f, h, insertion_order, node): f first so it stays the
                # real priority, then h to break equal-f ties towards the goal.
                heapq.heappush(frontier, (priority, h, next(tie_breaker), neighbour))

    return PathResult.not_found(
        algorithm=ALGORITHM_NAME,
        start=start,
        goal=goal,
        reason=f"No route exists from {start} to {goal}; the goal is unreachable",
        nodes_expanded=nodes_expanded,
        planning_time_ms=elapsed_ms(),
    )
