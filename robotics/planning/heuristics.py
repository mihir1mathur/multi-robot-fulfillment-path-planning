"""Heuristics: cheap guesses of the remaining distance to the goal.

WHAT IS A HEURISTIC?
--------------------
A heuristic is an estimate. In path planning, h(n) estimates how far cell `n`
still is from the goal, WITHOUT doing any searching. It is what lets A* prefer
cells that look promising instead of spreading out blindly in all directions.

The estimate must be cheap: it is computed for every cell the search touches,
so anything expensive would cost more than the search it saves.
"""

from __future__ import annotations

from robotics.warehouse.grid import Position


def manhattan_distance(a: Position, b: Position) -> float:
    """Estimate the number of moves between two cells, ignoring obstacles.

        manhattan(a, b) = |a.row - b.row| + |a.col - b.col|

    WHY IT IS CALLED "MANHATTAN" DISTANCE
    -------------------------------------
    Manhattan's streets form a grid. To walk from one corner to another you
    cannot cut diagonally through the buildings - you go so many blocks across
    and so many blocks up. That total number of blocks is this distance.

    WHY IT IS THE RIGHT HEURISTIC FOR THIS PROJECT
    ----------------------------------------------
    It matches the movement rules exactly:

      * robots move only UP, DOWN, LEFT and RIGHT,
      * every orthogonal step costs exactly the same (MOVE_COST = 1),
      * diagonal movement is not allowed.

    So to get from a cell 3 rows and 2 columns away from the goal, a robot must
    make at least 3 + 2 = 5 moves. It can never do it in fewer.

    WHY THAT MATTERS: ADMISSIBILITY
    -------------------------------
    A heuristic is ADMISSIBLE if it never OVER-estimates the true remaining
    cost. Manhattan distance is admissible here because obstacles can only
    force a robot to take a LONGER route, never a shorter one - so the true
    remaining cost is always at least the Manhattan distance.

    Admissibility is what guarantees A* still returns a genuinely shortest
    path. An over-estimating heuristic could make A* skip the real best route
    because it wrongly looked expensive.

    (This heuristic is also CONSISTENT: moving to a neighbour changes the
    estimate by at most 1, which is exactly the cost of that move. Consistency
    is what makes it safe to finalise a cell the first time it is popped, which
    is what both planners do.)

    IMPLEMENTATION NOTE
    -------------------
    The arithmetic itself lives on `Position.manhattan_distance_to`, which the
    warehouse foundation already provides and tests. Re-typing it here would
    create a second place to get it wrong. This function exists as a separate,
    independently testable planner-level entry point, and converts to float so
    that heuristic values and path costs share one numeric type.
    """
    return float(a.manhattan_distance_to(b))


def zero_heuristic(a: Position, b: Position) -> float:
    """A heuristic that always guesses zero. Never used by the planners.

    It exists because it makes a genuinely useful point testable: A* with a
    zero heuristic has no idea which direction the goal is in, so it behaves
    exactly like Dijkstra. Running A* with this heuristic and comparing it
    against the real Dijkstra implementation demonstrates that the two
    algorithms differ only in the priority they sort the frontier by.

    It is trivially admissible - zero never over-estimates anything - which is
    also why Dijkstra always returns an optimal path.
    """
    return 0.0
