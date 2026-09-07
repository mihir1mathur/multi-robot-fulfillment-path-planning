"""Checking planning requests before searching, and checking paths afterwards.

TWO DIFFERENT JOBS IN ONE MODULE
--------------------------------
1. BEFORE the search: `check_planning_request` decides whether the question
   even makes sense. Both planners call it, so "start is on a wall" means the
   same thing and produces the same message whichever algorithm was asked.

2. AFTER the search: the path validator independently re-checks the answer.

WHY VALIDATE A PATH WE JUST COMPUTED?
-------------------------------------
Because "the planner returned a path" and "the path is actually walkable" are
different claims. A subtle bug - an off-by-one in reconstruction, a neighbour
function that forgets an obstacle, a cost that disagrees with the path - would
still produce a confident-looking list of cells.

The validator knows nothing about how the path was produced. It just walks the
cells and asks: does this start where I asked, end where I asked, stay inside
the warehouse, avoid every obstacle, move one orthogonal step at a time, and
cost what the planner claimed? That independence is the point. It is why the
tests can assert that EVERY successful path from EITHER planner passes.
"""

from __future__ import annotations

from typing import List, Optional

from robotics.exceptions import InvalidPositionError, PathValidationError
from robotics.planning.path_result import PathResult
from robotics.warehouse.grid import MOVE_COST
from robotics.warehouse.grid import Position
from robotics.warehouse.warehouse import Warehouse


def check_planning_request(
    start: Position, goal: Position, warehouse: Warehouse
) -> Optional[str]:
    """Decide whether a planning request can possibly succeed.

    Returns:
        None if the request is worth searching, otherwise a human-readable
        reason why no path can exist.

    Raises:
        InvalidPositionError: if start or goal lies outside the warehouse.

    WHY OUT-OF-BOUNDS RAISES BUT A BLOCKED CELL DOES NOT
    ----------------------------------------------------
    They are different kinds of problem, so they get different treatment - and
    this matches how the existing warehouse foundation already behaves.

      * A coordinate outside the warehouse is a PROGRAMMING ERROR. Cell (99,99)
        of a 12x12 warehouse does not exist and never will. The foundation
        already raises `InvalidPositionError` for commands about cells that do
        not exist, so the planner does too, loudly.

      * A start or goal on a BLOCKED cell is a legitimate runtime OUTCOME, not
        a bug. A dynamic obstacle can appear on a drop-off point at any moment,
        and the correct answer to "route me there" is then simply "there is no
        path, because the goal is blocked". Returning that as an unsuccessful
        result lets a caller handle it without exception handling.
    """
    if not warehouse.in_bounds(start):
        raise InvalidPositionError(
            f"Planning start {start} is outside warehouse '{warehouse.name}' "
            f"({warehouse.height} rows x {warehouse.width} cols)"
        )
    if not warehouse.in_bounds(goal):
        raise InvalidPositionError(
            f"Planning goal {goal} is outside warehouse '{warehouse.name}' "
            f"({warehouse.height} rows x {warehouse.width} cols)"
        )

    if not warehouse.is_traversable(start):
        return f"Start {start} is blocked by an obstacle"
    if not warehouse.is_traversable(goal):
        return f"Goal {goal} is blocked by an obstacle"

    return None


def find_path_problems(result: PathResult, warehouse: Warehouse) -> List[str]:
    """Independently check a planner's answer. Empty list means it is valid.

    Returning a LIST rather than a single bool or a raised error is deliberate:
    when something is wrong it is far more useful to see every problem at once
    ("cell 4 is inside a rack" AND "reported cost 9 but the path needs 11
    moves") than to fix them one at a time.

    Checks performed on a SUCCESSFUL result:
        1. the path is not empty
        2. the first cell is the requested start
        3. the last cell is the requested goal
        4. every cell is inside the warehouse
        5. every cell is traversable in the current warehouse snapshot
        6. every consecutive pair is exactly one orthogonal step apart
           (this is also what rules out illegal jumps and diagonals)
        7. no cell is visited twice
        8. the reported total_cost equals the moves the path actually needs

    Checks performed on an UNSUCCESSFUL result:
        the path is empty, no cost is reported, and a reason is given.
    """
    problems: List[str] = []

    if not result.success:
        if result.path:
            problems.append(
                f"unsuccessful result should carry an empty path, "
                f"got {len(result.path)} cells"
            )
        if result.total_cost is not None:
            problems.append(
                f"unsuccessful result should have total_cost None, "
                f"got {result.total_cost}"
            )
        if not result.failure_reason:
            problems.append("unsuccessful result should explain why it failed")
        return problems

    path = result.path

    if not path:
        problems.append("successful result has an empty path")
        return problems  # nothing else can be checked

    if path[0] != result.start:
        problems.append(f"path starts at {path[0]}, expected {result.start}")
    if path[-1] != result.goal:
        problems.append(f"path ends at {path[-1]}, expected {result.goal}")

    for index, cell in enumerate(path):
        if not warehouse.in_bounds(cell):
            problems.append(f"cell {index} {cell} is outside the warehouse")
        elif not warehouse.is_traversable(cell):
            problems.append(f"cell {index} {cell} is blocked by an obstacle")

    for index in range(len(path) - 1):
        current, following = path[index], path[index + 1]
        if not current.is_adjacent_to(following):
            problems.append(
                f"illegal move from cell {index} {current} to cell "
                f"{index + 1} {following}: not one orthogonal step apart"
            )

    if len(set(path)) != len(path):
        problems.append("path visits the same cell more than once")

    expected_cost = (len(path) - 1) * MOVE_COST
    if result.total_cost != expected_cost:
        problems.append(
            f"reported total_cost {result.total_cost} does not match the "
            f"{len(path) - 1} moves in the path (expected {expected_cost})"
        )

    return problems


def is_valid_path(result: PathResult, warehouse: Warehouse) -> bool:
    """True if the result passes every check in `find_path_problems`."""
    return not find_path_problems(result, warehouse)


def assert_valid_path(result: PathResult, warehouse: Warehouse) -> None:
    """Raise if the result is not valid. Used where a bug must not pass silently.

    Raises:
        PathValidationError: listing every problem found.
    """
    problems = find_path_problems(result, warehouse)
    if problems:
        joined = "; ".join(problems)
        raise PathValidationError(
            f"{result.algorithm} produced an invalid result for "
            f"{result.start} -> {result.goal}: {joined}"
        )
