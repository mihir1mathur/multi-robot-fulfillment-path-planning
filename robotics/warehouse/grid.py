"""The 2D grid: the lowest-level model of the warehouse floor.

COORDINATE CONVENTION (used everywhere in this project)
======================================================
A location is a `Position(row, col)`.

    * `row` grows DOWNWARD  -> row 0 is the TOP row of the printed map
    * `col` grows RIGHTWARD -> col 0 is the LEFT-most column

    col ->   0   1   2   3
    row 0 [ . ][ . ][ . ][ . ]
    row 1 [ . ][ # ][ . ][ . ]
    row 2 [ . ][ . ][ . ][ . ]

We deliberately do NOT use (x, y). Screen/matrix code and printed maps are
naturally row-major, and mixing (x, y) with (row, col) in the same codebase is
one of the most common sources of silent bugs. One convention, everywhere.

WHY A GRID AT ALL?
------------------
A real warehouse floor is continuous space. Reasoning about continuous space
requires geometry, robot shapes and motion models. By snapping the floor to
fixed-size square cells we get a *graph*: each cell is a node, each legal move
between neighbouring cells is an edge. Graph search algorithms (A*, Dijkstra -
future work) then apply directly. This is the standard first abstraction in
warehouse robotics, and it keeps this layer small enough to test exhaustively.

MOVEMENT MODEL
--------------
Four directions only: UP, DOWN, LEFT, RIGHT. No diagonals.
See `Direction` for why.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterator, List, Set

from robotics.exceptions import InvalidPositionError

# Cost of one legal orthogonal move, in "grid distance" units.
# Movement uses a uniform cost of 1 per move. The Dijkstra/A* planners added
# later will reuse this same constant so path costs and robot odometry agree.
MOVE_COST = 1.0


@dataclass(frozen=True, order=True)
class Position:
    """An immutable (row, col) cell coordinate.

    Frozen (immutable) on purpose: positions are used as dictionary keys and
    set members (for obstacles and occupancy). A mutable object would silently
    corrupt those lookups if someone changed it after insertion.
    """

    row: int
    col: int

    def manhattan_distance_to(self, other: "Position") -> int:
        """Number of orthogonal steps needed if nothing were in the way.

        Called "Manhattan" distance because, like walking city blocks, you may
        only travel along the grid lines - never diagonally through a building.
        This is the natural distance measure for 4-directional movement and
        will become the A* heuristic once path planning is added.
        """
        return abs(self.row - other.row) + abs(self.col - other.col)

    def is_adjacent_to(self, other: "Position") -> bool:
        """True if `other` is exactly one orthogonal step away.

        Note this is False for the same cell (distance 0) and False for
        diagonals (distance 2 under Manhattan distance).
        """
        return self.manhattan_distance_to(other) == 1

    def __str__(self) -> str:
        return f"({self.row}, {self.col})"


class Direction(Enum):
    """The four legal movement directions, stored as (row_delta, col_delta).

    WHY NO DIAGONALS IN THIS PROJECT?
    ---------------------------------
    1. Cost honesty: a diagonal step is sqrt(2) ~= 1.414 cells long, not 1.
       Allowing diagonals while charging 1 per move would make path costs lie.
    2. Corner cutting: a diagonal move can slip between two obstacle corners
       that a physically real robot could not fit through.
    3. Comparability: with uniform cost-1 edges, Dijkstra and A* are far
       easier to compare and to explain when they are added.
    """

    UP = (-1, 0)      # decreasing row = towards the top of the printed map
    DOWN = (1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)

    @property
    def row_delta(self) -> int:
        return self.value[0]

    @property
    def col_delta(self) -> int:
        return self.value[1]

    def apply_to(self, position: Position) -> Position:
        """Return the cell reached by stepping one cell in this direction.

        The result is NOT validated - it may be outside the grid. Bounds
        checking is the grid's job, not the direction's.
        """
        return Position(position.row + self.row_delta, position.col + self.col_delta)


class Grid:
    """A rectangular grid of cells, each either free or blocked.

    The grid knows only about geometry and traversability. It knows nothing
    about robots, tasks, shelves or charging stations - those live one layer
    up in `Warehouse`. Keeping this class tiny is what makes it trivially
    testable and reusable by the path planners added later.
    """

    def __init__(self, width: int, height: int) -> None:
        """Create an all-free grid.

        Args:
            width: number of COLUMNS (horizontal size).
            height: number of ROWS (vertical size).

        Raises:
            ValueError: if either dimension is not a positive integer.
        """
        if width <= 0 or height <= 0:
            raise ValueError(
                f"Grid dimensions must be positive, got width={width}, height={height}"
            )

        self.width = width
        self.height = height

        # Cells that cannot be driven over. The grid stores them as a plain
        # set of positions; it does not care *why* a cell is blocked. The
        # Warehouse layer decides whether a block came from a wall, a rack or
        # a temporary spill.
        self._blocked_cells: Set[Position] = set()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def in_bounds(self, position: Position) -> bool:
        """True if the coordinate lies inside the warehouse rectangle."""
        return 0 <= position.row < self.height and 0 <= position.col < self.width

    def is_blocked(self, position: Position) -> bool:
        """True if the cell is explicitly marked as not drivable.

        Out-of-bounds cells are reported as blocked as well: from a robot's
        point of view "there is no floor there" and "there is a wall there"
        have the same consequence.
        """
        if not self.in_bounds(position):
            return True
        return position in self._blocked_cells

    def is_traversable(self, position: Position) -> bool:
        """True if a robot is allowed to stand on this cell.

        "Traversable" == inside the grid AND not blocked. Robot-to-robot
        occupancy is deliberately NOT considered here: occupancy changes every
        step and is owned by the simulator, whereas the grid describes the
        static floor plan.
        """
        return self.in_bounds(position) and position not in self._blocked_cells

    def neighbors(self, position: Position) -> List[Position]:
        """Return the traversable cells reachable in one orthogonal step.

        This is the function a graph search calls to "expand" a node. Cells
        that are out of bounds or blocked are filtered out, so a planner never
        has to re-check them.

        Raises:
            InvalidPositionError: if `position` itself is outside the grid.
        """
        self._require_in_bounds(position)

        neighbours: List[Position] = []
        for direction in Direction:
            candidate = direction.apply_to(position)
            if self.is_traversable(candidate):
                neighbours.append(candidate)
        return neighbours

    def all_positions(self) -> Iterator[Position]:
        """Yield every cell in the grid, top-left to bottom-right."""
        for row in range(self.height):
            for col in range(self.width):
                yield Position(row, col)

    @property
    def blocked_cells(self) -> Set[Position]:
        """A copy of the blocked cells, so callers cannot mutate our state."""
        return set(self._blocked_cells)

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------
    def block(self, position: Position) -> None:
        """Mark a cell as not drivable. Blocking twice is harmless.

        Raises:
            InvalidPositionError: if the cell is outside the warehouse.
        """
        self._require_in_bounds(position)
        self._blocked_cells.add(position)

    def unblock(self, position: Position) -> None:
        """Mark a cell as drivable again. Unblocking a free cell is harmless.

        Raises:
            InvalidPositionError: if the cell is outside the warehouse.
        """
        self._require_in_bounds(position)
        self._blocked_cells.discard(position)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _require_in_bounds(self, position: Position) -> None:
        if not self.in_bounds(position):
            raise InvalidPositionError(
                f"Position {position} is outside the warehouse "
                f"(valid rows 0..{self.height - 1}, valid cols 0..{self.width - 1})"
            )

    def __repr__(self) -> str:
        return (
            f"Grid(width={self.width}, height={self.height}, "
            f"blocked={len(self._blocked_cells)})"
        )
