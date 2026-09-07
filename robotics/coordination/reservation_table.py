"""A space-time reservation table: who has claimed which cell/edge, and when.

THE ONE IDEA
------------
When a robot's timed path is fixed, every (cell, timestep) it occupies and
every edge it traverses is RESERVED. A later robot's planner then treats a
reserved (cell, timestep) exactly like a wall that only exists at that instant.

TWO KINDS OF RESERVATION
------------------------
VERTEX   (cell, t)          "a robot is standing on `cell` at timestep t"
EDGE     (from, to, t)      "a robot moves from `from` to `to` during the
                             transition between timestep t and t+1"

THE EDGE TIMESTEP CONVENTION (fixed, used everywhere)
----------------------------------------------------
A robot that is at cell A at time t and at cell B at time t+1 reserves the
edge as `(A, B, t)` - keyed by the EARLIER timestep.

    A SWAP (head-on) conflict is: robot X does A->B at t, robot Y does B->A at
    the same t. Y's move reserves `(B, A, t)`; we detect the clash by looking
    for the REVERSE edge `(A, B, t)` already in the table.

GOAL HOLD
---------
When a robot arrives at its goal at time T, it does not vanish - it parks
there. So the table also reserves the goal cell for every timestep from T
through the planning horizon, which stops a later robot from planning a route
straight through an occupied goal.

`conflict_checks` counts every reservation lookup, so the benchmark can report
how much conflict checking the coordination actually did.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from robotics.coordination.timed_path import TimedPath
from robotics.warehouse.grid import Position

VertexKey = Tuple[Position, int]
EdgeKey = Tuple[Position, Position, int]


class ReservationTable:
    """Deterministic hash-based store of vertex and edge reservations."""

    def __init__(self) -> None:
        self._vertices: Dict[VertexKey, str] = {}
        self._edges: Dict[EdgeKey, str] = {}
        self._query_count = 0

    # ------------------------------------------------------------------
    # Writing reservations
    # ------------------------------------------------------------------
    def reserve_vertex(self, cell: Position, timestep: int, robot_id: str) -> None:
        """Claim `cell` at `timestep` for `robot_id`.

        Re-reserving the same key for the SAME robot is harmless (a robot may
        reserve its own start cell twice, for instance). Re-reserving it for a
        DIFFERENT robot is a bug in the caller and raises.
        """
        key = (cell, timestep)
        existing = self._vertices.get(key)
        if existing is not None and existing != robot_id:
            raise ValueError(
                f"vertex {cell} at t={timestep} already reserved by "
                f"'{existing}', cannot give it to '{robot_id}'"
            )
        self._vertices[key] = robot_id

    def reserve_edge(
        self, from_cell: Position, to_cell: Position, timestep: int, robot_id: str
    ) -> None:
        """Claim the move `from_cell -> to_cell` during the t -> t+1 transition."""
        key = (from_cell, to_cell, timestep)
        existing = self._edges.get(key)
        if existing is not None and existing != robot_id:
            raise ValueError(
                f"edge {from_cell}->{to_cell} at t={timestep} already reserved "
                f"by '{existing}', cannot give it to '{robot_id}'"
            )
        self._edges[key] = robot_id

    def reserve_timed_path(self, timed_path: TimedPath, horizon: int) -> None:
        """Reserve every vertex and edge of a successful timed path, plus the
        goal hold from arrival through the horizon.

        Raises:
            ValueError: if the path is unsuccessful, or a reservation clashes
                with another robot (which would mean the planner produced a
                conflicting route - a bug worth failing loudly on).
        """
        if not timed_path.success:
            raise ValueError(
                f"cannot reserve an unsuccessful timed path for "
                f"'{timed_path.robot_id}'"
            )

        robot_id = timed_path.robot_id
        steps = timed_path.steps

        for step in steps:
            self.reserve_vertex(step.position, step.timestep, robot_id)

        for earlier, later in zip(steps, steps[1:]):
            if earlier.position != later.position:  # a move, not a WAIT
                self.reserve_edge(
                    earlier.position, later.position, earlier.timestep, robot_id
                )

        goal = steps[-1].position
        for timestep in range(timed_path.arrival_time + 1, horizon + 1):
            self.reserve_vertex(goal, timestep, robot_id)

    # ------------------------------------------------------------------
    # Reading reservations
    # ------------------------------------------------------------------
    def vertex_reserver(self, cell: Position, timestep: int) -> Optional[str]:
        """The robot holding `cell` at `timestep`, or None."""
        self._query_count += 1
        return self._vertices.get((cell, timestep))

    def edge_reserver(
        self, from_cell: Position, to_cell: Position, timestep: int
    ) -> Optional[str]:
        """The robot holding the exact edge `from_cell -> to_cell` at `timestep`."""
        self._query_count += 1
        return self._edges.get((from_cell, to_cell, timestep))

    def is_vertex_free(
        self, cell: Position, timestep: int, for_robot: Optional[str] = None
    ) -> bool:
        """True if `cell` at `timestep` is unreserved, or reserved by `for_robot`."""
        reserver = self.vertex_reserver(cell, timestep)
        return reserver is None or reserver == for_robot

    def is_swap_conflict(
        self, from_cell: Position, to_cell: Position, timestep: int,
        for_robot: Optional[str] = None,
    ) -> bool:
        """True if moving `from_cell -> to_cell` during t -> t+1 would swap with
        another robot moving the opposite way (`to_cell -> from_cell`) at the
        same time.
        """
        reverse_reserver = self.edge_reserver(to_cell, from_cell, timestep)
        return reverse_reserver is not None and reverse_reserver != for_robot

    def is_move_allowed(
        self, from_cell: Position, to_cell: Position, timestep: int, for_robot: str
    ) -> bool:
        """The combined check the planner uses for one candidate MOVE.

        Allowed only if:
          * the destination cell is free at t+1 (no vertex conflict), and
          * the move does not swap with an opposing robot (no edge conflict).
        WAITs (from_cell == to_cell) are checked with `is_vertex_free` alone.
        """
        if from_cell == to_cell:
            return self.is_vertex_free(to_cell, timestep + 1, for_robot)
        if not self.is_vertex_free(to_cell, timestep + 1, for_robot):
            return False
        if self.is_swap_conflict(from_cell, to_cell, timestep, for_robot):
            return False
        return True

    # ------------------------------------------------------------------
    # Releasing reservations (used by dynamic recovery)
    # ------------------------------------------------------------------
    def reservations_owned_by(self, robot_id: str) -> int:
        """How many vertex + edge reservations `robot_id` currently holds."""
        vertices = sum(1 for owner in self._vertices.values() if owner == robot_id)
        edges = sum(1 for owner in self._edges.values() if owner == robot_id)
        return vertices + edges

    def release_robot(self, robot_id: str) -> int:
        """Drop every reservation held by `robot_id`. Returns how many were dropped.

        Used when a robot's future route has been invalidated (a new obstacle,
        a robot failure): its stale future reservations must not linger, or
        another robot's replan would treat empty cells as blocked.
        """
        released = self.reservations_owned_by(robot_id)
        self._vertices = {
            key: owner for key, owner in self._vertices.items() if owner != robot_id
        }
        self._edges = {
            key: owner for key, owner in self._edges.items() if owner != robot_id
        }
        return released

    def block_cell(
        self, cell: Position, from_timestep: int, to_timestep: int, owner: str
    ) -> int:
        """Reserve `cell` for every timestep in [from_timestep, to_timestep].

        Used to pin a FAILED robot in place: it stopped where it is and is now
        a fixed obstacle for the rest of the coordination episode. Returns the
        number of vertex reservations added.
        """
        added = 0
        for timestep in range(from_timestep, to_timestep + 1):
            self.reserve_vertex(cell, timestep, owner)
            added += 1
        return added

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    @property
    def conflict_checks(self) -> int:
        """How many reservation lookups have been made against this table."""
        return self._query_count

    @property
    def vertex_reservation_count(self) -> int:
        return len(self._vertices)

    @property
    def edge_reservation_count(self) -> int:
        return len(self._edges)

    def __repr__(self) -> str:
        return (
            f"ReservationTable(vertices={len(self._vertices)}, "
            f"edges={len(self._edges)}, queries={self._query_count})"
        )
