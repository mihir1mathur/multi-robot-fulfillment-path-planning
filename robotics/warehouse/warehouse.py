"""The Warehouse: a grid plus the meaning of each cell.

The `Grid` answers "can a robot stand here?". The `Warehouse` answers
"what is here, and why?" - is this cell a shelf, a pickup point, a packing
station, a charger, or a blocked rack?

MODELLING ASSUMPTIONS - stated explicitly so they can be defended
=================================================================
1. A cell has at most ONE role (aisle, storage, pickup, dropoff, charging).
   Real warehouses can overlap roles; a single role per cell keeps the map
   printable and the tests unambiguous.

2. Cells with a role are DRIVABLE. A robot must be able to stand on a pickup
   point to pick from it, and on a charger to charge. Physical shelving that a
   robot cannot enter is modelled as a STATIC OBSTACLE, not as a storage cell.
   So "storage cell" here means "the drivable cell from which stored goods are
   handled", not "the solid rack itself".

3. Static obstacles are pushed down into the grid's blocked set, because they
   never change. Dynamic obstacles are tracked separately by the warehouse,
   because replanning logic needs to ask two different questions:
       - "is this cell permanently blocked?"  (plan once, reuse forever)
       - "is this cell blocked right now?"    (may force a replan)

4. There is no notion of time yet. A dynamic obstacle is either present or
   absent; it does not have a schedule.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional, Set

from robotics.exceptions import InvalidPositionError, SimulationError
from robotics.warehouse.grid import Grid, Position
from robotics.warehouse.obstacle import Obstacle, ObstacleType


class LocationType(Enum):
    """The role a warehouse cell plays.

    AISLE
        Plain drivable floor. The default for every cell. Robots use aisles to
        travel between the interesting locations.
    STORAGE
        A shelf / storage position: where inventory lives and is handled from.
    PICKUP
        Where a robot collects an item at the start of a fulfillment task.
    DROPOFF
        Packing / despatch station: where a robot delivers the item.
    CHARGING
        A charging station. A robot may only recharge while standing here.
    """

    AISLE = "aisle"
    STORAGE = "storage"
    PICKUP = "pickup"
    DROPOFF = "dropoff"
    CHARGING = "charging"


# Roles that describe a purpose-built cell. A static obstacle may not be placed
# on one of these, because it would make part of the warehouse unusable in a
# way that is almost always a configuration mistake rather than an intention.
_FUNCTIONAL_LOCATIONS = frozenset(
    {
        LocationType.STORAGE,
        LocationType.PICKUP,
        LocationType.DROPOFF,
        LocationType.CHARGING,
    }
)


class Warehouse:
    """Owns the floor plan: the grid, the named locations and the obstacles."""

    def __init__(self, width: int, height: int, name: str = "warehouse") -> None:
        """Create an empty warehouse: all cells are free AISLE cells.

        Args:
            width: number of columns.
            height: number of rows.
            name: label used in logs and demo output.
        """
        self.name = name
        self.grid = Grid(width=width, height=height)

        # Only non-AISLE cells are stored. Anything absent from this map is an
        # ordinary aisle, which keeps memory proportional to the interesting
        # parts of the warehouse rather than to its area.
        self._locations: Dict[Position, LocationType] = {}

        # Obstacles are keyed by ID so they can be removed by name later.
        self._static_obstacles: Dict[str, Obstacle] = {}
        self._dynamic_obstacles: Dict[str, Obstacle] = {}

    # ------------------------------------------------------------------
    # Basic geometry (delegated to the grid)
    # ------------------------------------------------------------------
    @property
    def width(self) -> int:
        return self.grid.width

    @property
    def height(self) -> int:
        return self.grid.height

    def in_bounds(self, position: Position) -> bool:
        """True if the coordinate is inside the warehouse rectangle."""
        return self.grid.in_bounds(position)

    # ------------------------------------------------------------------
    # Named locations
    # ------------------------------------------------------------------
    def mark_location(self, position: Position, location_type: LocationType) -> None:
        """Give a cell a role (storage, pickup, dropoff, charging, aisle).

        Raises:
            InvalidPositionError: if the cell is outside the warehouse, or if a
                functional role is requested for a cell already blocked by a
                static obstacle (a robot could never reach it).
        """
        self._require_in_bounds(position)

        if location_type in _FUNCTIONAL_LOCATIONS and self.is_blocked_by_static(position):
            raise InvalidPositionError(
                f"Cannot mark {position} as {location_type.value}: the cell is "
                f"blocked by a static obstacle and no robot could reach it"
            )

        if location_type is LocationType.AISLE:
            # AISLE is the default, so "marking" a cell as aisle simply removes
            # any special role it had.
            self._locations.pop(position, None)
        else:
            self._locations[position] = location_type

    def get_location_type(self, position: Position) -> LocationType:
        """Return the role of a cell (AISLE if it has no special role).

        Raises:
            InvalidPositionError: if the cell is outside the warehouse.
        """
        self._require_in_bounds(position)
        return self._locations.get(position, LocationType.AISLE)

    def positions_of_type(self, location_type: LocationType) -> List[Position]:
        """All cells with the given role, in a stable (sorted) order.

        Sorted output matters: tests and printed reports must not change from
        run to run just because a set happened to iterate differently.
        """
        if location_type is LocationType.AISLE:
            return sorted(
                position
                for position in self.grid.all_positions()
                if position not in self._locations
            )
        return sorted(
            position
            for position, role in self._locations.items()
            if role is location_type
        )

    @property
    def storage_cells(self) -> List[Position]:
        return self.positions_of_type(LocationType.STORAGE)

    @property
    def pickup_locations(self) -> List[Position]:
        return self.positions_of_type(LocationType.PICKUP)

    @property
    def dropoff_locations(self) -> List[Position]:
        return self.positions_of_type(LocationType.DROPOFF)

    @property
    def charging_stations(self) -> List[Position]:
        return self.positions_of_type(LocationType.CHARGING)

    def is_charging_station(self, position: Position) -> bool:
        """True if a robot standing here is allowed to recharge."""
        return self.get_location_type(position) is LocationType.CHARGING

    # ------------------------------------------------------------------
    # Static obstacles
    # ------------------------------------------------------------------
    def add_static_obstacle(
        self, obstacle_id: str, position: Position, description: str = ""
    ) -> Obstacle:
        """Permanently block a cell (wall, pillar, solid rack).

        Raises:
            InvalidPositionError: cell out of bounds, or already carries a
                functional role such as a charging station.
            SimulationError: the obstacle ID is already in use.
        """
        self._require_in_bounds(position)
        self._require_unused_obstacle_id(obstacle_id)

        role = self.get_location_type(position)
        if role in _FUNCTIONAL_LOCATIONS:
            raise InvalidPositionError(
                f"Cannot place static obstacle '{obstacle_id}' on {position}: "
                f"the cell is a {role.value} location"
            )

        obstacle = Obstacle(
            obstacle_id=obstacle_id,
            position=position,
            obstacle_type=ObstacleType.STATIC,
            description=description,
        )
        self._static_obstacles[obstacle_id] = obstacle
        # Static obstacles never change, so they are pushed into the grid
        # itself: any planner that only ever sees the grid still avoids them.
        self.grid.block(position)
        return obstacle

    def remove_static_obstacle(self, obstacle_id: str) -> Obstacle:
        """Remove a permanent obstacle (e.g. the racking was torn out).

        Raises:
            SimulationError: no static obstacle with that ID exists.
        """
        if obstacle_id not in self._static_obstacles:
            raise SimulationError(f"No static obstacle with id '{obstacle_id}'")

        obstacle = self._static_obstacles.pop(obstacle_id)
        self.grid.unblock(obstacle.position)
        return obstacle

    def is_blocked_by_static(self, position: Position) -> bool:
        """True if a permanent obstacle sits on this cell."""
        return any(
            obstacle.position == position for obstacle in self._static_obstacles.values()
        )

    @property
    def static_obstacles(self) -> List[Obstacle]:
        return sorted(self._static_obstacles.values(), key=lambda o: o.obstacle_id)

    # ------------------------------------------------------------------
    # Dynamic obstacles
    # ------------------------------------------------------------------
    def add_dynamic_obstacle(
        self, obstacle_id: str, position: Position, description: str = ""
    ) -> Obstacle:
        """Temporarily block a cell (spill, cart, cordon).

        Unlike static obstacles, a dynamic obstacle MAY be placed on a
        functional cell - a box really can be dropped in front of a charger,
        and the replanning logic added later must handle exactly that.

        Raises:
            InvalidPositionError: cell out of bounds, or already permanently
                blocked (blocking a wall is meaningless).
            SimulationError: the obstacle ID is already in use.
        """
        self._require_in_bounds(position)
        self._require_unused_obstacle_id(obstacle_id)

        if self.is_blocked_by_static(position):
            raise InvalidPositionError(
                f"Cannot place dynamic obstacle '{obstacle_id}' on {position}: "
                f"the cell is already permanently blocked"
            )

        obstacle = Obstacle(
            obstacle_id=obstacle_id,
            position=position,
            obstacle_type=ObstacleType.DYNAMIC,
            description=description,
        )
        self._dynamic_obstacles[obstacle_id] = obstacle
        return obstacle

    def remove_dynamic_obstacle(self, obstacle_id: str) -> Obstacle:
        """Clear a temporary obstacle (the spill was mopped up).

        Raises:
            SimulationError: no dynamic obstacle with that ID exists.
        """
        if obstacle_id not in self._dynamic_obstacles:
            raise SimulationError(f"No dynamic obstacle with id '{obstacle_id}'")
        return self._dynamic_obstacles.pop(obstacle_id)

    def is_blocked_by_dynamic(self, position: Position) -> bool:
        """True if a temporary obstacle currently sits on this cell."""
        return any(
            obstacle.position == position
            for obstacle in self._dynamic_obstacles.values()
        )

    @property
    def dynamic_obstacles(self) -> List[Obstacle]:
        return sorted(self._dynamic_obstacles.values(), key=lambda o: o.obstacle_id)

    def obstacle_at(self, position: Position) -> Optional[Obstacle]:
        """Return the obstacle occupying a cell, static first, else None."""
        for obstacle in self._static_obstacles.values():
            if obstacle.position == position:
                return obstacle
        for obstacle in self._dynamic_obstacles.values():
            if obstacle.position == position:
                return obstacle
        return None

    # ------------------------------------------------------------------
    # Traversability
    # ------------------------------------------------------------------
    def is_traversable(self, position: Position) -> bool:
        """True if a robot may stand here RIGHT NOW, ignoring other robots.

        This combines the two obstacle layers:
            inside the warehouse AND not statically blocked AND not
            dynamically blocked.

        Robot-to-robot occupancy is deliberately excluded: it belongs to the
        simulator, which is the only object that knows where every robot is.
        """
        return self.grid.is_traversable(position) and not self.is_blocked_by_dynamic(
            position
        )

    def neighbors(self, position: Position) -> List[Position]:
        """Currently reachable cells one orthogonal step away.

        Same idea as `Grid.neighbors`, but it also filters out cells hidden
        behind dynamic obstacles.
        """
        self._require_in_bounds(position)
        return [
            candidate
            for candidate in self.grid.neighbors(position)
            if not self.is_blocked_by_dynamic(candidate)
        ]

    def blocked_cells(self) -> Set[Position]:
        """Every cell blocked right now, by either kind of obstacle."""
        blocked = self.grid.blocked_cells
        blocked.update(o.position for o in self._dynamic_obstacles.values())
        return blocked

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _require_in_bounds(self, position: Position) -> None:
        if not self.grid.in_bounds(position):
            raise InvalidPositionError(
                f"Position {position} is outside warehouse '{self.name}' "
                f"({self.height} rows x {self.width} cols)"
            )

    def _require_unused_obstacle_id(self, obstacle_id: str) -> None:
        if obstacle_id in self._static_obstacles or obstacle_id in self._dynamic_obstacles:
            raise SimulationError(f"Obstacle id '{obstacle_id}' is already in use")

    def __repr__(self) -> str:
        return (
            f"Warehouse(name='{self.name}', rows={self.height}, cols={self.width}, "
            f"static_obstacles={len(self._static_obstacles)}, "
            f"dynamic_obstacles={len(self._dynamic_obstacles)})"
        )
