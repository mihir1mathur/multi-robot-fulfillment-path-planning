"""Obstacles: cells a robot must not drive onto.

TWO KINDS OF OBSTACLE
---------------------
STATIC obstacle
    Part of the building: a wall, a support pillar, a fixed storage rack.
    It is present when the warehouse is built and it never moves. A planner
    may bake static obstacles into the map once and reuse that map forever.

DYNAMIC obstacle
    Something temporary: a dropped box, a maintenance cart, a wet-floor
    cordon, a human walking through. It appears and disappears while the
    simulation is running.

WHY SEPARATE THEM?
------------------
The distinction only starts to *matter* once routes are being planned, but it
must exist in the data model from day one:

  * A route computed against static obstacles stays valid indefinitely.
  * A route computed against dynamic obstacles can become invalid at any
    moment, which is exactly what triggers REPLANNING.

This module implements the representation and add/remove behaviour only. There
is no replanning, no obstacle movement and no time model yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from robotics.warehouse.grid import Position


class ObstacleType(Enum):
    """Whether an obstacle is permanent or temporary."""

    STATIC = "static"
    DYNAMIC = "dynamic"


@dataclass(frozen=True)
class Obstacle:
    """One blocked cell, with enough metadata to explain and undo it.

    Frozen because an obstacle's identity is its (id, position, type); if a
    temporary obstacle "moves", the honest model is to remove it and add a new
    one rather than mutating it in place.

    Attributes:
        obstacle_id: unique human-readable name, e.g. "rack-A3" or "spill-1".
        position: the cell that is blocked.
        obstacle_type: STATIC or DYNAMIC.
        description: free-text note shown in logs and demo output.
    """

    obstacle_id: str
    position: Position
    obstacle_type: ObstacleType
    description: str = ""

    @property
    def is_static(self) -> bool:
        return self.obstacle_type is ObstacleType.STATIC

    @property
    def is_dynamic(self) -> bool:
        return self.obstacle_type is ObstacleType.DYNAMIC

    def __str__(self) -> str:
        label = self.description or self.obstacle_id
        return f"{self.obstacle_type.value} obstacle '{label}' at {self.position}"
