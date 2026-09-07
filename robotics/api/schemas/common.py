"""Shared schema building blocks: coordinates, enums, the error envelope."""

from __future__ import annotations

from enum import Enum
from typing import Any, List, Optional

from pydantic import BaseModel, Field

from robotics.robots.robot_state import RobotStatus
from robotics.tasks.task import TaskPriority, TaskStatus


class Coordinate(BaseModel):
    """A grid cell. Rows grow downward, columns rightward; (0, 0) is top-left."""

    row: int = Field(ge=0, description="0-based row index (top = 0)")
    col: int = Field(ge=0, description="0-based column index (left = 0)")

    def as_tuple(self) -> tuple[int, int]:
        return (self.row, self.col)


class PlanningAlgorithm(str, Enum):
    astar = "astar"
    dijkstra = "dijkstra"


class AllocationAlgorithm(str, Enum):
    greedy = "greedy"
    cp_sat = "cp_sat"


# Re-expose the domain enums as string enums for request validation, so a
# client that sends an unknown status gets a 422 with the list of valid values.
RobotStatusName = Enum(  # type: ignore[misc]
    "RobotStatusName", {s.name: s.value for s in RobotStatus}, type=str
)
TaskStatusName = Enum(  # type: ignore[misc]
    "TaskStatusName", {s.name: s.value for s in TaskStatus}, type=str
)
TaskPriorityName = Enum(  # type: ignore[misc]
    "TaskPriorityName", {p.name.lower(): p.name.lower() for p in TaskPriority}, type=str
)


class WarehouseSpec(BaseModel):
    """Optional custom warehouse for a compute request.

    Omit it entirely to use the default fixed 12x12 profile (racks, storage,
    pickups, chargers). Supplying it gives a plain rectangle plus any
    ``static_obstacles``.
    """

    width: int = Field(gt=0, le=200)
    height: int = Field(gt=0, le=200)
    static_obstacles: List[Coordinate] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    """Uniform error body for every 4xx / 5xx the API produces itself."""

    error: str = Field(description="short machine-readable error code")
    message: str = Field(description="human-readable explanation")
    details: Optional[Any] = Field(default=None)
