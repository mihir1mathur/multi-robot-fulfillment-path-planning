"""Robot request / response schemas."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from robotics.api.schemas.common import Coordinate, RobotStatusName
from robotics.persistence.models import RobotRecord


class RobotCreate(BaseModel):
    robot_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_\-]+$")
    position: Coordinate
    battery_level: float = Field(default=100.0, ge=0.0, le=100.0)
    payload_capacity: float = Field(default=10.0, ge=0.0)
    current_payload: float = Field(default=0.0, ge=0.0)
    status: RobotStatusName = RobotStatusName.IDLE  # type: ignore[attr-defined]


class RobotUpdate(BaseModel):
    """All fields optional - only the ones present are changed (PATCH semantics)."""

    position: Optional[Coordinate] = None
    battery_level: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    payload_capacity: Optional[float] = Field(default=None, ge=0.0)
    current_payload: Optional[float] = Field(default=None, ge=0.0)
    status: Optional[RobotStatusName] = None  # type: ignore[valid-type]
    assigned_task_id: Optional[str] = Field(default=None, max_length=64)
    clear_assigned_task: bool = False


class RobotRead(BaseModel):
    robot_id: str
    position: Coordinate
    battery_level: float
    payload_capacity: float
    current_payload: float
    status: str
    assigned_task_id: Optional[str]

    @classmethod
    def from_record(cls, record: RobotRecord) -> "RobotRead":
        return cls(
            robot_id=record.robot_id,
            position=Coordinate(row=record.row, col=record.col),
            battery_level=record.battery_level,
            payload_capacity=record.payload_capacity,
            current_payload=record.current_payload,
            status=record.status,
            assigned_task_id=record.assigned_task_id,
        )


class RobotList(BaseModel):
    robots: List[RobotRead]
    count: int
