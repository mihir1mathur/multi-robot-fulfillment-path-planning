"""Task request / response schemas."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from robotics.api.schemas.common import Coordinate, TaskPriorityName, TaskStatusName
from robotics.persistence.models import TaskRecord


class TaskCreate(BaseModel):
    task_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_\-]+$")
    pickup: Coordinate
    dropoff: Coordinate
    payload_weight: float = Field(default=1.0, gt=0.0, description="item weight in kg")
    priority: TaskPriorityName = TaskPriorityName.normal  # type: ignore[attr-defined]


class TaskUpdate(BaseModel):
    """PATCH semantics. A status change is validated against the task lifecycle."""

    status: Optional[TaskStatusName] = None  # type: ignore[valid-type]
    assigned_robot_id: Optional[str] = Field(default=None, max_length=64)
    priority: Optional[TaskPriorityName] = None  # type: ignore[valid-type]
    payload_weight: Optional[float] = Field(default=None, gt=0.0)
    failure_reason: Optional[str] = Field(default=None, max_length=500)


class TaskRead(BaseModel):
    task_id: str
    pickup: Coordinate
    dropoff: Coordinate
    payload_weight: float
    priority: str
    status: str
    assigned_robot_id: Optional[str]
    failure_reason: Optional[str]

    @classmethod
    def from_record(cls, record: TaskRecord) -> "TaskRead":
        return cls(
            task_id=record.task_id,
            pickup=Coordinate(row=record.pickup_row, col=record.pickup_col),
            dropoff=Coordinate(row=record.dropoff_row, col=record.dropoff_col),
            payload_weight=record.payload_weight,
            priority=record.priority,
            status=record.status,
            assigned_robot_id=record.assigned_robot_id,
            failure_reason=record.failure_reason,
        )


class TaskList(BaseModel):
    tasks: List[TaskRead]
    count: int
