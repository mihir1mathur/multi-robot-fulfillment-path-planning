"""Task-allocation request / response schemas."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, model_validator

from robotics.api.schemas.common import (
    AllocationAlgorithm,
    Coordinate,
    TaskPriorityName,
)
from robotics.services.allocation_service import (
    AllocationOutcome,
    RobotSpec,
    TaskSpec,
)


class AllocationRobot(BaseModel):
    robot_id: str = Field(min_length=1, max_length=64)
    position: Coordinate
    battery_level: float = Field(default=100.0, ge=0.0, le=100.0)
    payload_capacity: float = Field(default=10.0, ge=0.0)

    def to_spec(self) -> RobotSpec:
        return RobotSpec(
            robot_id=self.robot_id,
            row=self.position.row,
            col=self.position.col,
            battery_level=self.battery_level,
            payload_capacity=self.payload_capacity,
        )


class AllocationTask(BaseModel):
    task_id: str = Field(min_length=1, max_length=64)
    pickup: Coordinate
    dropoff: Coordinate
    payload_weight: float = Field(default=1.0, gt=0.0)
    priority: TaskPriorityName = TaskPriorityName.normal  # type: ignore[attr-defined]

    def to_spec(self) -> TaskSpec:
        return TaskSpec(
            task_id=self.task_id,
            pickup_row=self.pickup.row,
            pickup_col=self.pickup.col,
            dropoff_row=self.dropoff.row,
            dropoff_col=self.dropoff.col,
            payload_weight=self.payload_weight,
            priority=self.priority.value,
        )


class AllocationRequest(BaseModel):
    algorithm: AllocationAlgorithm = AllocationAlgorithm.greedy
    robots: List[AllocationRobot] = Field(default_factory=list)
    tasks: List[AllocationTask] = Field(default_factory=list)
    from_persisted: bool = Field(
        default=False,
        description="use the persisted idle robots and pending tasks instead of "
        "the inline lists",
    )
    commit: bool = Field(
        default=False,
        description="apply the assignments to the persisted robots/tasks "
        "(only valid with from_persisted=true)",
    )

    @model_validator(mode="after")
    def _check_inputs(self) -> "AllocationRequest":
        if not self.from_persisted and (not self.robots or not self.tasks):
            raise ValueError(
                "provide both robots and tasks, or set from_persisted=true"
            )
        return self


class AllocationAssignment(BaseModel):
    robot_id: str
    task_id: str
    robot_to_pickup_cost: int
    pickup_to_dropoff_cost: int
    total_estimated_cost: int


class AllocationResponse(BaseModel):
    run_id: int
    algorithm: str
    solver_status: str
    assignments: List[AllocationAssignment]
    unassigned_task_ids: List[str]
    total_estimated_cost: int
    eligible_pair_count: int
    infeasible_pair_count: int
    solve_time_ms: float
    committed: bool
    applied: List[str]

    @classmethod
    def from_outcome(cls, outcome: AllocationOutcome) -> "AllocationResponse":
        result = outcome.result
        return cls(
            run_id=outcome.run_id,
            algorithm=result.algorithm,
            solver_status=result.solver_status,
            assignments=[
                AllocationAssignment(
                    robot_id=a.robot_id,
                    task_id=a.task_id,
                    robot_to_pickup_cost=a.robot_to_pickup_cost,
                    pickup_to_dropoff_cost=a.pickup_to_dropoff_cost,
                    total_estimated_cost=a.total_estimated_cost,
                )
                for a in result.assignments
            ],
            unassigned_task_ids=list(result.unassigned_task_ids),
            total_estimated_cost=result.total_estimated_cost,
            eligible_pair_count=result.eligible_pair_count,
            infeasible_pair_count=result.infeasible_pair_count,
            solve_time_ms=result.solve_time_ms,
            committed=outcome.committed,
            applied=list(outcome.applied),
        )
