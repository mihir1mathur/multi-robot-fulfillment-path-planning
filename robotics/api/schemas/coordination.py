"""Multi-robot coordination request / response schemas."""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from robotics.api.schemas.common import Coordinate, WarehouseSpec
from robotics.services.coordination_service import (
    CoordinationOutcome,
    CoordinationRobotSpec,
)


class CoordinationRobot(BaseModel):
    robot_id: str = Field(min_length=1, max_length=64)
    start: Coordinate
    goal: Coordinate

    def to_spec(self) -> CoordinationRobotSpec:
        return CoordinationRobotSpec(
            robot_id=self.robot_id,
            start_row=self.start.row,
            start_col=self.start.col,
            goal_row=self.goal.row,
            goal_col=self.goal.col,
        )


class CoordinationRequest(BaseModel):
    robots: List[CoordinationRobot] = Field(min_length=1)
    priority_order: Optional[List[str]] = Field(
        default=None,
        description="robot ids in the order to plan them; default is sorted ids",
    )
    horizon: Optional[int] = Field(default=None, gt=0)
    warehouse: Optional[WarehouseSpec] = None


class TimedStepModel(BaseModel):
    row: int
    col: int
    timestep: int


class CoordinatedRoute(BaseModel):
    robot_id: str
    success: bool
    steps: List[TimedStepModel]
    arrival_time: int
    move_count: int
    wait_count: int
    failure_reason: Optional[str]


class CoordinationResponse(BaseModel):
    run_id: int
    success: bool
    robot_order: List[str]
    planned_robot_count: int
    failed_robot_ids: List[str]
    routes: Dict[str, CoordinatedRoute]
    makespan: int
    total_move_steps: int
    total_wait_steps: int
    reservation_lookups: int = Field(
        description="reservation-table lookups made while planning (a work "
        "measure, not a count of real-world collisions)"
    )
    planning_time_ms: float
    failure_reason: Optional[str]

    @classmethod
    def from_outcome(cls, outcome: CoordinationOutcome) -> "CoordinationResponse":
        result = outcome.result
        routes: Dict[str, CoordinatedRoute] = {}
        for rid, tp in result.timed_paths.items():
            routes[rid] = CoordinatedRoute(
                robot_id=rid,
                success=tp.success,
                steps=[
                    TimedStepModel(row=s.position.row, col=s.position.col, timestep=s.timestep)
                    for s in tp.steps
                ],
                arrival_time=tp.arrival_time,
                move_count=tp.move_count,
                wait_count=tp.wait_count,
                failure_reason=tp.failure_reason,
            )
        return cls(
            run_id=outcome.run_id,
            success=result.success,
            robot_order=list(result.robot_order),
            planned_robot_count=result.planned_robot_count,
            failed_robot_ids=list(result.failed_robot_ids),
            routes=routes,
            makespan=result.makespan,
            total_move_steps=result.total_move_steps,
            total_wait_steps=result.total_wait_steps,
            reservation_lookups=result.conflict_checks,
            planning_time_ms=result.planning_time_ms,
            failure_reason=result.failure_reason,
        )
