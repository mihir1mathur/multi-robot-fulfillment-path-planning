"""Path-planning request / response schemas."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from robotics.api.schemas.common import Coordinate, PlanningAlgorithm, WarehouseSpec
from robotics.planning.path_result import PathResult
from robotics.services.planning_service import PlanningOutcome


class PathPlanRequest(BaseModel):
    start: Coordinate
    goal: Coordinate
    algorithm: PlanningAlgorithm = PlanningAlgorithm.astar
    warehouse: Optional[WarehouseSpec] = None
    dynamic_obstacles: List[Coordinate] = Field(
        default_factory=list,
        description="cells blocked for this request only (a spill, a cart)",
    )


class PathPlanResponse(BaseModel):
    run_id: int
    algorithm: str
    success: bool
    start: Coordinate
    goal: Coordinate
    path: List[Coordinate]
    total_cost: Optional[float]
    nodes_expanded: int
    planning_time_ms: float
    failure_reason: Optional[str]

    @classmethod
    def from_outcome(cls, outcome: PlanningOutcome) -> "PathPlanResponse":
        result: PathResult = outcome.result
        return cls(
            run_id=outcome.run_id,
            algorithm=outcome.algorithm,
            success=result.success,
            start=Coordinate(row=result.start.row, col=result.start.col),
            goal=Coordinate(row=result.goal.row, col=result.goal.col),
            path=[Coordinate(row=p.row, col=p.col) for p in result.path],
            total_cost=result.total_cost,
            nodes_expanded=result.nodes_expanded,
            planning_time_ms=result.planning_time_ms,
            failure_reason=result.failure_reason,
        )
