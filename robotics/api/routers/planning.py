"""Path-planning endpoint.

    POST /planning/path   run A* or Dijkstra for one robot between two cells

"No route exists" is a legitimate result: the response is 200 with
``success=false`` and the planner's reason, not a 4xx / 5xx.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from robotics.api.dependencies import get_session, require_role
from robotics.api.security import Role
from robotics.api.schemas.planning import PathPlanRequest, PathPlanResponse
from robotics.services.planning_service import PlanningService
from robotics.warehouse.grid import Position

router = APIRouter(
    prefix="/planning", tags=["planning"],
    dependencies=[Depends(require_role(Role.operator))],
)


@router.post("/path", response_model=PathPlanResponse)
def plan_path(
    body: PathPlanRequest, session: Session = Depends(get_session)
) -> PathPlanResponse:
    spec = body.warehouse
    outcome = PlanningService(session).plan_path(
        start_row=body.start.row,
        start_col=body.start.col,
        goal_row=body.goal.row,
        goal_col=body.goal.col,
        algorithm=body.algorithm.value,
        warehouse_width=spec.width if spec else None,
        warehouse_height=spec.height if spec else None,
        static_obstacles=(
            [Position(c.row, c.col) for c in spec.static_obstacles] if spec else None
        ),
        dynamic_obstacles=[Position(c.row, c.col) for c in body.dynamic_obstacles],
    )
    return PathPlanResponse.from_outcome(outcome)
