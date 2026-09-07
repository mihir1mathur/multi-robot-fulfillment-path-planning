"""Multi-robot coordination endpoint.

    POST /coordination/plan   prioritized space-time planning for a fleet

If the priority order boxes a robot in, that robot appears in
``failed_robot_ids`` with a reason and ``success`` is false - a 200 response
that reports the partial result honestly. Prioritized planning is not globally
optimal MAPF and the response does not pretend otherwise.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from robotics.api.dependencies import get_session, require_role
from robotics.api.security import Role
from robotics.api.schemas.coordination import (
    CoordinationRequest,
    CoordinationResponse,
)
from robotics.services.coordination_service import CoordinationService
from robotics.warehouse.grid import Position

router = APIRouter(
    prefix="/coordination", tags=["coordination"],
    dependencies=[Depends(require_role(Role.operator))],
)


@router.post("/plan", response_model=CoordinationResponse)
def plan_coordination(
    body: CoordinationRequest, session: Session = Depends(get_session)
) -> CoordinationResponse:
    spec = body.warehouse
    outcome = CoordinationService(session).coordinate(
        robots=[r.to_spec() for r in body.robots],
        priority_order=body.priority_order,
        horizon=body.horizon,
        warehouse_width=spec.width if spec else None,
        warehouse_height=spec.height if spec else None,
        static_obstacles=(
            [Position(c.row, c.col) for c in spec.static_obstacles] if spec else None
        ),
    )
    return CoordinationResponse.from_outcome(outcome)
