"""Recovery endpoints.

    POST /recovery/obstacle        an obstacle appears mid-episode -> replan
    POST /recovery/robot-failure   a robot goes OFFLINE mid-episode -> recover
                                   the task, re-coordinate the others.
                                   ``commit=true`` writes a successful
                                   failure->spare reassignment back to the
                                   persisted robots/tasks rows.

Both first coordinate the supplied fleet, then apply the disruption through the
existing `RecoveryManager`. If recovery is impossible (no alternate route, no
spare robot) the response reports ``safe_stop=true`` honestly - it is not a
5xx. If the fleet cannot even be coordinated, that is a domain failure and the
response says so.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from robotics.api.dependencies import get_session, require_role
from robotics.api.security import Role
from robotics.api.schemas.recovery import (
    ObstacleRecoveryRequest,
    RecoveryResponse,
    RobotFailureRecoveryRequest,
)
from robotics.services.errors import DomainFailureError
from robotics.services.recovery_service import RecoveryService
from robotics.warehouse.grid import Position

router = APIRouter(
    prefix="/recovery", tags=["recovery"],
    dependencies=[Depends(require_role(Role.operator))],
)


def _warehouse_kwargs(body) -> dict:
    spec = body.warehouse
    if spec is None:
        return {}
    return {
        "warehouse_width": spec.width,
        "warehouse_height": spec.height,
        "static_obstacles": [Position(c.row, c.col) for c in spec.static_obstacles],
    }


@router.post("/obstacle", response_model=RecoveryResponse)
def recover_obstacle(
    body: ObstacleRecoveryRequest, session: Session = Depends(get_session)
) -> RecoveryResponse:
    service = RecoveryService(session)
    try:
        outcome = service.recover_from_obstacle(
            robots=[r.to_spec() for r in body.robots],
            obstacle_row=body.obstacle.row,
            obstacle_col=body.obstacle.col,
            timestep=body.timestep,
            priority_order=body.priority_order,
            horizon=body.horizon,
            recovery_enabled=body.recovery_enabled,
            **_warehouse_kwargs(body),
        )
    except DomainFailureError as failure:
        return _domain_failure_response("dynamic_obstacle", failure)
    return RecoveryResponse.from_outcome(outcome)


@router.post("/robot-failure", response_model=RecoveryResponse)
def recover_robot_failure(
    body: RobotFailureRecoveryRequest, session: Session = Depends(get_session)
) -> RecoveryResponse:
    service = RecoveryService(session)
    try:
        outcome = service.recover_from_robot_failure(
            robots=[r.to_spec() for r in body.robots],
            failed_robot_id=body.failed_robot_id,
            timestep=body.timestep,
            priority_order=body.priority_order,
            horizon=body.horizon,
            recovery_enabled=body.recovery_enabled,
            commit=body.commit,
            **_warehouse_kwargs(body),
        )
    except DomainFailureError as failure:
        return _domain_failure_response("robot_offline", failure)
    return RecoveryResponse.from_outcome(outcome)


def _domain_failure_response(
    disruption_type: str, failure: DomainFailureError
) -> RecoveryResponse:
    """A legitimate 'cannot even plan this' answer, returned as a normal body."""
    return RecoveryResponse(
        run_id=0,
        disruption_type=disruption_type,
        recovery_enabled=True,
        recovery_attempted=False,
        recovery_success=False,
        safe_stop=True,
        resumed=False,
        affected_robot_ids=[],
        affected_task_ids=[],
        replanning_attempted=False,
        replanning_success=False,
        task_reassignment_attempted=False,
        task_reassignment_success=False,
        reassigned_from=None,
        reassigned_to=None,
        robot_replans=[],
        unresolved_vertex_conflicts=0,
        unresolved_edge_conflicts=0,
        total_recovery_latency_ms=0.0,
        execution={},
        failure_reason=failure.message,
    )
