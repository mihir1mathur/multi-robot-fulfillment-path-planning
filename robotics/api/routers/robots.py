"""Robot CRUD endpoints.

    POST   /robots            create a robot
    GET    /robots            list robots (optional ?status=)
    GET    /robots/{id}       fetch one          -> 404 if missing
    PATCH  /robots/{id}       partial update     -> 404 / 409 / 422
    DELETE /robots/{id}       remove a robot row -> 204, or 404 if missing

Validation and conflict rules live in `RobotService`; this module only maps
HTTP <-> service calls.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session

from robotics.api.dependencies import get_session, require_role
from robotics.api.schemas.robots import (
    RobotCreate,
    RobotList,
    RobotRead,
    RobotUpdate,
)
from robotics.api.security import Role
from robotics.services.robot_service import RobotService

router = APIRouter(prefix="/robots", tags=["robots"])
_viewer = Depends(require_role(Role.viewer))
_operator = Depends(require_role(Role.operator))


@router.post(
    "", response_model=RobotRead, status_code=status.HTTP_201_CREATED,
    dependencies=[_operator],
)
def create_robot(
    body: RobotCreate, session: Session = Depends(get_session)
) -> RobotRead:
    record = RobotService(session).create_robot(
        robot_id=body.robot_id,
        row=body.position.row,
        col=body.position.col,
        battery_level=body.battery_level,
        payload_capacity=body.payload_capacity,
        current_payload=body.current_payload,
        status=body.status.value,
    )
    return RobotRead.from_record(record)


@router.get("", response_model=RobotList, dependencies=[_viewer])
def list_robots(
    status: Optional[str] = Query(default=None),
    session: Session = Depends(get_session),
) -> RobotList:
    records = RobotService(session).list_robots(status=status)
    return RobotList(
        robots=[RobotRead.from_record(r) for r in records], count=len(records)
    )


@router.get("/{robot_id}", response_model=RobotRead, dependencies=[_viewer])
def get_robot(robot_id: str, session: Session = Depends(get_session)) -> RobotRead:
    return RobotRead.from_record(RobotService(session).get_robot(robot_id))


@router.patch("/{robot_id}", response_model=RobotRead, dependencies=[_operator])
def update_robot(
    robot_id: str, body: RobotUpdate, session: Session = Depends(get_session)
) -> RobotRead:
    service = RobotService(session)
    record = service.update_robot(
        robot_id,
        row=body.position.row if body.position else None,
        col=body.position.col if body.position else None,
        battery_level=body.battery_level,
        payload_capacity=body.payload_capacity,
        current_payload=body.current_payload,
        status=body.status.value if body.status else None,
        assigned_task_id=body.assigned_task_id,
        clear_assigned_task=body.clear_assigned_task,
    )
    return RobotRead.from_record(record)


@router.delete(
    "/{robot_id}", status_code=status.HTTP_204_NO_CONTENT, dependencies=[_operator]
)
def delete_robot(robot_id: str, session: Session = Depends(get_session)) -> None:
    RobotService(session).delete_robot(robot_id)
