"""Task CRUD endpoints.

    POST   /tasks           create a pending task
    GET    /tasks           list tasks (optional ?status=)
    GET    /tasks/{id}      fetch one           -> 404 if missing
    PATCH  /tasks/{id}      partial update / lifecycle transition
                            -> 409 for an illegal transition, 422 for bad input
    DELETE /tasks/{id}      remove a task row   -> 204, or 404 if missing
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session

from robotics.api.dependencies import get_session, require_role
from robotics.api.schemas.tasks import TaskCreate, TaskList, TaskRead, TaskUpdate
from robotics.api.security import Role
from robotics.services.task_service import TaskService

router = APIRouter(prefix="/tasks", tags=["tasks"])
_viewer = Depends(require_role(Role.viewer))
_operator = Depends(require_role(Role.operator))


@router.post(
    "", response_model=TaskRead, status_code=status.HTTP_201_CREATED,
    dependencies=[_operator],
)
def create_task(
    body: TaskCreate, session: Session = Depends(get_session)
) -> TaskRead:
    record = TaskService(session).create_task(
        task_id=body.task_id,
        pickup_row=body.pickup.row,
        pickup_col=body.pickup.col,
        dropoff_row=body.dropoff.row,
        dropoff_col=body.dropoff.col,
        priority=body.priority.value,
        payload_weight=body.payload_weight,
    )
    return TaskRead.from_record(record)


@router.get("", response_model=TaskList, dependencies=[_viewer])
def list_tasks(
    status: Optional[str] = Query(default=None),
    session: Session = Depends(get_session),
) -> TaskList:
    records = TaskService(session).list_tasks(status=status)
    return TaskList(
        tasks=[TaskRead.from_record(r) for r in records], count=len(records)
    )


@router.get("/{task_id}", response_model=TaskRead, dependencies=[_viewer])
def get_task(task_id: str, session: Session = Depends(get_session)) -> TaskRead:
    return TaskRead.from_record(TaskService(session).get_task(task_id))


@router.patch("/{task_id}", response_model=TaskRead, dependencies=[_operator])
def update_task(
    task_id: str, body: TaskUpdate, session: Session = Depends(get_session)
) -> TaskRead:
    record = TaskService(session).update_task(
        task_id,
        status=body.status.value if body.status else None,
        assigned_robot_id=body.assigned_robot_id,
        priority=body.priority.value if body.priority else None,
        payload_weight=body.payload_weight,
        failure_reason=body.failure_reason,
    )
    return TaskRead.from_record(record)


@router.delete(
    "/{task_id}", status_code=status.HTTP_204_NO_CONTENT, dependencies=[_operator]
)
def delete_task(task_id: str, session: Session = Depends(get_session)) -> None:
    TaskService(session).delete_task(task_id)
