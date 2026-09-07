"""Robot management: create / read / list / update persisted robots.

Every write goes through the domain `Robot` model first, so a row can never
be stored in a state the simulation would reject (battery out of range,
payload over capacity, a cell that is a wall). The database keeps the state;
the domain model keeps the rules.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

from sqlalchemy.orm import Session

from robotics.persistence.models import RobotRecord
from robotics.persistence.repositories import RobotRepository
from robotics.robots.robot import Robot
from robotics.robots.robot_state import RobotStatus
from robotics.services.errors import (
    ResourceConflictError,
    ResourceNotFoundError,
    ValidationFailedError,
)
from robotics.services.world import build_warehouse
from robotics.warehouse.grid import Position

logger = logging.getLogger("robotics.services.robot")


class RobotService:
    """Domain operations on the persisted fleet."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self.robots = RobotRepository(session)
        self._warehouse = build_warehouse()  # default profile - the world robots live in

    # ------------------------------------------------------------------
    def list_robots(self, status: Optional[str] = None) -> Sequence[RobotRecord]:
        if status is not None and status not in {s.value for s in RobotStatus}:
            raise ValidationFailedError(f"unknown robot status '{status}'")
        return self.robots.list(status=status)

    def get_robot(self, robot_id: str) -> RobotRecord:
        record = self.robots.get(robot_id)
        if record is None:
            raise ResourceNotFoundError(f"no robot with id '{robot_id}'")
        return record

    def delete_robot(self, robot_id: str) -> None:
        """Remove a robot row. 404 if it does not exist.

        The ``tasks.assigned_robot_id`` foreign key is ``ON DELETE SET NULL``,
        so any task still pointing at this robot is released rather than left
        dangling. Used to tear a scenario down (the demo dashboard owns its
        fleet).
        """
        record = self.get_robot(robot_id)
        self.robots.delete(record)
        logger.info("robot.deleted", extra={"robot_id": robot_id})

    # ------------------------------------------------------------------
    def create_robot(
        self,
        robot_id: str,
        row: int,
        col: int,
        battery_level: float = 100.0,
        payload_capacity: float = 10.0,
        current_payload: float = 0.0,
        status: str = "idle",
    ) -> RobotRecord:
        if self.robots.exists(robot_id):
            raise ResourceConflictError(f"robot '{robot_id}' already exists")

        position = Position(row, col)
        self._require_placeable(position, ignore_robot_id=None)
        # Construct the domain object purely to validate the numeric invariants.
        self._as_domain(robot_id, position, battery_level, payload_capacity, current_payload, status)

        record = RobotRecord(
            robot_id=robot_id,
            row=row,
            col=col,
            battery_level=battery_level,
            payload_capacity=payload_capacity,
            current_payload=current_payload,
            status=status,
            assigned_task_id=None,
        )
        self.robots.add(record)
        logger.info(
            "robot.created", extra={"robot_id": robot_id, "position": [row, col]}
        )
        return record

    # ------------------------------------------------------------------
    def update_robot(
        self,
        robot_id: str,
        *,
        row: Optional[int] = None,
        col: Optional[int] = None,
        battery_level: Optional[float] = None,
        payload_capacity: Optional[float] = None,
        current_payload: Optional[float] = None,
        status: Optional[str] = None,
        assigned_task_id: Optional[str] = None,
        clear_assigned_task: bool = False,
    ) -> RobotRecord:
        record = self.get_robot(robot_id)

        new_row = record.row if row is None else row
        new_col = record.col if col is None else col
        new_battery = record.battery_level if battery_level is None else battery_level
        new_capacity = (
            record.payload_capacity if payload_capacity is None else payload_capacity
        )
        new_payload = (
            record.current_payload if current_payload is None else current_payload
        )
        new_status = record.status if status is None else status

        position = Position(new_row, new_col)
        if (new_row, new_col) != (record.row, record.col):
            self._require_placeable(position, ignore_robot_id=robot_id)

        self._as_domain(
            robot_id, position, new_battery, new_capacity, new_payload, new_status
        )

        record.row = new_row
        record.col = new_col
        record.battery_level = new_battery
        record.payload_capacity = new_capacity
        record.current_payload = new_payload
        record.status = new_status
        if clear_assigned_task:
            record.assigned_task_id = None
        elif assigned_task_id is not None:
            record.assigned_task_id = assigned_task_id

        self.session.flush()
        logger.info("robot.updated", extra={"robot_id": robot_id})
        return record

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _as_domain(
        self,
        robot_id: str,
        position: Position,
        battery_level: float,
        payload_capacity: float,
        current_payload: float,
        status: str,
    ) -> Robot:
        try:
            robot_status = RobotStatus(status)
        except ValueError as error:
            raise ValidationFailedError(str(error)) from error
        try:
            return Robot(
                robot_id=robot_id,
                position=position,
                battery_level=battery_level,
                payload_capacity=payload_capacity,
                current_payload=current_payload,
                status=robot_status,
            )
        except ValueError as error:
            raise ValidationFailedError(str(error)) from error

    def _require_placeable(
        self, position: Position, ignore_robot_id: Optional[str]
    ) -> None:
        if not self._warehouse.in_bounds(position):
            raise ValidationFailedError(
                f"position {(position.row, position.col)} is outside the warehouse"
            )
        if not self._warehouse.is_traversable(position):
            raise ValidationFailedError(
                f"position {(position.row, position.col)} is blocked by an obstacle"
            )
        for other in self.robots.list():
            if other.robot_id == ignore_robot_id:
                continue
            if (other.row, other.col) == (position.row, position.col):
                raise ResourceConflictError(
                    f"position {(position.row, position.col)} is occupied by "
                    f"robot '{other.robot_id}'"
                )
