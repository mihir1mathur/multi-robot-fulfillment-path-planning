"""Building the in-memory domain world the algorithms need.

The planning / allocation / coordination / recovery code all work on live
`Warehouse`, `Robot`, `Task` and `WarehouseSimulator` objects - not database
rows. This module is the one place that converts:

    RobotRecord / TaskRecord  (rows)          -> Robot / Task  (domain objects)
    a warehouse spec (or the default profile)  -> Warehouse
    all of the above                           -> WarehouseSimulator

DEFAULT WAREHOUSE PROFILE
------------------------
When a request does not describe a warehouse, the API uses the same fixed
12x12 layout the demo and tests use (`build_sample_warehouse`): racks, storage,
pickup / drop-off points and chargers, all reachable. This keeps coordinates
meaningful and every run reproducible.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

from robotics.persistence.models import RobotRecord, TaskRecord
from robotics.robots.robot import Robot
from robotics.robots.robot_state import RobotStatus
from robotics.services.errors import ValidationFailedError
from robotics.simulation.scenario import build_sample_warehouse
from robotics.simulation.simulator import WarehouseSimulator
from robotics.tasks.task import Task, TaskPriority, TaskStatus
from robotics.warehouse.grid import Position
from robotics.warehouse.warehouse import Warehouse

DEFAULT_WAREHOUSE_PROFILE = "default-12x12"


def build_warehouse(
    width: Optional[int] = None,
    height: Optional[int] = None,
    static_obstacles: Optional[Sequence[Position]] = None,
) -> Warehouse:
    """Return a `Warehouse`.

    With no arguments, the fixed sample profile. With ``width``/``height``,
    a plain rectangular warehouse plus any ``static_obstacles``.
    """
    if width is None and height is None and not static_obstacles:
        return build_sample_warehouse()

    if width is None or height is None:
        raise ValidationFailedError(
            "a custom warehouse needs both width and height"
        )
    if width <= 0 or height <= 0:
        raise ValidationFailedError("warehouse width and height must be positive")

    warehouse = Warehouse(width=width, height=height, name="api-warehouse")
    for index, cell in enumerate(static_obstacles or ()):
        if not warehouse.in_bounds(cell):
            raise ValidationFailedError(
                f"static obstacle {(cell.row, cell.col)} is outside the warehouse"
            )
        warehouse.add_static_obstacle(f"obstacle-{index}", cell, "api static obstacle")
    return warehouse


def robot_from_record(record: RobotRecord) -> Robot:
    """Convert a persisted robot row into a domain `Robot` (validating it)."""
    try:
        status = RobotStatus(record.status)
    except ValueError as error:
        raise ValidationFailedError(f"robot '{record.robot_id}': {error}") from error
    try:
        return Robot(
            robot_id=record.robot_id,
            position=Position(record.row, record.col),
            battery_level=record.battery_level,
            payload_capacity=record.payload_capacity,
            current_payload=record.current_payload,
            status=status,
            assigned_task_id=record.assigned_task_id,
        )
    except ValueError as error:
        raise ValidationFailedError(
            f"robot '{record.robot_id}' has invalid stored state: {error}"
        ) from error


def task_from_record(record: TaskRecord) -> Task:
    """Convert a persisted task row into a domain `Task` (validating it)."""
    try:
        priority = TaskPriority[record.priority.upper()]
    except KeyError as error:
        raise ValidationFailedError(
            f"task '{record.task_id}' has unknown priority '{record.priority}'"
        ) from error
    try:
        status = TaskStatus(record.status)
    except ValueError as error:
        raise ValidationFailedError(f"task '{record.task_id}': {error}") from error
    try:
        task = Task(
            task_id=record.task_id,
            pickup_location=Position(record.pickup_row, record.pickup_col),
            dropoff_location=Position(record.dropoff_row, record.dropoff_col),
            priority=priority,
            payload_weight=record.payload_weight,
        )
    except ValueError as error:
        raise ValidationFailedError(
            f"task '{record.task_id}' has invalid stored state: {error}"
        ) from error
    # Re-apply the persisted lifecycle state without re-validating transitions
    # (the DB row is the source of truth here, not a fresh PENDING task).
    task.status = status
    task.assigned_robot_id = record.assigned_robot_id
    task.failure_reason = record.failure_reason
    return task


def build_simulator(
    warehouse: Warehouse,
    robots: Iterable[Robot] = (),
    tasks: Iterable[Task] = (),
) -> WarehouseSimulator:
    """Assemble a `WarehouseSimulator`, surfacing placement errors as 422s."""
    from robotics.exceptions import RoboticsError

    simulator = WarehouseSimulator(warehouse)
    for robot in robots:
        try:
            simulator.add_robot(robot)
        except RoboticsError as error:
            raise ValidationFailedError(str(error)) from error
    for task in tasks:
        try:
            simulator.add_task(task)
        except RoboticsError as error:
            raise ValidationFailedError(str(error)) from error
    return simulator
