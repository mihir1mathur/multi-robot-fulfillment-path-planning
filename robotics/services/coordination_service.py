"""Multi-robot coordination service: run the EXISTING prioritized coordinator.

The service builds the warehouse and the {robot: start} / {robot: goal} maps
the coordinator expects, calls `MultiRobotCoordinator.plan`, records the run,
and returns the coordinator's own `CoordinationResult`.

This is PRIORITIZED planning (plan robots in a fixed order, reserve each path
before the next). It is deterministic and explainable; it is NOT globally
optimal multi-agent path finding, and the API does not claim to be.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence

from sqlalchemy.orm import Session

from robotics.coordination.coordination_result import CoordinationResult
from robotics.coordination.coordinator import MultiRobotCoordinator
from robotics.persistence.models import CoordinationRun
from robotics.persistence.repositories import RunRepository
from robotics.services.errors import ValidationFailedError
from robotics.services.world import build_warehouse
from robotics.warehouse.grid import Position

logger = logging.getLogger("robotics.services.coordination")


@dataclass
class CoordinationRobotSpec:
    robot_id: str
    start_row: int
    start_col: int
    goal_row: int
    goal_col: int


@dataclass
class CoordinationOutcome:
    result: CoordinationResult
    run_id: int


class CoordinationService:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.runs = RunRepository(session)

    def coordinate(
        self,
        robots: Sequence[CoordinationRobotSpec],
        priority_order: Optional[List[str]] = None,
        horizon: Optional[int] = None,
        warehouse_width: Optional[int] = None,
        warehouse_height: Optional[int] = None,
        static_obstacles: Optional[Sequence[Position]] = None,
    ) -> CoordinationOutcome:
        if not robots:
            raise ValidationFailedError("coordination needs at least one robot")
        ids = [r.robot_id for r in robots]
        if len(ids) != len(set(ids)):
            raise ValidationFailedError("robot ids must be unique")

        warehouse = build_warehouse(
            warehouse_width, warehouse_height, static_obstacles
        )
        starts = {r.robot_id: Position(r.start_row, r.start_col) for r in robots}
        goals = {r.robot_id: Position(r.goal_row, r.goal_col) for r in robots}

        for rid, cell in {**starts, **goals}.items():
            if not warehouse.in_bounds(cell):
                raise ValidationFailedError(
                    f"robot '{rid}': cell {(cell.row, cell.col)} is outside the warehouse"
                )

        if priority_order is not None and sorted(priority_order) != sorted(ids):
            raise ValidationFailedError(
                "priority_order must be a permutation of the robot ids"
            )

        coordinator = MultiRobotCoordinator(warehouse)
        result = coordinator.plan(starts, goals, priority_order, horizon)

        run = CoordinationRun(
            robot_count=len(robots),
            success=result.success,
            planned_robot_count=result.planned_robot_count,
            failed_robot_ids=list(result.failed_robot_ids),
            makespan=result.makespan,
            total_move_steps=result.total_move_steps,
            total_wait_steps=result.total_wait_steps,
            conflict_checks=result.conflict_checks,
            planning_time_ms=result.planning_time_ms,
            failure_reason=result.failure_reason,
            result_json=result.to_dict(),
        )
        self.runs.add_coordination_run(run)
        logger.info(
            "coordination.run",
            extra={
                "robot_count": len(robots),
                "success": result.success,
                "planned": result.planned_robot_count,
            },
        )
        return CoordinationOutcome(result=result, run_id=run.id)
