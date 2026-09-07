"""Path-planning service: run the EXISTING A* / Dijkstra planner via the API.

This service does not contain a search algorithm. It:

    1. builds the `Warehouse` the request describes (default profile, plus any
       transient obstacles),
    2. calls `robotics.planning.PLANNERS[algorithm]`,
    3. records the run in ``planning_runs``,
    4. returns the planner's own `PathResult` plus the new run id.

A "no route exists" answer is a legitimate result, not an error - it is stored
and returned with ``success=False`` and the planner's reason.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence

from sqlalchemy.orm import Session

from robotics.exceptions import InvalidPositionError
from robotics.persistence.models import PlanningRun
from robotics.persistence.repositories import RunRepository
from robotics.planning import PLANNERS
from robotics.planning.path_result import PathResult
from robotics.services.errors import ValidationFailedError
from robotics.services.world import build_warehouse
from robotics.warehouse.grid import Position

logger = logging.getLogger("robotics.services.planning")

SUPPORTED_ALGORITHMS = tuple(PLANNERS.keys())  # ("dijkstra", "astar")


@dataclass
class PlanningOutcome:
    result: PathResult
    run_id: int
    algorithm: str


class PlanningService:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.runs = RunRepository(session)

    def plan_path(
        self,
        start_row: int,
        start_col: int,
        goal_row: int,
        goal_col: int,
        algorithm: str = "astar",
        warehouse_width: Optional[int] = None,
        warehouse_height: Optional[int] = None,
        static_obstacles: Optional[Sequence[Position]] = None,
        dynamic_obstacles: Optional[Sequence[Position]] = None,
    ) -> PlanningOutcome:
        if algorithm not in PLANNERS:
            raise ValidationFailedError(
                f"unsupported algorithm '{algorithm}'; use one of "
                f"{', '.join(SUPPORTED_ALGORITHMS)}"
            )

        warehouse = build_warehouse(
            warehouse_width, warehouse_height, static_obstacles
        )
        for index, cell in enumerate(dynamic_obstacles or ()):
            if not warehouse.in_bounds(cell):
                raise ValidationFailedError(
                    f"dynamic obstacle {(cell.row, cell.col)} is outside the warehouse"
                )
            if warehouse.is_blocked_by_static(cell):
                continue
            warehouse.add_dynamic_obstacle(
                f"api-dynamic-{index}", cell, "api transient obstacle"
            )

        start = Position(start_row, start_col)
        goal = Position(goal_row, goal_col)

        if not warehouse.in_bounds(start):
            raise ValidationFailedError(
                f"start {(start_row, start_col)} is outside the warehouse"
            )
        if not warehouse.in_bounds(goal):
            raise ValidationFailedError(
                f"goal {(goal_row, goal_col)} is outside the warehouse"
            )

        planner = PLANNERS[algorithm]
        try:
            result = planner(start, goal, warehouse)
        except InvalidPositionError as error:
            raise ValidationFailedError(str(error)) from error

        run = PlanningRun(
            algorithm=algorithm,
            start_row=start_row,
            start_col=start_col,
            goal_row=goal_row,
            goal_col=goal_col,
            success=result.success,
            total_cost=result.total_cost,
            nodes_expanded=result.nodes_expanded,
            planning_time_ms=result.planning_time_ms,
            failure_reason=result.failure_reason,
            result_json=result.to_dict(),
        )
        self.runs.add_planning_run(run)
        logger.info(
            "planning.run",
            extra={
                "algorithm": algorithm,
                "success": result.success,
                "nodes_expanded": result.nodes_expanded,
            },
        )
        return PlanningOutcome(result=result, run_id=run.id, algorithm=algorithm)
