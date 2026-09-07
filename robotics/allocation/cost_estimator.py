"""Estimating what it would cost a robot to do a task.

THE ESTIMATE
------------
A fulfillment task is "collect from the pickup cell P, deliver to the drop-off
cell D". A robot standing at R that takes the task must drive:

    R --> P     (go and get the item)
    P --> D     (carry it to the packing station)

so a simple, defensible cost is

    cost(R, task) = path_cost(R, P) + path_cost(P, D)

measured in grid moves, using the SAME A* planner the rest of the system uses.
We do not use a straight-line (Euclidean or Manhattan) guess: it would ignore
racks and walls, and the whole point of having a planner is that it does not.

INFEASIBLE PAIRINGS
-------------------
If either leg has no route - the pickup is walled off, say - then this robot
simply cannot do this task, and `estimate` returns None. The allocator must
never select a None-cost pairing.

THE CACHE
---------
Allocation asks for many costs and the same cell pairs come up again and again
(every robot is priced against every task's pickup; every task's P->D leg is
identical for all robots). The planner is deterministic and the warehouse does
not change during an allocation cycle, so a plain dict keyed by
(start, goal) -> cost is a safe, order-independent cache. It turns an O(robots x
tasks) pricing pass into far fewer planner calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

from robotics.planning.astar import plan as plan_astar
from robotics.planning.path_result import PathResult
from robotics.robots.robot import Robot
from robotics.tasks.task import Task
from robotics.warehouse.grid import Position
from robotics.warehouse.warehouse import Warehouse

Planner = Callable[[Position, Position, Warehouse], PathResult]


@dataclass(frozen=True)
class AssignmentCost:
    """The priced-out cost of one robot doing one task, in grid moves.

    Attributes:
        robot_to_pickup: moves from the robot's cell to the pickup cell.
        pickup_to_dropoff: moves from the pickup cell to the drop-off cell.
        total: the sum - the number the optimiser minimises.
    """

    robot_to_pickup: int
    pickup_to_dropoff: int

    @property
    def total(self) -> int:
        return self.robot_to_pickup + self.pickup_to_dropoff


class CostEstimator:
    """Prices robot-task pairings using the real path planner, with a cache."""

    def __init__(self, warehouse: Warehouse, planner: Planner = plan_astar) -> None:
        """
        Args:
            warehouse: the floor plan to plan across. Assumed unchanged for the
                lifetime of this estimator (one allocation cycle).
            planner: the path planner to price with. Defaults to A*. Injectable
                so a test can substitute a counting stub.
        """
        self.warehouse = warehouse
        self._planner = planner
        self._cache: Dict[Tuple[Position, Position], Optional[int]] = {}
        self._planner_calls = 0

    # ------------------------------------------------------------------
    # Path cost with caching
    # ------------------------------------------------------------------
    def path_cost(self, start: Position, goal: Position) -> Optional[int]:
        """Number of moves on the shortest route, or None if unreachable.

        Uses unit-cost moves, so the cost is a whole number and is returned as
        an int (CP-SAT needs integer coefficients).
        """
        key = (start, goal)
        if key not in self._cache:
            self._planner_calls += 1
            result = self._planner(start, goal, self.warehouse)
            self._cache[key] = int(result.total_cost) if result.success else None
        return self._cache[key]

    # ------------------------------------------------------------------
    # Robot-task pricing
    # ------------------------------------------------------------------
    def estimate(self, robot: Robot, task: Task) -> Optional[AssignmentCost]:
        """Price `robot` doing `task`, or None if the pairing is infeasible.

        "Infeasible" here means only "no walkable route exists for one of the
        legs". Payload and battery feasibility are the allocator's job, not the
        cost estimator's - keeping this class about geometry alone.
        """
        to_pickup = self.path_cost(robot.position, task.pickup_location)
        if to_pickup is None:
            return None

        pickup_to_dropoff = self.path_cost(
            task.pickup_location, task.dropoff_location
        )
        if pickup_to_dropoff is None:
            return None

        return AssignmentCost(
            robot_to_pickup=to_pickup, pickup_to_dropoff=pickup_to_dropoff
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    @property
    def planner_calls(self) -> int:
        """How many times the underlying planner actually ran (cache misses)."""
        return self._planner_calls

    @property
    def cache_size(self) -> int:
        return len(self._cache)
