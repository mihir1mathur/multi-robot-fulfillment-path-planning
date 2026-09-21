"""The WarehouseSimulator: the one object that sees the whole world.

RESPONSIBILITY
--------------
`Warehouse` knows the map. `Robot` knows itself. `Task` knows the job.
None of them can answer "is this move legal?", because that question needs
all three at once. The simulator is the coordinator that holds the registries
and enforces the rules that span objects:

    * robot IDs and task IDs are unique
    * a robot may only be placed on a reachable, unoccupied cell
    * two robots may never occupy the same cell
    * a move must respect bounds, obstacles, occupancy and battery
    * a task may only be given to an available robot that can carry it

WHAT IT IS NOT
--------------
It is not a path planner and it does not step time forward. There is no
`tick()`: every move is requested explicitly by the caller, one cell at a time.
A time-stepped loop with simultaneous robot movement introduces head-on swaps,
deadlock and reservation problems, which are a separate body of work and would
be dishonest to half-solve now.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from robotics.exceptions import (
    InvalidMoveError,
    InvalidPositionError,
    SimulationError,
)
from robotics.robots.robot import Robot
from robotics.robots.robot_state import RobotStatus
from robotics.tasks.task import Task, TaskStatus
from robotics.warehouse.grid import Position
from robotics.warehouse.obstacle import Obstacle
from robotics.warehouse.warehouse import Warehouse


class WarehouseSimulator:
    """Coordinates a warehouse, its robots and its tasks."""

    def __init__(self, warehouse: Warehouse) -> None:
        self.warehouse = warehouse

        # Dictionaries keyed by ID give O(1) lookup and make duplicate IDs
        # impossible to miss. Insertion order is preserved by Python dicts,
        # so reports come out in a stable, reproducible order.
        self._robots: Dict[str, Robot] = {}
        self._tasks: Dict[str, Task] = {}

    # ------------------------------------------------------------------
    # Registries
    # ------------------------------------------------------------------
    @property
    def robots(self) -> List[Robot]:
        """All robots, in the order they were added."""
        return list(self._robots.values())

    @property
    def tasks(self) -> List[Task]:
        """All tasks, in the order they were added."""
        return list(self._tasks.values())

    def get_robot(self, robot_id: str) -> Robot:
        """Look up a robot.

        Raises:
            SimulationError: if no robot has that ID.
        """
        if robot_id not in self._robots:
            raise SimulationError(f"No robot with id '{robot_id}'")
        return self._robots[robot_id]

    def get_task(self, task_id: str) -> Task:
        """Look up a task.

        Raises:
            SimulationError: if no task has that ID.
        """
        if task_id not in self._tasks:
            raise SimulationError(f"No task with id '{task_id}'")
        return self._tasks[task_id]

    # ------------------------------------------------------------------
    # Robots
    # ------------------------------------------------------------------
    def add_robot(self, robot: Robot) -> Robot:
        """Register a robot and place it on the map.

        Raises:
            SimulationError: duplicate robot ID, or the starting cell is
                already occupied by another robot.
            InvalidPositionError: the starting cell is outside the warehouse
                or is blocked by an obstacle.
        """
        if robot.robot_id in self._robots:
            raise SimulationError(f"Robot id '{robot.robot_id}' is already registered")

        if not self.warehouse.in_bounds(robot.position):
            raise InvalidPositionError(
                f"Cannot place robot '{robot.robot_id}' at {robot.position}: "
                f"outside the warehouse"
            )
        if not self.warehouse.is_traversable(robot.position):
            raise InvalidPositionError(
                f"Cannot place robot '{robot.robot_id}' at {robot.position}: "
                f"the cell is blocked by an obstacle"
            )

        occupant = self.robot_at(robot.position)
        if occupant is not None:
            raise SimulationError(
                f"Cannot place robot '{robot.robot_id}' at {robot.position}: "
                f"robot '{occupant.robot_id}' is already there"
            )

        self._robots[robot.robot_id] = robot
        return robot

    def remove_robot(self, robot_id: str) -> Robot:
        """Take a robot out of the fleet.

        Any task it was holding is released back to PENDING, so the work is
        not silently lost. (This is the seed of the failure-recovery behaviour
        that automatic fault detection will build on.)

        Raises:
            SimulationError: if no robot has that ID.
        """
        robot = self.get_robot(robot_id)

        if robot.assigned_task_id is not None:
            task = self._tasks.get(robot.assigned_task_id)
            if task is not None and task.is_open:
                if task.status is TaskStatus.IN_PROGRESS:
                    task.fail(f"Robot '{robot_id}' was removed from the fleet")
                else:
                    task.unassign()
            robot.clear_task()

        del self._robots[robot_id]
        return robot

    def robot_at(self, position: Position) -> Optional[Robot]:
        """Return the robot standing on a cell, or None.

        Occupancy is recomputed from the robots themselves on every call
        rather than kept in a cached set. With a small fleet the cost is
        negligible, and it removes a whole class of bugs where the cache and
        the robots' real positions drift apart. If the fleet ever grows large
        enough for this to matter, the cache can be added behind this same
        method without changing any caller.

        NOTE: an OFFLINE robot still occupies its cell. A broken robot does
        not stop being physically present.
        """
        for robot in self._robots.values():
            if robot.position == position:
                return robot
        return None

    def occupied_cells(self) -> Set[Position]:
        """Every cell currently holding a robot."""
        return {robot.position for robot in self._robots.values()}

    # ------------------------------------------------------------------
    # Movement
    # ------------------------------------------------------------------
    def validate_move(self, robot_id: str, destination: Position) -> Optional[str]:
        """Check a single-cell move without performing it.

        Returns:
            None if the move is legal, otherwise a human-readable reason why
            it is not.

        THE ORDER OF THESE CHECKS IS DELIBERATE. Each one is cheaper and more
        fundamental than the next, and each later check assumes the earlier
        ones passed:

            1. Does the robot exist and is it able to move at all?
            2. Is the destination inside the warehouse?      (bounds)
            3. Is it exactly one orthogonal step away?       (adjacency)
            4. Is it permanently blocked?                    (static obstacle)
            5. Is it temporarily blocked?                    (dynamic obstacle)
            6. Is another robot standing there?              (occupancy)
            7. Does the robot have enough battery?           (resources)

        Checking bounds before anything else matters: several of the later
        checks would otherwise be asked about a cell that does not exist.
        """
        robot = self.get_robot(robot_id)

        if robot.status is RobotStatus.OFFLINE:
            return f"Robot '{robot_id}' is offline and cannot move"

        if not self.warehouse.in_bounds(destination):
            return (
                f"Destination {destination} is outside the warehouse "
                f"({self.warehouse.height} rows x {self.warehouse.width} cols)"
            )

        if not robot.position.is_adjacent_to(destination):
            return (
                f"Destination {destination} is not adjacent to the robot's "
                f"current cell {robot.position}; only one UP/DOWN/LEFT/RIGHT "
                f"step is allowed"
            )

        if self.warehouse.is_blocked_by_static(destination):
            return f"Destination {destination} is blocked by a static obstacle"

        if self.warehouse.is_blocked_by_dynamic(destination):
            return f"Destination {destination} is blocked by a dynamic obstacle"

        occupant = self.robot_at(destination)
        if occupant is not None:
            return (
                f"Destination {destination} is already occupied by robot "
                f"'{occupant.robot_id}'"
            )

        if not robot.has_battery_for_move():
            return (
                f"Robot '{robot_id}' has {robot.battery_level:.1f}% battery, "
                f"which is not enough for a move costing "
                f"{robot.battery_drain_per_move:.1f}%"
            )

        return None

    def move_robot(self, robot_id: str, destination: Position) -> Robot:
        """Move a robot one cell, if that move is legal.

        This is NOT navigation. The caller supplies the next cell; the
        simulator only decides whether that single step is allowed. Automatic
        route finding (A*, Dijkstra) is later work.

        On success the robot's position, step count, distance and battery are
        updated. On failure nothing changes except the robot's `last_error`,
        which records the reason for diagnostics.

        Raises:
            InvalidMoveError: with the specific reason the move was rejected.
        """
        robot = self.get_robot(robot_id)

        reason = self.validate_move(robot_id, destination)
        if reason is not None:
            robot.last_error = reason
            raise InvalidMoveError(reason)

        # The robot re-checks adjacency and battery itself. That duplication is
        # intentional: Robot.move_to must be safe to call directly (in unit
        # tests, or by a future component), so it cannot rely on a caller
        # having validated anything.
        robot.move_to(destination)
        return robot

    # ------------------------------------------------------------------
    # Battery
    # ------------------------------------------------------------------
    def charge_robot(self, robot_id: str, amount: float = 100.0) -> float:
        """Recharge a robot that is standing on a charging station.

        The location rule lives here rather than in `Robot` because only the
        simulator can see both the robot and the map.

        Returns:
            The percentage points actually added.

        Raises:
            SimulationError: the robot is not on a charging station.
        """
        robot = self.get_robot(robot_id)

        if not self.warehouse.is_charging_station(robot.position):
            raise SimulationError(
                f"Robot '{robot_id}' cannot charge at {robot.position}: "
                f"that cell is not a charging station"
            )

        robot.set_status(RobotStatus.CHARGING)
        return robot.recharge(amount)

    # ------------------------------------------------------------------
    # Tasks
    # ------------------------------------------------------------------
    def add_task(self, task: Task) -> Task:
        """Register a fulfillment task.

        Both endpoints are validated against the map, because a task that
        points at a wall can never be completed and should be rejected as
        early as possible - at creation, not halfway through execution.

        Raises:
            SimulationError: duplicate task ID.
            InvalidPositionError: an endpoint is outside the warehouse or is
                permanently blocked.
        """
        if task.task_id in self._tasks:
            raise SimulationError(f"Task id '{task.task_id}' is already registered")

        for label, position in (
            ("pickup", task.pickup_location),
            ("dropoff", task.dropoff_location),
        ):
            if not self.warehouse.in_bounds(position):
                raise InvalidPositionError(
                    f"Task '{task.task_id}' {label} location {position} is "
                    f"outside the warehouse"
                )
            if self.warehouse.is_blocked_by_static(position):
                raise InvalidPositionError(
                    f"Task '{task.task_id}' {label} location {position} is "
                    f"permanently blocked and can never be reached"
                )

        self._tasks[task.task_id] = task
        return task

    def assign_task_manually(self, task_id: str, robot_id: str) -> Task:
        """Give a specific task to a specific robot.

        "Manually" is the important word: a human (or a test, or the demo)
        chooses the pairing. Choosing pairings automatically so that the whole
        fleet finishes as fast as possible is an optimisation problem and is
        explicitly later work.

        Raises:
            SimulationError: the task is not PENDING, the robot is not
                available, or the load is heavier than the robot's capacity.
        """
        task = self.get_task(task_id)
        robot = self.get_robot(robot_id)

        if task.status is not TaskStatus.PENDING:
            raise SimulationError(
                f"Task '{task_id}' cannot be assigned: its status is "
                f"{task.status.value}, expected pending"
            )
        if not robot.is_available:
            raise SimulationError(
                f"Robot '{robot_id}' is not available (status={robot.status}, "
                f"assigned_task_id={robot.assigned_task_id})"
            )
        if not robot.can_carry(task.payload_weight):
            raise SimulationError(
                f"Robot '{robot_id}' cannot carry task '{task_id}': "
                f"{task.payload_weight}kg exceeds the remaining capacity of "
                f"{robot.remaining_capacity}kg"
            )

        # Order matters: the robot is updated first because its `assign_task`
        # performs the last availability check and may raise. If it does, the
        # task is left untouched and the two objects cannot disagree.
        robot.assign_task(task.task_id)
        task.assign_to(robot.robot_id)
        return task

    def pending_tasks(self) -> List[Task]:
        """Tasks nobody is working on yet, most urgent first.

        Ties are broken by task ID so the order is fully deterministic.
        """
        pending = [t for t in self._tasks.values() if t.status is TaskStatus.PENDING]
        return sorted(pending, key=lambda t: (-int(t.priority), t.task_id))

    # ------------------------------------------------------------------
    # Obstacles
    # ------------------------------------------------------------------
    def add_dynamic_obstacle(
        self, obstacle_id: str, position: Position, description: str = ""
    ) -> Obstacle:
        """Place a temporary obstacle (spill, cart, cordon).

        Raises:
            SimulationError: a robot is standing on that cell. Dropping a box
                onto an occupied cell is not something the model can
                represent, so it is rejected rather than silently allowed.
            InvalidPositionError: the cell is out of bounds or already
                permanently blocked.
        """
        occupant = self.robot_at(position)
        if occupant is not None:
            raise SimulationError(
                f"Cannot place dynamic obstacle '{obstacle_id}' at {position}: "
                f"robot '{occupant.robot_id}' is standing there"
            )
        return self.warehouse.add_dynamic_obstacle(obstacle_id, position, description)

    def remove_dynamic_obstacle(self, obstacle_id: str) -> Obstacle:
        """Clear a temporary obstacle.

        Raises:
            SimulationError: no dynamic obstacle with that ID exists.
        """
        return self.warehouse.remove_dynamic_obstacle(obstacle_id)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def get_current_state(self) -> Dict[str, Any]:
        """A complete, plain-data snapshot of the simulation.

        Returning dictionaries rather than objects means this can be printed,
        written to a JSON log, compared in a test, or served by an API later
        without any of those callers depending on the simulator's classes.
        """
        return {
            "warehouse": {
                "name": self.warehouse.name,
                "rows": self.warehouse.height,
                "cols": self.warehouse.width,
                "storage_cells": len(self.warehouse.storage_cells),
                "pickup_locations": [(p.row, p.col) for p in self.warehouse.pickup_locations],
                "dropoff_locations": [
                    (p.row, p.col) for p in self.warehouse.dropoff_locations
                ],
                "charging_stations": [
                    (p.row, p.col) for p in self.warehouse.charging_stations
                ],
                "static_obstacles": [
                    (o.position.row, o.position.col) for o in self.warehouse.static_obstacles
                ],
                "dynamic_obstacles": [
                    {
                        "obstacle_id": o.obstacle_id,
                        "position": (o.position.row, o.position.col),
                        "description": o.description,
                    }
                    for o in self.warehouse.dynamic_obstacles
                ],
            },
            "robots": [robot.to_dict() for robot in self._robots.values()],
            "tasks": [task.to_dict() for task in self._tasks.values()],
        }

    def __repr__(self) -> str:
        return (
            f"WarehouseSimulator(warehouse='{self.warehouse.name}', "
            f"robots={len(self._robots)}, tasks={len(self._tasks)})"
        )
