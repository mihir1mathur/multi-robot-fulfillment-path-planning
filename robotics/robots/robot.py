"""The Robot model.

A `Robot` is a small bag of state plus the rules for changing that state
safely. It is deliberately IGNORANT of the warehouse:

    A robot knows where IT is.
    It does NOT know where the walls, the other robots or the shelves are.

WHY THAT SEPARATION?
--------------------
If `Robot` held a reference to the `Warehouse`, every robot could silently
read and modify the shared map, and a bug in one robot could corrupt the world
for all the others. Instead the `WarehouseSimulator` is the single object
allowed to compare a robot against the world. The robot only enforces the
rules it can check on its own (am I being asked to teleport? do I have enough
battery?), and does its own bookkeeping.

This is the same separation you would want if each robot later became a
separate process talking over a network: the robot owns its own state, the
coordinator owns the shared world.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from robotics.exceptions import InvalidMoveError, SimulationError
from robotics.robots.robot_state import AVAILABLE_STATUSES, RobotStatus
from robotics.warehouse.grid import MOVE_COST, Position

# Battery is modelled as a simple percentage from 0 to 100.
MIN_BATTERY = 0.0
MAX_BATTERY = 100.0

# Default battery cost of one orthogonal move, in percentage points.
# ASSUMPTION: this is a made-up, deliberately simple number. It is NOT based on
# any real robot's power draw. Its only purpose is to give the features built
# on top of it (battery-aware task allocation, failure recovery) a value that
# changes as robots work. A real system would measure joules, not
# percent-per-cell.
DEFAULT_BATTERY_DRAIN_PER_MOVE = 1.0


@dataclass
class Robot:
    """One autonomous mobile robot in the fleet.

    Attributes:
        robot_id: unique name, e.g. "R1". Used everywhere as the robot's key.
        position: the cell the robot currently occupies, as (row, col).
        battery_level: remaining charge as a percentage, 0-100.
        payload_capacity: heaviest load the robot may carry, in kilograms.
        current_payload: weight currently carried, in kilograms.
        status: what the robot is doing (see RobotStatus).
        assigned_task_id: the task this robot is responsible for, or None.
        distance_travelled: total grid distance covered since creation. With
            uniform cost-1 moves this equals steps_taken, but it is kept
            separate because turn costs or heavier terrain would make the two
            numbers diverge.
        steps_taken: number of successful single-cell moves.
        battery_drain_per_move: percentage points consumed per successful move.
    """

    robot_id: str
    position: Position
    battery_level: float = MAX_BATTERY
    payload_capacity: float = 10.0
    current_payload: float = 0.0
    status: RobotStatus = RobotStatus.IDLE
    assigned_task_id: Optional[str] = None
    distance_travelled: float = 0.0
    steps_taken: int = 0
    battery_drain_per_move: float = DEFAULT_BATTERY_DRAIN_PER_MOVE
    # `field` with compare=False keeps this out of equality checks; it is
    # diagnostic information, not part of the robot's identity.
    last_error: Optional[str] = field(default=None, compare=False)

    def __post_init__(self) -> None:
        """Reject impossible robots at construction time.

        Validating here (rather than trusting the caller) means every Robot
        that exists anywhere in the program is already known to be sane.
        """
        if not self.robot_id:
            raise ValueError("robot_id must be a non-empty string")
        if self.payload_capacity < 0:
            raise ValueError(
                f"payload_capacity must be >= 0, got {self.payload_capacity}"
            )
        if not 0 <= self.current_payload <= self.payload_capacity:
            raise ValueError(
                f"current_payload {self.current_payload} must be between 0 and "
                f"payload_capacity {self.payload_capacity}"
            )
        if self.battery_drain_per_move < 0:
            raise ValueError("battery_drain_per_move must be >= 0")
        if not MIN_BATTERY <= self.battery_level <= MAX_BATTERY:
            raise ValueError(
                f"battery_level must be between {MIN_BATTERY} and {MAX_BATTERY}, "
                f"got {self.battery_level}"
            )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    @property
    def is_available(self) -> bool:
        """True if the robot can be given a new task right now."""
        return self.status in AVAILABLE_STATUSES and self.assigned_task_id is None

    @property
    def remaining_capacity(self) -> float:
        """How much more weight the robot could still take on."""
        return self.payload_capacity - self.current_payload

    def has_battery_for_move(self) -> bool:
        """True if one more move would not push the battery below zero."""
        return self.battery_level >= self.battery_drain_per_move

    def can_carry(self, weight: float) -> bool:
        """True if adding `weight` would stay within the payload capacity."""
        return weight <= self.remaining_capacity

    # ------------------------------------------------------------------
    # Movement
    # ------------------------------------------------------------------
    def move_to(self, destination: Position) -> None:
        """Move exactly one orthogonal cell and update the robot's bookkeeping.

        This is NOT path planning. The caller (the simulator) decides where the
        robot should step next; this method only performs the two checks a
        robot can make from its own knowledge, then updates its own numbers.

        Checks performed here:
            1. The destination is exactly one orthogonal step away
               (no teleporting, no diagonals).
            2. The battery can pay for the move.

        Checks performed by the simulator BEFORE calling this:
            bounds, static obstacles, dynamic obstacles, other robots.

        Raises:
            InvalidMoveError: if either check fails. The robot's state is left
                completely unchanged when a move is rejected.
        """
        if not self.position.is_adjacent_to(destination):
            raise InvalidMoveError(
                f"Robot '{self.robot_id}' cannot move from {self.position} to "
                f"{destination}: destination is not an adjacent cell "
                f"(only UP/DOWN/LEFT/RIGHT by one cell is allowed)"
            )

        if not self.has_battery_for_move():
            raise InvalidMoveError(
                f"Robot '{self.robot_id}' cannot move: battery is "
                f"{self.battery_level:.1f}% but a move costs "
                f"{self.battery_drain_per_move:.1f}%"
            )

        self.position = destination
        self.steps_taken += 1
        self.distance_travelled += MOVE_COST
        self._drain_battery(self.battery_drain_per_move)
        self.status = RobotStatus.MOVING
        self.last_error = None

    # ------------------------------------------------------------------
    # Battery
    # ------------------------------------------------------------------
    def recharge(self, amount: float = MAX_BATTERY) -> float:
        """Add charge, never exceeding 100%.

        The CALLER is responsible for checking that the robot is standing on a
        charging station - the robot itself cannot see the map. The simulator
        enforces that rule in `WarehouseSimulator.charge_robot`.

        Args:
            amount: percentage points to add. Defaults to a full charge.

        Returns:
            The number of percentage points actually added (may be less than
            requested because the battery saturates at 100%).

        Raises:
            ValueError: if `amount` is negative.
        """
        if amount < 0:
            raise ValueError(f"Recharge amount must be >= 0, got {amount}")

        before = self.battery_level
        self.battery_level = min(MAX_BATTERY, self.battery_level + amount)
        return self.battery_level - before

    def _drain_battery(self, amount: float) -> None:
        """Consume charge, clamped so the battery can never go negative.

        Clamping rather than raising keeps the invariant "0 <= battery <= 100"
        true no matter what the caller does. A robot that hits exactly 0 simply
        stops being able to move, which `has_battery_for_move` reports.
        """
        self.battery_level = max(MIN_BATTERY, self.battery_level - amount)

    # ------------------------------------------------------------------
    # Payload
    # ------------------------------------------------------------------
    def load(self, weight: float) -> None:
        """Take on cargo.

        Raises:
            ValueError: negative weight.
            SimulationError: the load would exceed the payload capacity.
        """
        if weight < 0:
            raise ValueError(f"Load weight must be >= 0, got {weight}")
        if not self.can_carry(weight):
            raise SimulationError(
                f"Robot '{self.robot_id}' cannot carry {weight}kg: "
                f"capacity {self.payload_capacity}kg, already carrying "
                f"{self.current_payload}kg"
            )
        self.current_payload += weight

    def unload(self) -> float:
        """Put down everything the robot is carrying and return that weight."""
        unloaded = self.current_payload
        self.current_payload = 0.0
        return unloaded

    # ------------------------------------------------------------------
    # Status and task bookkeeping
    # ------------------------------------------------------------------
    def set_status(self, status: RobotStatus) -> None:
        """Change the robot's status.

        This deliberately does not police which transitions are legal. A full
        state machine would be premature: we do not yet have the behaviours
        (picking, charging cycles, fault recovery) that the transitions would
        describe, and an invented rule set would only have to be rewritten.
        The states that DO have rules today - assignment and movement - are
        enforced by the dedicated methods above and below.
        """
        if not isinstance(status, RobotStatus):
            raise TypeError(f"status must be a RobotStatus, got {type(status).__name__}")
        self.status = status

    def assign_task(self, task_id: str) -> None:
        """Attach a task to this robot and mark it ASSIGNED.

        Raises:
            SimulationError: the robot is not available (busy, blocked,
                offline, or already holding a task).
        """
        if not self.is_available:
            raise SimulationError(
                f"Robot '{self.robot_id}' is not available for a new task "
                f"(status={self.status}, assigned_task_id={self.assigned_task_id})"
            )
        self.assigned_task_id = task_id
        self.status = RobotStatus.ASSIGNED

    def clear_task(self) -> None:
        """Detach the current task and return the robot to IDLE."""
        self.assigned_task_id = None
        self.status = RobotStatus.IDLE

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """A plain-data snapshot, for printing, logging and future APIs."""
        return {
            "robot_id": self.robot_id,
            "position": (self.position.row, self.position.col),
            "status": self.status.value,
            "battery_level": round(self.battery_level, 2),
            "payload_capacity": self.payload_capacity,
            "current_payload": self.current_payload,
            "assigned_task_id": self.assigned_task_id,
            "steps_taken": self.steps_taken,
            "distance_travelled": self.distance_travelled,
        }

    def __str__(self) -> str:
        return (
            f"Robot {self.robot_id} at {self.position} | {self.status} | "
            f"battery {self.battery_level:.1f}% | "
            f"payload {self.current_payload}/{self.payload_capacity}kg | "
            f"steps {self.steps_taken}"
        )
