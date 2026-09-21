"""Fulfillment tasks.

WHAT IS A FULFILLMENT TASK?
---------------------------
A customer orders an item. Somewhere in the warehouse that item sits on a
shelf. Somebody has to fetch it and bring it to the packing station so it can
be boxed and shipped. That single unit of work - "collect from cell A, deliver
to cell B" - is called a Task.

    PICKUP cell  --------- robot carries the item --------->  DROPOFF cell

A task is pure DATA plus a lifecycle. It does not know about the warehouse
map, it does not choose a robot, and it does not plan a route. That keeps it
easy to store in a database later (Task maps almost one-to-one onto a table
row) and easy to hand to an optimiser.

CURRENT SCOPE
-------------
Tasks are created and assigned MANUALLY, to demonstrate and test the model.
Automatic assignment - deciding which robot should do which task so that the
whole fleet finishes fastest - is a constraint-optimisation problem and is
explicitly future work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Dict, Optional, Set

from robotics.exceptions import TaskError
from robotics.warehouse.grid import Position


class TaskPriority(IntEnum):
    """How urgent a task is.

    An IntEnum (rather than a bare int) removes a classic ambiguity: does
    "priority 1" mean most urgent or least urgent? Here the numbers are
    ordered so that a LARGER value is MORE urgent, and the names say so out
    loud. Because it is an IntEnum, tasks can be sorted directly:

        sorted(tasks, key=lambda t: t.priority, reverse=True)
    """

    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4

    def __str__(self) -> str:
        return self.name.lower()


class TaskStatus(Enum):
    """Where a task is in its life."""

    PENDING = "pending"
    """Created, not yet given to any robot. Waiting in the queue."""

    ASSIGNED = "assigned"
    """A specific robot is responsible for it, but work has not begun."""

    IN_PROGRESS = "in_progress"
    """The robot has started: it is travelling, picking or delivering."""

    COMPLETED = "completed"
    """The item reached the drop-off location. Terminal state."""

    FAILED = "failed"
    """The task could not be finished (e.g. the robot went offline).
    Terminal state; recovery logic may re-create the work as a fresh task."""

    def __str__(self) -> str:
        return self.value


# The only status changes allowed. Writing them down as data (instead of
# scattering `if` statements through the code) means the rules can be read,
# tested and extended in one place - and it makes an illegal transition such
# as COMPLETED -> PENDING impossible rather than merely unlikely.
_ALLOWED_TRANSITIONS: Dict[TaskStatus, Set[TaskStatus]] = {
    TaskStatus.PENDING: {TaskStatus.ASSIGNED, TaskStatus.FAILED},
    TaskStatus.ASSIGNED: {TaskStatus.IN_PROGRESS, TaskStatus.PENDING, TaskStatus.FAILED},
    TaskStatus.IN_PROGRESS: {TaskStatus.COMPLETED, TaskStatus.FAILED},
    TaskStatus.COMPLETED: set(),
    TaskStatus.FAILED: set(),
}


@dataclass
class Task:
    """One pickup-and-deliver job.

    Attributes:
        task_id: unique name, e.g. "T1".
        pickup_location: cell where the item is collected.
        dropoff_location: cell where the item must be delivered.
        priority: how urgent the task is (see TaskPriority).
        payload_weight: weight of the item in kilograms. Used to check that a
            robot's payload capacity is large enough.
        status: lifecycle state (see TaskStatus).
        assigned_robot_id: the robot responsible, or None while PENDING.
        failure_reason: why the task failed, if it did.
    """

    task_id: str
    pickup_location: Position
    dropoff_location: Position
    priority: TaskPriority = TaskPriority.NORMAL
    payload_weight: float = 1.0
    status: TaskStatus = TaskStatus.PENDING
    assigned_robot_id: Optional[str] = None
    failure_reason: Optional[str] = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError("task_id must be a non-empty string")
        if self.payload_weight <= 0:
            raise ValueError(
                f"payload_weight must be greater than 0, got {self.payload_weight}"
            )
        if self.pickup_location == self.dropoff_location:
            # A task whose pickup equals its drop-off is almost certainly a
            # configuration mistake, and it would make later distance and
            # cost calculations meaningless.
            raise ValueError(
                f"Task '{self.task_id}': pickup and dropoff cannot be the same "
                f"cell {self.pickup_location}"
            )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    @property
    def is_open(self) -> bool:
        """True while the task still needs work (not COMPLETED or FAILED)."""
        return self.status not in (TaskStatus.COMPLETED, TaskStatus.FAILED)

    @property
    def manhattan_length(self) -> int:
        """Shortest possible number of moves from pickup to drop-off.

        This ignores obstacles entirely, so it is a LOWER BOUND on the real
        travel cost, never the answer. It is useful today only as a rough
        sanity figure in reports; a path planner computes the true cost.
        """
        return self.pickup_location.manhattan_distance_to(self.dropoff_location)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def assign_to(self, robot_id: str) -> None:
        """Mark the task as the responsibility of a specific robot."""
        self._transition_to(TaskStatus.ASSIGNED)
        self.assigned_robot_id = robot_id

    def unassign(self) -> None:
        """Return the task to the pending queue (e.g. its robot went offline)."""
        self._transition_to(TaskStatus.PENDING)
        self.assigned_robot_id = None

    def start(self) -> None:
        """Mark work as begun. The task must already be assigned to a robot."""
        if self.assigned_robot_id is None:
            raise TaskError(
                f"Task '{self.task_id}' cannot start: no robot is assigned to it"
            )
        self._transition_to(TaskStatus.IN_PROGRESS)

    def complete(self) -> None:
        """Mark the item as delivered. Terminal."""
        self._transition_to(TaskStatus.COMPLETED)

    def fail(self, reason: str) -> None:
        """Mark the task as impossible to finish, recording why. Terminal."""
        self._transition_to(TaskStatus.FAILED)
        self.failure_reason = reason

    def _transition_to(self, new_status: TaskStatus) -> None:
        """Apply a status change, refusing anything the lifecycle forbids.

        Raises:
            TaskError: if the transition is not in `_ALLOWED_TRANSITIONS`.
        """
        allowed = _ALLOWED_TRANSITIONS[self.status]
        if new_status not in allowed:
            allowed_names = ", ".join(sorted(s.value for s in allowed)) or "nothing"
            raise TaskError(
                f"Task '{self.task_id}' cannot go from {self.status.value} to "
                f"{new_status.value} (allowed from {self.status.value}: {allowed_names})"
            )
        self.status = new_status

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """A plain-data snapshot, for printing, logging and future APIs."""
        return {
            "task_id": self.task_id,
            "pickup_location": (self.pickup_location.row, self.pickup_location.col),
            "dropoff_location": (self.dropoff_location.row, self.dropoff_location.col),
            "priority": self.priority.name.lower(),
            "payload_weight": self.payload_weight,
            "status": self.status.value,
            "assigned_robot_id": self.assigned_robot_id,
        }

    def __str__(self) -> str:
        robot = self.assigned_robot_id or "-"
        return (
            f"Task {self.task_id} {self.pickup_location} -> {self.dropoff_location} | "
            f"{self.status} | priority {self.priority} | "
            f"{self.payload_weight}kg | robot {robot}"
        )
