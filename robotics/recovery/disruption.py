"""Disruption events: the things that go wrong AFTER planning.

Two event classes are supported:

    DynamicObstacleEvent   a cell becomes blocked at a given timestep
    RobotFailureEvent      a robot goes OFFLINE at a given timestep

Events are deterministic - they carry a fixed timestep and target - so a test
or the benchmark can reproduce exactly the same disruption every run. A
`DisruptionSchedule` is just a sorted, queryable list of them.

There is deliberately no message bus / broker here (Kafka, Redis, Celery). A
disruption is a small immutable value the resilient executor checks for at the
start of each timestep.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Sequence, Union

from robotics.warehouse.grid import Position


class DisruptionType(Enum):
    """What kind of thing went wrong."""

    DYNAMIC_OBSTACLE = "dynamic_obstacle"
    ROBOT_OFFLINE = "robot_offline"


class RecoveryTrigger(Enum):
    """Why a recovery attempt was started.

    A single disruption can raise more than one trigger: a robot failure
    (`ROBOT_OFFLINE`) also invalidates any route that ran through the failed
    robot's cell (`ROUTE_INVALIDATED`) and any reservation it still held
    (`RESERVATION_INVALIDATED`).
    """

    DYNAMIC_OBSTACLE = "dynamic_obstacle"
    ROBOT_OFFLINE = "robot_offline"
    ROUTE_INVALIDATED = "route_invalidated"
    RESERVATION_INVALIDATED = "reservation_invalidated"


@dataclass(frozen=True)
class DynamicObstacleEvent:
    """A new obstacle appears on `position` at `timestep`."""

    timestep: int
    position: Position
    obstacle_id: str = ""

    @property
    def kind(self) -> DisruptionType:
        return DisruptionType.DYNAMIC_OBSTACLE

    def resolved_obstacle_id(self) -> str:
        if self.obstacle_id:
            return self.obstacle_id
        return f"disruption-{self.position.row}-{self.position.col}-t{self.timestep}"

    def __str__(self) -> str:
        return f"obstacle at {self.position} @ t{self.timestep}"


@dataclass(frozen=True)
class RobotFailureEvent:
    """`robot_id` goes OFFLINE at `timestep` (it stops where it is)."""

    timestep: int
    robot_id: str

    @property
    def kind(self) -> DisruptionType:
        return DisruptionType.ROBOT_OFFLINE

    def __str__(self) -> str:
        return f"robot {self.robot_id} fails @ t{self.timestep}"


DisruptionEvent = Union[DynamicObstacleEvent, RobotFailureEvent]


@dataclass
class DisruptionSchedule:
    """A deterministic, ordered set of disruption events.

    Events are stored sorted by (timestep, a stable secondary key) so
    `events_at` returns them in a fixed order.
    """

    events: List[DisruptionEvent] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.events = sorted(self.events, key=self._sort_key)

    @staticmethod
    def _sort_key(event: DisruptionEvent):
        if isinstance(event, RobotFailureEvent):
            return (event.timestep, 0, event.robot_id)
        return (event.timestep, 1, (event.position.row, event.position.col))

    def add(self, event: DisruptionEvent) -> None:
        self.events.append(event)
        self.events.sort(key=self._sort_key)

    def events_at(self, timestep: int) -> List[DisruptionEvent]:
        """Every event scheduled for exactly `timestep`."""
        return [e for e in self.events if e.timestep == timestep]

    @property
    def last_timestep(self) -> int:
        return max((e.timestep for e in self.events), default=0)

    def counts(self) -> Dict[str, int]:
        return {
            "dynamic_obstacle": sum(
                1 for e in self.events if e.kind is DisruptionType.DYNAMIC_OBSTACLE
            ),
            "robot_offline": sum(
                1 for e in self.events if e.kind is DisruptionType.ROBOT_OFFLINE
            ),
            "total": len(self.events),
        }

    @classmethod
    def of(cls, *events: DisruptionEvent) -> "DisruptionSchedule":
        return cls(list(events))

    def __len__(self) -> int:
        return len(self.events)

    def __iter__(self):
        return iter(self.events)
