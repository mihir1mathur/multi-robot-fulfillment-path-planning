"""A route through space AND time.

WHY A NEW TYPE (and not just re-using PathResult)?
-------------------------------------------------
`PathResult` answers "which cells, in which order?" - a purely SPATIAL answer.
It is exactly right for single-robot planning and stays unchanged.

Coordination needs one more axis: WHEN. "R1 is at (2,3)" is not enough to keep
robots apart; "R1 is at (2,3) at timestep 4" is. So a `TimedPath` is a list of
`(cell, timestep)` pairs.

THE TIMESTEP CONVENTION (fixed here, used everywhere)
----------------------------------------------------
* Timesteps are integers. Step i of the path is at time `start_time + i`.
* Consecutive steps are one timestep apart: t, t+1, t+2, ...
* A step to the SAME cell as the previous step is a WAIT (the robot stayed put
  for one timestep).
* A step to an ADJACENT cell is a MOVE.

    [ ((0,0), 0), ((0,1), 1), ((0,1), 2), ((0,2), 3) ]
        move        move        WAIT        move
    start_time 0, arrival_time 3, move_count 3, wait_count 1

`start_time` is usually 0 (all robots start together), but the type allows a
later start so a robot could be told to hold at its start cell first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from robotics.warehouse.grid import Position


@dataclass(frozen=True)
class TimedStep:
    """One (cell, timestep) pair on a robot's timed route."""

    position: Position
    timestep: int

    def __str__(self) -> str:
        return f"t{self.timestep}:{self.position}"


@dataclass
class TimedPath:
    """One robot's conflict-free route through space and time.

    Attributes:
        robot_id: the robot this route is for.
        steps: the (cell, timestep) pairs, start to goal inclusive. Empty when
            planning failed.
        success: True if a conflict-free route to the goal was found.
        failure_reason: why planning failed, or None on success.

    Derived (see properties): start_time, arrival_time, move_count, wait_count,
    makespan.
    """

    robot_id: str
    steps: List[TimedStep] = field(default_factory=list)
    success: bool = False
    failure_reason: Optional[str] = None

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def found(cls, robot_id: str, steps: List[TimedStep]) -> "TimedPath":
        """Build a successful timed path. Steps must be time-consecutive."""
        return cls(robot_id=robot_id, steps=list(steps), success=True)

    @classmethod
    def failed(cls, robot_id: str, reason: str) -> "TimedPath":
        """Build a failed timed path carrying the reason."""
        return cls(robot_id=robot_id, steps=[], success=False, failure_reason=reason)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    @property
    def start(self) -> Optional[Position]:
        return self.steps[0].position if self.steps else None

    @property
    def goal(self) -> Optional[Position]:
        return self.steps[-1].position if self.steps else None

    @property
    def start_time(self) -> int:
        return self.steps[0].timestep if self.steps else 0

    @property
    def arrival_time(self) -> int:
        """The timestep at which the robot reaches (and stays at) the goal."""
        return self.steps[-1].timestep if self.steps else 0

    @property
    def move_count(self) -> int:
        """Steps where the robot changed cell."""
        return sum(
            1
            for a, b in zip(self.steps, self.steps[1:])
            if a.position != b.position
        )

    @property
    def wait_count(self) -> int:
        """Steps where the robot stayed in the same cell (WAIT actions)."""
        return sum(
            1
            for a, b in zip(self.steps, self.steps[1:])
            if a.position == b.position
        )

    @property
    def makespan(self) -> int:
        """How many timesteps this route spans (arrival_time - start_time)."""
        return self.arrival_time - self.start_time

    def position_at(self, timestep: int) -> Optional[Position]:
        """Where the robot is at `timestep`.

        Before start_time -> the start cell. After arrival_time -> the goal cell
        (the robot has arrived and holds its goal). Returns None only for an
        unsuccessful path.
        """
        if not self.steps:
            return None
        if timestep <= self.start_time:
            return self.steps[0].position
        if timestep >= self.arrival_time:
            return self.steps[-1].position
        return self.steps[timestep - self.start_time].position

    def is_wait_at(self, index: int) -> bool:
        """True if step `index` is a WAIT (same cell as step index-1)."""
        if index <= 0 or index >= len(self.steps):
            return False
        return self.steps[index].position == self.steps[index - 1].position

    def positions(self) -> List[Position]:
        """Just the cells, in order (start to goal, WAITs repeat a cell)."""
        return [step.position for step in self.steps]

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "robot_id": self.robot_id,
            "success": self.success,
            "steps": [[s.position.row, s.position.col, s.timestep] for s in self.steps],
            "start_time": self.start_time,
            "arrival_time": self.arrival_time,
            "move_count": self.move_count,
            "wait_count": self.wait_count,
            "makespan": self.makespan,
            "failure_reason": self.failure_reason,
        }

    def __str__(self) -> str:
        if not self.success:
            return f"{self.robot_id}: NO TIMED PATH ({self.failure_reason})"
        return (
            f"{self.robot_id}: {self.start} -> {self.goal} | "
            f"arrive t{self.arrival_time} | {self.move_count} moves, "
            f"{self.wait_count} waits"
        )
