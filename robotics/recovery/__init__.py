"""Dynamic replanning and robot fault recovery.

WHAT THIS PACKAGE ADDS
----------------------
The coordination layer plans conflict-free timed routes for a fleet and drives
them forward one synchronised timestep at a time. If the world changes AFTER
planning - a new obstacle appears, or a robot breaks down - the coordinated
executor stops SAFELY, but it does not repair the plan.

This package turns "stop safely" into "detect, repair, resume":

    initial plan
        |
        v
    coordinated execution
        |
        v
    a disruption event fires  (obstacle appears / robot goes OFFLINE)
        |
        v
    detect which robots / tasks are affected
        |
        v
    release the affected robots' stale future reservations
        |
        +---------------------------+
        |                           |
   obstacle disruption        robot failure
        |                           |
   replan affected route(s)    release the task, reassign it if a
   from CURRENT position       feasible robot exists, then plan its route
        |                           |
        +------------+--------------+
                     |
              re-coordinate the affected subset
                     |
                     v
              resume execution   -- or, if repair is impossible, SAFE STOP

SAFETY FIRST
------------
A disruption must never make the executor knowingly step into a blocked cell or
onto another robot. If recovery cannot produce a conflict-free continuation,
the affected robot stops where it is and the episode reports a safe stop.

DETERMINISTIC AND EXPLAINABLE
----------------------------
No randomness, no LLM in the planning path. Replanning is Space-Time A* from the
robot's current cell; reassignment reuses the existing greedy / CP-SAT
allocator. Everything is reproducible.
"""

from robotics.recovery.disruption import (
    DisruptionEvent,
    DisruptionSchedule,
    DisruptionType,
    DynamicObstacleEvent,
    RecoveryTrigger,
    RobotFailureEvent,
)
from robotics.recovery.dynamic_replanner import DynamicReplanner
from robotics.recovery.recovery_manager import RecoveryManager
from robotics.recovery.recovery_result import (
    RecoveryEventResult,
    ResilientExecutionResult,
    RobotReplan,
)

__all__ = [
    "DisruptionType",
    "RecoveryTrigger",
    "DisruptionEvent",
    "DynamicObstacleEvent",
    "RobotFailureEvent",
    "DisruptionSchedule",
    "DynamicReplanner",
    "RecoveryManager",
    "RobotReplan",
    "RecoveryEventResult",
    "ResilientExecutionResult",
]
