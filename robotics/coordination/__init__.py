"""Multi-robot coordination: conflict-free routes through shared space.

THE PROBLEM
-----------
The path planner (`robotics.planning`) routes ONE robot at a time and ignores
where the other robots are. Two such routes can collide:

    * two robots want the SAME cell at the SAME moment      (vertex conflict)
    * two robots try to SWAP cells in one step, head-on      (edge / swap conflict)

The route executor (`robotics.execution`) already refuses a step onto an
occupied cell, so nothing physically overlaps - but a robot whose way is
blocked simply STOPS. On the allocation benchmark that happened to ~8% of the
robots driven to their pickups.

THE FIX IN THIS PACKAGE
-----------------------
Plan in SPACE **and** TIME. A route becomes a list of (cell, timestep) pairs,
so "R1 is at (2,3) at t=4" is a precise statement another robot's plan can be
made to respect.

    ReservationTable   remembers which (cell, timestep) and which (edge, timestep)
                       are already taken
    SpaceTimePlanner   A* over (cell, timestep) states, with a WAIT action,
                       that never plans onto a reserved cell or across a
                       reserved edge
    MultiRobotCoordinator
                       plans the robots one at a time in a fixed priority order,
                       reserving each robot's timed path before planning the next
    CoordinatedExecutor
                       drives the robots forward one synchronised timestep at a
                       time, validating the whole joint step before moving anyone

WHAT THIS IS NOT
----------------
This is PRIORITIZED planning, not globally optimal Multi-Agent Path Finding.
It is simple, deterministic and explainable. Its weakness: the priority order
can matter - an early robot can box a later one in, and a plan can fail even
when some other ordering would have worked. That trade-off is documented.

It also does NOT replan. If an obstacle appears AFTER coordination, execution
detects it and stops safely; repairing the plan is a later capability.
"""

from robotics.coordination.conflicts import (
    ConflictValidator,
    find_coordination_problems,
    is_conflict_free,
)
from robotics.coordination.coordination_result import CoordinationResult
from robotics.coordination.coordinated_executor import (
    CoordinatedExecutionResult,
    CoordinatedExecutor,
)
from robotics.coordination.coordinator import MultiRobotCoordinator, default_horizon
from robotics.coordination.reservation_table import ReservationTable
from robotics.coordination.space_time_planner import SpaceTimePlanner
from robotics.coordination.timed_path import TimedPath, TimedStep

__all__ = [
    "TimedStep",
    "TimedPath",
    "ReservationTable",
    "SpaceTimePlanner",
    "MultiRobotCoordinator",
    "default_horizon",
    "CoordinationResult",
    "CoordinatedExecutor",
    "CoordinatedExecutionResult",
    "ConflictValidator",
    "find_coordination_problems",
    "is_conflict_free",
]
