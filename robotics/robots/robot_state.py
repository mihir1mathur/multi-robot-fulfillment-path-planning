"""Robot status values.

WHY AN ENUM INSTEAD OF STRINGS?
-------------------------------
If status were a plain string, `robot.status == "idel"` would silently be
False forever and nobody would notice. With an enum, a typo like
`RobotStatus.IDEL` fails immediately with an AttributeError, and editors can
autocomplete the valid values. Enums turn a whole class of runtime bugs into
instant, obvious errors.

WHAT "STATE" MEANS HERE
-----------------------
"State" is simply the answer to "what is this robot doing right now?".
Higher layers make decisions from it: a task allocator will only consider
robots that are IDLE, and failure-recovery logic will look for robots that are
BLOCKED or OFFLINE.

HONESTY NOTE
------------
The code as it stands only ever produces IDLE, ASSIGNED, MOVING, CHARGING,
BLOCKED and OFFLINE. PICKING and DROPPING are declared because the task
lifecycle needs them, but nothing currently drives a robot into them - the
pick/drop execution loop is future work.
"""

from enum import Enum


class RobotStatus(Enum):
    """What a robot is doing at this instant."""

    IDLE = "idle"
    """Powered on, free, waiting for work. The only state in which a robot may
    be given a new task."""

    ASSIGNED = "assigned"
    """A task has been given to the robot but it has not started moving yet."""

    MOVING = "moving"
    """The robot is driving between cells. Set after a successful single-cell
    move."""

    PICKING = "picking"
    """The robot is at the pickup location and is loading an item.
    Declared for the task lifecycle; not exercised yet."""

    DROPPING = "dropping"
    """The robot is at the drop-off location and is unloading an item.
    Declared for the task lifecycle; not exercised yet."""

    CHARGING = "charging"
    """The robot is parked on a charging station and its battery is rising."""

    BLOCKED = "blocked"
    """The robot cannot make progress: the way ahead is obstructed or its
    battery is too low to move. It is still online and can be recovered."""

    OFFLINE = "offline"
    """The robot is out of service (fault, or removed from the fleet). It must
    not be given work and must not be treated as an obstacle by a planner
    without an explicit decision to do so."""

    def __str__(self) -> str:
        return self.value


# States in which a robot is healthy enough to accept new work. Kept next to
# the enum so that "what counts as available?" is defined in exactly one place.
AVAILABLE_STATUSES = frozenset({RobotStatus.IDLE})

# States that mean the robot needs human or supervisory attention.
UNHEALTHY_STATUSES = frozenset({RobotStatus.BLOCKED, RobotStatus.OFFLINE})
