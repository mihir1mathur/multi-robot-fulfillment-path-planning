"""Robot modelling: attributes, status values and single-step movement."""

from robotics.robots.robot import (
    DEFAULT_BATTERY_DRAIN_PER_MOVE,
    MAX_BATTERY,
    MIN_BATTERY,
    Robot,
)
from robotics.robots.robot_state import (
    AVAILABLE_STATUSES,
    UNHEALTHY_STATUSES,
    RobotStatus,
)

__all__ = [
    "Robot",
    "RobotStatus",
    "AVAILABLE_STATUSES",
    "UNHEALTHY_STATUSES",
    "MIN_BATTERY",
    "MAX_BATTERY",
    "DEFAULT_BATTERY_DRAIN_PER_MOVE",
]
