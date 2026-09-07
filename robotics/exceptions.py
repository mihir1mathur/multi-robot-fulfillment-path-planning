"""Exception types used across the project.

WHY a dedicated module?
-----------------------
Every layer (grid, warehouse, robot, simulator) needs to reject bad input.
If each layer invented its own error type, callers and tests would have to
import from many places and would end up catching bare `Exception`, which
hides real bugs. One small shared hierarchy means a caller can catch
`RoboticsError` to handle "anything my simulation rejected", or catch a
specific subclass when it cares about the exact reason.

All exceptions carry a human-readable message explaining *why* the operation
was rejected. That message is what the demo prints, and it is what a future
API layer would return to a client.
"""


class RoboticsError(Exception):
    """Base class for every error raised by this project."""


class InvalidPositionError(RoboticsError):
    """A coordinate is outside the warehouse, or otherwise not usable.

    Example: asking the grid to block cell (99, 99) in a 10x10 warehouse.
    """


class InvalidMoveError(RoboticsError):
    """A requested single-cell robot move is not legal.

    Reasons include: the destination is out of bounds, not adjacent to the
    robot's current cell, blocked by an obstacle, already occupied by another
    robot, or the robot does not have enough battery left.
    """


class TaskError(RoboticsError):
    """A task was used in a way its lifecycle does not allow.

    Example: assigning a task that has already been completed.
    """


class SimulationError(RoboticsError):
    """The simulator was asked to do something inconsistent.

    Example: registering two robots with the same ID, or looking up a robot
    that was never added.
    """


class PathValidationError(RoboticsError):
    """A path produced by a planner failed an independent correctness check.

    This should never happen in normal use. It exists so that a planner bug -
    a route that cuts through a wall, skips a cell, or reports a cost that
    disagrees with its own path - fails loudly at the point of detection
    rather than being executed.
    """
