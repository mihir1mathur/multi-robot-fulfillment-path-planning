"""Text rendering of the simulation.

WHY A SEPARATE MODULE?
----------------------
Printing is a presentation concern. If `Warehouse` or `Robot` knew how to draw
themselves, then changing the map's appearance would mean editing the domain
model, and the domain model would carry code that has nothing to do with the
rules of the warehouse. Keeping rendering outside means the same model could
later be drawn by a web page or a GUI with no changes at all.

Output is deliberately plain ASCII: it prints identically in every terminal,
including Windows consoles that do not handle Unicode box characters well.
"""

from __future__ import annotations

from typing import Dict, List

from robotics.simulation.simulator import WarehouseSimulator
from robotics.warehouse.grid import Position
from robotics.warehouse.warehouse import LocationType

# One character per cell. Later entries in `render_map` override earlier ones,
# so the drawing order below is what decides who wins when two things share a
# cell (a robot standing on a charging station shows as the robot).
_LOCATION_SYMBOLS: Dict[LocationType, str] = {
    LocationType.AISLE: ".",
    LocationType.STORAGE: "S",
    LocationType.PICKUP: "P",
    LocationType.DROPOFF: "D",
    LocationType.CHARGING: "C",
}

_STATIC_OBSTACLE_SYMBOL = "#"
_DYNAMIC_OBSTACLE_SYMBOL = "x"

LEGEND = (
    "Legend: . aisle   S storage   P pickup   D dropoff   C charging   "
    "# static obstacle   x dynamic obstacle   0-9 robot"
)


def render_map(simulator: WarehouseSimulator) -> str:
    """Draw the warehouse as an ASCII map with row and column rulers.

    Robots are drawn as their index in the fleet (0, 1, 2, ...) so that the
    map stays exactly one character wide per cell however long the robot IDs
    are. `render_robot_legend` explains which digit is which robot.
    """
    warehouse = simulator.warehouse

    # Robot index lookup, built once instead of scanning the fleet per cell.
    robot_symbols: Dict[Position, str] = {
        robot.position: str(index % 10)
        for index, robot in enumerate(simulator.robots)
    }

    lines: List[str] = []

    # Column ruler, e.g. "     0 1 2 3 4". Only the last digit of each column
    # number is shown, which is enough for the small maps used here.
    column_header = "    " + " ".join(str(col % 10) for col in range(warehouse.width))
    lines.append(column_header)

    for row in range(warehouse.height):
        cells: List[str] = []
        for col in range(warehouse.width):
            position = Position(row, col)
            cells.append(_symbol_for(simulator, position, robot_symbols))
        lines.append(f"{row:2d}  " + " ".join(cells))

    return "\n".join(lines)


def _symbol_for(
    simulator: WarehouseSimulator,
    position: Position,
    robot_symbols: Dict[Position, str],
) -> str:
    """Decide the single character shown for one cell.

    Priority, most important first: robot, dynamic obstacle, static obstacle,
    location role. A robot wins because "where are my robots right now?" is
    the question this map exists to answer.
    """
    if position in robot_symbols:
        return robot_symbols[position]
    if simulator.warehouse.is_blocked_by_dynamic(position):
        return _DYNAMIC_OBSTACLE_SYMBOL
    if simulator.warehouse.is_blocked_by_static(position):
        return _STATIC_OBSTACLE_SYMBOL
    return _LOCATION_SYMBOLS[simulator.warehouse.get_location_type(position)]


def render_robot_legend(simulator: WarehouseSimulator) -> str:
    """Map the digits used on the map back to robot IDs."""
    if not simulator.robots:
        return "No robots in the fleet."
    entries = [
        f"{index % 10} = {robot.robot_id}"
        for index, robot in enumerate(simulator.robots)
    ]
    return "Robots on map: " + ",  ".join(entries)


def render_robot_table(simulator: WarehouseSimulator) -> str:
    """A fixed-width table of every robot's state."""
    header = (
        f"{'ROBOT':<8}{'POSITION':<12}{'STATUS':<12}{'BATTERY':>9}"
        f"{'PAYLOAD':>12}{'STEPS':>7}{'DISTANCE':>10}  TASK"
    )
    lines = [header, "-" * len(header)]

    for robot in simulator.robots:
        payload = f"{robot.current_payload:g}/{robot.payload_capacity:g}kg"
        lines.append(
            f"{robot.robot_id:<8}"
            f"{str(robot.position):<12}"
            f"{robot.status.value:<12}"
            f"{robot.battery_level:>8.1f}%"
            f"{payload:>12}"
            f"{robot.steps_taken:>7}"
            f"{robot.distance_travelled:>10.1f}"
            f"  {robot.assigned_task_id or '-'}"
        )
    return "\n".join(lines)


def render_task_table(simulator: WarehouseSimulator) -> str:
    """A fixed-width table of every task's state."""
    header = (
        f"{'TASK':<8}{'PICKUP':<12}{'DROPOFF':<12}{'PRIORITY':<10}"
        f"{'WEIGHT':>9}{'STATUS':>14}  ROBOT"
    )
    lines = [header, "-" * len(header)]

    for task in simulator.tasks:
        lines.append(
            f"{task.task_id:<8}"
            f"{str(task.pickup_location):<12}"
            f"{str(task.dropoff_location):<12}"
            f"{str(task.priority):<10}"
            f"{task.payload_weight:>7g}kg"
            f"{task.status.value:>14}"
            f"  {task.assigned_robot_id or '-'}"
        )
    return "\n".join(lines)


def render_full_state(simulator: WarehouseSimulator) -> str:
    """Everything a human needs to verify the simulation by eye."""
    sections = [
        render_map(simulator),
        "",
        LEGEND,
        render_robot_legend(simulator),
        "",
        render_robot_table(simulator),
        "",
        render_task_table(simulator),
    ]
    return "\n".join(sections)
