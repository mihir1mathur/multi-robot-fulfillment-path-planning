"""A fixed, reproducible sample warehouse used by the demo and the tests.

REPRODUCIBILITY
---------------
Running the demo twice must produce byte-for-byte identical output. If it did
not, no measurement taken from it could be trusted and no test could assert
against it. Two things guarantee that here:

  1. The layout (racks, pickup points, drop-offs, chargers, robot start cells)
     is written out literally. Nothing about it is random.
  2. The only randomised quantity - each robot's starting battery - is drawn
     from a `random.Random(SEED)` instance with a fixed seed, so it produces
     the same sequence on every run and every machine.

Point 2 also matters for a reason beyond the demo: a fleet where every robot
starts at exactly 100% would make battery-aware task allocation meaningless,
because every robot would always look identical.

THE LAYOUT (12 rows x 12 cols)
------------------------------
Storage racks are solid blocks robots cannot drive through, arranged in a
regular pattern with aisles between them - the same idea as supermarket
shelving.

         col  0  1  2  3  4  5  6  7  8  9 10 11
    row 0     C  .  .  .  .  .  .  .  .  .  .  C
    row 1     .  .  .  .  .  .  .  .  .  .  .  .
    row 2     .  S  #  #  P  #  #  .  #  #  .  .
    row 3     .  .  #  #  S  #  #  .  #  #  .  .
    row 4     .  .  .  .  .  .  .  .  .  .  .  .
    row 5     .  S  #  #  .  #  #  .  #  #  .  .
    row 6     .  .  #  #  .  #  #  S  #  #  .  .
    row 7     .  .  .  .  .  .  .  .  .  .  .  .
    row 8     .  .  #  #  .  #  #  P  #  #  S  .
    row 9     .  .  #  #  S  #  #  .  #  #  .  .
    row 10    .  .  .  .  .  .  .  .  .  .  .  .
    row 11    .  D  .  .  .  .  .  .  .  .  D  .

    C charging   S storage   P pickup   D dropoff   # rack (static obstacle)

Every aisle connects to every other aisle, so no location is walled off.
"""

from __future__ import annotations

import random
from typing import List, Tuple

from robotics.robots.robot import Robot
from robotics.simulation.simulator import WarehouseSimulator
from robotics.tasks.task import Task, TaskPriority
from robotics.warehouse.grid import Position
from robotics.warehouse.warehouse import LocationType, Warehouse

# Fixed seed for every random draw in this module.
SEED = 42

WAREHOUSE_NAME = "demo-warehouse"
WAREHOUSE_ROWS = 12
WAREHOUSE_COLS = 12

# Rows and columns occupied by storage racks. A cell is solid rack only where a
# rack row and a rack column cross; everything else stays as a driveable aisle.
RACK_ROWS: Tuple[int, ...] = (2, 3, 5, 6, 8, 9)
RACK_COLS: Tuple[int, ...] = (2, 3, 5, 6, 8, 9)

# Driveable cells from which stored goods are handled (see the assumptions in
# warehouse.py: the solid rack is the obstacle, the storage cell is the spot a
# robot stands on to work with it).
STORAGE_CELLS: Tuple[Position, ...] = (
    Position(2, 1),
    Position(3, 4),
    Position(5, 1),
    Position(6, 7),
    Position(8, 10),
    Position(9, 4),
)

PICKUP_CELLS: Tuple[Position, ...] = (Position(2, 4), Position(8, 7))
DROPOFF_CELLS: Tuple[Position, ...] = (Position(11, 1), Position(11, 10))
CHARGING_CELLS: Tuple[Position, ...] = (Position(0, 0), Position(0, 11))

# Robot start cells, chosen to be plain aisle cells so the starting state does
# not accidentally block a charger or a pickup point.
ROBOT_START_CELLS: Tuple[Tuple[str, Position, float], ...] = (
    # (robot_id, start position, payload capacity in kg)
    ("R1", Position(0, 1), 10.0),
    ("R2", Position(0, 4), 10.0),
    ("R3", Position(0, 5), 5.0),
    ("R4", Position(10, 10), 15.0),
)

# The single dynamic obstacle the demo starts with: a spill in the aisle
# directly below robot R2.
SPILL_OBSTACLE_ID = "spill-1"
SPILL_POSITION = Position(1, 4)

# Battery levels are drawn from this range so that robots differ from one
# another without any of them starting unusably low.
MIN_START_BATTERY = 55
MAX_START_BATTERY = 100


def build_sample_warehouse() -> Warehouse:
    """Build the fixed 12x12 floor plan: racks, storage, pickups, chargers."""
    warehouse = Warehouse(
        width=WAREHOUSE_COLS, height=WAREHOUSE_ROWS, name=WAREHOUSE_NAME
    )

    # Racks first. They must exist before the functional locations are marked,
    # because Warehouse refuses to mark a permanently blocked cell as a pickup
    # point - which is exactly the safety check we want exercised here.
    for row in RACK_ROWS:
        for col in RACK_COLS:
            warehouse.add_static_obstacle(
                obstacle_id=f"rack-r{row}c{col}",
                position=Position(row, col),
                description="storage rack",
            )

    for position in STORAGE_CELLS:
        warehouse.mark_location(position, LocationType.STORAGE)
    for position in PICKUP_CELLS:
        warehouse.mark_location(position, LocationType.PICKUP)
    for position in DROPOFF_CELLS:
        warehouse.mark_location(position, LocationType.DROPOFF)
    for position in CHARGING_CELLS:
        warehouse.mark_location(position, LocationType.CHARGING)

    return warehouse


def build_sample_robots(seed: int = SEED) -> List[Robot]:
    """Create the fleet, with reproducible starting battery levels."""
    rng = random.Random(seed)

    robots: List[Robot] = []
    for robot_id, start_position, capacity in ROBOT_START_CELLS:
        robots.append(
            Robot(
                robot_id=robot_id,
                position=start_position,
                battery_level=float(rng.randint(MIN_START_BATTERY, MAX_START_BATTERY)),
                payload_capacity=capacity,
            )
        )
    return robots


def build_sample_tasks() -> List[Task]:
    """Create the sample fulfillment tasks.

    Every payload weight fits at least one robot in the fleet, so all of these
    tasks are assignable. Weights and priorities differ on purpose: an
    optimiser needs tasks that are genuinely distinguishable, otherwise any
    assignment looks as good as any other.
    """
    return [
        Task(
            task_id="T1",
            pickup_location=Position(2, 4),
            dropoff_location=Position(11, 1),
            priority=TaskPriority.HIGH,
            payload_weight=3.0,
        ),
        Task(
            task_id="T2",
            pickup_location=Position(8, 7),
            dropoff_location=Position(11, 10),
            priority=TaskPriority.URGENT,
            payload_weight=7.5,
        ),
        Task(
            task_id="T3",
            pickup_location=Position(2, 1),
            dropoff_location=Position(11, 10),
            priority=TaskPriority.NORMAL,
            payload_weight=2.0,
        ),
        Task(
            task_id="T4",
            pickup_location=Position(5, 1),
            dropoff_location=Position(11, 1),
            priority=TaskPriority.LOW,
            payload_weight=4.5,
        ),
    ]


def build_sample_simulator(
    seed: int = SEED, include_dynamic_obstacle: bool = True
) -> WarehouseSimulator:
    """Assemble the complete sample scenario, ready to use.

    Args:
        seed: seed for the robots' starting battery levels.
        include_dynamic_obstacle: whether to start with the aisle spill in
            place. Tests that want a clean map can turn it off.

    Returns:
        A simulator holding the warehouse, four robots and four pending tasks.
        No task is assigned yet and no robot has moved.
    """
    simulator = WarehouseSimulator(build_sample_warehouse())

    for robot in build_sample_robots(seed=seed):
        simulator.add_robot(robot)

    for task in build_sample_tasks():
        simulator.add_task(task)

    if include_dynamic_obstacle:
        simulator.add_dynamic_obstacle(
            obstacle_id=SPILL_OBSTACLE_ID,
            position=SPILL_POSITION,
            description="liquid spill, cleaning crew notified",
        )

    return simulator
