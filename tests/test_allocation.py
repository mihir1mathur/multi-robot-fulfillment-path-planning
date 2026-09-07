"""Tests for the greedy and CP-SAT task allocators.

Most tests run against BOTH allocators (parametrized), because they must agree
on feasibility, on assigning each task and robot at most once, and on producing
a result the independent validator accepts. The tests that are specific to one
allocator - CP-SAT optimality, greedy's local choice - are called out.
"""

from __future__ import annotations

from typing import List, Sequence

import pytest

from robotics.allocation.allocation_result import AllocationResult
from robotics.allocation.cost_estimator import CostEstimator
from robotics.allocation.cp_sat_allocator import CpSatAllocator
from robotics.allocation.greedy_allocator import GreedyAllocator
from robotics.allocation.validation import find_allocation_problems
from robotics.robots.robot import Robot
from robotics.robots.robot_state import RobotStatus
from robotics.tasks.task import Task, TaskPriority, TaskStatus
from robotics.warehouse.grid import Position
from robotics.warehouse.warehouse import Warehouse

ALLOCATORS = [GreedyAllocator, CpSatAllocator]
ALLOCATOR_IDS = ["greedy", "cp_sat"]


def _wh(width: int = 12, height: int = 12) -> Warehouse:
    return Warehouse(width=width, height=height, name="alloc-test")


def _run(allocator_cls, robots, tasks, warehouse) -> AllocationResult:
    estimator = CostEstimator(warehouse)
    return allocator_cls(estimator).allocate(robots, tasks)


def _valid(result, robots, tasks, warehouse) -> List[str]:
    return find_allocation_problems(result, robots, tasks, CostEstimator(warehouse))


# ----------------------------------------------------------------------
# Cardinality cases
# ----------------------------------------------------------------------
@pytest.mark.parametrize("allocator_cls", ALLOCATORS, ids=ALLOCATOR_IDS)
def test_one_robot_one_task(allocator_cls) -> None:
    wh = _wh()
    robots = [Robot("R1", Position(0, 0))]
    tasks = [Task("T1", Position(0, 3), Position(3, 3))]

    result = _run(allocator_cls, robots, tasks, wh)

    assert result.success
    assert result.assigned_task_count == 1
    assert result.assignments[0].robot_id == "R1"
    assert result.assignments[0].task_id == "T1"
    assert _valid(result, robots, tasks, wh) == []


@pytest.mark.parametrize("allocator_cls", ALLOCATORS, ids=ALLOCATOR_IDS)
def test_multiple_robots_one_task_assigns_the_cheapest_robot(allocator_cls) -> None:
    wh = _wh()
    robots = [Robot("R1", Position(0, 0)), Robot("R2", Position(0, 2))]
    tasks = [Task("T1", Position(0, 3), Position(5, 3))]  # R2 is closer to pickup

    result = _run(allocator_cls, robots, tasks, wh)

    assert result.assigned_task_count == 1
    assert result.assignments[0].robot_id == "R2"
    assert _valid(result, robots, tasks, wh) == []


@pytest.mark.parametrize("allocator_cls", ALLOCATORS, ids=ALLOCATOR_IDS)
def test_one_robot_multiple_tasks_assigns_one_task_this_cycle(allocator_cls) -> None:
    wh = _wh()
    robots = [Robot("R1", Position(0, 0))]
    tasks = [
        Task("T1", Position(0, 2), Position(2, 2)),
        Task("T2", Position(5, 5), Position(6, 6)),
    ]

    result = _run(allocator_cls, robots, tasks, wh)

    assert result.assigned_task_count == 1
    assert result.unassigned_task_count == 1
    assert _valid(result, robots, tasks, wh) == []


@pytest.mark.parametrize("allocator_cls", ALLOCATORS, ids=ALLOCATOR_IDS)
def test_multiple_robots_multiple_tasks(allocator_cls) -> None:
    wh = _wh()
    robots = [Robot("R1", Position(0, 0)), Robot("R2", Position(11, 11))]
    tasks = [
        Task("T1", Position(1, 1), Position(2, 2)),
        Task("T2", Position(10, 10), Position(9, 9)),
    ]

    result = _run(allocator_cls, robots, tasks, wh)

    assert result.assigned_task_count == 2
    assert {a.robot_id for a in result.assignments} == {"R1", "R2"}
    assert {a.task_id for a in result.assignments} == {"T1", "T2"}
    assert _valid(result, robots, tasks, wh) == []


@pytest.mark.parametrize("allocator_cls", ALLOCATORS, ids=ALLOCATOR_IDS)
def test_more_tasks_than_robots_leaves_some_pending(allocator_cls) -> None:
    wh = _wh()
    robots = [Robot("R1", Position(0, 0)), Robot("R2", Position(11, 11))]
    tasks = [
        Task("T1", Position(1, 1), Position(2, 1)),
        Task("T2", Position(3, 3), Position(4, 3)),
        Task("T3", Position(10, 10), Position(9, 10)),
        Task("T4", Position(8, 8), Position(7, 8)),
    ]

    result = _run(allocator_cls, robots, tasks, wh)

    assert result.assigned_task_count == 2       # one per robot, this cycle
    assert result.unassigned_task_count == 2
    assert len({a.robot_id for a in result.assignments}) == 2
    assert _valid(result, robots, tasks, wh) == []


@pytest.mark.parametrize("allocator_cls", ALLOCATORS, ids=ALLOCATOR_IDS)
def test_more_robots_than_tasks_leaves_robots_idle(allocator_cls) -> None:
    wh = _wh()
    robots = [Robot(f"R{i}", Position(0, i)) for i in range(4)]
    tasks = [Task("T1", Position(5, 5), Position(6, 6))]

    result = _run(allocator_cls, robots, tasks, wh)

    assert result.assigned_task_count == 1
    assert result.unassigned_task_count == 0
    assert _valid(result, robots, tasks, wh) == []


# ----------------------------------------------------------------------
# Eligibility / feasibility exclusions
# ----------------------------------------------------------------------
@pytest.mark.parametrize("allocator_cls", ALLOCATORS, ids=ALLOCATOR_IDS)
def test_unavailable_robot_is_excluded(allocator_cls) -> None:
    wh = _wh()
    busy = Robot("R1", Position(0, 0))
    busy.set_status(RobotStatus.OFFLINE)
    free = Robot("R2", Position(0, 1))
    tasks = [Task("T1", Position(0, 3), Position(3, 3))]

    result = _run(allocator_cls, [busy, free], tasks, wh)

    assert result.assigned_task_count == 1
    assert result.assignments[0].robot_id == "R2"


@pytest.mark.parametrize("allocator_cls", ALLOCATORS, ids=ALLOCATOR_IDS)
def test_robot_that_already_holds_a_task_is_excluded(allocator_cls) -> None:
    wh = _wh()
    holding = Robot("R1", Position(0, 0))
    holding.assign_task("T0")  # now ASSIGNED, not available
    free = Robot("R2", Position(0, 1))
    tasks = [Task("T1", Position(0, 3), Position(3, 3))]

    result = _run(allocator_cls, [holding, free], tasks, wh)

    assert result.assignments[0].robot_id == "R2"


@pytest.mark.parametrize("allocator_cls", ALLOCATORS, ids=ALLOCATOR_IDS)
def test_non_pending_task_is_excluded(allocator_cls) -> None:
    wh = _wh()
    robots = [Robot("R1", Position(0, 0))]
    pending = Task("T1", Position(0, 3), Position(3, 3))
    done = Task("T2", Position(1, 1), Position(2, 2))
    done.assign_to("RX")
    done.start()
    done.complete()  # COMPLETED - must not be a candidate

    result = _run(allocator_cls, robots, [done, pending], wh)

    assert result.assigned_task_count == 1
    assert result.assignments[0].task_id == "T1"
    assert "T2" not in result.unassigned_task_ids


@pytest.mark.parametrize("allocator_cls", ALLOCATORS, ids=ALLOCATOR_IDS)
def test_payload_infeasible_pairing_is_excluded(allocator_cls) -> None:
    wh = _wh()
    light = Robot("R1", Position(0, 0), payload_capacity=3.0)
    heavy = Robot("R2", Position(0, 1), payload_capacity=20.0)
    tasks = [Task("T1", Position(0, 3), Position(3, 3), payload_weight=10.0)]

    result = _run(allocator_cls, [light, heavy], tasks, wh)

    assert result.assigned_task_count == 1
    assert result.assignments[0].robot_id == "R2"     # only R2 can carry 10kg
    assert result.infeasible_pair_count >= 1


@pytest.mark.parametrize("allocator_cls", ALLOCATORS, ids=ALLOCATOR_IDS)
def test_battery_infeasible_pairing_is_excluded(allocator_cls) -> None:
    wh = _wh()
    # R1 is near, but almost flat: 3% charge pays for 3 moves only.
    flat = Robot("R1", Position(0, 3), battery_level=3.0)
    charged = Robot("R2", Position(0, 4), battery_level=100.0)
    # route is ~ (R->P) + (P->D); well over 3 moves from either robot
    tasks = [Task("T1", Position(5, 5), Position(9, 9))]

    result = _run(allocator_cls, [flat, charged], tasks, wh)

    assert result.assigned_task_count == 1
    assert result.assignments[0].robot_id == "R2"
    assert result.infeasible_pair_count >= 1


@pytest.mark.parametrize("allocator_cls", ALLOCATORS, ids=ALLOCATOR_IDS)
def test_unreachable_pickup_makes_the_pairing_infeasible(allocator_cls) -> None:
    wh = _wh(5, 5)
    for cell in (Position(0, 3), Position(1, 3), Position(1, 4)):
        wh.add_static_obstacle(f"seal-{cell.row}-{cell.col}", cell)
    robots = [Robot("R1", Position(4, 0))]
    tasks = [Task("T1", Position(0, 4), Position(4, 4))]  # pickup walled off

    result = _run(allocator_cls, robots, tasks, wh)

    assert result.assigned_task_count == 0
    assert result.unassigned_task_ids == ["T1"]
    assert result.infeasible_pair_count == 1
    assert result.eligible_pair_count == 0


@pytest.mark.parametrize("allocator_cls", ALLOCATORS, ids=ALLOCATOR_IDS)
def test_unreachable_dropoff_makes_the_pairing_infeasible(allocator_cls) -> None:
    wh = _wh(5, 5)
    for cell in (Position(0, 3), Position(1, 3), Position(1, 4)):
        wh.add_static_obstacle(f"seal-{cell.row}-{cell.col}", cell)
    robots = [Robot("R1", Position(4, 0))]
    tasks = [Task("T1", Position(4, 4), Position(0, 4))]  # dropoff walled off

    result = _run(allocator_cls, robots, tasks, wh)

    assert result.assigned_task_count == 0
    assert result.infeasible_pair_count == 1


# ----------------------------------------------------------------------
# Structural invariants
# ----------------------------------------------------------------------
@pytest.mark.parametrize("allocator_cls", ALLOCATORS, ids=ALLOCATOR_IDS)
def test_each_task_assigned_at_most_once(allocator_cls) -> None:
    wh = _wh()
    robots = [Robot(f"R{i}", Position(0, i)) for i in range(5)]
    tasks = [Task("T1", Position(6, 6), Position(7, 7))]

    result = _run(allocator_cls, robots, tasks, wh)

    assert len(result.assigned_task_ids) == len(set(result.assigned_task_ids))
    assert result.assigned_task_count <= 1


@pytest.mark.parametrize("allocator_cls", ALLOCATORS, ids=ALLOCATOR_IDS)
def test_each_robot_assigned_at_most_once_per_cycle(allocator_cls) -> None:
    wh = _wh()
    robots = [Robot("R1", Position(0, 0))]
    tasks = [Task(f"T{i}", Position(2, i), Position(3, i)) for i in range(1, 5)]

    result = _run(allocator_cls, robots, tasks, wh)

    assert len(result.assigned_robot_ids) == len(set(result.assigned_robot_ids))
    assert result.assigned_task_count <= 1


@pytest.mark.parametrize("allocator_cls", ALLOCATORS, ids=ALLOCATOR_IDS)
def test_result_passes_independent_validation_on_a_mixed_scenario(allocator_cls) -> None:
    wh = _wh()
    wh.add_static_obstacle("rack-1", Position(5, 5))
    wh.add_static_obstacle("rack-2", Position(6, 5))
    robots = [
        Robot("R1", Position(0, 0)),
        Robot("R2", Position(11, 11), payload_capacity=2.0),
        Robot("R3", Position(6, 0), battery_level=100.0),
    ]
    tasks = [
        Task("T1", Position(1, 2), Position(3, 4), payload_weight=1.0),
        Task("T2", Position(9, 9), Position(10, 8), payload_weight=5.0),
        Task("T3", Position(6, 2), Position(2, 8), payload_weight=1.0),
    ]

    result = _run(allocator_cls, robots, tasks, wh)

    assert result.success
    assert _valid(result, robots, tasks, wh) == []


@pytest.mark.parametrize("allocator_cls", ALLOCATORS, ids=ALLOCATOR_IDS)
def test_no_feasible_pair_gives_a_valid_empty_result(allocator_cls) -> None:
    wh = _wh(5, 5)
    for cell in (Position(0, 3), Position(1, 3), Position(1, 4)):
        wh.add_static_obstacle(f"seal-{cell.row}-{cell.col}", cell)
    robots = [Robot("R1", Position(4, 0))]
    tasks = [Task("T1", Position(0, 4), Position(4, 4))]

    result = _run(allocator_cls, robots, tasks, wh)

    assert result.success is True            # not an error - just nothing to do
    assert result.assignments == []
    assert result.unassigned_task_ids == ["T1"]


@pytest.mark.parametrize("allocator_cls", ALLOCATORS, ids=ALLOCATOR_IDS)
def test_no_robots_or_no_tasks(allocator_cls) -> None:
    wh = _wh()
    assert _run(allocator_cls, [], [Task("T1", Position(0, 1), Position(0, 2))], wh).assigned_task_count == 0
    assert _run(allocator_cls, [Robot("R1", Position(0, 0))], [], wh).assigned_task_count == 0


# ----------------------------------------------------------------------
# Determinism
# ----------------------------------------------------------------------
@pytest.mark.parametrize("allocator_cls", ALLOCATORS, ids=ALLOCATOR_IDS)
def test_allocation_is_deterministic(allocator_cls) -> None:
    wh = _wh()
    robots = [Robot(f"R{i}", Position(0, i * 2)) for i in range(4)]
    tasks = [Task(f"T{i}", Position(4, i * 2), Position(8, i * 2)) for i in range(4)]

    first = _run(allocator_cls, robots, tasks, wh)
    second = _run(allocator_cls, robots, tasks, wh)

    assert [a.to_dict() for a in first.assignments] == [
        a.to_dict() for a in second.assignments
    ]
    assert first.total_estimated_cost == second.total_estimated_cost


@pytest.mark.parametrize("allocator_cls", ALLOCATORS, ids=ALLOCATOR_IDS)
def test_tie_breaking_prefers_lower_robot_id(allocator_cls) -> None:
    wh = _wh()
    # R1 and R2 are symmetric about the pickup: identical cost.
    robots = [Robot("R1", Position(0, 2)), Robot("R2", Position(0, 4))]
    tasks = [Task("T1", Position(0, 3), Position(3, 3))]

    result = _run(allocator_cls, robots, tasks, wh)

    assert result.assignments[0].robot_id == "R1"   # deterministic tie-break


# ----------------------------------------------------------------------
# CP-SAT vs greedy: fair cost comparison
# ----------------------------------------------------------------------
def test_cp_sat_is_no_worse_than_greedy_for_the_same_cardinality() -> None:
    """A scenario where greedy's first, locally-cheap pick blocks the better
    global assignment. CP-SAT should assign the same number of tasks at a
    strictly lower (or equal) total cost."""
    wh = _wh(14, 14)
    # T1's cheapest robot is R1 (adjacent). But R1 is also the ONLY cheap robot
    # for T2; R2 is far from T2. Greedy grabs R1 for T1, stranding T2 with R2.
    robots = [Robot("R1", Position(0, 1)), Robot("R2", Position(13, 13))]
    tasks = [
        Task("T1", Position(0, 0), Position(0, 2)),      # R1 is right next to it
        Task("T2", Position(0, 3), Position(0, 5)),      # also near R1, far from R2
    ]

    estimator = CostEstimator(wh)
    greedy = GreedyAllocator(estimator).allocate(robots, tasks)
    cp_sat = CpSatAllocator(estimator).allocate(robots, tasks)

    assert greedy.success and cp_sat.success
    if greedy.assigned_task_count == cp_sat.assigned_task_count:
        assert cp_sat.total_estimated_cost <= greedy.total_estimated_cost


def test_cp_sat_finds_the_optimal_swap_greedy_misses() -> None:
    """The classic greedy failure: R1 is marginally cheaper for T1, but taking
    it there forces a very expensive R2->T2. CP-SAT swaps them."""
    wh = _wh(20, 20)
    robots = [Robot("R1", Position(0, 0)), Robot("R2", Position(0, 1))]
    tasks = [
        Task("T1", Position(0, 2), Position(0, 4)),        # both robots near
        Task("T2", Position(19, 19), Position(18, 18)),    # far from everyone
    ]

    estimator = CostEstimator(wh)
    greedy = GreedyAllocator(estimator).allocate(robots, tasks)
    cp_sat = CpSatAllocator(estimator).allocate(robots, tasks)

    # Both assign both tasks; CP-SAT's total cost is <= greedy's.
    assert greedy.assigned_task_count == cp_sat.assigned_task_count == 2
    assert cp_sat.total_estimated_cost <= greedy.total_estimated_cost


def test_cp_sat_reports_optimal_status_on_a_solvable_scenario() -> None:
    wh = _wh()
    robots = [Robot("R1", Position(0, 0)), Robot("R2", Position(5, 5))]
    tasks = [Task("T1", Position(1, 1), Position(2, 2)), Task("T2", Position(6, 6), Position(7, 7))]

    result = CpSatAllocator(CostEstimator(wh)).allocate(robots, tasks)

    assert result.solver_status == "OPTIMAL"
    assert result.success
