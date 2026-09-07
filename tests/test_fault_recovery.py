"""Tests for robot-failure (OFFLINE) recovery and task reassignment."""

from __future__ import annotations

from robotics.allocation import CostEstimator, GreedyAllocator, commit_allocation
from robotics.coordination.coordinator import MultiRobotCoordinator
from robotics.recovery.disruption import DisruptionSchedule, RobotFailureEvent
from robotics.recovery.recovery_manager import RecoveryManager
from robotics.robots.robot import Robot
from robotics.robots.robot_state import RobotStatus
from robotics.simulation.simulator import WarehouseSimulator
from robotics.tasks.task import Task, TaskPriority, TaskStatus
from robotics.warehouse.grid import Position
from robotics.warehouse.warehouse import Warehouse


def P(r, c):
    return Position(r, c)


def _sim_with(warehouse, robot_cells):
    sim = WarehouseSimulator(warehouse)
    for rid, cell in robot_cells:
        sim.add_robot(Robot(rid, cell))
    return sim


def _allocate_and_coordinate(sim, warehouse):
    estimator = CostEstimator(warehouse)
    allocation = GreedyAllocator(estimator).allocate(sim.robots, sim.tasks)
    commit_allocation(sim, allocation)
    goals = {
        a.robot_id: sim.get_task(a.task_id).pickup_location
        for a in allocation.assignments
    }
    coordination = MultiRobotCoordinator(warehouse).plan_for(sim, goals)
    return coordination, goals


# ----------------------------------------------------------------------
# Scenario F - robot fails before pickup, task reassigned
# ----------------------------------------------------------------------
def test_robot_failure_before_pickup_reassigns_the_task() -> None:
    wh = Warehouse(width=10, height=10, name="F")
    sim = _sim_with(wh, [("R1", P(0, 0)), ("R2", P(9, 9))])
    sim.add_task(Task("T1", P(0, 6), P(2, 6), TaskPriority.NORMAL, 2.0))
    coordination, _ = _allocate_and_coordinate(sim, wh)

    result = RecoveryManager(sim).run(
        coordination, DisruptionSchedule.of(RobotFailureEvent(3, "R1"))
    )

    assert result.success
    r1 = sim.get_robot("R1")
    assert r1.status is RobotStatus.OFFLINE
    assert r1.assigned_task_id is None
    assert r1.position == P(0, 3)                         # stayed where it failed
    task = sim.get_task("T1")
    assert task.status is TaskStatus.ASSIGNED             # NOT falsely completed
    assert task.assigned_robot_id == "R2"                 # reassigned
    assert sim.get_robot("R2").position == P(0, 6)        # replacement reached pickup

    event = result.recovery_events[0]
    assert event.task_reassignment_attempted is True
    assert event.task_reassignment_success is True
    assert event.reassigned_from == "R1"
    assert event.reassigned_to == "R2"
    assert event.recovery_success is True


def test_failed_robot_never_moves_after_going_offline() -> None:
    wh = Warehouse(width=8, height=8, name="Fmove")
    sim = _sim_with(wh, [("R1", P(0, 0)), ("R2", P(7, 7))])
    sim.add_task(Task("T1", P(0, 5), P(2, 5), TaskPriority.NORMAL, 2.0))
    coordination, _ = _allocate_and_coordinate(sim, wh)

    result = RecoveryManager(sim).run(
        coordination, DisruptionSchedule.of(RobotFailureEvent(2, "R1"))
    )
    # R1 did exactly 2 moves before it failed, then none
    assert result.per_robot["R1"]["moves"] == 2
    assert result.per_robot["R1"]["offline"] is True
    assert sim.get_robot("R1").position == P(0, 2)


# ----------------------------------------------------------------------
# Scenario G - no feasible replacement -> truthful failure
# ----------------------------------------------------------------------
def test_no_feasible_replacement_reports_recovery_failure() -> None:
    wh = Warehouse(width=8, height=8, name="G")
    sim = _sim_with(wh, [("R1", P(0, 0))])                # the ONLY robot
    sim.add_task(Task("T1", P(0, 5), P(2, 5), TaskPriority.NORMAL, 2.0))
    coordination, _ = _allocate_and_coordinate(sim, wh)

    result = RecoveryManager(sim).run(
        coordination, DisruptionSchedule.of(RobotFailureEvent(3, "R1"))
    )

    assert result.success is False
    assert result.safe_stop is True
    task = sim.get_task("T1")
    assert task.status is TaskStatus.PENDING              # released, not completed
    assert task.assigned_robot_id is None
    event = result.recovery_events[0]
    assert event.task_reassignment_attempted is True
    assert event.task_reassignment_success is False
    assert event.safe_stop is True
    assert "no feasible replacement" in event.failure_reason


def test_payload_infeasible_replacement_is_rejected() -> None:
    wh = Warehouse(width=8, height=8, name="Gpay")
    sim = _sim_with(wh, [
        ("R1", P(0, 0)),                                  # will fail
        ("R2", P(7, 7), ),
    ])
    # give R2 a small payload cap so it cannot carry the heavy task
    sim.get_robot("R2").payload_capacity = 3.0
    sim.add_task(Task("T1", P(0, 5), P(2, 5), TaskPriority.NORMAL, 8.0))  # 8kg
    coordination, _ = _allocate_and_coordinate(sim, wh)

    result = RecoveryManager(sim).run(
        coordination, DisruptionSchedule.of(RobotFailureEvent(3, "R1"))
    )
    event = result.recovery_events[0]
    assert event.task_reassignment_attempted is True
    assert event.task_reassignment_success is False       # R2 can't carry 8kg
    assert sim.get_task("T1").status is TaskStatus.PENDING


def test_battery_infeasible_replacement_is_rejected() -> None:
    wh = Warehouse(width=12, height=12, name="Gbat")
    sim = _sim_with(wh, [("R1", P(0, 0)), ("R2", P(11, 11))])
    sim.get_robot("R2").battery_level = 4.0               # only ~4 moves of charge
    sim.add_task(Task("T1", P(0, 5), P(2, 5), TaskPriority.NORMAL, 2.0))
    coordination, _ = _allocate_and_coordinate(sim, wh)

    result = RecoveryManager(sim).run(
        coordination, DisruptionSchedule.of(RobotFailureEvent(3, "R1"))
    )
    event = result.recovery_events[0]
    assert event.task_reassignment_success is False       # R2 too far for its charge
    assert sim.get_task("T1").status is TaskStatus.PENDING


# ----------------------------------------------------------------------
# Scenario H - failed robot's cell blocks another robot, which re-coordinates
# ----------------------------------------------------------------------
def test_failed_robot_cell_blocks_others_which_replan_around_it() -> None:
    wh = Warehouse(width=9, height=9, name="H")
    starts = {"R1": P(4, 0), "R2": P(4, 8)}
    goals = {"R1": P(4, 8), "R2": P(4, 0)}                # opposite ends of row 4
    coordination = MultiRobotCoordinator(wh).plan(starts, goals)
    sim = _sim_with(wh, sorted(starts.items()))

    result = RecoveryManager(sim).run(
        coordination, DisruptionSchedule.of(RobotFailureEvent(4, "R1"))
    )

    r1 = sim.get_robot("R1")
    assert r1.status is RobotStatus.OFFLINE
    failed_cell = r1.position
    assert sim.get_robot("R2").position != failed_cell    # never drove onto it
    # R2 either reached its goal by routing around, or safe-stopped - never a crash
    event = result.recovery_events[0]
    assert "R1" in event.affected_robot_ids
    assert result.vertex_conflicts == 0 and result.edge_conflicts == 0
    assert result.validation_problems == []


# ----------------------------------------------------------------------
# Scenario I - one robot fails, an unaffected robot keeps going
# ----------------------------------------------------------------------
def test_unaffected_robot_is_not_replanned_when_another_fails() -> None:
    wh = Warehouse(width=10, height=10, name="I")
    starts = {"R1": P(0, 0), "R2": P(9, 0)}               # parallel, no interaction
    goals = {"R1": P(0, 9), "R2": P(9, 9)}
    coordination = MultiRobotCoordinator(wh).plan(starts, goals)
    sim = _sim_with(wh, sorted(starts.items()))

    result = RecoveryManager(sim).run(
        coordination, DisruptionSchedule.of(RobotFailureEvent(3, "R1"))
    )

    assert sim.get_robot("R2").position == P(9, 9)        # unaffected, finished
    assert result.per_robot["R2"]["reached_goal"] is True
    event = result.recovery_events[0]
    assert event.affected_robot_ids == ["R1"]             # R2 not in the affected set
    assert not any(r.robot_id == "R2" for r in event.robot_replans)


# ----------------------------------------------------------------------
# Scenario J - multiple pending tasks: use the allocator, not a hard-coded pick
# ----------------------------------------------------------------------
def test_reassignment_uses_the_allocator_over_multiple_candidates() -> None:
    wh = Warehouse(width=12, height=12, name="J")
    sim = _sim_with(wh, [
        ("R1", P(0, 0)),      # takes T1 (assigned by hand), then fails
        ("R2", P(0, 11)),     # far from T1's pickup
        ("R3", P(2, 6)),      # close to T1's pickup -> allocator should pick R3
    ])
    sim.add_task(Task("T1", P(0, 6), P(6, 6), TaskPriority.NORMAL, 2.0))
    sim.assign_task_manually("T1", "R1")                  # force R1 to own T1
    goals = {"R1": sim.get_task("T1").pickup_location}
    coordination = MultiRobotCoordinator(wh).plan_for(sim, goals)

    result = RecoveryManager(sim).run(
        coordination, DisruptionSchedule.of(RobotFailureEvent(2, "R1"))
    )
    event = result.recovery_events[0]
    assert event.task_reassignment_success is True
    # the allocator minimises travel, so the nearer idle robot (R3) wins over R2
    assert event.reassigned_to == "R3"
    assert sim.get_task("T1").assigned_robot_id == "R3"


# ----------------------------------------------------------------------
# Scenario K - the failed robot stops on a cell a still-live robot's planned
# route runs through at a LATER timestep. Building the recovery reservation
# table used to raise ValueError (the failed-robot pin collided with that
# robot's stale reservation); recovery must instead replan the live robot.
# ----------------------------------------------------------------------
def test_recovery_handles_failed_robot_cell_with_future_reservation() -> None:
    wh = Warehouse(width=9, height=9, name="K")
    sim = _sim_with(wh, [
        ("R1", P(0, 4)),     # drives down column 4, owns T1, fails at the crossing
        ("R2", P(3, 0)),     # drives across row 3 - passes (3, 4) one step later
        ("R3", P(8, 0)),     # idle spare, feasible replacement for T1
    ])
    sim.add_task(Task("T1", P(5, 4), P(6, 4), TaskPriority.NORMAL, 2.0))
    sim.assign_task_manually("T1", "R1")

    coordination = MultiRobotCoordinator(wh).plan(
        {"R1": P(0, 4), "R2": P(3, 0)},
        {"R1": P(5, 4), "R2": P(3, 8)},
    )
    # R1 is at (3, 4) at t=3; R2's planned route occupies (3, 4) at t=4.
    assert coordination.timed_paths["R2"].position_at(4) == P(3, 4)

    result = RecoveryManager(sim).run(          # must not raise ValueError
        coordination, DisruptionSchedule.of(RobotFailureEvent(3, "R1"))
    )

    r1 = sim.get_robot("R1")
    assert r1.status is RobotStatus.OFFLINE
    failed_cell = r1.position
    assert failed_cell == P(3, 4)
    # no live robot is ever driven onto the failed robot's cell
    assert sim.get_robot("R2").position != failed_cell
    assert sim.get_robot("R3").position != failed_cell

    event = result.recovery_events[0]
    assert "R1" in event.affected_robot_ids
    assert "R2" in event.affected_robot_ids               # replanned around the pin
    assert event.task_reassignment_attempted is True
    assert event.task_reassignment_success is True        # T1 -> the spare R3
    assert event.reassigned_from == "R1"
    assert event.reassigned_to == "R3"
    assert sim.get_task("T1").assigned_robot_id == "R3"
    assert sim.get_task("T1").status is TaskStatus.ASSIGNED   # never falsely completed

    # the executed trace is independently conflict-free
    assert result.vertex_conflicts == 0 and result.edge_conflicts == 0
    assert result.validation_problems == []
    assert event.unresolved_vertex_conflicts == 0
    assert event.unresolved_edge_conflicts == 0


# ----------------------------------------------------------------------
# State-machine correctness
# ----------------------------------------------------------------------
def test_task_follows_valid_state_transitions_during_recovery() -> None:
    wh = Warehouse(width=10, height=10, name="states")
    sim = _sim_with(wh, [("R1", P(0, 0)), ("R2", P(9, 9))])
    sim.add_task(Task("T1", P(0, 6), P(2, 6), TaskPriority.NORMAL, 2.0))
    coordination, _ = _allocate_and_coordinate(sim, wh)
    task = sim.get_task("T1")
    assert task.status is TaskStatus.ASSIGNED

    RecoveryManager(sim).run(
        coordination, DisruptionSchedule.of(RobotFailureEvent(3, "R1"))
    )
    # ASSIGNED -> PENDING -> ASSIGNED is the only path taken; never COMPLETED,
    # never an illegal jump
    assert task.status is TaskStatus.ASSIGNED
    assert task.assigned_robot_id == "R2"


def test_unknown_failed_robot_id_is_handled() -> None:
    wh = Warehouse(width=7, height=7, name="unknown")
    coordination = MultiRobotCoordinator(wh).plan({"R1": P(3, 0)}, {"R1": P(3, 6)})
    sim = _sim_with(wh, [("R1", P(3, 0))])
    result = RecoveryManager(sim).run(
        coordination, DisruptionSchedule.of(RobotFailureEvent(2, "GHOST"))
    )
    # the ghost event is recorded but does nothing; R1 still finishes
    assert result.success
    assert sim.get_robot("R1").position == P(3, 6)
