"""Tests for dynamic-obstacle replanning during coordinated execution."""

from __future__ import annotations

from robotics.coordination.coordinator import MultiRobotCoordinator
from robotics.recovery.disruption import DisruptionSchedule, DynamicObstacleEvent
from robotics.recovery.recovery_manager import RecoveryManager
from robotics.robots.robot import Robot
from robotics.simulation.simulator import WarehouseSimulator
from robotics.warehouse.grid import Position
from robotics.warehouse.warehouse import Warehouse


def P(r, c):
    return Position(r, c)


def _run(warehouse, starts, goals, events, priority_order=None):
    coordination = MultiRobotCoordinator(warehouse).plan(
        starts, goals, priority_order=priority_order
    )
    sim = WarehouseSimulator(warehouse)
    for rid, cell in sorted(starts.items()):
        sim.add_robot(Robot(rid, cell))
    result = RecoveryManager(sim).run(coordination, DisruptionSchedule.of(*events))
    return sim, coordination, result


# ----------------------------------------------------------------------
# Scenario A - obstacle on the remaining route, an alternate exists
# ----------------------------------------------------------------------
def test_obstacle_on_route_with_an_alternate_is_recovered() -> None:
    wh = Warehouse(width=7, height=7, name="A")
    sim, _, result = _run(
        wh, {"R1": P(3, 0)}, {"R1": P(3, 6)},
        [DynamicObstacleEvent(3, P(3, 4))],   # R1 is at (3,3) at t=3
    )
    assert result.success
    assert result.safe_stop is False
    assert sim.get_robot("R1").position == P(3, 6)
    assert result.per_robot["R1"]["reached_goal"] is True
    assert len(result.recovery_events) == 1
    event = result.recovery_events[0]
    assert event.recovery_success is True
    assert event.replanning_attempted is True
    assert event.replanning_success is True
    assert event.reservations_released > 0
    assert event.reservations_created > 0


def test_replanning_starts_from_the_current_position_not_the_original_start() -> None:
    wh = Warehouse(width=7, height=7, name="A2")
    sim, _, result = _run(
        wh, {"R1": P(3, 0)}, {"R1": P(3, 6)},
        [DynamicObstacleEvent(3, P(3, 4))],
    )
    replan = result.recovery_events[0].robot_replans[0]
    assert replan.current_position == P(3, 3)          # where R1 actually was
    assert replan.current_position != P(3, 0)          # NOT the original start
    assert replan.new_path[0] == P(3, 3)


def test_already_traversed_cells_are_not_replayed() -> None:
    wh = Warehouse(width=7, height=7, name="A3")
    sim, _, result = _run(
        wh, {"R1": P(3, 0)}, {"R1": P(3, 6)},
        [DynamicObstacleEvent(3, P(3, 4))],
    )
    # R1's executed move count == 3 already done + the (longer) replanned tail.
    # It must never be more than "3 + new_tail_length".
    replan = result.recovery_events[0].robot_replans[0]
    assert result.per_robot["R1"]["moves"] == 3 + replan.new_length


def test_the_obstacle_cell_is_never_entered() -> None:
    wh = Warehouse(width=7, height=7, name="A4")
    sim, coord, result = _run(
        wh, {"R1": P(3, 0)}, {"R1": P(3, 6)},
        [DynamicObstacleEvent(3, P(3, 4))],
    )
    assert result.validation_problems == []
    # the executed trace never puts R1 on (3,4)
    # (reconstruct from per_robot final + the fact that validation passed and
    #  the cell became a warehouse obstacle)
    assert not wh.is_traversable(P(3, 4))              # obstacle really was added
    assert sim.get_robot("R1").position == P(3, 6)


# ----------------------------------------------------------------------
# Scenario B - obstacle off the remaining route -> no replanning
# ----------------------------------------------------------------------
def test_obstacle_off_route_causes_no_replanning() -> None:
    wh = Warehouse(width=7, height=7, name="B")
    sim, _, result = _run(
        wh, {"R1": P(3, 0)}, {"R1": P(3, 6)},
        [DynamicObstacleEvent(2, P(0, 0))],
    )
    assert result.success
    assert len(result.recovery_events) == 1
    event = result.recovery_events[0]
    assert event.replanning_attempted is False
    assert event.affected_robot_ids == []
    assert event.recovery_success is True
    assert sim.get_robot("R1").position == P(3, 6)


# ----------------------------------------------------------------------
# Scenario C - obstacle blocks the route, no alternative -> safe stop
# ----------------------------------------------------------------------
def test_no_alternate_route_leads_to_a_safe_stop() -> None:
    wh = Warehouse(width=6, height=1, name="C")         # 1-wide corridor
    sim, _, result = _run(
        wh, {"R1": P(0, 0)}, {"R1": P(0, 5)},
        [DynamicObstacleEvent(2, P(0, 3))],             # R1 at (0,2), (0,3) blocked
    )
    assert result.success is False
    assert result.safe_stop is True
    assert sim.get_robot("R1").position == P(0, 2)      # stopped before the block
    assert sim.get_robot("R1").position != P(0, 3)
    event = result.recovery_events[0]
    assert event.replanning_attempted is True
    assert event.replanning_success is False
    assert event.safe_stop is True
    assert event.failure_reason is not None
    assert result.vertex_conflicts == 0 and result.edge_conflicts == 0


# ----------------------------------------------------------------------
# Scenario D - obstacle invalidates one robot of a coordinated pair
# ----------------------------------------------------------------------
def test_coordinated_single_robot_replan_keeps_others_untouched() -> None:
    wh = Warehouse(width=9, height=9, name="D")
    starts = {"R1": P(4, 0), "R2": P(0, 4)}
    goals = {"R1": P(4, 8), "R2": P(8, 4)}
    sim, coord, result = _run(
        wh, starts, goals, [DynamicObstacleEvent(3, P(4, 5))],
    )
    assert result.success
    assert result.robots_reached_goal == 2
    event = result.recovery_events[0]
    assert event.affected_robot_ids == ["R1"]          # R2 not disturbed
    assert event.unresolved_vertex_conflicts == 0
    assert event.unresolved_edge_conflicts == 0
    assert result.validation_problems == []


# ----------------------------------------------------------------------
# Scenario E - obstacle invalidates more than one robot's route
# ----------------------------------------------------------------------
def test_obstacle_invalidating_two_routes_re_coordinates_the_subset() -> None:
    wh = Warehouse(width=9, height=9, name="E")
    starts = {"R1": P(4, 0), "R2": P(0, 4)}
    goals = {"R1": P(4, 8), "R2": P(8, 4)}             # cross at (4,4)
    sim, coord, result = _run(
        wh, starts, goals, [DynamicObstacleEvent(2, P(4, 4))],
    )
    assert result.success
    assert result.robots_reached_goal == 2
    event = result.recovery_events[0]
    assert set(event.affected_robot_ids) == {"R1", "R2"}
    assert all(r.success for r in event.robot_replans)
    assert event.unresolved_vertex_conflicts == 0
    assert event.unresolved_edge_conflicts == 0
    assert result.validation_problems == []


# ----------------------------------------------------------------------
# Determinism
# ----------------------------------------------------------------------
_TIMING_KEYS = {
    "replanning_time_ms", "replanning_latency_ms", "recoordination_latency_ms",
    "total_recovery_latency_ms", "execution_time_ms",
}


def _strip_timing(obj):
    if isinstance(obj, dict):
        return {k: _strip_timing(v) for k, v in obj.items() if k not in _TIMING_KEYS}
    if isinstance(obj, list):
        return [_strip_timing(v) for v in obj]
    return obj


def test_recovery_is_deterministic() -> None:
    wh1 = Warehouse(width=9, height=9, name="det")
    wh2 = Warehouse(width=9, height=9, name="det")
    starts = {"R1": P(4, 0), "R2": P(0, 4)}
    goals = {"R1": P(4, 8), "R2": P(8, 4)}
    ev = [DynamicObstacleEvent(2, P(4, 4))]
    _, _, a = _run(wh1, starts, goals, ev)
    _, _, b = _run(wh2, starts, goals, ev)
    # logical outcome is identical; only wall-clock latency fields differ
    assert _strip_timing(a.to_dict()) == _strip_timing(b.to_dict())


def test_obstacle_cannot_be_placed_on_a_robot_is_reported_not_crashing() -> None:
    wh = Warehouse(width=7, height=7, name="occ")
    # R1 is at (3,3) at t=3; try to drop the obstacle exactly there
    sim, _, result = _run(
        wh, {"R1": P(3, 0)}, {"R1": P(3, 6)},
        [DynamicObstacleEvent(3, P(3, 3))],
    )
    event = result.recovery_events[0]
    assert "not applied" in (event.failure_reason or "")
    assert result.success                              # R1 still finishes
