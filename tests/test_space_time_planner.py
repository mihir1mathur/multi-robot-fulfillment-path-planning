"""Tests for Space-Time A*."""

from __future__ import annotations

import pytest

from robotics.coordination.reservation_table import ReservationTable
from robotics.coordination.space_time_planner import (
    REASON_GOAL_HELD,
    REASON_HORIZON,
    REASON_SPATIALLY_UNREACHABLE,
    SpaceTimePlanner,
)
from robotics.coordination.timed_path import TimedStep
from robotics.planning.astar import plan as plan_spatial
from robotics.warehouse.grid import Position
from robotics.warehouse.warehouse import Warehouse


def P(r, c):
    return Position(r, c)


def _wh(w=8, h=8):
    return Warehouse(width=w, height=h, name="stp-test")


def _plan(wh, start, goal, reservations=None, robot_id="R1", start_time=0, horizon=60):
    return SpaceTimePlanner(wh).plan(
        start, goal, reservations or ReservationTable(), robot_id, start_time, horizon
    )


# ----------------------------------------------------------------------
# With no reservations it is just a shortest route
# ----------------------------------------------------------------------
def test_one_robot_on_an_empty_grid() -> None:
    tp = _plan(_wh(), P(0, 0), P(0, 3))
    assert tp.success
    assert tp.positions() == [P(0, 0), P(0, 1), P(0, 2), P(0, 3)]
    assert tp.move_count == 3
    assert tp.wait_count == 0


def test_cost_matches_the_spatial_shortest_route_when_unreserved() -> None:
    wh = _wh()
    for goal in (P(5, 5), P(7, 0), P(3, 6)):
        spatial = plan_spatial(P(0, 0), goal, wh)
        timed = _plan(wh, P(0, 0), goal)
        assert timed.success
        assert timed.arrival_time == spatial.total_cost


def test_start_equals_goal() -> None:
    tp = _plan(_wh(), P(2, 2), P(2, 2))
    assert tp.success
    assert tp.positions() == [P(2, 2)]
    assert tp.move_count == 0 and tp.wait_count == 0


# ----------------------------------------------------------------------
# Reservations change the route
# ----------------------------------------------------------------------
def test_routes_around_a_permanently_reserved_cell() -> None:
    wh = _wh()
    reservations = ReservationTable()
    # (0,2) is taken for a long time - the robot must detour off row 0
    for t in range(0, 40):
        reservations.reserve_vertex(P(0, 2), t, "OTHER")
    tp = _plan(wh, P(0, 0), P(0, 4), reservations)
    assert tp.success
    assert P(0, 2) not in tp.positions()


def test_waits_for_a_temporary_reservation_to_clear() -> None:
    wh = _wh()
    reservations = ReservationTable()
    reservations.reserve_vertex(P(0, 2), 2, "OTHER")  # only blocked at t=2
    tp = _plan(wh, P(0, 0), P(0, 3), reservations)
    assert tp.success
    assert tp.wait_count == 1                     # one WAIT to let the cell clear
    assert tp.arrival_time == 4                   # 3-move route + 1 wait
    assert P(0, 2) not in [s.position for s in tp.steps if s.timestep == 2]


def test_avoids_a_vertex_conflict_with_a_reserved_path() -> None:
    wh = _wh()
    reservations = ReservationTable()
    # OTHER occupies the diagonal-ish line the robot would cross
    for cell, t in [(P(2, 0), 0), (P(2, 1), 1), (P(2, 2), 2), (P(2, 3), 3)]:
        reservations.reserve_vertex(cell, t, "OTHER")
    tp = _plan(wh, P(0, 2), P(4, 2), reservations)
    assert tp.success
    # never on OTHER's cell at OTHER's timestep
    for step in tp.steps:
        assert reservations.vertex_reserver(step.position, step.timestep) != "OTHER"


def test_avoids_an_edge_swap() -> None:
    wh = _wh()
    reservations = ReservationTable()
    # OTHER moves (1,1) -> (1,2) during t=0 -> t=1
    reservations.reserve_vertex(P(1, 1), 0, "OTHER")
    reservations.reserve_vertex(P(1, 2), 1, "OTHER")
    reservations.reserve_edge(P(1, 1), P(1, 2), 0, "OTHER")
    # our robot at (1,2) wants (1,1) - the straight move would swap
    tp = _plan(wh, P(1, 2), P(1, 0), reservations)
    assert tp.success
    # it must not do (1,2)->(1,1) during t=0->t=1
    assert not (tp.positions()[0] == P(1, 2) and tp.positions()[1] == P(1, 1))


# ----------------------------------------------------------------------
# Obstacles
# ----------------------------------------------------------------------
def test_respects_a_static_obstacle() -> None:
    wh = _wh()
    wh.add_static_obstacle("rack", P(0, 2))
    tp = _plan(wh, P(0, 1), P(0, 3))
    assert tp.success
    assert P(0, 2) not in tp.positions()
    assert tp.arrival_time == 4  # around the rack


def test_respects_a_preexisting_dynamic_obstacle() -> None:
    wh = _wh()
    wh.add_dynamic_obstacle("spill", P(0, 2), "spill")
    tp = _plan(wh, P(0, 1), P(0, 3))
    assert tp.success
    assert P(0, 2) not in tp.positions()


# ----------------------------------------------------------------------
# Failure modes - distinguishable reasons
# ----------------------------------------------------------------------
def test_spatially_unreachable_goal() -> None:
    wh = Warehouse(width=5, height=5, name="sealed")
    for cell in (P(0, 3), P(1, 3), P(1, 4)):
        wh.add_static_obstacle(f"seal-{cell.row}-{cell.col}", cell)
    tp = _plan(wh, P(4, 0), P(0, 4))
    assert tp.success is False
    assert tp.failure_reason == REASON_SPATIALLY_UNREACHABLE


def test_temporally_blocked_goal_reports_the_hold_reason() -> None:
    wh = _wh()
    reservations = ReservationTable()
    for t in range(0, 61):
        reservations.reserve_vertex(P(0, 3), t, "OTHER")  # goal held forever
    tp = _plan(wh, P(0, 0), P(0, 3), reservations)
    assert tp.success is False
    assert tp.failure_reason == REASON_GOAL_HELD


def test_horizon_exceeded_reports_the_horizon_reason() -> None:
    wh = _wh()
    reservations = ReservationTable()
    # wall off row 0 cols 2 for the whole (small) horizon so the only route is a
    # detour that does not fit
    for t in range(0, 8):
        for r in range(0, 8):
            if r != 0:
                reservations.reserve_vertex(P(r, 2), t, "OTHER")
    tp = SpaceTimePlanner(wh).plan(
        P(0, 0), P(0, 4), reservations, "R1", start_time=0, horizon=3
    )
    assert tp.success is False
    assert tp.failure_reason == REASON_HORIZON


# ----------------------------------------------------------------------
# Determinism / validity
# ----------------------------------------------------------------------
def test_is_deterministic() -> None:
    wh = _wh()
    reservations = ReservationTable()
    reservations.reserve_vertex(P(0, 2), 2, "OTHER")
    first = _plan(wh, P(0, 0), P(0, 5), reservations)
    reservations2 = ReservationTable()
    reservations2.reserve_vertex(P(0, 2), 2, "OTHER")
    second = _plan(wh, P(0, 0), P(0, 5), reservations2)
    assert [s.__dict__ for s in first.steps] == [s.__dict__ for s in second.steps]


def test_produces_a_timed_path_with_consecutive_timesteps_and_unit_steps() -> None:
    wh = _wh()
    reservations = ReservationTable()
    reservations.reserve_vertex(P(3, 3), 3, "OTHER")
    tp = _plan(wh, P(0, 0), P(6, 6), reservations)
    assert tp.success
    for i, step in enumerate(tp.steps):
        assert step.timestep == i
    for a, b in zip(tp.steps, tp.steps[1:]):
        assert a.position.manhattan_distance_to(b.position) in (0, 1)


def test_goal_stays_reservable_from_the_arrival_time() -> None:
    wh = _wh()
    reservations = ReservationTable()
    # goal (0,3) is blocked only up to t=2; robot must arrive at t>=3
    for t in range(0, 3):
        reservations.reserve_vertex(P(0, 3), t, "OTHER")
    tp = _plan(wh, P(0, 0), P(0, 3), reservations)
    assert tp.success
    assert tp.arrival_time >= 3


def test_out_of_bounds_and_blocked_endpoints_fail_cleanly() -> None:
    wh = _wh(5, 5)
    assert _plan(wh, P(-1, 0), P(0, 0)).success is False
    assert _plan(wh, P(0, 0), P(9, 9)).success is False
    wh.add_static_obstacle("rack", P(2, 2))
    assert _plan(wh, P(2, 2), P(0, 0)).success is False
    assert _plan(wh, P(0, 0), P(2, 2)).success is False
