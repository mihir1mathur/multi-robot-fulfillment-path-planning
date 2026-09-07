"""Tests for the reservation-release helpers used by dynamic recovery."""

from __future__ import annotations

from robotics.coordination.reservation_table import ReservationTable
from robotics.coordination.timed_path import TimedPath, TimedStep
from robotics.recovery.dynamic_replanner import (
    build_future_reservations,
    remaining_path_problem,
    route_touches_cell,
    splice,
)
from robotics.warehouse.grid import Position
from robotics.warehouse.warehouse import Warehouse


def P(r, c):
    return Position(r, c)


def _tp(rid, cells_times):
    return TimedPath.found(rid, [TimedStep(P(r, c), t) for (r, c), t in cells_times])


# ----------------------------------------------------------------------
# ReservationTable.release_robot / reservations_owned_by / block_cell
# ----------------------------------------------------------------------
def test_release_robot_drops_only_that_robots_reservations() -> None:
    table = ReservationTable()
    table.reserve_timed_path(_tp("R1", [((1, 1), 0), ((1, 2), 1), ((1, 3), 2)]), horizon=5)
    table.reserve_timed_path(_tp("R2", [((3, 1), 0), ((3, 2), 1)]), horizon=5)

    before_r1 = table.reservations_owned_by("R1")
    assert before_r1 > 0
    released = table.release_robot("R1")

    assert released == before_r1
    assert table.reservations_owned_by("R1") == 0
    assert table.vertex_reserver(P(3, 1), 0) == "R2"          # R2 untouched
    assert table.vertex_reserver(P(1, 2), 1) is None          # R1 gone


def test_block_cell_pins_a_failed_robot() -> None:
    table = ReservationTable()
    added = table.block_cell(P(4, 4), 3, 8, "__failed__R2")

    assert added == 6                                          # t = 3..8
    for t in range(3, 9):
        assert table.vertex_reserver(P(4, 4), t) == "__failed__R2"
    assert table.vertex_reserver(P(4, 4), 2) is None
    # another robot cannot move onto it
    assert table.is_move_allowed(P(4, 3), P(4, 4), 4, "R9") is False


def test_release_robot_returns_zero_for_an_unknown_robot() -> None:
    table = ReservationTable()
    assert table.release_robot("ghost") == 0


# ----------------------------------------------------------------------
# build_future_reservations
# ----------------------------------------------------------------------
def test_build_future_reservations_only_covers_from_the_given_timestep() -> None:
    paths = {
        "R1": _tp("R1", [((0, 0), 0), ((0, 1), 1), ((0, 2), 2), ((0, 3), 3)]),
        "R2": _tp("R2", [((5, 0), 0), ((5, 1), 1), ((5, 2), 2)]),
    }
    table = build_future_reservations(paths, from_timestep=2, horizon=6)

    assert table.vertex_reserver(P(0, 0), 0) is None          # in the past
    assert table.vertex_reserver(P(0, 2), 2) == "R1"          # from t=2 on
    assert table.vertex_reserver(P(0, 3), 3) == "R1"
    assert table.vertex_reserver(P(0, 3), 6) == "R1"          # goal hold


def test_build_future_reservations_excludes_named_robots() -> None:
    paths = {
        "R1": _tp("R1", [((0, 0), 0), ((0, 1), 1)]),
        "R2": _tp("R2", [((1, 0), 0), ((1, 1), 1)]),
    }
    table = build_future_reservations(paths, 0, 5, exclude=["R1"])
    assert table.reservations_owned_by("R1") == 0
    assert table.reservations_owned_by("R2") > 0


def test_build_future_reservations_pins_a_parked_robot() -> None:
    # R2's path ended at t=1; from t=3 it is parked and must still be reserved
    paths = {"R2": _tp("R2", [((2, 2), 0), ((2, 3), 1)])}
    table = build_future_reservations(paths, from_timestep=3, horizon=6)
    for t in range(3, 7):
        assert table.vertex_reserver(P(2, 3), t) == "R2"


# ----------------------------------------------------------------------
# remaining_path_problem
# ----------------------------------------------------------------------
def test_remaining_path_problem_none_when_clear() -> None:
    wh = Warehouse(width=6, height=6, name="rp")
    tp = _tp("R1", [((0, 0), 0), ((0, 1), 1), ((0, 2), 2), ((0, 3), 3)])
    assert remaining_path_problem(tp, from_timestep=1, warehouse=wh) is None


def test_remaining_path_problem_detects_a_new_obstacle() -> None:
    wh = Warehouse(width=6, height=6, name="rp")
    wh.add_dynamic_obstacle("spill", P(0, 2), "spill")
    tp = _tp("R1", [((0, 0), 0), ((0, 1), 1), ((0, 2), 2), ((0, 3), 3)])
    problem = remaining_path_problem(tp, from_timestep=1, warehouse=wh)
    assert problem is not None and "(0, 2)" in problem


def test_remaining_path_problem_ignores_the_already_travelled_prefix() -> None:
    wh = Warehouse(width=6, height=6, name="rp")
    wh.add_dynamic_obstacle("spill", P(0, 1), "spill")   # on a PAST cell
    tp = _tp("R1", [((0, 0), 0), ((0, 1), 1), ((0, 2), 2)])
    # from t=2 the robot is at (0,2) - the (0,1) obstacle is behind it
    assert remaining_path_problem(tp, from_timestep=2, warehouse=wh) is None


def test_remaining_path_problem_detects_a_new_reservation_clash() -> None:
    wh = Warehouse(width=6, height=6, name="rp")
    tp = _tp("R1", [((0, 0), 0), ((0, 1), 1), ((0, 2), 2)])
    table = ReservationTable()
    table.reserve_vertex(P(0, 2), 2, "R9")
    problem = remaining_path_problem(tp, 1, wh, table)
    assert problem is not None and "R9" in problem


# ----------------------------------------------------------------------
# route_touches_cell / splice
# ----------------------------------------------------------------------
def test_route_touches_cell_respects_the_from_timestep() -> None:
    tp = _tp("R1", [((0, 0), 0), ((0, 1), 1), ((0, 2), 2)])
    assert route_touches_cell(tp, P(0, 1), from_timestep=0) is True
    assert route_touches_cell(tp, P(0, 1), from_timestep=2) is False  # in the past
    assert route_touches_cell(tp, P(0, 2), from_timestep=2) is True


def test_splice_joins_history_and_future_without_duplicating_the_seam() -> None:
    history = [TimedStep(P(0, 0), 0), TimedStep(P(0, 1), 1), TimedStep(P(1, 1), 2)]
    future = _tp("R1", [((1, 1), 2), ((1, 2), 3), ((1, 3), 4)])
    spliced = splice(history, future)
    assert spliced.positions() == [P(0, 0), P(0, 1), P(1, 1), P(1, 2), P(1, 3)]
    assert [s.timestep for s in spliced.steps] == [0, 1, 2, 3, 4]
