"""Tests for the space-time reservation table."""

from __future__ import annotations

import pytest

from robotics.coordination.reservation_table import ReservationTable
from robotics.coordination.timed_path import TimedPath, TimedStep
from robotics.warehouse.grid import Position


def P(r, c):
    return Position(r, c)


# ----------------------------------------------------------------------
# Vertex reservations
# ----------------------------------------------------------------------
def test_vertex_reservation_insert_and_lookup() -> None:
    table = ReservationTable()
    table.reserve_vertex(P(1, 1), 3, "R1")

    assert table.vertex_reserver(P(1, 1), 3) == "R1"
    assert table.vertex_reserver(P(1, 1), 4) is None
    assert table.vertex_reserver(P(1, 2), 3) is None


def test_is_vertex_free_respects_the_asking_robot() -> None:
    table = ReservationTable()
    table.reserve_vertex(P(2, 2), 5, "R1")

    assert table.is_vertex_free(P(2, 2), 5) is False
    assert table.is_vertex_free(P(2, 2), 5, for_robot="R2") is False
    assert table.is_vertex_free(P(2, 2), 5, for_robot="R1") is True   # its own
    assert table.is_vertex_free(P(2, 2), 6) is True                    # other time


def test_same_cell_different_timesteps_do_not_conflict() -> None:
    table = ReservationTable()
    table.reserve_vertex(P(0, 0), 1, "R1")
    table.reserve_vertex(P(0, 0), 2, "R2")

    assert table.vertex_reserver(P(0, 0), 1) == "R1"
    assert table.vertex_reserver(P(0, 0), 2) == "R2"


def test_reserving_same_key_for_same_robot_is_harmless() -> None:
    table = ReservationTable()
    table.reserve_vertex(P(1, 1), 3, "R1")
    table.reserve_vertex(P(1, 1), 3, "R1")  # no raise
    assert table.vertex_reserver(P(1, 1), 3) == "R1"


def test_reserving_same_key_for_different_robot_raises() -> None:
    table = ReservationTable()
    table.reserve_vertex(P(1, 1), 3, "R1")
    with pytest.raises(ValueError):
        table.reserve_vertex(P(1, 1), 3, "R2")


# ----------------------------------------------------------------------
# Edge reservations and swap detection
# ----------------------------------------------------------------------
def test_edge_reservation_insert_and_lookup() -> None:
    table = ReservationTable()
    table.reserve_edge(P(1, 1), P(1, 2), 4, "R1")

    assert table.edge_reserver(P(1, 1), P(1, 2), 4) == "R1"
    assert table.edge_reserver(P(1, 2), P(1, 1), 4) is None  # reverse not reserved
    assert table.edge_reserver(P(1, 1), P(1, 2), 5) is None  # other time


def test_reverse_edge_is_a_swap_conflict() -> None:
    table = ReservationTable()
    # R1 moves (1,1) -> (1,2) during t=4 -> t=5
    table.reserve_edge(P(1, 1), P(1, 2), 4, "R1")

    # R2 wants to move the opposite way (1,2) -> (1,1) during the same step
    assert table.is_swap_conflict(P(1, 2), P(1, 1), 4, for_robot="R2") is True
    # a different timestep is fine
    assert table.is_swap_conflict(P(1, 2), P(1, 1), 5, for_robot="R2") is False
    # the same direction is not a swap (it is a vertex conflict, handled elsewhere)
    assert table.is_swap_conflict(P(1, 1), P(1, 2), 4, for_robot="R2") is False


def test_is_move_allowed_combines_vertex_and_swap_checks() -> None:
    table = ReservationTable()
    table.reserve_vertex(P(2, 3), 6, "R1")          # R1 sits on (2,3) at t=6
    table.reserve_edge(P(0, 0), P(0, 1), 2, "R1")   # R1 crosses (0,0)->(0,1) at t=2

    # R2 moving into (2,3) arriving at t=6 -> blocked by the vertex reservation
    assert table.is_move_allowed(P(2, 2), P(2, 3), 5, "R2") is False
    # R2 moving (0,1)->(0,0) during t=2 -> blocked by the swap
    assert table.is_move_allowed(P(0, 1), P(0, 0), 2, "R2") is False
    # an unrelated move is fine
    assert table.is_move_allowed(P(4, 4), P(4, 5), 1, "R2") is True


def test_wait_is_allowed_check_uses_the_next_timestep_vertex() -> None:
    table = ReservationTable()
    table.reserve_vertex(P(1, 1), 4, "R1")
    # R2 waiting at (1,1) from t=3 to t=4 lands on the reserved (1,1)@4
    assert table.is_move_allowed(P(1, 1), P(1, 1), 3, "R2") is False
    # waiting from t=4 to t=5 is fine
    assert table.is_move_allowed(P(1, 1), P(1, 1), 4, "R2") is True


# ----------------------------------------------------------------------
# Reserving a whole timed path + goal hold
# ----------------------------------------------------------------------
def _straight_path(robot_id: str) -> TimedPath:
    return TimedPath.found(
        robot_id,
        [TimedStep(P(1, 1), 0), TimedStep(P(1, 2), 1), TimedStep(P(1, 3), 2)],
    )


def test_reserve_timed_path_claims_vertices_edges_and_goal_hold() -> None:
    table = ReservationTable()
    table.reserve_timed_path(_straight_path("R1"), horizon=6)

    # vertices along the path
    assert table.vertex_reserver(P(1, 1), 0) == "R1"
    assert table.vertex_reserver(P(1, 2), 1) == "R1"
    assert table.vertex_reserver(P(1, 3), 2) == "R1"
    # edges along the path
    assert table.edge_reserver(P(1, 1), P(1, 2), 0) == "R1"
    assert table.edge_reserver(P(1, 2), P(1, 3), 1) == "R1"
    # goal hold: the goal cell reserved from arrival (t=2) through the horizon
    for t in range(2, 7):
        assert table.vertex_reserver(P(1, 3), t) == "R1"
    assert table.vertex_reserver(P(1, 3), 7) is None  # beyond the horizon


def test_reserve_timed_path_with_a_wait_reserves_no_edge_for_the_wait() -> None:
    table = ReservationTable()
    waiting = TimedPath.found(
        "R1",
        [
            TimedStep(P(0, 0), 0),
            TimedStep(P(0, 1), 1),
            TimedStep(P(0, 1), 2),  # WAIT
            TimedStep(P(0, 2), 3),
        ],
    )
    table.reserve_timed_path(waiting, horizon=5)

    assert table.edge_reserver(P(0, 1), P(0, 1), 1) is None  # no self-edge
    assert table.vertex_reserver(P(0, 1), 1) == "R1"
    assert table.vertex_reserver(P(0, 1), 2) == "R1"          # the wait
    assert table.edge_reserver(P(0, 1), P(0, 2), 2) == "R1"


def test_reserving_an_unsuccessful_path_raises() -> None:
    table = ReservationTable()
    with pytest.raises(ValueError):
        table.reserve_timed_path(TimedPath.failed("R1", "no route"), horizon=5)


def test_conflict_check_counter_increments_on_lookups() -> None:
    table = ReservationTable()
    assert table.conflict_checks == 0
    table.vertex_reserver(P(0, 0), 0)
    table.edge_reserver(P(0, 0), P(0, 1), 0)
    table.is_vertex_free(P(0, 0), 0)
    assert table.conflict_checks == 3


def test_reservation_table_is_deterministic() -> None:
    def build():
        t = ReservationTable()
        t.reserve_timed_path(_straight_path("R1"), horizon=6)
        return (
            t.vertex_reservation_count,
            t.edge_reservation_count,
            t.vertex_reserver(P(1, 3), 5),
        )

    assert build() == build()
