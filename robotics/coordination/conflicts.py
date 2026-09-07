"""Independently checking that a set of timed paths is actually conflict-free.

Same philosophy as the path validator in `robotics.planning.validation`: "the
coordinator returned timed paths" and "those timed paths are safe to execute"
are different claims. A bug in the reservation logic - a missed reverse edge, an
off-by-one in a timestep - would still produce confident-looking timed paths.

This validator knows NOTHING about reservation tables or priority order. It
takes the finished timed paths and replays them against each other, cell by
cell and timestep by timestep, asking simple questions:

    does every robot start where it should?
    is every cell in bounds and walkable?
    is every step a single orthogonal move or a WAIT?
    do two robots ever share a cell at the same timestep?     (vertex conflict)
    do two robots ever swap cells in one step?                 (edge/swap conflict)
    once a robot parks on its goal, does anyone drive through it?

Tests, the demo and the benchmark all run planner output through this before
treating it as safe.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

from robotics.coordination.timed_path import TimedPath
from robotics.exceptions import RoboticsError
from robotics.warehouse.grid import Position
from robotics.warehouse.warehouse import Warehouse

TimedPaths = Union[Sequence[TimedPath], Dict[str, TimedPath]]


class CoordinationValidationError(RoboticsError):
    """A coordinated plan failed the independent conflict check."""


def _as_list(timed_paths: TimedPaths) -> List[TimedPath]:
    if isinstance(timed_paths, dict):
        return list(timed_paths.values())
    return list(timed_paths)


def find_coordination_problems(
    timed_paths: TimedPaths,
    warehouse: Warehouse,
    expected_starts: Optional[Dict[str, Position]] = None,
    check_obstacles: bool = True,
) -> List[str]:
    """Return every problem with a set of timed paths. Empty list means safe.

    Args:
        timed_paths: the coordinator's output - a list or {robot_id: TimedPath}.
            Only SUCCESSFUL paths are checked; failed ones are ignored here (the
            coordinator reports those separately).
        warehouse: the floor plan to check bounds / obstacles against, as it is
            NOW.
        expected_starts: optional {robot_id: cell}. When given, each robot's
            first step must be at its expected start cell.
        check_obstacles: when True (default), every cell must be traversable in
            the CURRENT warehouse. The coordinated executor sets this False for
            its pre-flight check - a dynamic obstacle that appeared after
            planning is an execution-time condition it detects per timestep,
            not a flaw in the plan's structure.
    """
    problems: List[str] = []
    paths = [tp for tp in _as_list(timed_paths) if tp.success]

    if not paths:
        return problems

    # ---- per-path checks -------------------------------------------------
    for tp in paths:
        rid = tp.robot_id

        if not tp.steps:
            problems.append(f"{rid}: successful timed path has no steps")
            continue

        if expected_starts is not None and rid in expected_starts:
            if tp.steps[0].position != expected_starts[rid]:
                problems.append(
                    f"{rid}: starts at {tp.steps[0].position}, expected "
                    f"{expected_starts[rid]}"
                )

        # timesteps must be consecutive integers
        for i, step in enumerate(tp.steps):
            want = tp.steps[0].timestep + i
            if step.timestep != want:
                problems.append(
                    f"{rid}: step {i} has timestep {step.timestep}, expected {want}"
                )

        # every cell in bounds and walkable
        for i, step in enumerate(tp.steps):
            if not warehouse.in_bounds(step.position):
                problems.append(
                    f"{rid}: step {i} {step.position} is outside the warehouse"
                )
            elif check_obstacles and not warehouse.is_traversable(step.position):
                problems.append(
                    f"{rid}: step {i} {step.position} is blocked by an obstacle"
                )

        # every transition is one orthogonal move or a WAIT
        for i in range(len(tp.steps) - 1):
            a, b = tp.steps[i].position, tp.steps[i + 1].position
            gap = a.manhattan_distance_to(b)
            if gap not in (0, 1):
                problems.append(
                    f"{rid}: illegal transition from step {i} {a} to step "
                    f"{i + 1} {b} (distance {gap}; must be a WAIT or one step)"
                )

    # ---- cross-path checks (vertex + edge conflicts) -------------------
    horizon = max(tp.arrival_time for tp in paths)
    start_time = min(tp.start_time for tp in paths)

    # vertex conflicts: two robots on the same cell at the same timestep.
    for t in range(start_time, horizon + 1):
        seen: Dict[Position, str] = {}
        for tp in paths:
            pos = tp.position_at(t)
            if pos is None:
                continue
            if pos in seen:
                problems.append(
                    f"vertex conflict at {pos}, t={t}: robots "
                    f"'{seen[pos]}' and '{tp.robot_id}'"
                )
            else:
                seen[pos] = tp.robot_id

    # edge / swap conflicts: two robots trade places in one step.
    for t in range(start_time, horizon):
        moves: Dict[tuple, str] = {}  # (from, to) -> robot_id
        for tp in paths:
            a = tp.position_at(t)
            b = tp.position_at(t + 1)
            if a is None or b is None or a == b:
                continue
            moves[(a, b)] = tp.robot_id
        for (a, b), rid in moves.items():
            other = moves.get((b, a))
            if other is not None and other != rid and rid < other:
                problems.append(
                    f"edge/swap conflict between '{rid}' and '{other}' over "
                    f"{a}<->{b} during t={t}->{t + 1}"
                )

    # ---- goal occupation policy --------------------------------------
    # After a robot arrives, its goal is its permanent parking spot for the
    # rest of the episode; no other robot may be on that cell afterwards.
    for tp in paths:
        goal = tp.steps[-1].position
        for other in paths:
            if other.robot_id == tp.robot_id:
                continue
            for t in range(tp.arrival_time, horizon + 1):
                if other.position_at(t) == goal:
                    problems.append(
                        f"goal-occupation violation: '{other.robot_id}' is on "
                        f"'{tp.robot_id}' goal {goal} at t={t} "
                        f"(arrived t={tp.arrival_time})"
                    )
                    break

    return problems


def is_conflict_free(
    timed_paths: TimedPaths,
    warehouse: Warehouse,
    expected_starts: Optional[Dict[str, Position]] = None,
) -> bool:
    """True if the timed paths pass every check."""
    return not find_coordination_problems(timed_paths, warehouse, expected_starts)


def assert_conflict_free(
    timed_paths: TimedPaths,
    warehouse: Warehouse,
    expected_starts: Optional[Dict[str, Position]] = None,
) -> None:
    """Raise if the timed paths are not conflict-free. For invariant checks."""
    problems = find_coordination_problems(timed_paths, warehouse, expected_starts)
    if problems:
        raise CoordinationValidationError(
            "coordinated plan is not conflict-free: " + "; ".join(problems)
        )


class ConflictValidator:
    """Object wrapper, for callers that prefer to hold a configured validator."""

    def __init__(self, warehouse: Warehouse) -> None:
        self.warehouse = warehouse

    def find_problems(
        self,
        timed_paths: TimedPaths,
        expected_starts: Optional[Dict[str, Position]] = None,
    ) -> List[str]:
        return find_coordination_problems(timed_paths, self.warehouse, expected_starts)

    def is_conflict_free(
        self,
        timed_paths: TimedPaths,
        expected_starts: Optional[Dict[str, Position]] = None,
    ) -> bool:
        return is_conflict_free(timed_paths, self.warehouse, expected_starts)
