"""Driving a fleet of robots along their timed paths, one timestep at a time.

THE KEY DIFFERENCE FROM SINGLE-ROBOT EXECUTION
---------------------------------------------
`RouteExecutor` drives ONE robot to completion. If you called it for R1 then
R2, R1 could finish parked on a cell R2 needs, and R2 would stop. That is
exactly the failure coordination is meant to remove.

`CoordinatedExecutor` advances a shared clock. At each timestep it:

    PHASE 1 - VALIDATE the whole joint move
        * each robot's intended cell is in bounds, walkable NOW, adjacent (or a
          WAIT), and the robot has battery for a move
        * no two robots want the same cell            (vertex conflict)
        * no two robots swap cells                     (edge / swap conflict)

    PHASE 2 - COMMIT
        only if PHASE 1 passed for EVERY robot, apply all the moves

If PHASE 1 fails, NOTHING is moved that timestep and execution stops safely.
This "validate first, commit second" split is what prevents the partial-mutation
bug where R1 moves, R2 then fails, and the world is left inconsistent.

BATTERY / DISTANCE SEMANTICS
---------------------------
A MOVE goes through `Robot.move_to`, so it updates position, step count,
odometer and battery exactly like every other move in the project. A WAIT
consumes a timestep but adds no distance and no odometry, and - because the
existing battery model only charges for movement - no battery. That WAIT costs
no battery is a simulation simplification, noted as such.

NO REPLANNING
-------------
If a dynamic obstacle appears on a robot's next cell during execution, PHASE 1
catches it and execution stops. Repairing the plan is a later capability.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from robotics.coordination.conflicts import find_coordination_problems
from robotics.coordination.coordination_result import CoordinationResult
from robotics.coordination.timed_path import TimedPath, TimedStep
from robotics.robots.robot_state import RobotStatus
from robotics.warehouse.grid import Position


@dataclass
class RobotExecutionSummary:
    """How one robot fared during a coordinated run."""

    robot_id: str
    start: Position
    goal: Position
    final_position: Position
    reached_goal: bool
    moves: int
    waits: int
    distance_travelled: float
    battery_before: float
    battery_after: float

    @property
    def battery_consumed(self) -> float:
        return self.battery_before - self.battery_after

    def to_dict(self) -> Dict[str, Any]:
        return {
            "robot_id": self.robot_id,
            "start": [self.start.row, self.start.col],
            "goal": [self.goal.row, self.goal.col],
            "final_position": [self.final_position.row, self.final_position.col],
            "reached_goal": self.reached_goal,
            "moves": self.moves,
            "waits": self.waits,
            "distance_travelled": self.distance_travelled,
            "battery_before": round(self.battery_before, 4),
            "battery_after": round(self.battery_after, 4),
            "battery_consumed": round(self.battery_consumed, 4),
        }


@dataclass
class CoordinatedExecutionResult:
    """The outcome of `CoordinatedExecutor.execute`."""

    success: bool
    executed_robot_ids: List[str] = field(default_factory=list)
    not_executed_robot_ids: List[str] = field(default_factory=list)
    per_robot: Dict[str, RobotExecutionSummary] = field(default_factory=dict)
    robots_reached_goal: int = 0
    total_move_steps: int = 0
    total_wait_steps: int = 0
    makespan_executed: int = 0
    total_distance: float = 0.0
    total_battery_consumed: float = 0.0
    vertex_conflicts: int = 0
    edge_conflicts: int = 0
    validation_problems: List[str] = field(default_factory=list)
    execution_time_ms: float = 0.0
    failure_reason: Optional[str] = None

    @property
    def validated_conflict_free(self) -> bool:
        return not self.validation_problems

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "executed_robot_ids": list(self.executed_robot_ids),
            "not_executed_robot_ids": list(self.not_executed_robot_ids),
            "robots_reached_goal": self.robots_reached_goal,
            "total_move_steps": self.total_move_steps,
            "total_wait_steps": self.total_wait_steps,
            "makespan_executed": self.makespan_executed,
            "total_distance": round(self.total_distance, 4),
            "total_battery_consumed": round(self.total_battery_consumed, 4),
            "vertex_conflicts": self.vertex_conflicts,
            "edge_conflicts": self.edge_conflicts,
            "validation_problems": list(self.validation_problems),
            "execution_time_ms": round(self.execution_time_ms, 6),
            "failure_reason": self.failure_reason,
            "per_robot": {rid: s.to_dict() for rid, s in self.per_robot.items()},
        }

    def __str__(self) -> str:
        head = "OK" if self.success else f"STOPPED ({self.failure_reason})"
        return (
            f"coordinated execution {head}: {self.robots_reached_goal}/"
            f"{len(self.executed_robot_ids)} reached goal | "
            f"{self.total_move_steps} moves, {self.total_wait_steps} waits | "
            f"{self.vertex_conflicts} vertex + {self.edge_conflicts} edge "
            f"conflicts during execution"
        )


class CoordinatedExecutor:
    """Executes a `CoordinationResult`'s timed paths in synchronised timesteps."""

    def __init__(self, simulator) -> None:
        self.simulator = simulator

    # ------------------------------------------------------------------
    def execute(
        self, coordination: CoordinationResult
    ) -> CoordinatedExecutionResult:
        """Drive every successfully-planned robot along its timed path.

        Robots whose planning FAILED are not executed and are listed in
        `not_executed_robot_ids`. The successfully-planned subset is
        mutually conflict-free by construction, so it is safe to run.
        """
        started_ns = time.perf_counter_ns()

        def elapsed_ms() -> float:
            return (time.perf_counter_ns() - started_ns) / 1_000_000

        paths: Dict[str, TimedPath] = dict(coordination.successful_paths)
        not_executed = list(coordination.failed_robot_ids)

        # --- pre-flight: independent conflict check on the plan STRUCTURE ---
        # (`check_obstacles=False`: an obstacle that appeared after planning is
        # caught per-timestep during execution, not treated as a plan flaw.)
        problems = find_coordination_problems(
            paths, self.simulator.warehouse, check_obstacles=False
        )
        if problems:
            return CoordinatedExecutionResult(
                success=False,
                executed_robot_ids=[],
                not_executed_robot_ids=not_executed + sorted(paths),
                validation_problems=problems,
                execution_time_ms=elapsed_ms(),
                failure_reason=(
                    "the coordinated plan did not pass independent conflict "
                    "validation; nothing was executed"
                ),
            )

        # --- pre-flight: robots exist and stand at their planned start ---
        for robot_id, tp in sorted(paths.items()):
            try:
                robot = self.simulator.get_robot(robot_id)
            except Exception as error:  # SimulationError for unknown id
                return self._preflight_failure(
                    not_executed, paths, str(error), elapsed_ms()
                )
            if robot.status is RobotStatus.OFFLINE:
                return self._preflight_failure(
                    not_executed, paths,
                    f"robot '{robot_id}' is offline", elapsed_ms(),
                )
            if robot.position != tp.start:
                return self._preflight_failure(
                    not_executed, paths,
                    f"robot '{robot_id}' is at {robot.position} but its timed "
                    f"path starts at {tp.start}",
                    elapsed_ms(),
                )

        if not paths:
            return CoordinatedExecutionResult(
                success=True,
                executed_robot_ids=[],
                not_executed_robot_ids=not_executed,
                execution_time_ms=elapsed_ms(),
            )

        return self._run(paths, not_executed, elapsed_ms, started_ns)

    # ------------------------------------------------------------------
    def _run(self, paths, not_executed, elapsed_ms, started_ns):
        order = sorted(paths)
        makespan = max(tp.arrival_time for tp in paths.values())

        battery_before = {
            rid: self.simulator.get_robot(rid).battery_level for rid in order
        }
        odometer_before = {
            rid: self.simulator.get_robot(rid).distance_travelled for rid in order
        }
        wait_counts: Dict[str, int] = {rid: 0 for rid in order}
        move_counts: Dict[str, int] = {rid: 0 for rid in order}
        # executed trace, for the post-run invariant check
        trace: Dict[str, List[TimedStep]] = {
            rid: [TimedStep(paths[rid].start, 0)] for rid in order
        }

        vertex_conflicts = 0
        edge_conflicts = 0
        stop_reason: Optional[str] = None
        executed_timesteps = 0

        for t in range(1, makespan + 1):
            intents = self._intents_at(paths, order, t)

            # --- PHASE 1: validate the whole joint step ---
            ok, reason, vc, ec = self._validate_joint_step(order, intents)
            vertex_conflicts += vc
            edge_conflicts += ec
            if not ok:
                stop_reason = f"stopped before timestep {t}: {reason}"
                break

            # --- PHASE 2: commit every move/wait ---
            for robot_id in order:
                frm, to = intents[robot_id]
                arrival = paths[robot_id].arrival_time
                if frm == to:
                    # A WAIT the planner chose (before arrival) is coordination
                    # cost. Staying put AFTER arrival is just the robot being
                    # done while the others finish - not counted as anything.
                    if t <= arrival:
                        wait_counts[robot_id] += 1
                else:
                    self.simulator.get_robot(robot_id).move_to(to)
                    move_counts[robot_id] += 1
                trace[robot_id].append(TimedStep(to, t))

            executed_timesteps = t

        # --- assemble per-robot summaries ---
        per_robot: Dict[str, RobotExecutionSummary] = {}
        reached = 0
        total_distance = 0.0
        total_battery = 0.0
        for robot_id in order:
            robot = self.simulator.get_robot(robot_id)
            tp = paths[robot_id]
            distance = robot.distance_travelled - odometer_before[robot_id]
            consumed = battery_before[robot_id] - robot.battery_level
            arrived = robot.position == tp.goal
            if arrived:
                reached += 1
            total_distance += distance
            total_battery += consumed
            per_robot[robot_id] = RobotExecutionSummary(
                robot_id=robot_id,
                start=tp.start,
                goal=tp.goal,
                final_position=robot.position,
                reached_goal=arrived,
                moves=move_counts[robot_id],
                waits=wait_counts[robot_id],
                distance_travelled=distance,
                battery_before=battery_before[robot_id],
                battery_after=robot.battery_level,
            )

        # --- independent check of what ACTUALLY happened ---
        executed_paths = {
            rid: TimedPath.found(rid, steps) for rid, steps in trace.items()
        }
        validation_problems = find_coordination_problems(
            executed_paths, self.simulator.warehouse
        )

        success = stop_reason is None and not validation_problems

        return CoordinatedExecutionResult(
            success=success,
            executed_robot_ids=order,
            not_executed_robot_ids=not_executed,
            per_robot=per_robot,
            robots_reached_goal=reached,
            total_move_steps=sum(move_counts.values()),
            total_wait_steps=sum(wait_counts.values()),
            makespan_executed=executed_timesteps,
            total_distance=total_distance,
            total_battery_consumed=total_battery,
            vertex_conflicts=vertex_conflicts,
            edge_conflicts=edge_conflicts,
            validation_problems=validation_problems,
            execution_time_ms=elapsed_ms(),
            failure_reason=stop_reason
            or (
                "executed trace failed independent validation"
                if validation_problems
                else None
            ),
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _intents_at(paths, order, t):
        """{robot_id: (from_cell, to_cell)} for timestep t."""
        intents = {}
        for robot_id in order:
            tp = paths[robot_id]
            frm = tp.position_at(t - 1)
            to = tp.position_at(t)
            intents[robot_id] = (frm, to)
        return intents

    def _validate_joint_step(self, order, intents):
        """PHASE 1. Returns (ok, reason, vertex_conflicts, edge_conflicts)."""
        warehouse = self.simulator.warehouse

        # per-robot feasibility
        for robot_id in order:
            frm, to = intents[robot_id]
            robot = self.simulator.get_robot(robot_id)

            if robot.status is RobotStatus.OFFLINE:
                return False, f"robot '{robot_id}' went offline", 0, 0

            if robot.position != frm:
                return (
                    False,
                    f"robot '{robot_id}' is at {robot.position}, expected {frm}",
                    0, 0,
                )

            if frm == to:  # WAIT
                if not warehouse.is_traversable(frm):
                    return (
                        False,
                        f"robot '{robot_id}' cannot wait at {frm}: an obstacle "
                        f"appeared there",
                        0, 0,
                    )
                continue

            # MOVE
            if not warehouse.in_bounds(to):
                return False, f"robot '{robot_id}' move to {to} is out of bounds", 0, 0
            if not warehouse.is_traversable(to):
                return (
                    False,
                    f"robot '{robot_id}' next cell {to} is blocked by an "
                    f"obstacle that appeared after planning",
                    0, 0,
                )
            if not frm.is_adjacent_to(to):
                return (
                    False,
                    f"robot '{robot_id}' move {frm}->{to} is not one step",
                    0, 0,
                )
            if not robot.has_battery_for_move():
                return (
                    False,
                    f"robot '{robot_id}' has {robot.battery_level:.1f}% battery, "
                    f"not enough to move",
                    0, 0,
                )

        # joint: vertex conflict
        destinations: Dict[Position, str] = {}
        for robot_id in order:
            _, to = intents[robot_id]
            if to in destinations:
                return (
                    False,
                    f"vertex conflict: '{destinations[to]}' and '{robot_id}' "
                    f"both want {to}",
                    1, 0,
                )
            destinations[to] = robot_id

        # joint: edge / swap conflict
        moves = {
            robot_id: intents[robot_id]
            for robot_id in order
            if intents[robot_id][0] != intents[robot_id][1]
        }
        for robot_id, (frm, to) in moves.items():
            for other_id, (o_frm, o_to) in moves.items():
                if other_id <= robot_id:
                    continue
                if frm == o_to and to == o_frm:
                    return (
                        False,
                        f"edge/swap conflict: '{robot_id}' and '{other_id}' "
                        f"trade {frm}<->{to}",
                        0, 1,
                    )

        return True, None, 0, 0

    def _preflight_failure(self, not_executed, paths, reason, elapsed):
        return CoordinatedExecutionResult(
            success=False,
            executed_robot_ids=[],
            not_executed_robot_ids=not_executed + sorted(paths),
            execution_time_ms=elapsed,
            failure_reason=f"pre-flight check failed: {reason}",
        )
