"""Autonomous Multi-Robot Fulfillment & Dynamic Path Planning System.

A deterministic, testable 2D warehouse simulation foundation.

Sub-packages
------------
robotics.warehouse   : the map (grid, warehouse locations, obstacles)
robotics.robots      : the robots (attributes, status, single-step movement)
robotics.tasks       : fulfillment tasks (pickup -> dropoff jobs)
robotics.simulation  : the coordinator (simulator, sample scenario, renderer)

Nothing in this package performs path planning. It models the world and
validates single, externally-supplied moves.
"""

__version__ = "0.1.0"
