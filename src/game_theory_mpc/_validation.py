# SPDX-License-Identifier: MIT
# Copyright 2026 Zdeněk Bouček
"""Validation shared by the public controllers."""
import numpy as np


def positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def nonnegative_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def positive_number(value, name):
    if not np.isscalar(value) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def states_and_targets(states, targets, num_agents, num_targets=None):
    num_targets = num_agents if num_targets is None else num_targets
    states = np.asarray(states, dtype=float)
    if states.shape != (num_agents, 6) or not np.all(np.isfinite(states)):
        raise ValueError(f"current_states must be finite with shape ({num_agents}, 6)")
    if len(targets) != num_targets:
        raise ValueError(f"expected {num_targets} target positions")
    result = []
    for target in targets:
        target = np.asarray(target, dtype=float)
        if target.shape not in {(3,), (6,)} or not np.all(np.isfinite(target)):
            raise ValueError("each target must be a finite (3,) or (6,) vector")
        result.append(target)
    return states, result


def obstacles_array(obstacles, max_obstacles=None):
    if obstacles is None:
        return None
    arr = np.asarray(obstacles, dtype=float)
    if arr.size == 0:
        return np.empty((0, 3))
    if arr.ndim != 2 or arr.shape[1] < 3 or not np.all(np.isfinite(arr)):
        raise ValueError("obstacles must be finite rows containing at least x, y, z")
    if max_obstacles is not None and len(arr) > max_obstacles:
        raise ValueError("obstacle count exceeds max_obstacles")
    return arr
