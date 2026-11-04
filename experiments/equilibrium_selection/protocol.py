# SPDX-License-Identifier: MIT
# Copyright 2026 Zdeněk Bouček
"""Deterministic short-horizon study design: article IV-A.

This tracking/effort/repulsion game deliberately omits the flight controller's
velocity and terminal terms. Geometry, perturbations and common starts are
fixed here; model.py and solvers.py implement evaluation and optimization."""
from dataclasses import asdict, dataclass
from itertools import combinations

import numpy as np


@dataclass(frozen=True)
class Protocol:
    agents: int = 3
    horizon: int = 5
    dt: float = 0.1
    bound: float = 5.0
    separation: float = 0.5
    effort_weight: float = 0.1
    repulsion_offset: float = 0.01
    solver_tolerance: float = 1e-12
    max_evaluations: int = 1000
    solve_limit_s: float = 30.0
    kkt_tolerance: float = 1e-6
    feasibility_tolerance: float = 1e-8
    activity_tolerance: float = 1e-6
    deviation_iterations: int = 200
    deviation_limit_s: float = 5.0
    unique_control_tolerance: float = 1e-5
    seed: int = 20261023

    def as_dict(self):
        return asdict(self)


PROTOCOL = Protocol()
METHODS = ("gne", "vgne", "potential")
GEOMETRIES = ("head_on", "crossing", "three_way")
INITIALIZATIONS = ("zero", "braking", "lateral_plus", "lateral_minus",
                   "vertical_plus", "vertical_minus")
LOOP_STARTS = ("braking", "lateral_plus")
PAIRS = tuple(combinations(range(3), 2))


def minimum_distance(states):
    return float(min(np.linalg.norm(states[i, :3] - states[j, :3]) for i, j in PAIRS))


def nominal(geometry):
    if geometry == "head_on":
        states = [[-.3, 0, 1, .2, 0, 0], [.3, 0, 1, -.1, 0, 0],
                  [0, 2, 1, 0, 0, 0]]
        targets = [[10, 0, 1], [-6, 0, 1], [0, 2, 1]]
    elif geometry == "crossing":
        states = [[-.45, 0, 1, .2, 0, 0], [0, -.45, 1.1, 0, .15, 0],
                  [2, 2, 1.4, 0, 0, 0]]
        targets = [[8, 0, 1], [0, 8, 1.1], [2, 2, 1.4]]
    elif geometry == "three_way":
        angles = np.deg2rad([0, 120, 240])
        radial = np.column_stack([np.cos(angles), np.sin(angles)])
        states = np.zeros((3, 6))
        states[:, :2], states[:, 2] = .45 * radial, [.95, 1, 1.05]
        states[:, 3:5] = -.15 * radial
        targets = np.column_stack([-6 * radial, states[:, 2]])
    else:
        raise ValueError(f"Unknown geometry: {geometry}")
    return np.array(states, dtype=float), np.array(targets, dtype=float)


def instance(geometry, perturbation=0):
    """Rejection draws stay in one SeedSequence stream; nominal is index zero."""
    states, targets = nominal(geometry)
    draws = 0
    if perturbation:
        if perturbation not in range(1, 5):
            raise ValueError("Perturbation must be in 0..4")
        rng = np.random.default_rng(np.random.SeedSequence(
            [PROTOCOL.seed, GEOMETRIES.index(geometry), perturbation]))
        while True:
            draws += 1
            perturbed = states.copy()
            perturbed[:, :3] += rng.normal(0, .01, (3, 3))
            perturbed[:, 3:] += rng.normal(0, .02, (3, 3))
            if minimum_distance(perturbed) >= .55:
                states = perturbed
                break
            if draws == 10000:
                raise RuntimeError("Perturbation rejection sampler exhausted")
    return {"geometry": geometry, "perturbation": perturbation,
            "states": states.tolist(), "targets": targets.tolist(),
            "seed_sequence": [PROTOCOL.seed, GEOMETRIES.index(geometry), perturbation],
            "rejection_draws": draws, "minimum_initial_distance_m": minimum_distance(states)}


def initial_controls(states, geometry, name, config=PROTOCOL):
    if name not in INITIALIZATIONS:
        raise ValueError(name)
    controls = np.zeros((3, config.horizon, 3))
    if name == "zero":
        return controls.ravel()
    controls[:, 0] = -np.asarray(states)[:, 3:] / config.dt
    # A COMMON horizontal axis perpendicular to nominal agent 0's route.
    # It is fixed across perturbations; the player signs are (+1,-1,0).
    base, targets = nominal(geometry)
    direction = targets[0, :2] - base[0, :2]
    lateral = np.array([-direction[1], direction[0], 0.]) / np.linalg.norm(direction)
    if name.startswith("lateral") or name.startswith("vertical"):
        axis = lateral if name.startswith("lateral") else np.array([0., 0., 1.])
        sign = 1 if name.endswith("plus") else -1
        controls += .5 * sign * np.array([1, -1, 0])[:, None, None] * axis
    return np.clip(controls, -config.bound, config.bound).ravel()


def shift_controls(controls, config=PROTOCOL):
    controls = np.asarray(controls).reshape(3, config.horizon, 3)
    return np.concatenate([controls[:, 1:], np.zeros((3, 1, 3))], axis=1).ravel()


def plan(phase):
    count, steps = (1, 20) if phase == "pilot" else (5, 60)
    return {"phase": phase, "static_attempts": 3 * count * 6 * 3,
            "methods": METHODS, "initializations": INITIALIZATIONS, "loop_starts": LOOP_STARTS,
            "protocol": PROTOCOL.as_dict(),
            "episodes": 3 * count * 2 * 3, "updates_per_episode": steps,
            "maximum_loop_attempts": 3 * count * 2 * 3 * steps,
            "instances": [instance(g, p) for g in GEOMETRIES for p in range(count)]}
