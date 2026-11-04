# SPDX-License-Identifier: MIT
# Copyright 2026 Zdeněk Bouček
"""One vGNE update of the article's blocked flight planner, without physics.

Run from the package root: python examples/cooperative.py.
This demonstrates the API; full flights use python -m game_theory_mpc.
"""
import jax
import numpy as np
from game_theory_mpc import CooperativeEquilibriumMPC, CooperativeWeights

jax.config.update("jax_enable_x64", True)
states = np.array(
    [[-0.3, 0, 1, 0.2, 0, 0], [0.3, 0, 1, -0.1, 0, 0], [0, 2, 1, 0, 0, 0]]
)
targets = np.array([[10.0, 0, 1], [-6.0, 0, 1], [0.0, -2, 1]])
planner = CooperativeEquilibriumMPC(
    3,
    (1, 1, 2, 4, 6, 6),
    method="vgne",
    cooperative_weights=CooperativeWeights(1.0, 1.0, 0.1),
    velocity_weight=1.0,
    terminal_scale=1.0,
    input_bound=2.0,
    speed_bound=2.0,
    jerk_bound=6.0,
)
planner.prepare(states, targets)
controls, predictions, info = planner.solve(states, targets)
print("Model-feasible:", info["feasible"])
print("Numerically accepted:", info["validation"]["accepted"])
print("First accelerations [m/s²]:\n", controls[:, 0])
assert controls.shape == (3, 20, 3) and predictions.shape == (3, 20, 6)
assert np.isfinite(controls).all()
# No input is committed here: this example does not execute a plant update.
# An application checks its execution policy, applies an admitted acceleration,
# then calls planner.commit_controls(the_acceleration_actually_applied).
