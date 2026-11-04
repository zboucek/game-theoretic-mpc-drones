# SPDX-License-Identifier: MIT
# Copyright 2026 Zdeněk Bouček
"""One blocked pursuit-evasion call using the article's response planner."""
import numpy as np
from game_theory_mpc import BlockedGameMPC, AdversarialWeights

states = np.array([[2.0, 2.0, 1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
planner = BlockedGameMPC(
    2,
    (1, 1, 2),
    num_pursuers=1,
    velocity_weight=5.0,
    terminal_scale=1.0,
    adversarial_weights=AdversarialWeights(goal=100.0, evasion=10.0),
    input_bound=2.0,
    speed_bound=2.0,
    jerk_bound=6.0,
    response_sweeps=5,
)
controls, predictions, info = planner.solve(states, [[3.0, 1.0, 1.0]])
print("Pursuer/evader first accelerations [m/s²]:", controls[:, 0])
print("Feasible candidate:", info["feasible"])
assert controls.shape == (2, 4, 3) and np.isfinite(controls).all()
# Check an execution envelope before applying inputs, then commit actual inputs.
