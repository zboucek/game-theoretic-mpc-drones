# SPDX-License-Identifier: MIT
# Copyright 2026 Zdeněk Bouček
"""Two-agent tracking with the fast linear controller; no simulator required."""
import numpy as np
from game_theory_mpc import LinearNashMPCController

planner = LinearNashMPCController(2, horizon=5)
states = np.array([[0.0, 0.0, 0.5, 0.0, 0.0, 0.0], [1.0, 0.0, 0.5, 0.0, 0.0, 0.0]])
targets = np.array([[0.0, 0.2, 1.0], [1.0, 0.2, 1.0]])
for _ in range(20):
    _, predicted, info = planner.solve(states, targets)
    states = np.array([p[0] for p in predicted])
print("Final positions [m]:", states[:, :3])
assert np.all(np.isfinite(states))
