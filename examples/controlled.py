# SPDX-License-Identifier: MIT
# Copyright 2026 Zdeněk Bouček
"""Short three-agent controlled game with independent diagnostics."""
import numpy as np
from experiments.equilibrium_selection.protocol import (
    Protocol,
    nominal,
    initial_controls,
)
from experiments.equilibrium_selection.solvers import Solvers

protocol = Protocol(horizon=2, max_evaluations=100)
states, targets = nominal("head_on")
solvers = Solvers(protocol)
assert solvers.derivative_check(states, targets)["passed"]
for method in ("gne", "vgne", "potential"):
    result = solvers.solve(
        method,
        states,
        targets,
        initial_controls(states, "head_on", "braking", protocol),
    )
    print(method, "numerically accepted:", result["validation"]["accepted"])
    assert result["validation_error"] is None and np.isfinite(result["controls"]).all()
