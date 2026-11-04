# SPDX-License-Identifier: MIT
# Copyright 2026 Zdeněk Bouček
"""Construct planners and independently evaluate their configured models."""
import numpy as np
from .blocked_game_mpc import BlockedGameMPC, CooperativeWeights, AdversarialWeights
from .cooperative_equilibrium_mpc import (
    CooperativeEquilibriumMPC,
    CooperativeDerivatives,
)


def make_model(config, method, execution=None, recovery=False):
    c = config
    options = dict(
        num_pursuers=c.pursuers,
        dt=c.dt,
        obstacles=c.obstacles,
        velocity_weight=c.velocity_weight,
        terminal_scale=c.terminal_scale,
        separation=c.separation,
        input_bound=c.input_bound,
        max_iterations=c.max_iterations,
        response_sweeps=c.response_sweeps,
        adversarial_weights=AdversarialWeights(**c.adversarial_weights),
        cooperative_weights=CooperativeWeights(
            c.position_weight, c.repulsion_weight, c.effort_weight
        ),
        speed_bound=min(
            c.speed_bound + execution.recovery_speed_margin,
            execution.physical_speed_bound,
        )
        if recovery
        else c.speed_bound,
        jerk_bound=execution.emergency_jerk_bound if recovery else c.jerk_bound,
        altitude_bounds=execution.physical_altitude_bounds
        if recovery
        else c.altitude_bounds,
    )
    if method in ("gne", "vgne"):
        return CooperativeEquilibriumMPC(
            len(c.states), c.blocks, method=method, max_sweeps=c.max_sweeps, **options
        )
    return BlockedGameMPC(len(c.states), c.blocks, **options)


def check_model(model, controls, states, targets, derivatives=False):
    """Check NumPy costs/constraints and optional independent analytic derivatives."""
    u = np.asarray(controls).ravel()
    p = model.parameters(states, targets)
    numeric = model.evaluate_numpy(u, states, targets)
    np.testing.assert_allclose(
        model.costs(u, p), numeric["costs"], atol=2e-8, rtol=2e-9
    )
    np.testing.assert_allclose(model.constraints(u, p), numeric["g"], atol=2e-10)
    result = dict(
        cost_max_error=float(np.max(np.abs(model.costs(u, p) - numeric["costs"]))),
        constraint_max_error=float(
            np.max(np.abs(model.constraints(u, p) - numeric["g"]))
        ),
    )
    if derivatives and not model.adversarial:
        analytic = CooperativeDerivatives(model).evaluate(u, states, targets)
        np.testing.assert_allclose(
            model.potential_gradient(u, p), analytic["gradient"], atol=2e-8, rtol=2e-8
        )
        np.testing.assert_allclose(
            model.constraint_jacobian(u, p), analytic["jac"], atol=2e-9
        )
        result["derivatives_checked"] = True
    return result
