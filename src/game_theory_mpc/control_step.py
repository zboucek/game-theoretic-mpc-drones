# SPDX-License-Identifier: MIT
# Copyright 2026 Zdeněk Bouček
"""Solve, retry and admit one nominal plant update using the flight policy."""
import numpy as np
from experiments.equilibrium_mpc_execution import first_interval_extrema
from .experiment_model import make_model, check_model


def control_step(model, recovery, config, execution, method, states, targets):
    previous = model._previous_controls.copy()
    controls, prediction, info = model.solve(states, targets)
    if execution.reserve_retry and not info["feasible"]:
        tightened = info
        if recovery is None:
            recovery = make_model(config, method, execution, True)
        recovery._warm = np.asarray(info["blocked_controls"]).copy()
        recovery._solved = True
        recovery.commit_controls(previous)
        controls, prediction, info = recovery.solve(states, targets)
        info["tightened_attempt"] = tightened
        model._warm = recovery._warm.copy()
    used = recovery if "tightened_attempt" in info else model
    independent = check_model(used, info["blocked_controls"], states, targets)
    extent = first_interval_extrema(states, controls[:, 0], config.dt)
    speed = max(
        np.linalg.norm(states[:, 3:], axis=1).max(),
        np.linalg.norm(prediction[:, 0, 3:], axis=1).max(),
    )
    admitted = bool(
        info["feasible"]
        and extent["first_interval_minimum_altitude_m"]
        >= execution.physical_altitude_bounds[0] - 1e-8
        and extent["first_interval_maximum_altitude_m"]
        <= execution.physical_altitude_bounds[1] + 1e-8
        and speed <= execution.physical_speed_bound + 1e-8
        and (
            not model.hard_pairs
            or extent["first_interval_minimum_distance_m"]
            >= execution.physical_separation - 1e-8
        )
    )
    record = dict(
        info,
        states=states,
        targets=targets,
        controls=controls,
        prediction=prediction,
        previous_controls=previous,
        first_interval=extent,
        independent=independent,
        applied=admitted,
        equilibrium_accepted=bool(info.get("validation", {}).get("accepted", False)),
    )
    return controls, prediction, record, recovery
