# SPDX-License-Identifier: MIT
# Copyright 2026 Zdeněk Bouček
"""Nominal first-interval execution checks for article IV-B and V-A/B.

The flight and weight drivers call first_interval_extrema to check motion
between prediction nodes. Nominal feasibility, numerical equilibrium
acceptance and the decision to execute an input remain separate.
"""
import numpy as np
from dataclasses import replace
from itertools import combinations

from experiments.equilibrium_selection.protocol import PAIRS, PROTOCOL


def first_interval_extrema(states, acceleration, dt=PROTOCOL.dt):
    states, acceleration = np.asarray(states), np.asarray(acceleration)
    minimum_distance = float("inf")
    for i, j in combinations(range(len(states)), 2):
        position = states[i, :3]-states[j, :3]
        velocity = states[i, 3:]-states[j, 3:]
        change = acceleration[i]-acceleration[j]
        squared = np.array([position@position, 2*position@velocity,
                            velocity@velocity+position@change, velocity@change, .25*change@change])
        derivative = np.trim_zeros(squared[1:]*np.arange(1, 5), "b")
        roots = np.polynomial.polynomial.polyroots(derivative) if len(derivative) > 1 else []
        candidates = [0., dt, *(float(r.real) for r in roots if abs(r.imag) < 1e-9 and 0 < r.real < dt)]
        minimum_distance = min(minimum_distance, float(np.sqrt(max(
            0., np.min(np.polynomial.polynomial.polyval(candidates, squared))))))
    heights = []
    for state, control in zip(states, acceleration):
        candidates = [0., dt]
        if abs(control[2]) > 1e-14:
            stationary = -state[5]/control[2]
            if 0 < stationary < dt:
                candidates.append(stationary)
        heights.extend(state[2]+state[5]*t+.5*control[2]*t*t for t in candidates)
    return dict(first_interval_minimum_distance_m=minimum_distance,
                first_interval_minimum_altitude_m=float(min(heights)),
                first_interval_maximum_altitude_m=float(max(heights)))


def execution_validation(record, policy="feasible", envelope=None, config=PROTOCOL):
    if policy not in ("strict", "feasible"):
        raise ValueError("Unknown MPC execution policy")
    controls = np.asarray(record["controls"], dtype=float).reshape(3, config.horizon, 3)
    states, prediction = np.asarray(record["states"], dtype=float), np.asarray(record["prediction"], dtype=float)
    finite = bool(all(np.isfinite(x).all() for x in (controls, states, prediction)))
    equilibrium = bool(record["validation"].get("accepted", False))
    result = dict(policy=policy, finite_primal=finite, equilibrium_accepted=equilibrium, apply=False)
    if not finite:
        result["reason"] = "nonfinite_primal"
        return result
    extrema = first_interval_extrema(states, controls[:, 0], config.dt)
    distances = np.stack([np.linalg.norm(prediction[i, :, :3]-prediction[j, :, :3], axis=1)
                          for i, j in PAIRS])
    height_low = envelope.minimum_altitude if envelope else -np.inf
    height_high = envelope.maximum_altitude if envelope else np.inf
    tolerance = config.feasibility_tolerance
    inputs_feasible = bool(np.max(np.abs(controls)) <= config.bound+tolerance)
    horizon_feasible = bool(distances.min() >= config.separation-tolerance
                            and prediction[:, :, 2].min() >= height_low-tolerance
                            and prediction[:, :, 2].max() <= height_high+tolerance)
    interval_feasible = bool(extrema["first_interval_minimum_distance_m"] >= config.separation-tolerance
                             and extrema["first_interval_minimum_altitude_m"] >= height_low-tolerance
                             and extrema["first_interval_maximum_altitude_m"] <= height_high+tolerance)
    primal_feasible = inputs_feasible and horizon_feasible and interval_feasible
    apply = primal_feasible and (equilibrium or policy == "feasible")
    result.update(extrema, inputs_feasible=inputs_feasible, horizon_physically_feasible=horizon_feasible,
                  first_interval_physically_feasible=interval_feasible,
                  nominal_primal_feasible=primal_feasible, apply=bool(apply),
                  reason=("equilibrium" if equilibrium else "feasible_approximation") if apply
                         else ("primal_infeasible" if not primal_feasible else "equilibrium_not_converged"))
    return result


def solve_with_budget(solvers, method, states, targets, guess, policy, envelope,
                      initial_evaluations=1000):
    """Expand the solve budget only if the initial candidate cannot be applied."""
    maximum = solvers.config
    first = replace(maximum, max_evaluations=min(initial_evaluations, maximum.max_evaluations),
                    solve_limit_s=min(30., maximum.solve_limit_s))
    try:
        solvers.config = first
        candidate = solvers.solve(method, states, targets, guess)
    finally:
        solvers.config = maximum
    decision = execution_validation(candidate, policy, envelope, maximum)
    stages = [dict(max_evaluations=first.max_evaluations, solve_limit_s=first.solve_limit_s,
                   solver=candidate["solver"], execution_validation=decision)]
    if not decision["apply"] and first.max_evaluations < maximum.max_evaluations:
        print(f"{method}: initial candidate is inadmissible; "
              f"retrying with up to {maximum.max_evaluations} evaluations", flush=True)
        extended = solvers.solve(method, states, targets, guess)
        stages.append(dict(max_evaluations=maximum.max_evaluations, solve_limit_s=maximum.solve_limit_s,
                           solver=extended["solver"],
                           execution_validation=execution_validation(extended, policy, envelope, maximum)))
        extended["preliminary_attempt"] = candidate
        candidate = extended
    candidate["budget_stages"] = stages
    candidate["total_solve_seconds"] = sum(stage["solver"]["solve_seconds"] for stage in stages)
    return candidate
